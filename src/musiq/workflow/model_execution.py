"""Execution logic for workflow models."""

from __future__ import annotations

import time
import numpy as np
from dataclasses import asdict
from typing import Any
from pathlib import Path

from musiq.analysis.state_utils import final_density_matrix, state_fidelity
from musiq.workflow.contracts import (
    AnalyserConfig,
    CircuitConfig,
    DeviceConfig,
    SolverConfig,
    Task,
    compose_workflow_task,
)
from musiq.workflow.output import resolve_writable_out_dir
from musiq.workflow.planner import build_execution_plan
from musiq.workflow.planner_study import StudyPlanner, StudyPlan, StudySample
from musiq.workflow.stages import (
    parse_compile_lower_model,
    run_analysis_stage,
    run_decode_stage,
    run_engine_stage,
)
from musiq.schemas.results import (
    AnalysisScope,
    CaseAnalysis,
    ComprehensiveAnalysis,
    ModelAnalysis,
    ParameterAxis,
    ParameterValues,
    ParametricAnalysis,
    ResultRef,
    RunProvenance,
    RunResult,
    MetricSweepValues,
)
from musiq.schemas.model import RunStatus, RunIdentity, ModelRun, RunArtifacts

from musiq.workflow.model_utils import (
    compact_runtime_details,
    public_value,
    safe_study_token,
    study_name,
    effective_analyser_payload,
    require_solver_id,
    require_analyser_id,
)
from musiq.common.id_generator import IDGenerator


def _merge_param_bindings(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any] | None:
    merged = dict(base or {})
    merged.update(dict(override or {}))
    return merged or None


def _extract_case_metric_terminal(metrics: dict[str, Any] | None, target_name: str) -> float:
    metric_map = dict(metrics or {})
    target = str(target_name).strip()
    if not target:
        return 0.0

    if target in metric_map:
        series = metric_map[target]
        values = getattr(series, "values", None)
        if isinstance(values, list) and values:
            return float(values[-1])
        if isinstance(values, dict):
            first_key = next(iter(values), None)
            if first_key is not None:
                val_list = list(values.get(first_key, []) or [])
                if val_list:
                    return float(val_list[-1])
        return 0.0

    lowered = target.lower()
    if lowered.startswith("final_p"):
        label = target[len("final_P") :]
        population = metric_map.get("population")
        values = getattr(population, "values", None)
        if isinstance(values, dict):
            state_values = list(values.get(label, []) or [])
            if state_values:
                return float(state_values[-1])
        return 0.0
    if lowered == "final_leakage":
        return _extract_case_metric_terminal(metric_map, "leakage")
    if lowered == "final_coherence_01":
        return _extract_case_metric_terminal(metric_map, "coherence_01")
    return 0.0


def _extract_final_fidelity(run_result: Any) -> float:
    param_values = dict(getattr(getattr(run_result, "parameters", None), "values", {}) or {})
    theta = param_values.get("theta")
    if theta is None:
        theta = dict(getattr(run_result, "runtime_metadata", {}) or {}).get("param_bindings", {}).get("theta")
    if theta is None:
        return 0.0

    trajectory = next(iter(dict(getattr(run_result, "trajectories", {}) or {}).values()), None)
    if trajectory is None:
        return 0.0

    try:
        theta_val = float(theta)
        rho = final_density_matrix(trajectory)
        target = np.array(
            [
                np.cos(theta_val / 2.0),
                -1j * np.sin(theta_val / 2.0),
                0.0,
            ],
            dtype=complex,
        )
        return float(state_fidelity(rho, target))
    except Exception:
        return 0.0

def run_one_solver_study(
    model: Any,
    *,
    solver_id: str,
    solver_cfg: SolverConfig,
    study: dict[str, Any],
    study_index: int | None,
    total_studies: int,
    tag: str | None = None,
) -> list[str]:
    """Orchestrate compilation and execution for one study step across all applicable runs."""
    # 1. Plan the expansion to find all compilation units (run_ids) that use this solver
    plan = StudyPlanner.plan(model)
    
    # Find all run_ids in the plan that belong to this solver
    target_run_ids = [
        rid for rid, samples in plan.run_groups.items()
        if samples and samples[0].solver_id == solver_id
    ]
    
    if not target_run_ids:
        raise RuntimeError(f"Could not resolve run_ids for solver {solver_id} from study plan")

    # 2. Process each run group (e.g., different tasks using the same solver)
    for rid in target_run_ids:
        if rid not in model.runs:
            # Get the first sample to use as a template for compilation.
            sample = plan.run_groups[rid][0]
            model.runs[rid] = execute_compilation_unit(model, sample, run_id=rid, tag=tag)

        run_obj = model.runs[rid]
        
        # 3. Execute all samples in this compilation unit
        for sample in plan.run_groups[rid]:
            run_sample(model, run_obj, sample)
        
        # Update run identity to reflect the specific study step being run
        run_obj.identity.study_index = study_index
        run_obj.identity.study_name = study.get("name") or IDGenerator.next_study_name(run_obj)
    
    return target_run_ids


def get_study_entries(solver_cfg: SolverConfig) -> list[tuple[int | None, dict[str, Any]]]:
    entries = [dict(step) for step in list(solver_cfg.study or []) if isinstance(step, dict)]
    if not entries:
        return [(None, {})]
    return [(idx, step) for idx, step in enumerate(entries)]

def clone_solver_cfg_with_single_study(
    solver_cfg: SolverConfig,
    *,
    study: dict[str, Any],
) -> SolverConfig:
    return SolverConfig(
        backend=type(solver_cfg.backend)(**asdict(solver_cfg.backend)),
        run=type(solver_cfg.run)(**asdict(solver_cfg.run)),
        frame=type(solver_cfg.frame)(**asdict(solver_cfg.frame)),
        study=[dict(study)] if study else None,
    )

def find_run_id(
    model: Any,
    *,
    solver_id: str,
    study_name_val: str | None = None,
) -> str | None:
    candidates: list[tuple[str, ModelRun]] = [
        (run_id, run_obj)
        for run_id, run_obj in model.runs.items()
        if run_obj.identity.solver_id == solver_id and run_obj.results
    ]
    if study_name_val is None:
        if len(candidates) == 1:
            return candidates[0][0]
        return None
    wanted = str(study_name_val).strip()
    for run_id, run_obj in candidates:
        if str(run_obj.identity.study_name or '').strip() == wanted:
            return run_id
    return None

def _nearest_centroid(point: complex, centroids: dict[str, complex]) -> str:
    if not centroids:
        return ""
    return min(centroids, key=lambda label: abs(point - centroids[label]))

def _get_study_label(bundle: Any, analysis: ModelAnalysis) -> str | None:
    study_name_val = str(bundle.identity.study_name or "").strip()
    if study_name_val:
        return study_name_val

    runtime_task = bundle.runtime_task
    if runtime_task is not None:
        task_input = getattr(runtime_task, "input", None)
        study_steps = list(getattr(task_input, "study", []) or []) if task_input is not None else []
        if study_steps:
            prep_state = dict(study_steps[0].get("prep_state", {}) or {})
            prep_label = str(prep_state.get("label", "") or "").strip()
            if prep_label:
                return prep_label
    
    iq_output = analysis.output.iq
    if iq_output:
        iq_payload = iq_output if isinstance(iq_output, dict) else asdict(iq_output)
        cm = iq_payload.get("confusion_matrix", {})
        labels = list(cm.get("labels", []) or [])
        if labels:
            return str(labels[0]).strip()
    
    return None

def build_multi_study_iq_summary(model: Any, analysis_items: list[tuple[Any, ModelAnalysis]]) -> dict[str, Any] | None:
    if len(analysis_items) <= 1:
        return None
    centroids: dict[str, complex] = {}
    clouds: dict[str, list[list[float]]] = {}
    study_map: dict[str, str] = {}
    noise_sigmas: list[float] = []
    for bundle, analysis in analysis_items:
        iq_output = analysis.output.iq
        if not iq_output:
            continue
        iq_payload = iq_output if isinstance(iq_output, dict) else asdict(iq_output)
        label = _get_study_label(bundle, analysis)
        if not label:
            continue
        centroid_map = dict(iq_payload.get('centroids', {}) or {})
        centroid_raw = None
        if label in centroid_map:
            centroid_raw = centroid_map.get(label)
        elif centroid_map:
            centroid_raw = next(iter(centroid_map.values()))
        
        if centroid_raw is None:
            centroids[label] = complex(0.0, 0.0)
            continue
        
        try:
            raw_val = np.asarray(centroid_raw)
            if raw_val.size < 2:
                centroids[label] = complex(0.0, 0.0)
                continue
            centroids[label] = complex(float(raw_val[0]), float(raw_val[1]))
        except (TypeError, ValueError, IndexError):
            centroids[label] = complex(0.0, 0.0)
            continue
        cloud_source = dict(iq_payload.get('synthetic_clouds', {}) or {})
        raw_cloud = cloud_source.get(label)
        if raw_cloud is None and cloud_source:
            raw_cloud = next(iter(cloud_source.values()))
        clouds[label] = [
            [float(point[0]), float(point[1])]
            for point in list(raw_cloud or [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        study_map[label] = str(bundle.identity.study_name or '')
        try:
            noise_sigmas.append(float(iq_payload.get('noise_sigma', 0.0) or 0.0))
        except Exception:
            pass
    labels = list(centroids.keys())
    if not labels:
        return None
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for i, label in enumerate(labels):
        for point_raw in clouds.get(label, []):
            point = complex(float(point_raw[0]), float(point_raw[1]))
            pred = _nearest_centroid(point, centroids)
            if pred in labels:
                confusion[i, labels.index(pred)] += 1
    assignment_fidelity = float(np.trace(confusion) / max(1, confusion.sum())) if confusion.size else 0.0
    pairwise_distances = [
        abs(centroids[a] - centroids[b])
        for i, a in enumerate(labels)
        for b in labels[i + 1 :]
    ]
    noise_sigma = float(np.mean(noise_sigmas)) if noise_sigmas else 0.0
    cluster_separation = float(min(pairwise_distances) / max(noise_sigma, 1.0e-12)) if pairwise_distances else 0.0
    snr = float((np.mean(pairwise_distances) if pairwise_distances else 0.0) / max(2.0 * noise_sigma, 1.0e-12))
    return {
        "schema_version": "1.0",
        "labels": labels,
        "centroids": {label: [float(val.real), float(val.imag)] for label, val in centroids.items()},
        "synthetic_clouds": clouds,
        "confusion_matrix": {"labels": labels, "values": confusion.astype(int).tolist()},
        "assignment_fidelity": assignment_fidelity,
        "cluster_separation": cluster_separation,
        "snr": snr,
        "study_map": study_map,
    }

def compose_runtime_task(
    model: Any,
    *,
    circuit_cfg: CircuitConfig,
    solver_cfg: SolverConfig,
    device_cfg: DeviceConfig,
    pulse_cfg: Any,
    analyser_cfg: AnalyserConfig | None,
    backend_source: str | None = None,
) -> Task:
    return compose_workflow_task(
        target=model.config.target,
        features=model.config.features,
        output=model.config.output,
        tags=model.config.tags,
        circuit_cfg=circuit_cfg,
        solver_cfg=solver_cfg,
        device_cfg=device_cfg,
        analyser_cfg=analyser_cfg,
        model_pulse=pulse_cfg,
        backend_source=backend_source,
    )

def execute_compilation_unit(
    model: Any,
    sample: StudySample,
    *,
    run_id: str | None = None,
    tag: str | None = None,
) -> ModelRun:
    """Handle the STAGE_PARSE part and create a ModelRun compilation unit."""
    # Determine which config resources to use
    circuit_cfg = model.config.circuits[sample.circuit_id]
    device_cfg = model.config.devices[sample.device_id]
    pulse_cfg = model.config.pulses[sample.pulse_id]
    solver_cfg = model.config.solvers[sample.solver_id]
    
    # Compose runtime task for compilation
    # For compilation, we use the default values; ParametricValue is handled at execution
    task = compose_runtime_task(
        model, 
        circuit_cfg=circuit_cfg,
        solver_cfg=solver_cfg, 
        device_cfg=device_cfg,
        pulse_cfg=pulse_cfg,
        analyser_cfg=None # Analysers are bound at analysis stage
    )
    
    # Use auto-incrementing run_id to ensure uniqueness unless the planner reserved one.
    run_id = run_id or IDGenerator.next_run_id(model, tag=tag)
    out = resolve_writable_out_dir(Path(task.output.out_dir))
    model.out_dir = str(out)
    
    started_at = time.perf_counter()
    parsed = parse_compile_lower_model(
        qasm_text=task.input.qasm_text,
        backend_path=task.input.backend_path,
        backend_config=task.input.backend_config,
        out=out,
        device=(task.input.device_model or task.input.device),
        pulse=task.input.pulse,
        frame=task.input.frame,
        analyser=task.input.analyser,
        study=task.input.study,
        schedule_policy=task.input.schedule_policy,
        reset_feedback_policy=task.input.reset_feedback_policy,
        noise=task.input.noise,
        solver_run={
            'dt_s': task.run.dt_s,
            't_end_s': task.run.t_end_s,
            't_padding_s': task.run.t_padding_s,
            'seed': task.run.seed,
            'ntraj': task.run.mcwf_ntraj,
            'qutip_options': task.run.qutip_options,
            'native_options': task.run.native_options,
            'backend_options': task.run.backend_options,
            'one_over_f_components': task.run.one_over_f_components,
        },
        solver_mode=task.run.solver_mode,
        param_bindings=task.input.param_bindings,
        persist_artifacts=task.output.persist_artifacts,
    )
    
    timings = {'build': time.perf_counter() - started_at}
    timings.update(parsed.get('timings', {}))

    run_obj = ModelRun(
        identity=RunIdentity(
            run_id=run_id,
            solver_id=sample.solver_id,
            study_name=None, # Handled by samples/results
            study_index=None,
        ),
        runtime_task=task,
        status=RunStatus.RUNNING,
        started_at=time.time(),
        artifacts=RunArtifacts(
            circuit=parsed['circuit'],
            normalized_circuit=parsed['normalized'],
            model_spec=parsed['model_spec'],
            pulse_ir=parsed['pulse_ir'],
            executable_model=parsed['executable'],
            compile_report=public_value(parsed['compile_report']),
            timings=timings,
        )
    )
    return run_obj

def run_sample(
    model: Any,
    run_obj: ModelRun,
    sample: StudySample,
) -> str:
    """Execute a single numerical sample within a compilation unit."""
    # Use auto-incrementing param_id
    param_id = IDGenerator.next_param_id(run_obj)
    
    sample_model_spec = run_obj.artifacts.model_spec
    sample_param_bindings = _merge_param_bindings(
        getattr(run_obj.runtime_task.input, "param_bindings", None),
        sample.params,
    )
    if sample.params:
        parsed = parse_compile_lower_model(
            qasm_text=run_obj.runtime_task.input.qasm_text,
            backend_path=run_obj.runtime_task.input.backend_path,
            backend_config=run_obj.runtime_task.input.backend_config,
            out=resolve_writable_out_dir(Path(run_obj.runtime_task.output.out_dir)),
            device=(run_obj.runtime_task.input.device_model or run_obj.runtime_task.input.device),
            pulse=run_obj.runtime_task.input.pulse,
            frame=run_obj.runtime_task.input.frame,
            analyser=run_obj.runtime_task.input.analyser,
            study=run_obj.runtime_task.input.study,
            schedule_policy=run_obj.runtime_task.input.schedule_policy,
            reset_feedback_policy=run_obj.runtime_task.input.reset_feedback_policy,
            noise=run_obj.runtime_task.input.noise,
            solver_run={
                'dt_s': run_obj.runtime_task.run.dt_s,
                't_end_s': run_obj.runtime_task.run.t_end_s,
                't_padding_s': run_obj.runtime_task.run.t_padding_s,
                'seed': run_obj.runtime_task.run.seed,
                'ntraj': run_obj.runtime_task.run.mcwf_ntraj,
                'qutip_options': run_obj.runtime_task.run.qutip_options,
                'native_options': run_obj.runtime_task.run.native_options,
                'backend_options': run_obj.runtime_task.run.backend_options,
                'one_over_f_components': run_obj.runtime_task.run.one_over_f_components,
            },
            solver_mode=run_obj.runtime_task.run.solver_mode,
            param_bindings=sample_param_bindings,
            persist_artifacts=False,
        )
        sample_model_spec = parsed["model_spec"]

    started_at = time.perf_counter()
    trajectory = run_engine_stage(
        model_spec=sample_model_spec,
        cfg=run_obj.runtime_task.input.backend_config,
        engine=run_obj.runtime_task.run.engine,
        allow_mock_fallback=run_obj.runtime_task.run.allow_mock_fallback,
        julia_bin=run_obj.runtime_task.run.julia_bin,
        julia_depot_path=run_obj.runtime_task.run.julia_depot_path,
        julia_timeout_s=run_obj.runtime_task.run.julia_timeout_s,
        mcwf_ntraj=run_obj.runtime_task.run.mcwf_ntraj,
    )
    
    # Handle Decoding if requested in the task
    plan = build_execution_plan(run_obj.runtime_task)
    decoded = {}
    if plan.run_decode:
        decoded = run_decode_stage(
            trajectory=trajectory,
            circuit=run_obj.artifacts.circuit,
            model_spec=run_obj.artifacts.model_spec,
            engine=run_obj.runtime_task.run.engine,
            cfg=run_obj.runtime_task.input.backend_config,
            prior_backend=run_obj.runtime_task.run.prior_backend,
            decoder=run_obj.runtime_task.run.decoder,
            decoder_options=run_obj.runtime_task.run.decoder_options,
        )
        run_obj.artifacts.decoder_outputs = decoded

    result = RunResult(
        result_id=f"{run_obj.identity.run_id}_{param_id}",
        parameters=ParameterValues(
            parameter_id=param_id,
            values=sample.params,
        ),
        provenance=RunProvenance(
            solver_id=run_obj.identity.solver_id,
            study_name=None,
            study_index=None,
        ),
        # Use auto-incrementing shot_id
        trajectories={IDGenerator.next_shot_id(run_obj): trajectory},
        runtime_metadata={
            'engine_used': trajectory.engine,
            'param_id': param_id,
            'param_bindings': dict(sample_param_bindings or {}),
        }
    )
    
    run_obj.results[param_id] = result
    return result.result_id

def run_study(
    model: Any,
    *,
    solver_id: str | None = None,
    study_name_val: str | None = None,
    study_index: int | None = None,
    tag: str | None = None,
) -> str:
    selected_solver_id = require_solver_id(model, solver_id)
    solver_cfg = model.solvers[selected_solver_id].config
    entries = get_study_entries(solver_cfg)
    chosen_index: int | None = None
    chosen_study: dict[str, Any] | None = None
    if study_name_val is not None:
        wanted = str(study_name_val).strip()
        for idx, step in entries:
            if str(study_name(step, idx) or '').strip() == wanted:
                chosen_index = idx
                chosen_study = dict(step)
                break
        if chosen_study is None:
            raise KeyError(f'Unknown study `{wanted}` for solver `{selected_solver_id}`.')
    elif study_index is not None:
        for idx, step in entries:
            if idx == study_index:
                chosen_index = idx
                chosen_study = dict(step)
                break
        if chosen_study is None:
            raise IndexError(f'Unknown study index `{study_index}` for solver `{selected_solver_id}`.')
    else:
        if len(entries) != 1:
            raise ValueError(f'study_name or study_index is required for solver `{selected_solver_id}` with multiple study steps.')
        chosen_index, chosen_study = entries[0]
    assert chosen_study is not None
    chosen_study_name = str(study_name(chosen_study, chosen_index) or "").strip()
    return run_one_solver_study(
        model,
        solver_id=selected_solver_id,
        solver_cfg=solver_cfg,
        study=chosen_study,
        study_index=chosen_index,
        total_studies=len(entries),
        tag=tag,
    )

def run_solver(model: Any, solver_id: str | None = None, tag: str | None = None) -> None:
    selected_solver_id = require_solver_id(model, solver_id)
    solver_cfg = model.solvers[selected_solver_id].config
    
    entries = get_study_entries(solver_cfg)
    for idx, step in entries:
        run_one_solver_study(
            model,
            solver_id=selected_solver_id,
            solver_cfg=solver_cfg,
            study=dict(step),
            study_index=idx,
            total_studies=len(entries),
            tag=tag,
        )

def run_analysis(model: Any, *, analyser_id: str | None = None, study_name_val: str | None = None, tag: str | None = None) -> None:
    selected_analyser_id = require_analyser_id(model, analyser_id)
    analyser_cfg = model.analysers[selected_analyser_id]
    selected_solver_id = require_solver_id(model, analyser_cfg.solver_id)
    matching_runs = [
        (run_id, run_obj)
        for run_id, run_obj in model.runs.items()
        if (
            run_obj.identity.solver_id == selected_solver_id 
            and run_obj.results 
            and any(res.trajectories for res in run_obj.results.values())
        )
    ]
    if study_name_val is not None:
        matching_runs = [
            (run_id, run_obj)
            for run_id, run_obj in matching_runs
            if str(run_obj.identity.study_name or '').strip() == str(study_name_val).strip()
        ]
    if not matching_runs:
        raise ValueError(f'Solver `{selected_solver_id}` has not been run yet.')

    per_study_analyses: list[tuple[Any, ModelAnalysis]] = []
    total_studies = len(matching_runs)
    for run_id, solver_run in matching_runs:
        cfg = getattr(solver_run.runtime_task, 'input', None).backend_config if solver_run.runtime_task else None
        if cfg is None:
            raise ValueError(f'Missing runtime task/backend config for solver `{selected_solver_id}`.')
        logical_error = None
        decoder_outputs = solver_run.artifacts.decoder_outputs
        if decoder_outputs:
            logical_error = decoder_outputs.get('logical_error')

        started_at = time.perf_counter()
        from musiq.workflow.contracts import build_effective_pulse_config
        
        # Analyze only the first available parameter result to maintain 1:1 run-to-case mapping
        if not solver_run.results:
            continue
            
        param_id = next(iter(solver_run.results))
        run_result = solver_run.results[param_id]
        
        # Use the first available trajectory regardless of the key (e.g., 'shot_0')
        trajectory = next(iter(run_result.trajectories.values()), None)
        if trajectory is None:
            continue

        analyzed = run_analysis_stage(
            trajectory=trajectory,
            model_spec=solver_run.artifacts.model_spec,
            pulse_ir=solver_run.artifacts.pulse_ir,
            pulse_cfg=build_effective_pulse_config(model.device, model.pulse),
            cfg=cfg,
            logical_error=logical_error,
            analyser_cfg=analyser_cfg.to_payload(),
            metric_registry=model.metric_registry,
        )
        
        output = analyzed.get('analysis')
        
        # Use auto-incrementing analysis ID based on scope
        analysis_run_id = IDGenerator.next_analysis_id(model, scope="case", tag=tag)
        
        analysis_result = ModelAnalysis(
            analysis_id=analysis_run_id,
            analyser_id=selected_analyser_id,
            input_results=[ResultRef(run_id=run_id, parameter_id=param_id)],
            scope=AnalysisScope.CASE,
            output=output,
        )
        model.analyses[analysis_run_id] = analysis_result
        per_study_analyses.append((solver_run, analysis_result))
        
        # Update timings in artifacts, not in result metadata
        if solver_run.artifacts:
            solver_run.artifacts.timings[f'analysis:{selected_analyser_id}'] = time.perf_counter() - started_at

    if not per_study_analyses:
        return

    if study_name_val is None:
        # Trigger ParametricAnalysis per run if a sweep is defined and the run contains multiple points
        param_cfg = model.config.parameter_list
        has_sweep_def = param_cfg is not None and len(param_cfg.parameters) > 0
        
        if has_sweep_def:
            for run_id, solver_run in matching_runs:
                if len(solver_run.results) <= 1:
                    continue
                
                # 1. Build the ParametricAnalysis object for this specific run
                parametric_out = ParametricAnalysis()
                summary_results = []

                # Define parameter axes from config
                for p_name, p_list in param_cfg.parameters.items():
                    if len(p_list.values) > 1:
                        parametric_out.parameters[p_name] = ParameterAxis(
                            parameter_name=p_name,
                            values=p_list.values,
                            unit=p_list.unit
                        )

                # Aggregate final values for parametric analysis
                metrics_sweep: dict[str, MetricSweepValues] = {}
                payload = analyser_cfg.to_payload()
                sweep_targets = payload.get("sweep_metrics") or payload.get("case_metrics") or []
                
                if sweep_targets:
                    axes = list(parametric_out.parameters.keys())
                    
                    # We need to iterate through all results in THIS run
                    # and align them with the parameter axes values.
                    # Current implementation assumes a simple linear sweep.
                    all_results = solver_run.results
                    
                    for target in sweep_targets:
                        target_name = target if isinstance(target, str) else target.get("name", "unknown")
                        values_list = []
                        
                        # In a real multi-dim sweep, we'd sort results by the axes values.
                        # For now, we assume results are added in the order of the sweep.
                        for param_id, run_result in all_results.items():
                            # We can't use CaseAnalysis because we didn't run run_analysis_stage for every point.
                            # We must calculate the metric on the fly or run the stage.
                            # To be efficient, if it's a known terminal metric, we extract it.
                            # Otherwise, we should ideally run the analyser for this point.
                            
                            # To correctly get metrics for ALL points, we need the analyst's logic.
                            # Since run_analysis_stage is the entry point, let's use it.
                            trajectory = next(iter(run_result.trajectories.values()), None)
                            if trajectory is None:
                                values_list.append(0.0)
                                continue
                            
                            # Run a quick analysis for this specific point
                            point_analysis = run_analysis_stage(
                                trajectory=trajectory,
                                model_spec=solver_run.artifacts.model_spec,
                                pulse_ir=solver_run.artifacts.pulse_ir,
                                pulse_cfg=None, # Using None as fallback or we could build it
                                cfg=getattr(solver_run.runtime_task, 'input', None).backend_config,
                                logical_error=None,
                                analyser_cfg=payload,
                                metric_registry=model.metric_registry,
                            )
                            
                            metrics_map = point_analysis.get('analysis', {}).get('metrics', {})
                            if str(target_name).strip().lower() == "final_fidelity":
                                values_list.append(_extract_final_fidelity(run_result))
                            else:
                                values_list.append(_extract_case_metric_terminal(metrics_map, target_name))
                            
                            summary_results.append(ResultRef(run_id=run_id, parameter_id=param_id))
                        
                        metrics_sweep[target_name] = MetricSweepValues(
                            metric_name=target_name,
                            dimensions=axes,
                            values=values_list
                        )
            
                # Re-calculating summary_results to be unique
                summary_results = list(dict.fromkeys(summary_results))

                parametric_out.metrics = metrics_sweep
                parametric_out.input_results = summary_results

                # 2. Handle higher-level summary (ComprehensiveAnalysis)
                # We pass the results of this specific run to the summary builder
                current_run_analyses = [
                    (solver_run, ModelAnalysis(
                        analysis_id="temp", 
                        analyser_id=selected_analyser_id, 
                        input_results=[ResultRef(run_id=run_id, parameter_id=pid)],
                        scope=AnalysisScope.CASE,
                        output=None # summary_iq_payload only needs identity/run_obj
                    )) 
                    for pid in solver_run.results
                ]
                summary_iq_payload = build_multi_study_iq_summary(model, current_run_analyses)
                
                if summary_iq_payload is not None:
                    comprehensive_out = ComprehensiveAnalysis(
                        parametric_analyses={"main": parametric_out},
                        cross_analysis={"iq_summary": summary_iq_payload},
                        input_sweeps=[ResultRef(run_id=run_id, parameter_id="summary")]
                    )
                    analysis_id = IDGenerator.next_analysis_id(model, scope="comprehensive", tag=tag)
                    model.analyses[analysis_id] = ModelAnalysis(
                        analysis_id=analysis_id,
                        analyser_id=selected_analyser_id,
                        input_results=summary_results,
                        scope=AnalysisScope.COMPREHENSIVE,
                        output=comprehensive_out,
                    )
                else:
                    analysis_id = IDGenerator.next_analysis_id(model, scope="parametric", tag=tag)
                    model.analyses[analysis_id] = ModelAnalysis(
                        analysis_id=analysis_id,
                        analyser_id=selected_analyser_id,
                        input_results=summary_results,
                        scope=AnalysisScope.PARAMETRIC,
                        output=parametric_out,
                    )

def run_all(model: Any) -> None:
    for solver_id in sorted(model.solvers.keys()):
        run_solver(model, solver_id)
    for analyser_id in sorted(model.analysers.keys()):
        run_analysis(model, analyser_id=analyser_id)

def run(model: Any) -> None:
    run_all(model)
