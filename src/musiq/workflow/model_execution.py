"""Execution logic for workflow models."""

from __future__ import annotations

import time
import numpy as np
from dataclasses import asdict, is_dataclass, replace
from typing import Any
from pathlib import Path

from musiq.analysis.definitions import collect_analysis_metrics
from musiq.analysis.common.state_utils import final_density_matrix, state_fidelity
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
    format_study_id,
    effective_analyser_payload,
    require_solver_id,
    require_analyser_id,
)
from musiq.common.id_generator import IDGenerator


def _merge_param_bindings(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any] | None:
    merged = dict(base or {})
    merged.update(dict(override or {}))
    return merged or None


def _looks_like_pulse_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    pulse_payload_keys = {
        "acquisition",
        "timing",
        "channels",
        "gate_duration_ns",
        "idle_duration_ns",
        "measure_duration_ns",
        "measure_amp",
        "measure_segments",
        "xy_freq_Hz",
        "single_qubit_shape",
        "single_qubit_rect_edge_ns",
    }
    return any(key in value for key in pulse_payload_keys)


def _assign_config_value(target: Any, field_name: str, value: Any) -> None:
    if target is None:
        return
    if isinstance(target, dict):
        target[field_name] = value
        return
    if hasattr(target, field_name):
        setattr(target, field_name, value)
        return
    extras = getattr(target, "extras", None)
    if extras is None:
        extras = {}
        setattr(target, "extras", extras)
    if isinstance(extras, dict):
        extras[field_name] = value


def _shallow_clone_config(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return replace(value)
    return value


def _dedupe_result_refs(refs: list[ResultRef]) -> list[ResultRef]:
    unique: list[ResultRef] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        key = (str(ref.run_id), str(ref.parameter_id))
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique


def _select_run_id(
    model: Any,
    solver_id: str,
    *,
    sample: StudySample | None,
    sibling_count: int,
    reserved_run_id: str | None,
    study: dict[str, Any],
    study_index: int | None,
    total_studies: int,
    tag: str | None,
) -> str | None:
    candidate = format_study_id(solver_id, study, study_index, total_studies)
    if sibling_count > 1:
        profile_token = safe_study_token(
            getattr(sample, "profile_id", None)
            or reserved_run_id
            or solver_id
        )
        candidate = f"{profile_token}__{candidate}"
    
    solver_runs = model.runs.get(solver_id, {})
    if candidate not in solver_runs:
        return candidate
    
    return IDGenerator.next_run_id(model, tag=candidate)


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


def _requested_sweep_targets(analyser_payload: dict[str, Any] | None) -> list[str | dict[str, Any]]:
    payload = dict(analyser_payload or {})
    targets = list(payload.get("sweep_metrics", []) or payload.get("parametric_metrics", []) or [])
    targets.extend(collect_analysis_metrics(payload, level="PARAMETRIC"))
    if targets:
        return targets
    return list(payload.get("case_metrics", []) or payload.get("metrics", []) or [])

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
    produced_run_ids: list[str] = []
    resolved_study_name = study.get("name") or study_name(study, study_index) or None
    for reserved_run_id in target_run_ids:
        sample = plan.run_groups[reserved_run_id][0]
        run_id = _select_run_id(
            model,
            solver_id=solver_id,
            sample=sample,
            sibling_count=len(target_run_ids),
            reserved_run_id=reserved_run_id,
            study=study,
            study_index=study_index,
            total_studies=total_studies,
            tag=tag,
        ) or reserved_run_id

        solver_runs = model.runs.setdefault(solver_id, {})
        if run_id not in solver_runs:
            solver_runs[run_id] = execute_compilation_unit(
                model,
                sample,
                solver_cfg_override=clone_solver_cfg_with_single_study(solver_cfg, study=study),
                run_id=run_id,
                tag=tag,
            )

        run_obj = solver_runs[run_id]
        run_obj.identity.run_id = run_id
        run_obj.identity.study_index = study_index
        run_obj.identity.study_name = resolved_study_name

        for sample in plan.run_groups[reserved_run_id]:
            run_sample(model, run_obj, sample)

        run_obj.status = RunStatus.COMPLETED
        run_obj.finished_at = time.time()
        produced_run_ids.append(run_id)

    return produced_run_ids


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
    solver_runs = model.runs.get(solver_id, {})
    candidates = [
        (run_id, run_obj)
        for run_id, run_obj in solver_runs.items()
        if run_obj.results
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
        analysis_output = getattr(analysis, "output", None)
        iq_output = getattr(analysis_output, "iq", None) if analysis_output is not None else None
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
    solver_cfg_override: SolverConfig | None = None,
    run_id: str | None = None,
    tag: str | None = None,
) -> ModelRun:
    """Handle the STAGE_PARSE part and create a ModelRun compilation unit."""
    # Determine which config resources to use
    circuit_cfg = model.config.circuits[sample.circuit_id]
    device_cfg = model.config.devices[sample.device_id]
    pulse_cfg = model.config.pulses[sample.pulse_id]
    solver_cfg = solver_cfg_override or model.config.solvers[sample.solver_id]
    
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
        # Handle namespaced overrides for device and pulse configs
        # We create local copies for this sample to avoid polluting the global model
        current_device = (run_obj.runtime_task.input.device_model or run_obj.runtime_task.input.device)
        current_pulse = run_obj.runtime_task.input.pulse
        
        # Use dataclass replace for shallow copy
        effective_device = _shallow_clone_config(current_device)
        
        # Pulse config is typically a dict[str, PulseConfig]
        effective_pulse = {}
        if isinstance(current_pulse, dict) and not _looks_like_pulse_payload(current_pulse):
            effective_pulse = {pid: _shallow_clone_config(cfg) for pid, cfg in current_pulse.items()}
        else:
            effective_pulse = _shallow_clone_config(current_pulse)

        # Apply overrides from sample.params
        for key, value in sample.params.items():
            if key.startswith("device:"):
                field_name = key[7:]
                if effective_device:
                    _assign_config_value(effective_device, field_name, value)
            elif key.startswith("pulse:"):
                parts = key.split(":")
                if len(parts) == 2:
                    field_name = parts[1]
                    if _looks_like_pulse_payload(effective_pulse):
                        _assign_config_value(effective_pulse, field_name, value)
                    elif isinstance(effective_pulse, dict):
                        target_pulse = effective_pulse.get(sample.pulse_id)
                        if target_pulse is None and len(effective_pulse) == 1:
                            target_pulse = next(iter(effective_pulse.values()))
                        if target_pulse is not None:
                            _assign_config_value(target_pulse, field_name, value)
                elif len(parts) == 3:
                    pid, field_name = parts[1], parts[2]
                    if isinstance(effective_pulse, dict) and pid in effective_pulse:
                        _assign_config_value(effective_pulse[pid], field_name, value)

        parsed = parse_compile_lower_model(
            qasm_text=run_obj.runtime_task.input.qasm_text,
            backend_path=run_obj.runtime_task.input.backend_path,
            backend_config=run_obj.runtime_task.input.backend_config,
            out=resolve_writable_out_dir(Path(run_obj.runtime_task.output.out_dir)),
            device=effective_device,
            pulse=effective_pulse,
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
            study_name=run_obj.identity.study_name,
            study_index=run_obj.identity.study_index,
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
) -> list[str]:
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
    return run_one_solver_study(
        model,
        solver_id=selected_solver_id,
        solver_cfg=solver_cfg,
        study=chosen_study,
        study_index=chosen_index,
        total_studies=len(entries),
        tag=tag,
    )

def run_solver(model: Any, solver_id: str | None = None, tag: str | None = None) -> list[str]:
    selected_solver_id = require_solver_id(model, solver_id)
    solver_cfg = model.solvers[selected_solver_id].config
    
    entries = get_study_entries(solver_cfg)
    all_run_ids = []
    for idx, step in entries:
        rids = run_one_solver_study(
            model,
            solver_id=selected_solver_id,
            solver_cfg=solver_cfg,
            study=dict(step),
            study_index=idx,
            total_studies=len(entries),
            tag=tag,
        )
        all_run_ids.extend(rids)
    return all_run_ids

def run_analysis(model: Any, *, analyser_id: str | None = None, study_name_val: str | None = None, tag: str | None = None, run_ids: list[str] | None = None) -> None:
    selected_analyser_id = require_analyser_id(model, analyser_id)
    analyser_cfg_obj = model.analysers[selected_analyser_id]
    analyser_payload = analyser_cfg_obj.to_payload()
    selected_solver_id = require_solver_id(model, analyser_cfg_obj.solver_id)
    
    matching_runs = [
        (run_id, run_obj)
        for run_id, run_obj in model.runs.get(selected_solver_id, {}).items()
        if (
            run_obj.results 
            and any(res.trajectories for res in run_obj.results.values())
        )
    ]
    if run_ids is not None:
        matching_runs = [item for item in matching_runs if item[0] in run_ids]
    if study_name_val is not None:
        matching_runs = [
            (run_id, run_obj)
            for run_id, run_obj in matching_runs
            if str(run_obj.identity.study_name or '').strip() == str(study_name_val).strip()
        ]
    if not matching_runs:
        raise ValueError(f'Solver `{selected_solver_id}` has not been run yet.')

    # --- Phase 1: Case-level Analysis ---
    # We collect results for ALL parameter points in the matching runs
    case_analyses_collected: list[tuple[Any, Any, CaseAnalysis]] = [] # (run_obj, param_id, analysis)
    
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
        pulse_cfg = build_effective_pulse_config(model.device, model.pulse)
        
        for param_id, run_result in solver_run.results.items():
            trajectory = next(iter(run_result.trajectories.values()), None)
            if trajectory is None:
                continue

            analyzed = run_analysis_stage(
                trajectory=trajectory,
                model_spec=solver_run.artifacts.model_spec,
                pulse_ir=solver_run.artifacts.pulse_ir,
                pulse_cfg=pulse_cfg,
                device_cfg=model.device,
                cfg=cfg,
                logical_error=logical_error,
                analyser_cfg=analyser_payload,
                metric_registry=model.metric_registry,
            )
            
            analysis_output = analyzed.get('analysis')
            analysis_run_id = IDGenerator.next_analysis_id(model, scope="case", tag=tag)
            
            analysis_result = ModelAnalysis(
                analysis_id=analysis_run_id,
                analyser_id=selected_analyser_id,
                input_results=[ResultRef(run_id=run_id, parameter_id=param_id)],
                scope=AnalysisScope.CASE,
                output=analysis_output,
            )
            model.analyses[analysis_run_id] = analysis_result
            case_analyses_collected.append((solver_run, param_id, analysis_result))
            
        if solver_run.artifacts:
            solver_run.artifacts.timings[f'analysis:{selected_analyser_id}'] = time.perf_counter() - started_at

    if not case_analyses_collected:
        return

    # --- Phase 2: Comprehensive/Global Analysis ---
    from musiq.analysis.dispatcher import dispatch_analysis
    analysis_steps = analyser_payload.get("analysis", [])
    comprehensive_steps = [s for s in analysis_steps if s.get("level") == "COMPREHENSIVE"]
    
    if comprehensive_steps:
        for step in comprehensive_steps:
            # Map legacy name to AnalysisKind
            name_to_kind = {"iq_analysis": "IQ"}
            kind = name_to_kind.get(step.get("name"), step.get("kind"))
            if not kind:
                continue

            try:
                # Prepare arguments for the Comprehensive handler
                # For IQ analysis, we gather case results as required by build_iq_analysis
                if kind == "IQ":
                    case_results_for_iq = []
                    labels = []
                    for solver_run, param_id, analysis in case_analyses_collected:
                        readout_data = getattr(analysis.output, "readout", None)
                        if readout_data and readout_data.integrated_points:
                            case_results_for_iq.append({"integrated_iq": list(readout_data.integrated_points)})
                            labels.append(getattr(solver_run.identity, "study_name", "unknown"))
                    
                    res = dispatch_analysis(
                        level="COMPREHENSIVE",
                        kind=kind,
                        case_results=case_results_for_iq,
                        labels=labels,
                        seed=int(getattr(model.config, "seed", 12345)),
                    )
                else:
                    # For other comprehensive types, pass the step extras as kwargs
                    res = dispatch_analysis(
                        level="COMPREHENSIVE",
                        kind=kind,
                        **step.get("extras", {}),
                    )

                comp_analysis_id = IDGenerator.next_analysis_id(model, scope="comprehensive", tag=tag)
                model.analyses[comp_analysis_id] = ModelAnalysis(
                    analysis_id=comp_analysis_id,
                    analyser_id=selected_analyser_id,
                    input_results=[ResultRef(run_id=r.identity.run_id, parameter_id="global") for r, _, _ in case_analyses_collected],
                    scope=AnalysisScope.COMPREHENSIVE,
                    output=res,
                )
            except KeyError:
                continue

    # --- Parametric Analysis (Maintained from original) ---
    if study_name_val is None:
        param_cfg = model.config.parameter_list
        has_sweep_def = param_cfg is not None and len(param_cfg.parameters) > 0
        if has_sweep_def:
            from musiq.workflow.contracts import build_effective_pulse_config
            for run_id, solver_run in matching_runs:
                if len(solver_run.results) <= 1:
                    continue
                
                parametric_out = ParametricAnalysis()
                summary_results = []
                axes_names = []
                for p_name, p_list in param_cfg.parameters.items():
                    if len(p_list.values) > 1:
                        axes_names.append(p_name)
                        parametric_out.parameters[p_name] = ParameterAxis(
                            parameter_name=p_name,
                            values=p_list.values,
                            unit=p_list.unit
                        )

                metrics_sweep: dict[str, MetricSweepValues] = {}
                sweep_targets = _requested_sweep_targets(analyser_payload)
                
                if sweep_targets:
                    sorted_results = []
                    if axes_names:
                        primary_axis = axes_names[0]
                        temp_list = []
                        for pid, res in solver_run.results.items():
                            val = res.parameters.values.get(primary_axis, 0.0)
                            temp_list.append((val, pid, res))
                        temp_list.sort(key=lambda x: x[0])
                        sorted_results = [(pid, res) for val, pid, res in temp_list]
                    else:
                        sorted_results = list(solver_run.results.items())
                    
                    for target in sweep_targets:
                        target_name = target if isinstance(target, str) else target.get("name", "unknown")
                        values_list = []
                        for param_id, run_result in sorted_results:
                            # Reuse results from Phase 1 (Case-level analysis) to avoid redundant execution
                            # Find the CaseAnalysis associated with this run and parameter
                            case_analysis = next(
                                (a for r, p, a in case_analyses_collected if r.identity.run_id == run_id and p == param_id),
                                None
                            )
                            
                            if case_analysis is None:
                                values_list.append(0.0)
                                continue
                            
                            metrics_map = dict(getattr(case_analysis.output, "metrics", {}) or {})
                            if str(target_name).strip().lower() == "final_fidelity":
                                values_list.append(_extract_final_fidelity(run_result))
                            else:
                                values_list.append(_extract_case_metric_terminal(metrics_map, target_name))
                            summary_results.append(ResultRef(run_id=run_id, parameter_id=param_id))
                        
                        metrics_sweep[target_name] = MetricSweepValues(
                            metric_name=target_name,
                            dimensions=axes_names,
                            values=values_list
                        )
                
                summary_results = _dedupe_result_refs(summary_results)
                parametric_out.metrics = metrics_sweep
                parametric_out.input_results = summary_results

                analysis_id = IDGenerator.next_analysis_id(model, scope="parametric", tag=tag)
                model.analyses[analysis_id] = ModelAnalysis(
                    analysis_id=analysis_id,
                    analyser_id=selected_analyser_id,
                    input_results=summary_results,
                    scope=AnalysisScope.PARAMETRIC,
                    output=parametric_out,
                )

def run_all(model: Any) -> None:
    for profile_id in sorted(model.config.profiles.keys()):
        run_profile(model, profile_id)
    return


def run(model: Any) -> None:
    run_all(model)

def run_profile(model: Any, profile_id: str, tag: str | None = None) -> None:
    """
    Run simulation for a specific profile.
    This method temporarily isolates the target profile to ensure only its 
    configuration is expanded by the StudyPlanner.
    """
    from musiq.workflow.model import Profile
    p_wrapper = Profile(model, str(profile_id), model.config.profiles[profile_id])
    
    # Backup original profiles
    original_profiles = dict(model.config.profiles)

    try:
        # Isolate target profile so StudyPlanner only generates samples for it
        model.config.profiles = {profile_id: p_wrapper.config}
        
        # Run the solver associated with this profile
        run_ids = p_wrapper.run_solver(tag=tag)
        
        # Trigger analysis for the results produced
        p_wrapper.run_analysis(tag=tag, run_ids=run_ids)
        
    finally:
        # Restore original profiles regardless of success/failure
        model.config.profiles = original_profiles
