from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from musiq.schemas.model import ModelRun, RunIdentity
from musiq.schemas.results import CaseAnalysis, MetricSeries, RunResult, RunProvenance, ParameterValues
from musiq.common.id_generator import IDGenerator
from musiq.workflow.contracts import AnalyserConfig, ProfileConfig, PulseConfig, SolverBackendConfig, SolverConfig, WorkflowRunOptions
from musiq.workflow.model_execution import (
    _extract_case_metric_terminal,
    _extract_final_fidelity,
    build_multi_study_iq_summary,
    clone_solver_cfg_with_single_study,
    find_run_id,
    get_study_entries,
    run_analysis,
    run_all,
    run_one_solver_study,
    run_profile,
    run_sample,
)
from musiq.workflow.planner_study import StudySample


def test_get_study_entries_returns_default_entry_when_absent():
    solver_cfg = SolverConfig()

    assert get_study_entries(solver_cfg) == [(None, {})]


def test_get_study_entries_enumerates_study_steps():
    solver_cfg = SolverConfig(study=[{"name": "ground"}, {"name": "excited"}])

    assert get_study_entries(solver_cfg) == [(0, {"name": "ground"}), (1, {"name": "excited"})]


def test_clone_solver_cfg_with_single_study_creates_detached_copy():
    solver_cfg = SolverConfig(
        backend=SolverBackendConfig(level="qubit"),
        run=WorkflowRunOptions(engine="qutip", solver_mode="me"),
        study=[{"name": "ground"}, {"name": "excited"}],
    )

    cloned = clone_solver_cfg_with_single_study(solver_cfg, study={"name": "excited", "time": {"dt_s": 1.0}})
    cloned.study[0]["name"] = "mutated"

    assert solver_cfg.study == [{"name": "ground"}, {"name": "excited"}]
    assert cloned.study == [{"name": "mutated", "time": {"dt_s": 1.0}}]
    assert cloned.run.engine == "qutip"
    assert cloned.backend.level == "qubit"


def test_find_run_id_returns_only_matching_solver_or_study():
    run_a = ModelRun(identity=RunIdentity(run_id="run_a", solver_id="solver_0"), runtime_task=None, results={"param_0": object()})
    run_b = ModelRun(
        identity=RunIdentity(run_id="run_b", solver_id="solver_0", study_name="excited"),
        runtime_task=None,
        results={"param_0": object()},
    )
    model = SimpleNamespace(runs={"run_a": run_a, "run_b": run_b})

    assert find_run_id(model, solver_id="solver_0", study_name_val="excited") == "run_b"
    assert find_run_id(model, solver_id="solver_0", study_name_val=None) is None


def test_build_multi_study_iq_summary_aggregates_centroids_and_confusion():
    bundle_ground = SimpleNamespace(identity=SimpleNamespace(study_name="ground"))
    bundle_excited = SimpleNamespace(identity=SimpleNamespace(study_name="excited"))
    analysis_ground = SimpleNamespace(
        output=SimpleNamespace(
            iq={
                "centroids": {"ground": [0.0, 0.0]},
                "synthetic_clouds": {"ground": [[0.0, 0.0], [0.05, 0.02]]},
                "noise_sigma": 0.1,
            }
        )
    )
    analysis_excited = SimpleNamespace(
        output=SimpleNamespace(
            iq={
                "centroids": {"excited": [1.0, 0.0]},
                "synthetic_clouds": {"excited": [[1.0, 0.0], [0.95, -0.02]]},
                "noise_sigma": 0.1,
            }
        )
    )

    summary = build_multi_study_iq_summary(
        None,
        [(bundle_ground, analysis_ground), (bundle_excited, analysis_excited)],
    )

    assert summary is not None
    assert summary["labels"] == ["ground", "excited"]
    assert summary["confusion_matrix"]["values"] == [[2, 0], [0, 2]]
    assert summary["assignment_fidelity"] == 1.0
    assert summary["study_map"] == {"ground": "ground", "excited": "excited"}


def test_extract_case_metric_terminal_derives_final_values_from_case_metrics():
    metrics = {
        "population": MetricSeries(values={"0": [0.9, 0.2], "1": [0.1, 0.8], "2": [0.0, 0.0]}),
        "leakage": MetricSeries(values=[0.0, 0.0]),
        "coherence_01": MetricSeries(values=[0.3, 0.4]),
    }

    assert _extract_case_metric_terminal(metrics, "final_P0") == 0.2
    assert _extract_case_metric_terminal(metrics, "final_P1") == 0.8
    assert _extract_case_metric_terminal(metrics, "final_leakage") == 0.0
    assert _extract_case_metric_terminal(metrics, "final_coherence_01") == 0.4


def test_extract_final_fidelity_uses_theta_and_final_density_matrix():
    theta = 1.5707963267948966
    amp0 = 2 ** -0.5
    amp1 = -1j * 2 ** -0.5
    rho = [
        [amp0 * np.conj(amp0), amp0 * np.conj(amp1), 0.0j],
        [amp1 * np.conj(amp0), amp1 * np.conj(amp1), 0.0j],
        [0.0j, 0.0j, 0.0j],
    ]
    run_result = SimpleNamespace(
        parameters=SimpleNamespace(values={"theta": theta}),
        runtime_metadata={},
        trajectories={
            "shot_0": SimpleNamespace(
                density_matrix={"snapshots": [rho]},
            )
        },
    )

    fidelity = _extract_final_fidelity(run_result)

    assert fidelity == pytest.approx(1.0)


def test_run_sample_recompiles_with_sample_param_bindings(monkeypatch):
    compile_calls = []

    def fake_parse_compile_lower_model(**kwargs):
        compile_calls.append(dict(kwargs.get("param_bindings") or {}))
        return {"model_spec": SimpleNamespace(bound_params=dict(kwargs.get("param_bindings") or {}))}

    def fake_run_engine_stage(*, model_spec, **kwargs):
        return SimpleNamespace(engine="fake", metadata={"bound_params": dict(model_spec.bound_params)})

    monkeypatch.setattr("musiq.workflow.model_execution.parse_compile_lower_model", fake_parse_compile_lower_model)
    monkeypatch.setattr("musiq.workflow.model_execution.run_engine_stage", fake_run_engine_stage)
    monkeypatch.setattr("musiq.workflow.model_execution.build_execution_plan", lambda task: SimpleNamespace(run_decode=False))

    run_obj = ModelRun(
        identity=RunIdentity(run_id="run_solver_0", solver_id="solver_0"),
        runtime_task=SimpleNamespace(
            input=SimpleNamespace(
                qasm_text="OPENQASM 3; qubit[1] q;",
                backend_path=None,
                backend_config=None,
                device_model=None,
                device={},
                pulse={},
                frame={},
                analyser={},
                study=None,
                schedule_policy=None,
                reset_feedback_policy=None,
                noise={},
                param_bindings={"theta": 0.0},
            ),
            run=SimpleNamespace(
                dt_s=None,
                t_end_s=None,
                t_padding_s=None,
                seed=123,
                mcwf_ntraj=1,
                qutip_options={},
                native_options={},
                backend_options={},
                one_over_f_components=None,
                solver_mode="me",
                engine="fake",
                allow_mock_fallback=True,
                julia_bin=None,
                julia_depot_path=None,
                julia_timeout_s=None,
                prior_backend=None,
                decoder=None,
                decoder_options={},
            ),
            output=SimpleNamespace(out_dir=str(Path("tests/.tmp/run-sample"))),
        ),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(bound_params={"theta": 0.0})),
    )

    sample = StudySample(
        profile_id=None,
        circuit_id="circuit_0",
        device_id="device_0",
        pulse_id="pulse_0",
        solver_id="solver_0",
        params={"theta": 1.25},
    )
    run_sample(SimpleNamespace(), run_obj, sample)

    assert compile_calls == [{"theta": 1.25}]
    stored_result = next(iter(run_obj.results.values()))
    assert stored_result.runtime_metadata["param_bindings"] == {"theta": 1.25}
    assert stored_result.provenance.study_name is None
    assert stored_result.provenance.study_index is None
    assert stored_result.trajectories["shot_0"].metadata["bound_params"] == {"theta": 1.25}


def test_run_sample_applies_two_segment_pulse_override_to_active_pulse(monkeypatch):
    compile_calls = []

    def fake_parse_compile_lower_model(**kwargs):
        pulse_cfg = kwargs.get("pulse")
        active_pulse = pulse_cfg["pulse_0"]
        compile_calls.append(dict(getattr(active_pulse, "extras", {}) or {}))
        return {"model_spec": SimpleNamespace(bound_params=dict(kwargs.get("param_bindings") or {}))}

    def fake_run_engine_stage(*, model_spec, **kwargs):
        return SimpleNamespace(engine="fake", metadata={"bound_params": dict(model_spec.bound_params)})

    monkeypatch.setattr("musiq.workflow.model_execution.parse_compile_lower_model", fake_parse_compile_lower_model)
    monkeypatch.setattr("musiq.workflow.model_execution.run_engine_stage", fake_run_engine_stage)
    monkeypatch.setattr("musiq.workflow.model_execution.build_execution_plan", lambda task: SimpleNamespace(run_decode=False))

    run_obj = ModelRun(
        identity=RunIdentity(run_id="run_solver_0", solver_id="solver_0"),
        runtime_task=SimpleNamespace(
            input=SimpleNamespace(
                qasm_text="OPENQASM 3; qubit[1] q;",
                backend_path=None,
                backend_config=None,
                device_model=None,
                device={},
                pulse={"pulse_0": PulseConfig(extras={"idle_duration_ns": 10.0})},
                frame={},
                analyser={},
                study=None,
                schedule_policy=None,
                reset_feedback_policy=None,
                noise={},
                param_bindings=None,
            ),
            run=SimpleNamespace(
                dt_s=None,
                t_end_s=None,
                t_padding_s=None,
                seed=123,
                mcwf_ntraj=1,
                qutip_options={},
                native_options={},
                backend_options={},
                one_over_f_components=None,
                solver_mode="me",
                engine="fake",
                allow_mock_fallback=True,
                julia_bin=None,
                julia_depot_path=None,
                julia_timeout_s=None,
                prior_backend=None,
                decoder=None,
                decoder_options={},
            ),
            output=SimpleNamespace(out_dir=str(Path("tests/.tmp/run-sample-pulse"))),
        ),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(bound_params={})),
    )

    sample = StudySample(
        profile_id=None,
        circuit_id="circuit_0",
        device_id="device_0",
        pulse_id="pulse_0",
        solver_id="solver_0",
        params={"pulse:idle_duration_ns": 42.0},
    )
    run_sample(SimpleNamespace(), run_obj, sample)

    assert compile_calls == [{"idle_duration_ns": 42.0}]


def test_run_sample_applies_device_override_when_runtime_device_is_plain_dict(monkeypatch):
    compile_calls = []

    def fake_parse_compile_lower_model(**kwargs):
        compile_calls.append(dict(kwargs.get("device") or {}))
        return {"model_spec": SimpleNamespace(bound_params=dict(kwargs.get("param_bindings") or {}))}

    def fake_run_engine_stage(*, model_spec, **kwargs):
        return SimpleNamespace(engine="fake", metadata={"bound_params": dict(model_spec.bound_params)})

    monkeypatch.setattr("musiq.workflow.model_execution.parse_compile_lower_model", fake_parse_compile_lower_model)
    monkeypatch.setattr("musiq.workflow.model_execution.run_engine_stage", fake_run_engine_stage)
    monkeypatch.setattr("musiq.workflow.model_execution.build_execution_plan", lambda task: SimpleNamespace(run_decode=False))

    run_obj = ModelRun(
        identity=RunIdentity(run_id="run_solver_0", solver_id="solver_0"),
        runtime_task=SimpleNamespace(
            input=SimpleNamespace(
                qasm_text="OPENQASM 3; qubit[1] q;",
                backend_path=None,
                backend_config=None,
                device_model=None,
                device={"T1_s": 1.0e-5},
                pulse={},
                frame={},
                analyser={},
                study=None,
                schedule_policy=None,
                reset_feedback_policy=None,
                noise={},
                param_bindings=None,
            ),
            run=SimpleNamespace(
                dt_s=None,
                t_end_s=None,
                t_padding_s=None,
                seed=123,
                mcwf_ntraj=1,
                qutip_options={},
                native_options={},
                backend_options={},
                one_over_f_components=None,
                solver_mode="me",
                engine="fake",
                allow_mock_fallback=True,
                julia_bin=None,
                julia_depot_path=None,
                julia_timeout_s=None,
                prior_backend=None,
                decoder=None,
                decoder_options={},
            ),
            output=SimpleNamespace(out_dir=str(Path("tests/.tmp/run-sample-device"))),
        ),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(bound_params={})),
    )

    sample = StudySample(
        profile_id=None,
        circuit_id="circuit_0",
        device_id="device_0",
        pulse_id="pulse_0",
        solver_id="solver_0",
        params={"device:T2_s": 8.0e-6},
    )
    run_sample(SimpleNamespace(), run_obj, sample)

    assert compile_calls == [{"T1_s": 1.0e-5, "T2_s": 8.0e-6}]


def test_run_analysis_builds_parametric_metrics_from_case_analysis_dataclass(monkeypatch):
    analyser_cfg = AnalyserConfig(solver_id="solver_0", sweep_metrics=["final_coherence_01"])
    run_obj = ModelRun(
        identity=RunIdentity(run_id="run_0", solver_id="solver_0"),
        runtime_task=SimpleNamespace(input=SimpleNamespace(backend_config=SimpleNamespace(seed=123))),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(), pulse_ir={}, timings={}, decoder_outputs={}),
        results={
            "param_0": RunResult(
                result_id="run_0_param_0",
                parameters=ParameterValues(parameter_id="param_0", values={"pulse:idle_duration_ns": 0.0}),
                provenance=RunProvenance(solver_id="solver_0"),
                trajectories={"shot_0": SimpleNamespace()},
                runtime_metadata={},
            ),
            "param_1": RunResult(
                result_id="run_0_param_1",
                parameters=ParameterValues(parameter_id="param_1", values={"pulse:idle_duration_ns": 10.0}),
                provenance=RunProvenance(solver_id="solver_0"),
                trajectories={"shot_0": SimpleNamespace()},
                runtime_metadata={},
            ),
        },
    )
    model = SimpleNamespace(
        analysers={"analyser_0": analyser_cfg},
        runs={"run_0": run_obj},
        analyses={},
        config=SimpleNamespace(
            parameter_list=SimpleNamespace(
                parameters={
                    "pulse:idle_duration_ns": SimpleNamespace(values=[0.0, 10.0], unit="ns"),
                }
            )
        ),
        metric_registry=None,
        device={},
        pulse={},
    )

    monkeypatch.setattr("musiq.workflow.model_execution.require_analyser_id", lambda model, analyser_id: analyser_id or "analyser_0")
    monkeypatch.setattr("musiq.workflow.model_execution.require_solver_id", lambda model, solver_id: solver_id or "solver_0")
    monkeypatch.setattr("musiq.workflow.contracts.build_effective_pulse_config", lambda device, pulse: {})
    monkeypatch.setattr(
        "musiq.workflow.model_execution.run_analysis_stage",
        lambda **kwargs: {
            "analysis": CaseAnalysis(
                metrics={
                    "coherence_01": MetricSeries(values=[0.75, 0.25 if kwargs["trajectory"] is run_obj.results["param_1"].trajectories["shot_0"] else 0.5])
                }
            )
        },
    )

    run_analysis(model, analyser_id="analyser_0")

    sweep_analysis = next(analysis for analysis in model.analyses.values() if analysis.scope.value == "parametric")
    assert sweep_analysis.output.metrics["final_coherence_01"].values == [0.5, 0.25]


def test_run_one_solver_study_uses_tag_as_run_key_and_sets_study_identity(monkeypatch):
    sample = StudySample(
        profile_id="Relaxation",
        circuit_id="circuit_0",
        device_id="device_0",
        pulse_id="pulse_0",
        solver_id="solver_0",
        params={},
    )
    model = SimpleNamespace(runs={})

    monkeypatch.setattr(
        "musiq.workflow.model_execution.StudyPlanner.plan",
        lambda current_model: SimpleNamespace(run_groups={"run_0": [sample]}),
    )

    created_run_ids = []

    def fake_execute_compilation_unit(current_model, current_sample, *, run_id=None, tag=None):
        created_run_ids.append(run_id)
        return ModelRun(
            identity=RunIdentity(run_id=run_id or "missing", solver_id=current_sample.solver_id),
            runtime_task=SimpleNamespace(),
            artifacts=SimpleNamespace(),
            results={},
        )

    def fake_run_sample(current_model, run_obj, current_sample):
        run_obj.results["param_0"] = RunResult(
            result_id=f"{run_obj.identity.run_id}_param_0",
            parameters=ParameterValues(parameter_id="param_0", values={}),
            provenance=RunProvenance(
                solver_id=run_obj.identity.solver_id,
                study_name=run_obj.identity.study_name,
                study_index=run_obj.identity.study_index,
            ),
            trajectories={},
            runtime_metadata={},
        )
        return "ok"

    monkeypatch.setattr("musiq.workflow.model_execution.execute_compilation_unit", fake_execute_compilation_unit)
    monkeypatch.setattr("musiq.workflow.model_execution.run_sample", fake_run_sample)

    run_ids = run_one_solver_study(
        model,
        solver_id="solver_0",
        solver_cfg=SolverConfig(),
        study={"name": "Relaxation"},
        study_index=0,
        total_studies=1,
        tag="T1",
    )

    assert run_ids == ["T1"]
    assert created_run_ids == ["T1"]
    assert list(model.runs.keys()) == ["T1"]
    assert model.runs["T1"].identity.run_id == "T1"
    assert model.runs["T1"].identity.study_name == "Relaxation"
    assert model.runs["T1"].identity.study_index == 0


def test_run_profile_delegates_to_solver_and_analysis_with_isolated_profile(monkeypatch):
    captured = {}
    model = SimpleNamespace(
        config=SimpleNamespace(
            profiles={
                "Relaxation": ProfileConfig(solver_id="solver_0", analyser_id="analyser_0"),
                "Other": ProfileConfig(solver_id="solver_1", analyser_id="analyser_1"),
            }
        )
    )

    def fake_run_solver(current_model, solver_id=None, tag=None):
        captured["solver_profiles"] = sorted(current_model.config.profiles.keys())
        captured["solver_id"] = solver_id
        captured["tag"] = tag
        return ["T1"]

    def fake_run_analysis(current_model, *, analyser_id=None, study_name_val=None, tag=None, run_ids=None):
        captured["analysis_profiles"] = sorted(current_model.config.profiles.keys())
        captured["analyser_id"] = analyser_id
        captured["analysis_tag"] = tag
        captured["run_ids"] = run_ids

    monkeypatch.setattr("musiq.workflow.model_execution.run_solver", fake_run_solver)
    monkeypatch.setattr("musiq.workflow.model_execution.run_analysis", fake_run_analysis)
    monkeypatch.setattr("musiq.workflow.model.run_solver", fake_run_solver)
    monkeypatch.setattr("musiq.workflow.model.run_analysis", fake_run_analysis)

    run_profile(model, "Relaxation", tag="T1")

    assert captured["solver_profiles"] == ["Relaxation"]
    assert captured["analysis_profiles"] == ["Relaxation"]
    assert captured["solver_id"] == "solver_0"
    assert captured["analyser_id"] == "analyser_0"
    assert captured["tag"] == "T1"
    assert captured["analysis_tag"] == "T1"
    assert captured["run_ids"] == ["T1"]
    assert sorted(model.config.profiles.keys()) == ["Other", "Relaxation"]


def test_run_all_dispatches_via_profiles_when_present(monkeypatch):
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(profiles={"b": ProfileConfig(), "a": ProfileConfig()}))

    monkeypatch.setattr(
        "musiq.workflow.model_execution.run_profile",
        lambda current_model, profile_id, tag=None: calls.append((profile_id, tag)),
    )

    run_all(model)

    assert calls == [("a", None), ("b", None)]


def test_id_generator_skips_existing_run_and_analysis_ids():
    model = SimpleNamespace(
        runs={"run_0": object(), "run_1": object()},
        analyses={"case_0": object(), "summary_0": object(), "sweep_0": object()},
    )

    assert IDGenerator.next_run_id(model) == "run_2"
    assert IDGenerator.next_analysis_id(model, scope="case") == "case_1"
    assert IDGenerator.next_analysis_id(model, scope="parametric") == "sweep_1"
    assert IDGenerator.next_analysis_id(model, scope="comprehensive") == "summary_1"
