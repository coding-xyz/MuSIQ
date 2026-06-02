from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from musiq.schemas.model import ModelRun, RunIdentity
from musiq.schemas.results import CaseAnalysis, MetricSeries, ReadoutAnalysis, RunResult, RunProvenance, ParameterValues, ShotData, Trajectory
from musiq.common.id_generator import IDGenerator
from musiq.analysis.case.readout.analysis import build_readout_analysis as build_case_readout_analysis
from musiq.analysis.metrics import resolve_metrics_payload
from musiq.analysis.readout_chain import build_readout_analysis
from musiq.workflow.contracts import AnalyserConfig, ProfileConfig, PulseConfig, SolverBackendConfig, SolverConfig, WorkflowRunOptions
from musiq.workflow.model_execution import (
    _extract_case_metric_terminal,
    _extract_final_fidelity,
    _requested_sweep_targets,
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
from musiq.workflow.model import Profile, _build_cartesian_profiles
from musiq.workflow.planner_study import StudySample
from musiq.workflow.stages import run_analysis_stage


def test_get_study_entries_returns_default_entry_when_absent():
    solver_cfg = SolverConfig()

    assert get_study_entries(solver_cfg) == [(None, {})]


def test_build_cartesian_profiles_collapses_singleton_to_default():
    profiles = _build_cartesian_profiles(
        circuits={"default": object()},
        devices={"default": object()},
        pulses={"default": object()},
        solvers={"solver_0": object()},
        analysers={"analyser_0": object()},
    )

    assert sorted(profiles.keys()) == ["default"]
    assert profiles["default"].solver_id == "solver_0"


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
            0: SimpleNamespace(
                density_matrix={"snapshots": [rho]},
            )
        },
    )

    fidelity = _extract_final_fidelity(run_result)

    assert fidelity == pytest.approx(1.0)


def test_resolve_metrics_payload_reads_case_metrics_from_hierarchical_single_qubit_analysis():
    rho0 = [
        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
    ]
    rho1 = [
        [0.2 + 0.0j, 0.3 + 0.0j, 0.0 + 0.0j],
        [0.3 + 0.0j, 0.7 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j],
    ]
    trajectory = SimpleNamespace(
        engine="qutip",
        times=[0.0, 1.0],
        density_matrix={"actual_kind": "density_matrix", "snapshots": [rho0, rho1]},
        wave_function={},
        classical={},
    )
    model_spec = SimpleNamespace(
        system=SimpleNamespace(
            num_qubits=1,
            transmon_levels=3,
            model_type="transmon_nlevel",
        )
    )

    metrics, observables, report = resolve_metrics_payload(
        trajectory,
        model_spec,
        {
            "analysis": [
                {
                    "name": "single_qubit_analysis",
                    "level": "CASE",
                    "metrics": ["population", "leakage", "coherence_01"],
                }
            ]
        },
    )

    assert sorted(metrics.keys()) == ["coherence_01", "leakage", "population"]
    assert metrics["population"].values["1"] == [0.0, 0.7]
    assert metrics["leakage"].values == [0.0, 0.1]
    assert metrics["coherence_01"].values == [0.0, 0.3]
    assert "coherence_01" in observables.values
    assert report.summary["metrics"] == ["population", "leakage", "coherence_01"]


def test_resolve_metrics_payload_emits_mean_and_std_for_multi_trajectory_population():
    rho_ground = [
        [1.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j],
    ]
    rho_excited = [
        [0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 1.0 + 0.0j],
    ]
    trajectory_0 = Trajectory(engine="qutip", times=[0.0], density_matrix=[rho_ground])
    trajectory_1 = Trajectory(engine="qutip", times=[0.0], density_matrix=[rho_excited])
    model_spec = SimpleNamespace(
        system=SimpleNamespace(
            num_qubits=1,
            transmon_levels=2,
            model_type="transmon_nlevel",
        )
    )

    metrics, observables, report = resolve_metrics_payload(
        trajectory_0,
        model_spec,
        {
            "analysis": [
                {
                    "name": "single_qubit_analysis",
                    "level": "CASE",
                    "metrics": ["population"],
                }
            ]
        },
        trajectories=[trajectory_0, trajectory_1],
    )

    assert sorted(metrics.keys()) == ["population_mean", "population_std"]
    assert metrics["population_mean"].values["0"] == [0.5]
    assert metrics["population_mean"].values["1"] == [0.5]
    assert metrics["population_std"].values["0"] == [0.5]
    assert metrics["population_std"].values["1"] == [0.5]
    assert "population_mean" in report.summary["metrics"]
    assert "population_std" in report.summary["metrics"]
    assert observables.values["samples"] == pytest.approx(1.0)


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
    assert stored_result.trajectories[0].metadata["bound_params"] == {"theta": 1.25}


def test_run_sample_expands_mcwf_runs_into_multiple_shots(monkeypatch):
    def fake_parse_compile_lower_model(**kwargs):
        return {"model_spec": SimpleNamespace(bound_params=dict(kwargs.get("param_bindings") or {}))}

    def fake_run_engine_stage(*, model_spec, **kwargs):
        return Trajectory(
            engine="qutip",
            times=[0.0, 1.0],
            wave_function={
                "actual_kind": "wave_function",
                "snapshots": [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
                "runs": [
                    [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]],
                ],
                "num_runs": 2,
            },
            metadata={"bound_params": dict(model_spec.bound_params)},
        )

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
                mcwf_ntraj=2,
                qutip_options={},
                native_options={},
                backend_options={},
                one_over_f_components=None,
                solver_mode="mcwf",
                engine="qutip",
                allow_mock_fallback=True,
                julia_bin=None,
                julia_depot_path=None,
                julia_timeout_s=None,
                prior_backend=None,
                decoder=None,
                decoder_options={},
            ),
            output=SimpleNamespace(out_dir=str(Path("tests/.tmp/run-sample-mcwf"))),
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

    stored_result = next(iter(run_obj.results.values()))
    assert list(stored_result.trajectories.keys()) == [0, 1]
    assert stored_result.trajectories[0].wave_function == [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ]
    assert stored_result.trajectories[1].wave_function == [
        [[0.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0]],
    ]
    assert stored_result.trajectories[0].metadata["trajectory_index"] == 0
    assert stored_result.trajectories[1].metadata["trajectory_index"] == 1


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


def test_run_sample_applies_pulse_override_to_flat_runtime_pulse_dict(monkeypatch):
    compile_calls = []

    def fake_parse_compile_lower_model(**kwargs):
        compile_calls.append(dict(kwargs.get("pulse") or {}))
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
                pulse={"gate_duration_ns": 15.0, "idle_duration_ns": 100.0, "xy_freq_Hz": 5.0e9},
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
            output=SimpleNamespace(out_dir=str(Path("tests/.tmp/run-sample-flat-pulse"))),
        ),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(bound_params={})),
    )

    sample = StudySample(
        profile_id=None,
        circuit_id="circuit_0",
        device_id="device_0",
        pulse_id="default",
        solver_id="solver_0",
        params={"pulse:idle_duration_ns": 42.0},
    )
    run_sample(SimpleNamespace(), run_obj, sample)

    assert compile_calls == [{"gate_duration_ns": 15.0, "idle_duration_ns": 42.0, "xy_freq_Hz": 5.0e9}]


def test_run_sample_applies_nested_typed_pulse_override(monkeypatch):
    compile_calls = []

    def fake_parse_compile_lower_model(**kwargs):
        compile_calls.append(dict(kwargs.get("pulse") or {}))
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
                pulse={
                    "defaults": {"xy_carrier_freq_Hz": 5.0e9},
                    "gates": {
                        "rx": {
                            "recipe_type": "rx",
                            "duration_ns": 20.0,
                            "amplitude_Hz": 12.5e6,
                        }
                    },
                },
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
            output=SimpleNamespace(out_dir=str(Path("tests/.tmp/run-sample-typed-pulse"))),
        ),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(bound_params={})),
    )

    sample = StudySample(
        profile_id=None,
        circuit_id="circuit_0",
        device_id="device_0",
        pulse_id="default",
        solver_id="solver_0",
        params={"pulse:gates.rx.duration_ns": 42.0},
    )
    run_sample(SimpleNamespace(), run_obj, sample)

    assert compile_calls == [
        {
            "defaults": {"xy_carrier_freq_Hz": 5.0e9},
            "gates": {
                "rx": {
                    "recipe_type": "rx",
                    "duration_ns": 42.0,
                    "amplitude_Hz": 12.5e6,
                }
            },
        }
    ]


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
                device={
                    "noise": {
                        "model": "markovian_lindblad",
                        "sources": [
                            {
                                "id": "q0_T1",
                                "kind": "markovian",
                                "targets": ["q0"],
                                "operator": "lowering",
                                "parameters": {"T1_s": 1.0e-5},
                            }
                        ],
                    }
                },
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
        params={"device:noise.model": "stochastic"},
    )
    run_sample(SimpleNamespace(), run_obj, sample)

    assert compile_calls == [
        {
            "noise": {
                "model": "stochastic",
                "sources": [
                    {
                        "id": "q0_T1",
                        "kind": "markovian",
                        "targets": ["q0"],
                        "operator": "lowering",
                        "parameters": {"T1_s": 1.0e-5},
                    }
                ],
            }
        }
    ]


def test_run_analysis_builds_parametric_metrics_from_case_analysis_dataclass(monkeypatch):
    analyser_cfg = AnalyserConfig(
        solver_id="solver_0",
        analysis=[
            {"name": "single_qubit_analysis", "level": "CASE", "metrics": ["coherence_01"]},
            {"name": "rabi_sweep_analysis", "level": "PARAMETRIC", "metrics": ["final_coherence_01"]},
        ],
    )
    run_obj = ModelRun(
        identity=RunIdentity(run_id="run_0", solver_id="solver_0"),
        runtime_task=SimpleNamespace(input=SimpleNamespace(backend_config=SimpleNamespace(seed=123))),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(), pulse_ir={}, timings={}, decoder_outputs={}),
        results={
            "param_0": RunResult(
                result_id="run_0_param_0",
                parameters=ParameterValues(parameter_id="param_0", values={"pulse:idle_duration_ns": 0.0}),
                provenance=RunProvenance(solver_id="solver_0"),
                trajectories={0: SimpleNamespace()},
                runtime_metadata={},
            ),
            "param_1": RunResult(
                result_id="run_0_param_1",
                parameters=ParameterValues(parameter_id="param_1", values={"pulse:idle_duration_ns": 10.0}),
                provenance=RunProvenance(solver_id="solver_0"),
                trajectories={0: SimpleNamespace()},
                runtime_metadata={},
            ),
        },
    )
    model = SimpleNamespace(
        analysers={"analyser_0": analyser_cfg},
        runs={"run_0": run_obj},
        analyses={},
        config=SimpleNamespace(
            parameter_sweep=SimpleNamespace(
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
                    "coherence_01": MetricSeries(values=[0.75, 0.25 if kwargs["trajectory"] is run_obj.results["param_1"].trajectories[0] else 0.5])
                }
            )
        },
    )

    run_analysis(model, analyser_id="analyser_0")

    sweep_analysis = next(analysis for analysis in model.analyses.values() if analysis.scope.value == "parametric")
    assert sweep_analysis.output.metrics["final_coherence_01"].values == [0.5, 0.25]


def test_run_analysis_averages_quantum_trajectories_before_case_metrics(monkeypatch):
    analyser_cfg = AnalyserConfig(
        solver_id="solver_0",
        analysis=[{"name": "single_qubit_analysis", "level": "CASE", "metrics": ["population"]}],
    )
    rho_shot_0 = [
        [1.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j],
    ]
    rho_shot_1 = [
        [0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 1.0 + 0.0j],
    ]
    run_obj = ModelRun(
        identity=RunIdentity(run_id="run_0", solver_id="solver_0"),
        runtime_task=SimpleNamespace(input=SimpleNamespace(backend_config=SimpleNamespace(seed=123))),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(), pulse_ir={}, timings={}, decoder_outputs={}),
        results={
            "param_0": RunResult(
                result_id="run_0_param_0",
                parameters=ParameterValues(parameter_id="param_0", values={}),
                provenance=RunProvenance(solver_id="solver_0"),
                trajectories={
                    0: Trajectory(engine="qutip", times=[0.0], density_matrix=[rho_shot_0]),
                    1: Trajectory(engine="qutip", times=[0.0], density_matrix=[rho_shot_1]),
                },
                runtime_metadata={},
            ),
        },
    )
    model = SimpleNamespace(
        analysers={"analyser_0": analyser_cfg},
        runs={"run_0": run_obj},
        analyses={},
        config=SimpleNamespace(parameter_sweep=None),
        metric_registry=None,
        device={},
        pulse={},
    )

    monkeypatch.setattr("musiq.workflow.model_execution.require_analyser_id", lambda model, analyser_id: analyser_id or "analyser_0")
    monkeypatch.setattr("musiq.workflow.model_execution.require_solver_id", lambda model, solver_id: solver_id or "solver_0")
    monkeypatch.setattr("musiq.workflow.contracts.build_effective_pulse_config", lambda device, pulse: {})

    def fake_run_analysis_stage(**kwargs):
        trajectory = kwargs["trajectory"]
        trajectories = kwargs["trajectories"]
        assert trajectory is run_obj.results["param_0"].trajectories[0]
        assert trajectories == [
            run_obj.results["param_0"].trajectories[0],
            run_obj.results["param_0"].trajectories[1],
        ]
        return {
            "analysis": CaseAnalysis(
                metrics={
                    "population_mean": MetricSeries(values={"0": [0.5], "1": [0.5]}),
                    "population_std": MetricSeries(values={"0": [0.5], "1": [0.5]}),
                }
            )
        }

    monkeypatch.setattr("musiq.workflow.model_execution.run_analysis_stage", fake_run_analysis_stage)

    run_analysis(model, analyser_id="analyser_0")

    case_analysis = next(analysis for analysis in model.analyses.values() if analysis.scope.value == "case")
    assert case_analysis.output.metrics["population_mean"].values["0"] == [0.5]
    assert case_analysis.output.metrics["population_mean"].values["1"] == [0.5]
    assert case_analysis.output.metrics["population_std"].values["0"] == [0.5]
    assert case_analysis.output.metrics["population_std"].values["1"] == [0.5]


def test_run_analysis_stage_state_analysis_uses_multi_trajectory_metric_names():
    rho_ground = [
        [1.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j],
    ]
    rho_excited = [
        [0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 1.0 + 0.0j],
    ]
    trajectory_0 = Trajectory(engine="qutip", times=[0.0], density_matrix=[rho_ground])
    trajectory_1 = Trajectory(engine="qutip", times=[0.0], density_matrix=[rho_excited])
    model_spec = SimpleNamespace(
        system=SimpleNamespace(
            num_qubits=1,
            transmon_levels=2,
            model_type="transmon_nlevel",
        )
    )

    analyzed = run_analysis_stage(
        trajectory=trajectory_0,
        trajectories=[trajectory_0, trajectory_1],
        model_spec=model_spec,
        pulse_ir=None,
        pulse_cfg={},
        device_cfg={},
        cfg=SimpleNamespace(seed=123, sweep=None),
        logical_error=None,
        analyser_cfg={
            "analysis": [
                {
                    "name": "state_analysis",
                    "level": "CASE",
                    "metrics": ["population"],
                }
            ]
        },
        metric_registry=None,
    )

    metrics = analyzed["analysis"].metrics
    assert "population_mean" in metrics
    assert "population_std" in metrics
    assert "population" not in metrics


def test_requested_sweep_targets_reads_hierarchical_parametric_metrics():
    targets = _requested_sweep_targets(
        {
            "analysis": [
                {"name": "single_qubit_analysis", "level": "CASE", "metrics": ["coherence_01"]},
                {"name": "rabi_sweep_analysis", "level": "PARAMETRIC", "metrics": ["final_coherence_01"]},
            ]
        }
    )

    assert targets == ["final_coherence_01"]


def test_build_readout_analysis_populates_shots_from_trajectory_payload():
    trajectory = SimpleNamespace(
        times=[0.0, 1.0e-9, 2.0e-9],
        classical={
            "readout": {
                "a_in": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                "cavity_a": [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
                "a_out": [[0.2, 0.1], [0.3, 0.1], [0.4, 0.1]],
                "shots": [
                    {
                        "trajectory_index": 0,
                        "hidden_state_label": "0",
                        "a_out": [[0.2, 0.1], [0.3, 0.1], [0.4, 0.1]],
                        "integrated_iq": [0.25, -0.5],
                    },
                    {
                        "trajectory_index": 1,
                        "hidden_state_label": "0",
                        "a_out": [[0.25, 0.05], [0.35, 0.05], [0.45, 0.05]],
                        "integrated_iq": [0.5, -0.25],
                    },
                ],
            }
        },
    )
    pulse_ir = SimpleNamespace(channels=[])
    readout = build_readout_analysis(
        trajectory=trajectory,
        model_spec=SimpleNamespace(payload={}),
        pulse_ir=pulse_ir,
        pulse_cfg={"acquisition": {}},
        device_cfg={
            "components": [
                {
                    "id": "ro0",
                    "parameters": {
                        "kappa_ext_Hz": 1.0,
                        "carrier_frequency_Hz": 5.0e9,
                        "lo_frequency_Hz": 5.0e9,
                        "if_frequency_Hz": 10.0e6,
                        "adc_sample_rate_Hz": 1.0e9,
                    },
                }
            ]
        },
        seed=123,
    )

    assert len(readout.shots) == 2
    assert readout.shots[0].integrated_iq == [0.25, -0.5]
    assert readout.shots[0].metadata["hidden_state_label"] == "0"
    assert readout.integrated_points == [complex(0.25, -0.5), complex(0.5, -0.25)]


def test_build_readout_analysis_derives_shot_iq_from_shot_trace_when_payload_missing():
    trajectory = SimpleNamespace(
        times=[0.0, 1.0, 2.0],
        classical={
            "readout": {
                "a_in": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                "cavity_a": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                "a_out": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                "shots": [
                    {
                        "hidden_state_label": "0",
                        "a_out": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "measured_voltage": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                    },
                    {
                        "hidden_state_label": "0",
                        "a_out": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        "measured_voltage": [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
                    },
                ],
            }
        },
    )
    pulse_ir = SimpleNamespace(
        channels=[
            SimpleNamespace(
                name="RO_0",
                pulses=[SimpleNamespace(shape="readout", params={"break_stage": "measure"}, t0_s=0.0, t1_s=2.0)],
            )
        ]
    )
    readout = build_readout_analysis(
        trajectory=trajectory,
        model_spec=SimpleNamespace(payload={}),
        pulse_ir=pulse_ir,
        pulse_cfg={"acquisition": {}, "measure_duration_ns": 2.0e9},
        device_cfg={"components": [{"id": "ro0", "parameters": {"adc_sample_rate_Hz": 1.0}}]},
        seed=123,
    )

    assert readout.shots[0].integrated_iq == [2.0, 0.0]
    assert readout.shots[1].integrated_iq == [4.0, 0.0]
    assert readout.integrated_points == [complex(2.0, 0.0), complex(4.0, 0.0)]


def test_build_readout_analysis_supports_nested_device_config_payload():
    trajectory = SimpleNamespace(
        times=[0.0, 0.5e-9, 1.0e-9],
        classical={
            "readout": {
                "a_in": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                "cavity_a": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                "a_out": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                "shots": [],
            }
        },
    )
    pulse_ir = SimpleNamespace(channels=[])
    readout = build_readout_analysis(
        trajectory=trajectory,
        model_spec=SimpleNamespace(payload={}),
        pulse_ir=pulse_ir,
        pulse_cfg={"acquisition": {}},
        device_cfg={
            "device": {
                "components": [
                    {
                        "id": "ro0",
                        "parameters": {
                            "carrier_frequency_Hz": 6.45e9,
                            "adc_sample_rate_Hz": 10.0e9,
                        },
                    }
                ]
            }
        },
        seed=123,
    )

    assert readout.chain_params["carrier_frequency_Hz"] == 6.45e9
    assert readout.demodulation["carrier_frequency_Hz"] == 6.45e9
    assert readout.demodulation["adc_sample_rate_Hz"] == 10.0e9


def test_build_readout_analysis_supports_nested_pulse_config_payload():
    trajectory = SimpleNamespace(
        times=[0.0, 1.0, 2.0],
        classical={
            "readout": {
                "a_in": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                "cavity_a": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                "a_out": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                "shots": [
                    {
                        "hidden_state_label": "0",
                        "measured_voltage": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                    },
                    {
                        "hidden_state_label": "1",
                        "measured_voltage": [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
                    },
                ],
            }
        },
    )
    pulse_ir = SimpleNamespace(
        channels=[
            SimpleNamespace(
                name="RO_0",
                pulses=[SimpleNamespace(shape="readout", params={"break_stage": "measure"}, t0_s=0.0, t1_s=2.0)],
            )
        ]
    )
    pulse_cfg = {
        "pulse": {
            "acquisition": {
                "extras": {
                    "start_delay_ns": 1.0e9,
                    "integration_window_ns": 1.0e9,
                }
            }
        }
    }
    device_cfg = {"components": [{"id": "ro0", "parameters": {"adc_sample_rate_Hz": 1.0}}]}

    readout = build_readout_analysis(
        trajectory=trajectory,
        model_spec=SimpleNamespace(payload={}),
        pulse_ir=pulse_ir,
        pulse_cfg=pulse_cfg,
        device_cfg=device_cfg,
        seed=123,
    )
    case_readout = build_case_readout_analysis(
        trajectory=trajectory,
        model_spec=SimpleNamespace(payload={}),
        pulse_ir=pulse_ir,
        pulse_cfg=pulse_cfg,
        device_cfg=device_cfg,
        seed=123,
    )

    assert readout.integrated_points == [complex(2.5, 0.0), complex(4.5, 0.0)]
    assert case_readout.integrated_points == [complex(2.5, 0.0), complex(4.5, 0.0)]


def test_readout_analysis_reconstruct_shot_coerces_scalar_chain_params():
    readout = ReadoutAnalysis(
        sim_times=[0.0, 0.5e-9, 1.0e-9],
        adc_times=[0.0, 0.5e-9, 1.0e-9],
        chain_params={
            "carrier_frequency_Hz": [6.45e9],
            "rf_phase_rad": [0.0],
            "lo_frequency_Hz": [5.8e9],
            "if_phase_rad": [0.0],
            "adc_noise_sigma": [0.0],
        },
        shots=[
            ShotData(
                timestamp=0.0,
                a_out=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            )
        ],
    )

    rf = readout.get_rf_signal(0)
    if_sig = readout.get_if_signal(0)
    adc = readout.get_adc_signal(0)

    assert len(rf) == 3
    assert len(if_sig) == 3
    assert len(adc) == 3


def test_run_analysis_builds_one_case_per_study_and_one_summary_with_shot_iq(monkeypatch):
    analyser_cfg = AnalyserConfig(
        solver_id="solver_0",
        analysis=[
            {"name": "readout_analysis", "level": "CASE"},
            {"name": "iq_analysis", "level": "COMPREHENSIVE"},
        ],
    )
    run_ground = ModelRun(
        identity=RunIdentity(run_id="prep_0", solver_id="solver_0", study_name="prep_0", study_index=0),
        runtime_task=SimpleNamespace(input=SimpleNamespace(backend_config=SimpleNamespace(seed=123))),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(), pulse_ir={}, timings={}, decoder_outputs={}),
        results={
            "param_0": RunResult(
                result_id="prep_0_param_0",
                parameters=ParameterValues(parameter_id="param_0", values={}),
                provenance=RunProvenance(solver_id="solver_0", study_name="prep_0", study_index=0),
                trajectories={0: SimpleNamespace()},
                runtime_metadata={},
            ),
        },
    )
    run_excited = ModelRun(
        identity=RunIdentity(run_id="prep_1", solver_id="solver_0", study_name="prep_1", study_index=1),
        runtime_task=SimpleNamespace(input=SimpleNamespace(backend_config=SimpleNamespace(seed=123))),
        artifacts=SimpleNamespace(model_spec=SimpleNamespace(), pulse_ir={}, timings={}, decoder_outputs={}),
        results={
            "param_0": RunResult(
                result_id="prep_1_param_0",
                parameters=ParameterValues(parameter_id="param_0", values={}),
                provenance=RunProvenance(solver_id="solver_0", study_name="prep_1", study_index=1),
                trajectories={0: SimpleNamespace()},
                runtime_metadata={},
            ),
        },
    )
    model = SimpleNamespace(
        analysers={"analyser_0": analyser_cfg},
        runs={"prep_0": run_ground, "prep_1": run_excited},
        analyses={},
        config=SimpleNamespace(parameter_sweep=None),
        metric_registry=None,
        device={},
        pulse={},
    )

    monkeypatch.setattr("musiq.workflow.model_execution.require_analyser_id", lambda model, analyser_id: analyser_id or "analyser_0")
    monkeypatch.setattr("musiq.workflow.model_execution.require_solver_id", lambda model, solver_id: solver_id or "solver_0")
    monkeypatch.setattr("musiq.workflow.contracts.build_effective_pulse_config", lambda device, pulse: {})

    def fake_run_analysis_stage(**kwargs):
        trajectory = kwargs["trajectory"]
        if trajectory is run_ground.results["param_0"].trajectories[0]:
            points = [complex(0.0, 0.0), complex(0.1, 0.0)]
        else:
            points = [complex(1.0, 0.0), complex(1.1, 0.0)]
        return {
            "analysis": CaseAnalysis(
                metrics={},
                readout=SimpleNamespace(integrated_points=points),
                iq=None,
            )
        }

    monkeypatch.setattr("musiq.workflow.model_execution.run_analysis_stage", fake_run_analysis_stage)

    run_analysis(model, analyser_id="analyser_0")

    case_analyses = [analysis for analysis in model.analyses.values() if analysis.scope.value == "case"]
    summaries = [analysis for analysis in model.analyses.values() if analysis.scope.value == "comprehensive"]

    assert len(case_analyses) == 2
    assert len(summaries) == 1
    assert summaries[0].output.confusion_matrix["values"] == [[2, 0], [0, 2]]


def test_run_one_solver_study_compiles_with_only_selected_study_step(monkeypatch):
    sample = StudySample(
        profile_id="default",
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

    captured = {}

    def fake_execute_compilation_unit(current_model, current_sample, *, solver_cfg_override=None, run_id=None, tag=None):
        captured["study"] = list(solver_cfg_override.study or [])
        return ModelRun(
            identity=RunIdentity(run_id=run_id or "prep_1", solver_id=current_sample.solver_id),
            runtime_task=SimpleNamespace(input=SimpleNamespace(study=list(solver_cfg_override.study or []))),
            artifacts=SimpleNamespace(),
            results={},
        )

    monkeypatch.setattr("musiq.workflow.model_execution.execute_compilation_unit", fake_execute_compilation_unit)
    monkeypatch.setattr("musiq.workflow.model_execution.run_sample", lambda current_model, run_obj, current_sample: "ok")

    run_one_solver_study(
        model,
        solver_id="solver_0",
        solver_cfg=SolverConfig(study=[{"name": "prep_0"}, {"name": "prep_1", "prep_state": {"label": "1"}}]),
        study={"name": "prep_1", "prep_state": {"label": "1"}},
        study_index=1,
        total_studies=2,
        tag=None,
    )

    assert captured["study"] == [{"name": "prep_1", "prep_state": {"label": "1"}}]


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

    def fake_execute_compilation_unit(current_model, current_sample, *, solver_cfg_override=None, run_id=None, tag=None):
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

    assert run_ids == ["Relaxation"]
    assert created_run_ids == ["Relaxation"]
    assert list(model.runs.keys()) == ["Relaxation"]
    assert model.runs["Relaxation"].identity.run_id == "Relaxation"
    assert model.runs["Relaxation"].identity.study_name == "Relaxation"
    assert model.runs["Relaxation"].identity.study_index == 0


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

    def fake_build_solver(current_model, solver_id=None, tag=None):
        captured["solver_profiles"] = sorted(current_model.config.profiles.keys())
        captured["solver_id"] = solver_id
        captured["tag"] = tag
        return ["Relaxation"]

    def fake_run_engine(current_model, solver_id=None, tag=None):
        captured["engine_profiles"] = sorted(current_model.config.profiles.keys())
        captured["engine_solver_id"] = solver_id
        captured["engine_tag"] = tag
        return ["Relaxation"]

    def fake_run_analysis(current_model, *, analyser_id=None, study_name_val=None, tag=None, run_ids=None):
        captured["analysis_profiles"] = sorted(current_model.config.profiles.keys())
        captured["analyser_id"] = analyser_id
        captured["analysis_tag"] = tag
        captured["run_ids"] = run_ids

    monkeypatch.setattr("musiq.workflow.model_execution.build_solver", fake_build_solver)
    monkeypatch.setattr("musiq.workflow.model_execution.run_engine", fake_run_engine)
    monkeypatch.setattr("musiq.workflow.model_execution.run_analysis", fake_run_analysis)
    monkeypatch.setattr("musiq.workflow.model.build_solver", fake_build_solver)
    monkeypatch.setattr("musiq.workflow.model.run_engine", fake_run_engine)
    monkeypatch.setattr("musiq.workflow.model.run_analysis", fake_run_analysis)

    run_profile(model, "Relaxation", tag="T1")

    assert captured["solver_profiles"] == ["Relaxation"]
    assert captured["engine_profiles"] == ["Relaxation"]
    assert captured["analysis_profiles"] == ["Relaxation"]
    assert captured["solver_id"] == "solver_0"
    assert captured["engine_solver_id"] == "solver_0"
    assert captured["analyser_id"] == "analyser_0"
    assert captured["tag"] == "T1"
    assert captured["engine_tag"] == "T1"
    assert captured["analysis_tag"] == "T1"
    assert captured["run_ids"] == ["Relaxation"]
    assert sorted(model.config.profiles.keys()) == ["Other", "Relaxation"]


def test_profile_run_solver_isolates_current_profile(monkeypatch):
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
        captured["profiles"] = sorted(current_model.config.profiles.keys())
        captured["solver_id"] = solver_id
        captured["tag"] = tag
        return ["Relaxation__study"]

    monkeypatch.setattr("musiq.workflow.model.run_solver", fake_run_solver)

    profile = Profile(model, "Relaxation", model.config.profiles["Relaxation"])
    run_ids = profile.run_solver(tag="T1")

    assert run_ids == ["Relaxation__study"]
    assert captured["profiles"] == ["Relaxation"]
    assert captured["solver_id"] == "solver_0"
    assert captured["tag"] == "T1"
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


def test_run_one_solver_study_disambiguates_profiles_with_same_study_name(monkeypatch):
    created_run_ids = []
    executed_circuits = {}
    model = SimpleNamespace(
        config=SimpleNamespace(
            profiles={
                "Relaxation": ProfileConfig(circuit_id="x", device_id="device_a", pulse_id="pulse_a", solver_id="solver_0", analyser_id="analyser_0"),
                "Dephasing": ProfileConfig(circuit_id="h", device_id="device_b", pulse_id="pulse_a", solver_id="solver_0", analyser_id="analyser_0"),
            },
            circuits={"x": object(), "h": object()},
            devices={"device_a": object(), "device_b": object()},
            pulses={"pulse_a": object()},
            solvers={"solver_0": object()},
            analysers={"analyser_0": object()},
            parameter_sweep=None,
        ),
        solvers={"solver_0": SimpleNamespace(config=SolverConfig())},
        runs={},
        out_dir=None,
    )

    def fake_execute_compilation_unit(_model, sample, solver_cfg_override=None, run_id=None, tag=None):
        created_run_ids.append(run_id)
        executed_circuits[run_id] = sample.circuit_id
        return ModelRun(
            identity=RunIdentity(run_id=run_id, solver_id=sample.solver_id),
            runtime_task=None,
            results={},
        )

    def fake_run_sample(_model, run_obj, sample):
        param_id = f"param_{len(run_obj.results)}"
        run_obj.results[f"param_{len(run_obj.results)}"] = RunResult(
            result_id=param_id,
            parameters=ParameterValues(parameter_id=param_id, values=dict(sample.params)),
            provenance=RunProvenance(solver_id=sample.solver_id),
            trajectories={0: object()},
        )
        return next(reversed(run_obj.results))

    monkeypatch.setattr("musiq.workflow.model_execution.execute_compilation_unit", fake_execute_compilation_unit)
    monkeypatch.setattr("musiq.workflow.model_execution.run_sample", fake_run_sample)

    run_ids = run_one_solver_study(
        model,
        solver_id="solver_0",
        solver_cfg=SolverConfig(),
        study={"name": "single_qubit_decoherence"},
        study_index=0,
        total_studies=1,
    )

    assert run_ids == ["x__device_a", "h__device_b"]
    assert created_run_ids == run_ids
    assert executed_circuits == {
        "x__device_a": "x",
        "h__device_b": "h",
    }


def test_id_generator_skips_existing_run_and_analysis_ids():
    model = SimpleNamespace(
        runs={"run_0": object(), "run_1": object()},
        analyses={"case_0": object(), "summary_0": object(), "sweep_0": object()},
    )

    assert IDGenerator.next_run_id(model) == "run_2"
    assert IDGenerator.next_analysis_id(model, scope="case") == "case_1"
    assert IDGenerator.next_analysis_id(model, scope="parametric") == "sweep_1"
    assert IDGenerator.next_analysis_id(model, scope="comprehensive") == "summary_1"
