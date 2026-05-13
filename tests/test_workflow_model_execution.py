from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from musiq.schemas.model import ModelRun, RunIdentity
from musiq.schemas.results import MetricSeries, RunResult, RunProvenance, ParameterValues
from musiq.workflow.contracts import SolverBackendConfig, SolverConfig, WorkflowRunOptions
from musiq.workflow.model_execution import (
    _extract_case_metric_terminal,
    _extract_final_fidelity,
    build_multi_study_iq_summary,
    clone_solver_cfg_with_single_study,
    find_run_id,
    get_study_entries,
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

    monkeypatch.setattr("workflow.model_execution.parse_compile_lower_model", fake_parse_compile_lower_model)
    monkeypatch.setattr("workflow.model_execution.run_engine_stage", fake_run_engine_stage)
    monkeypatch.setattr("workflow.model_execution.build_execution_plan", lambda task: SimpleNamespace(run_decode=False))

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

    sample = StudySample(task_id="task_0", device_id="device_0", pulse_id="pulse_0", solver_id="solver_0", params={"theta": 1.25})
    run_sample(SimpleNamespace(), run_obj, sample)

    assert compile_calls == [{"theta": 1.25}]
    stored_result = next(iter(run_obj.results.values()))
    assert stored_result.runtime_metadata["param_bindings"] == {"theta": 1.25}
    assert stored_result.trajectories["shot_0"].metadata["bound_params"] == {"theta": 1.25}
