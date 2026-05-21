from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from musiq.common.schemas import Trajectory
from musiq.schemas.model import ModelRun, ModelSpec, RunArtifacts, RunIdentity
from musiq.schemas.results import CaseAnalysis, MetricSeries, ResultRef
from musiq.schemas.solver import FrameSpec, SolverSpec, TimeSpec
from musiq.schemas.system import SystemSpec
from musiq.ui.result_summary import attach_compare_status, collect_pulse_metrics, summarize_workflow_result
from musiq.workflow.model import Model, ModelConfig


def test_collect_pulse_metrics_reads_first_matching_npz(tmp_path: Path):
    np.savez(
        tmp_path / "pulse_samples_extra.npz",
        XY_0_t=np.array([0.0, 1.0, 2.0]),
        XY_0_y=np.array([0.0, 1.0, 0.0]),
    )

    metrics = collect_pulse_metrics(tmp_path)

    assert metrics["XY_0_samples"] == 3.0
    assert metrics["XY_0_duration"] == 2.0
    assert metrics["XY_0_peak"] == 1.0
    assert metrics["XY_0_abs_area"] == 1.0


def test_summarize_workflow_result_flattens_model_run_and_analysis(tmp_path: Path):
    np.savez(
        tmp_path / "pulse_samples.npz",
        RO_0_t=np.array([0.0, 2.0]),
        RO_0_y=np.array([0.0, 1.0]),
    )

    trajectory = Trajectory(
        engine="qutip",
        times=[0.0, 1.0],
        density_matrix={
            "actual_kind": "density_matrix",
            "encoding": "complex",
            "snapshots": [
                [[0.9 + 0.0j, 0.0j], [0.0j, 0.1 + 0.0j]],
                [[0.2 + 0.0j, 0.0j], [0.0j, 0.8 + 0.0j]],
            ],
        },
        metadata={
            "num_qubits": 1,
            "model_dimension": 2,
            "details": {"solver_impl": "mesolve", "native_solver": True},
        },
    )

    run_bundle = SimpleNamespace(
        results={
            "param_0": SimpleNamespace(
                trajectories={"shot_0": trajectory},
                runtime_metadata={"solver_mode": "me", "details": {"solver_impl": "mesolve", "native_solver": True}},
            )
        },
        artifacts=RunArtifacts(
            model_spec=ModelSpec(
                solver=SolverSpec(engine="qutip", mode="me"),
                time=TimeSpec(dt_s=1.0, t_end_s=1.0),
                frame=FrameSpec(),
                system=SystemSpec(dimension=2),
            )
        ),
    )
    model = SimpleNamespace(
        runs={"solver_0": {"run_0": run_bundle}},
        analyses={
            "analyser_0": SimpleNamespace(
                analysis_id="analyser_0",
                analyser_id="analyser_0",
                input_results=[ResultRef(run_id="run_0", parameter_id="param_0")],
                output=CaseAnalysis(
                    metrics={
                        "population": MetricSeries(values={"0": [0.9, 0.2], "1": [0.1, 0.8]}),
                        "mean_excited": MetricSeries(values=[0.1, 0.45]),
                        "variance": MetricSeries(values=[0.0, 0.02]),
                    }
                ),
            )
        },
        out_dir=str(tmp_path),
    )
    model.find_analysis_for_run = MethodType(
        lambda self, wanted_run_id: next(
            (analysis for analysis in self.analyses.values() if any(ref.run_id == wanted_run_id for ref in analysis.input_results)),
            None,
        ),
        model,
    )

    row = summarize_workflow_result(
        model,
        task_tag="task1",
        task_title="Task 1",
        case_tag="baseline",
        engine="qutip",
        device={"qubit_freqs_Hz": [5.0e9]},
        noise={"model": "markovian_lindblad"},
        note="demo",
    )

    assert row["task"] == "task1"
    assert row["task_title"] == "Task 1"
    assert row["case"] == "baseline"
    assert row["solver_impl"] == "mesolve"
    assert row["solver"] == "me"
    assert row["RO_0_duration"] == 2.0
    assert row["final_p1_obs"] == 0.8
    assert row["final_p0_obs"] == pytest.approx(0.2)


def test_attach_compare_status_marks_mixed_encodings_for_review():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"task": "task1", "case": "case1", "state_encoding": "per_qubit_excited_probability"},
            {"task": "task1", "case": "case1", "state_encoding": "basis_population_single_qubit"},
        ]
    )

    annotated = attach_compare_status(df)

    assert set(annotated["compare_status"]) == {"semantic-review-needed"}
    assert set(annotated["compare_reason"]) == {"basis_population_single_qubit | per_qubit_excited_probability"}


def test_model_get_analysis_resolves_generated_case_id_by_analyser_and_study():
    model = Model(config=ModelConfig(circuits={}, devices={}, pulses={}, solvers={}, analysers={}))
    model.config.analysers["analyser_0"] = SimpleNamespace(solver_id="solver_0")
    model.runs["solver_0"] = {"run_0": SimpleNamespace(identity=SimpleNamespace(study_name="ground"))}
    model.analyses["case_0"] = SimpleNamespace(
        analysis_id="case_0",
        analyser_id="analyser_0",
        input_results=[ResultRef(run_id="run_0", parameter_id="param_0")],
        scope="case",
    )

    analysis = model.get_analysis(analyser_id="analyser_0", study_name="ground")

    assert analysis is model.analyses["case_0"]
