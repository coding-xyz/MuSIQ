from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt

from musiq.schemas.model import ModelRun, RunArtifacts, RunIdentity
from musiq.schemas.pulse import ChannelSpec, PulseIR, PulseSpec
from musiq.schemas.results import (
    CaseAnalysis,
    MetricSeries,
    MetricSweepValues,
    ModelAnalysis,
    ParameterAxis,
    ParameterValues,
    ParametricAnalysis,
    RunProvenance,
    RunResult,
    Trajectory,
)
from musiq.visualization import (
    integrated_heterodyne_iq,
    make_pulse_figure,
    make_report_figure,
    make_trajectory_figure,
    plot_case_final_population,
    plot_case_iq_cloud,
    plot_case_metrics,
    plot_grouped_bars,
    plot_iq_cloud,
    plot_pulse,
    plot_sweep_metrics,
)


def test_make_pulse_figure_uses_new_visualization_entrypoint():
    pulse_ir = PulseIR(
        t_end_s=20e-9,
        channels=[
            ChannelSpec(
                name="XY_0",
                pulses=[PulseSpec(t0_s=0.0, t1_s=20e-9, amp=0.25, shape="rect")],
            )
        ],
    )

    fig = make_pulse_figure(pulse_ir)

    assert fig.axes
    assert fig.axes[0].get_xlabel() == "time (ns)"
    plt.close(fig)


def test_make_trajectory_and_report_figures_render():
    trajectory = Trajectory(
        engine="mock",
        times=[0.0, 1.0, 2.0],
        classical={"readout": {"values": [[0.0], [0.5], [1.0]], "series_labels": ["p1"], "quantity": "population"}},
    )

    trajectory_fig = make_trajectory_figure(trajectory)
    report_fig = make_report_figure({"error_budget": {"coherence": 0.1, "leakage": 0.02}})

    assert trajectory_fig.axes[0].get_title() == "Trajectory (mock)"
    assert report_fig.axes[0].get_title() == "Error Budget"
    plt.close(trajectory_fig)
    plt.close(report_fig)


def test_integrated_iq_helpers_plot():
    times = np.linspace(0.0, 10e-9, 6)
    trajectory = Trajectory(
        times=list(times),
        classical={"readout": {"measurement_windows": [{"t0_s": float(times[2]), "t1_s": float(times[-1])}]}},
        measurements={
            "records": [
                {
                    "heterodyne_I": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    "heterodyne_Q": [0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
                }
            ]
        },
    )
    case = {"label": "case-0", "trajectory": trajectory}

    points = integrated_heterodyne_iq(case)
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_iq_cloud(ax, case)

    assert points.shape == (1, 2)
    assert np.isclose(points[0, 0], 1.0)
    assert np.isclose(points[0, 1], 0.5)
    assert ax.get_title() == "case-0"
    plt.close(fig)


def _build_fake_model():
    pulse_ir = PulseIR(
        t_end_s=20e-9,
        channels=[
            ChannelSpec(
                name="XY_0",
                pulses=[PulseSpec(t0_s=0.0, t1_s=20e-9, amp=0.25, shape="rect")],
            )
        ],
    )
    trajectory = Trajectory(
        engine="mock",
        times=[0.0, 10.0, 20.0],
        classical={"readout": {"measurement_windows": [{"t0_s": 10.0, "t1_s": 20.0}]}},
        measurements={
            "records": [
                {
                    "heterodyne_I": [0.0, 1.0, 1.0],
                    "heterodyne_Q": [0.0, 0.5, 0.5],
                }
            ]
        },
    )
    run_result = RunResult(
        result_id="result_0",
        parameters=ParameterValues(parameter_id="param_0", values={"pulse:gates.rx.duration_ns": 20.0}),
        provenance=RunProvenance(solver_id="solver_0", study_name="pulse_shape"),
        trajectories={"shot_0": trajectory},
    )
    run_obj = ModelRun(
        identity=RunIdentity(run_id="solver_0", solver_id="solver_0", study_name="pulse_shape"),
        runtime_task=SimpleNamespace(),
        artifacts=RunArtifacts(pulse_ir=pulse_ir),
        results={"param_0": run_result},
    )
    case_analysis = ModelAnalysis(
        analysis_id="case_0",
        analyser_id="analyser_0",
        input_results=[SimpleNamespace(run_id="solver_0", parameter_id="param_0")],
        scope=SimpleNamespace(value="case"),
        output=CaseAnalysis(
            metrics={
                "population": MetricSeries(
                    times=[0.0, 10.0, 20.0],
                    values={"0": [1.0, 0.4, 0.1], "1": [0.0, 0.5, 0.8], "2": [0.0, 0.1, 0.1]},
                ),
                "final_P0": MetricSeries(times=[20.0], values=[0.1]),
            }
        ),
    )
    sweep_analysis = ModelAnalysis(
        analysis_id="sweep_0",
        analyser_id="analyser_0",
        input_results=[],
        scope=SimpleNamespace(value="parametric"),
        output=ParametricAnalysis(
            parameters={
                "pulse:gates.rx.duration_ns": ParameterAxis(
                    parameter_name="pulse:gates.rx.duration_ns",
                    values=[10.0, 20.0, 30.0],
                    unit="ns",
                )
            },
            metrics={
                "final_P0": MetricSweepValues(
                    metric_name="final_P0",
                    dimensions=["pulse:gates.rx.duration_ns"],
                    values=[0.8, 0.5, 0.2],
                    unit=None,
                )
            },
        ),
    )
    return SimpleNamespace(
        runs={"solver_0": run_obj},
        analyses={"case_0": case_analysis, "sweep_0": sweep_analysis},
    )


def test_model_aware_plot_functions():
    model = _build_fake_model()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    plot_pulse(axes[0, 0], model, run_id="solver_0")
    plot_case_metrics(axes[0, 1], model, "case_0", "population")
    plot_sweep_metrics(axes[1, 0], model, "sweep_0", "final_p0")
    plot_case_final_population(axes[1, 1], model, "case_0", "population")

    assert axes[0, 0].get_xlabel() == "time (ns)"
    assert axes[0, 1].get_title() == "case_0: population"
    assert axes[1, 0].get_title() == "sweep_0: final_P0"
    assert axes[1, 1].get_ylabel() == "final population"
    plt.close(fig)


def test_plot_case_iq_cloud_from_model():
    model = _build_fake_model()
    fig, ax = plt.subplots(figsize=(4, 4))

    plot_case_iq_cloud(ax, model, "case_0")

    assert ax.get_title() == "case_0"
    plt.close(fig)


def test_plot_case_metrics_applies_default_and_per_series_styles():
    model = _build_fake_model()
    fig, ax = plt.subplots(figsize=(5, 3))

    plot_case_metrics(
        ax,
        model,
        "case_0",
        "population",
        series_keys=["0"],
        style={"alpha": 0.4},
        series_styles={"0": {"label": "ideal", "linestyle": "--", "color": "C2"}},
    )

    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == "ideal"
    assert ax.lines[0].get_linestyle() == "--"
    assert np.isclose(ax.lines[0].get_alpha(), 0.4)
    plt.close(fig)


def test_plot_grouped_bars_applies_group_styles():
    fig, ax = plt.subplots(figsize=(5, 3))

    plot_grouped_bars(
        ax,
        categories=["0", "1"],
        groups={"baseline": [0.1, 0.2], "ideal": [0.3, 0.4]},
        ylabel="population",
        style={"alpha": 0.5},
        group_styles={"ideal": {"label": "ideal ref", "color": "C3"}},
    )

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "baseline" in labels
    assert "ideal ref" in labels
    assert any(np.isclose(patch.get_alpha(), 0.5) for patch in ax.patches)
    plt.close(fig)
