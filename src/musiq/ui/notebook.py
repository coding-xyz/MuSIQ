"""Notebook helper utilities for the model-first workflow API."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import time

import numpy as np
import yaml

from musiq.visualization import (
    density_snapshots,
    final_level_population_table,
    integrated_heterodyne_iq,
    integrated_iq_mean_error,
    make_pulse_figure,
    make_report_figure,
    make_trajectory_figure,
    plot_iq_cloud,
    plot_iq_clouds,
    qubit_level_populations,
)
from musiq.workflow.model import Model, create_model


def _iter_model_runs(model: Model):
    for run_id, run_obj in model.runs.items():
        yield run_id, run_obj


def plot_default(model: Model) -> dict[str, object]:
    """Build the default figure bundle for a completed ``Model``.

    Args:
        model: A model that has already completed ``model.run()``.

    Returns:
        A mapping with optional matplotlib figures under
        ``pulses``, ``trajectory``, and ``report``.
    """
    if not model.runs:
        raise ValueError("plot_default expects a model that has already been run.")
    run_id, bundle = next(_iter_model_runs(model))
    result = next(iter(bundle.results.values()), None)
    trajectory = next(iter(result.trajectories.values()), None) if result is not None else None
    assert trajectory is not None

    report_payload = {}
    # Find the first analysis associated with this run
    analysis = model.find_analysis_for_run(run_id)
    if analysis and analysis.output:
        report_payload = dict(getattr(analysis.output, "report", {}) or {})

    return {
        "pulses": make_pulse_figure(bundle.artifacts.pulse_ir) if bundle.artifacts and bundle.artifacts.pulse_ir is not None else None,
        "trajectory": make_trajectory_figure(trajectory),
        "report": make_report_figure(report_payload),
    }


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return dict(payload)


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def _first_available_trajectory(model: Model, *, study_name: str | None = None):
    for getter in (
        lambda: model.get_trajectory(),
        lambda: model.get_trajectory(study_name=study_name) if study_name else None,
    ):
        try:
            trajectory = getter()
        except Exception:
            trajectory = None
        if trajectory is not None:
            return trajectory
    for _, bundle in _iter_model_runs(model):
        result = next(iter(bundle.results.values()), None)
        trajectory = next(iter(result.trajectories.values()), None) if result is not None else None
        if trajectory is not None:
            return trajectory
    keys = list(model.runs.keys())
    raise RuntimeError(f"musiq run finished but no trajectory was found; runs={keys}")


def _first_available_analysis(model: Model, *, study_name: str | None = None):
    for getter in (
        lambda: model.get_analysis(),
        lambda: model.get_analysis(study_name=study_name) if study_name else None,
    ):
        try:
            analysis = getter()
        except Exception:
            analysis = None
        if analysis is not None:
            return analysis
    for run_id, _ in _iter_model_runs(model):
        analysis = model.find_analysis_for_run(run_id)
        if analysis:
            return analysis
    return None


def run_circuit_case(
    circuit_config: str | Path,
    solver_config: str | Path,
    device_config: str | Path,
    *,
    pulse_config: str | Path | None = None,
    analyser_config: str | Path | None = None,
    label: str | None = None,
    suffix: str | None = None,
    param_bindings: dict[str, float] | None = None,
    pulse_updates: dict[str, Any] | None = None,
    seed_offset: int | None = None,
    generated_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Run one musiq circuit/device/solver bundle with optional overrides."""
    source_circuit = Path(circuit_config).resolve()
    source_solver = Path(solver_config).resolve()
    source_device = Path(device_config).resolve()
    source_pulse = Path(pulse_config).resolve() if pulse_config is not None else None
    source_analyser = Path(analyser_config).resolve() if analyser_config is not None else None
    token = str(suffix or int(time.time() * 1000))
    generated_root = Path(generated_dir or source_circuit.parent / "generated_configs").resolve()
    generated_circuit = source_circuit
    generated_solver = source_solver
    generated_pulse = source_pulse

    if seed_offset is not None:
        solver_payload = copy.deepcopy(_load_yaml(source_solver))
        solver_cfg = solver_payload.setdefault("solver", {})
        if "seed" in solver_cfg:
            solver_cfg["seed"] = int(solver_cfg["seed"]) + int(seed_offset)
            generated_solver = _write_yaml(generated_root / f"solver_{token}.yaml", solver_payload)

    if param_bindings is not None:
        circuit_payload = copy.deepcopy(_load_yaml(source_circuit))
        circuit_payload["param_bindings"] = {str(k): float(v) for k, v in param_bindings.items()}
        generated_circuit = _write_yaml(generated_root / f"circuit_{token}.yaml", circuit_payload)

    if pulse_updates:
        if source_pulse is None:
            raise ValueError(f"circuit bundle has no pulse_config: {source_circuit}")
        pulse_payload = copy.deepcopy(_load_yaml(source_pulse))
        pulse_payload.setdefault("pulse", {}).update(dict(pulse_updates))
        generated_pulse = _write_yaml(generated_root / f"pulse_{token}.yaml", pulse_payload)

    model = create_model(
        circuits=generated_circuit,
        solvers=generated_solver,
        devices=source_device,
        pulses=generated_pulse,
        analysers=source_analyser,
    )
    if out_dir is not None:
        model.config.output.out_dir = str(Path(out_dir).resolve())
    model.run()
    saved_dir = model.save()
    trajectory = _first_available_trajectory(model, study_name=study_name)
    return {
        "label": label or token,
        "circuit_config": generated_circuit,
        "out_dir": saved_dir,
        "model": model,
        "trajectory": trajectory,
        "analysis": _first_available_analysis(model, study_name=study_name),
    }


def run_param_sweep(
    circuit_config: str | Path,
    solver_config: str | Path,
    device_config: str | Path,
    param_name: str,
    values,
    *,
    pulse_config: str | Path | None = None,
    analyser_config: str | Path | None = None,
    labels: list[str] | None = None,
    seed_stride: int | None = 1,
    generated_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    study_name: str | None = None,
) -> list[dict[str, Any]]:
    """Run a musiq circuit bundle repeatedly while sweeping one QASM parameter."""
    cases = []
    for idx, value in enumerate(list(values)):
        token = f"{param_name}_{idx:02d}"
        label = labels[idx] if labels and idx < len(labels) else f"{param_name}={float(value):.3g}"
        cases.append(
            run_circuit_case(
                circuit_config,
                solver_config,
                device_config,
                pulse_config=pulse_config,
                analyser_config=analyser_config,
                label=label,
                suffix=token,
                param_bindings={param_name: float(value)},
                seed_offset=(idx * int(seed_stride)) if seed_stride is not None else None,
                generated_dir=generated_dir,
                out_dir=(Path(out_root) / token) if out_root is not None else None,
                study_name=study_name,
            )
        )
    return cases


def run_pulse_sweep(
    circuit_config: str | Path,
    solver_config: str | Path,
    device_config: str | Path,
    pulse_key: str,
    values,
    *,
    pulse_config: str | Path,
    analyser_config: str | Path | None = None,
    labels: list[str] | None = None,
    seed_stride: int | None = 1,
    generated_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    study_name: str | None = None,
) -> list[dict[str, Any]]:
    """Run a musiq circuit bundle repeatedly while sweeping one pulse config field."""
    cases = []
    for idx, value in enumerate(list(values)):
        token = f"{pulse_key}_{idx:02d}"
        label = labels[idx] if labels and idx < len(labels) else f"{pulse_key}={float(value):.3g}"
        cases.append(
            run_circuit_case(
                circuit_config,
                solver_config,
                device_config,
                pulse_config=pulse_config,
                analyser_config=analyser_config,
                label=label,
                suffix=token,
                pulse_updates={pulse_key: float(value)},
                seed_offset=(idx * int(seed_stride)) if seed_stride is not None else None,
                generated_dir=generated_dir,
                out_dir=(Path(out_root) / token) if out_root is not None else None,
                study_name=study_name,
            )
        )
    return cases


__all__ = [
    "plot_default",
    "run_circuit_case",
    "run_param_sweep",
    "run_pulse_sweep",
    "density_snapshots",
    "qubit_level_populations",
    "final_level_population_table",
    "integrated_heterodyne_iq",
    "integrated_iq_mean_error",
    "plot_iq_cloud",
    "plot_iq_clouds",
]
