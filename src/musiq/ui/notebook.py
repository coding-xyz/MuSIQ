"""Notebook helper utilities for the model-first workflow API."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import time

import numpy as np
import yaml

from musiq.pulse.visualize import plot_pulses, plot_report, plot_trajectory
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
        "pulses": plot_pulses(bundle.artifacts.pulse_ir) if bundle.artifacts and bundle.artifacts.pulse_ir is not None else None,
        "trajectory": plot_trajectory(trajectory),
        "report": plot_report(report_payload),
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


def density_snapshots(source: Any) -> np.ndarray:
    """Return density-matrix snapshots from a musiq case or trajectory."""
    trajectory = source.get("trajectory") if isinstance(source, dict) else source
    density_matrix = getattr(trajectory, "density_matrix", None)
    if isinstance(density_matrix, dict):
        return np.asarray(density_matrix.get("snapshots", []) or [], dtype=complex)
    return np.asarray(density_matrix or [], dtype=complex)


def _case_model_spec(case: dict[str, Any]):
    _, run_obj = next(_iter_model_runs(case["model"]))
    return run_obj.artifacts.model_spec


def qubit_level_populations(case: dict[str, Any], *, normalize: bool = True) -> np.ndarray:
    """Return transmon-level populations from density matrices in a musiq case."""
    rho = density_snapshots(case)
    if rho.size == 0:
        return np.zeros((0, 0), dtype=float)
    model_spec = _case_model_spec(case)
    levels = int(model_spec.system.transmon_levels or 2)
    cavity_dim = int(model_spec.system.cavity_nmax or 0) + 1
    pops = np.zeros((rho.shape[0], levels), dtype=float)
    for t_idx, mat in enumerate(rho):
        for c in range(cavity_dim):
            for level in range(levels):
                idx = c * levels + level
                if idx < mat.shape[0]:
                    pops[t_idx, level] += float(np.real(mat[idx, idx]))
    pops = np.clip(pops, 0.0, None)
    if normalize:
        norm = pops.sum(axis=1, keepdims=True)
        pops = np.divide(pops, norm, out=np.zeros_like(pops), where=norm > 0.0)
    return pops


def final_level_population_table(cases: list[dict[str, Any]]) -> np.ndarray:
    """Return one row of final level populations for each musiq case."""
    rows = []
    width = 0
    for case in cases:
        pops = qubit_level_populations(case)
        row = pops[-1] if pops.size else np.asarray([], dtype=float)
        width = max(width, int(row.size))
        rows.append(row)
    if width == 0:
        return np.zeros((len(rows), 0), dtype=float)
    return np.asarray([np.pad(row, (0, width - row.size), constant_values=np.nan) for row in rows], dtype=float)


def integrated_heterodyne_iq(case: dict[str, Any]) -> np.ndarray:
    """Integrate per-shot heterodyne I/Q records over the readout window."""
    trajectory = case["trajectory"]
    times = np.asarray(trajectory.times, dtype=float)
    records = list((trajectory.measurements or {}).get("records", []) or [])
    if times.size == 0 or not records:
        return np.zeros((0, 2), dtype=float)

    readout = dict((trajectory.classical or {}).get("readout", {}) or {})
    windows = list(readout.get("measurement_windows", []) or readout.get("readout_windows", []) or [])
    if windows:
        t0 = float(windows[-1].get("t0_s", times[0]))
        t1 = float(windows[-1].get("t1_s", times[-1]))
    else:
        t0 = float(times[int(0.55 * max(len(times) - 1, 0))])
        t1 = float(times[-1])
    mask = (times >= t0) & (times <= t1)

    points = []
    for record in records:
        i_vals = np.asarray(record.get("heterodyne_I", []), dtype=float)
        q_vals = np.asarray(record.get("heterodyne_Q", []), dtype=float)
        if i_vals.size != times.size or q_vals.size != times.size or not np.any(mask):
            continue
        duration = max(float(times[mask][-1] - times[mask][0]), np.finfo(float).eps)
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:
            integrate = getattr(np, "trapz")
        points.append([integrate(i_vals[mask], times[mask]) / duration, integrate(q_vals[mask], times[mask]) / duration])
    return np.asarray(points, dtype=float)


def integrated_iq_mean_error(cases: list[dict[str, Any]], *, error: str = "sem") -> dict[str, np.ndarray]:
    """Return per-case integrated I/Q mean and error bars from shot records."""
    means = []
    errors = []
    stds = []
    counts = []
    for case in cases:
        points = integrated_heterodyne_iq(case)
        counts.append(int(points.shape[0]))
        if points.size == 0:
            means.append([np.nan, np.nan])
            errors.append([np.nan, np.nan])
            stds.append([np.nan, np.nan])
            continue
        mean = np.nanmean(points, axis=0)
        std = np.nanstd(points, axis=0, ddof=1) if points.shape[0] > 1 else np.zeros(2, dtype=float)
        if error == "std":
            err = std
        elif error == "sem":
            err = std / max(float(np.sqrt(points.shape[0])), 1.0)
        else:
            raise ValueError("error must be 'sem' or 'std'")
        means.append(mean)
        errors.append(err)
        stds.append(std)
    return {
        "mean": np.asarray(means, dtype=float),
        "error": np.asarray(errors, dtype=float),
        "std": np.asarray(stds, dtype=float),
        "n": np.asarray(counts, dtype=int),
    }


def plot_iq_cloud(ax, case: dict[str, Any], title: str | None = None) -> None:
    """Plot integrated heterodyne I/Q points for one musiq case."""
    points = integrated_heterodyne_iq(case)
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=24, alpha=0.72)
        ax.scatter([points[:, 0].mean()], [points[:, 1].mean()], marker="x", s=90, color="black")
    ax.set_title(title or str(case.get("label", "")))
    ax.set_xlabel("integrated I")
    ax.set_ylabel("integrated Q")
    ax.axis("equal")


def plot_iq_clouds(ax, cases: list[dict[str, Any]], title: str | None = None) -> None:
    """Plot multiple integrated heterodyne I/Q clouds on one axis."""
    for case in cases:
        points = integrated_heterodyne_iq(case)
        label = str(case.get("label", ""))
        if not points.size:
            continue
        # ``Axes._get_lines.prop_cycler`` is not available in some matplotlib versions.
        # Prefer the stable helper and keep a safe fallback.
        if hasattr(ax._get_lines, "get_next_color"):
            color = ax._get_lines.get_next_color()
        else:
            color = None
        ax.scatter(points[:, 0], points[:, 1], s=20, alpha=0.45, color=color, label=label)
        ax.scatter([points[:, 0].mean()], [points[:, 1].mean()], marker="x", s=90, color=color)
    ax.set_title(title or "Integrated I/Q clouds")
    ax.set_xlabel("integrated I")
    ax.set_ylabel("integrated Q")
    ax.axis("equal")
    ax.legend()


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
