"""Trajectory plotting and loading helpers."""

from __future__ import annotations

from pathlib import Path
import json

from musiq.analysis.common.trajectory_semantics import state_rows
from musiq.common.schemas import Trajectory, json_restore


def plot_trajectory(ax, trajectory: Trajectory, *, title: str | None = None) -> None:
    """Plot available classical/state trajectory series onto one axis."""
    classical = dict(getattr(trajectory, "classical", {}) or {})
    plotted = False
    for key, payload in classical.items():
        if not isinstance(payload, dict):
            continue
        values = [list(row) for row in list(payload.get("values", []) or [])]
        if not values:
            continue
        labels = list(payload.get("series_labels", []) or [])
        quantity = str(payload.get("quantity", key))
        for idx in range(len(values[0])):
            label = labels[idx] if idx < len(labels) and labels[idx] else f"s{idx}"
            ax.plot(trajectory.times[: len(values)], [row[idx] for row in values], label=f"{quantity}:{label}")
        plotted = True
    fallback_rows = state_rows(trajectory)
    if not plotted and fallback_rows:
        for idx in range(len(fallback_rows[0])):
            ax.plot(trajectory.times[: len(fallback_rows)], [row[idx] for row in fallback_rows], label=f"state[{idx}]")
    ax.set_title(title or f"Trajectory ({trajectory.engine})")
    ax.set_xlabel("t")
    if ax.lines:
        ax.legend()


def make_trajectory_figure(trajectory: Trajectory):
    """Create a figure for a trajectory plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_trajectory(ax, trajectory)
    fig.tight_layout()
    return fig


def load_trajectory_h5(path: str | Path) -> Trajectory:
    """Load ``Trajectory`` from HDF5 file written by workflow artifacts."""
    import h5py

    with h5py.File(path, "r") as h5:
        times = h5["times"][:].tolist()
        engine = h5.attrs.get("engine", "unknown")
        schema_version = str(h5.attrs.get("trajectory_schema_version", "1.0"))
        metadata = {}
        wave_function = None
        density_matrix = None
        classical = {}
        measurements = {}
        if "metadata_json" in h5:
            raw = h5["metadata_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw:
                metadata = dict(json_restore(json.loads(str(raw))))
        if "wave_function_json" in h5:
            raw = h5["wave_function_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw:
                restored = json_restore(json.loads(str(raw)))
                wave_function = dict(restored) if isinstance(restored, dict) else restored
                wave_function = wave_function or None
        if "density_matrix_json" in h5:
            raw = h5["density_matrix_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw:
                restored = json_restore(json.loads(str(raw)))
                density_matrix = dict(restored) if isinstance(restored, dict) else restored
                density_matrix = density_matrix or None
        if "classical_json" in h5:
            raw = h5["classical_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw:
                classical = dict(json_restore(json.loads(str(raw))))
        if "measurements_json" in h5:
            raw = h5["measurements_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw:
                measurements = dict(json_restore(json.loads(str(raw))))
        for key in ("num_qubits", "model_dimension"):
            if key in h5.attrs and key not in metadata:
                metadata[key] = h5.attrs[key].item() if hasattr(h5.attrs[key], "item") else h5.attrs[key]
    return Trajectory(
        schema_version=schema_version,
        engine=str(engine),
        times=list(times),
        wave_function=wave_function,
        density_matrix=density_matrix,
        classical=classical,
        measurements=measurements,
        metadata=metadata,
    )
