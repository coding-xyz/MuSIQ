"""Coherence-based metrics for quantum states."""

from typing import Any
from musiq.common.schemas import ModelSpec, Trajectory
from musiq.schemas.results import MetricSeries
from musiq.analysis.common.metrics_utils import _basis_labels, _complex_scalar

def _coherence_series(trajectory: Trajectory, model_spec: ModelSpec, state_a: str = "0", state_b: str = "1") -> dict[str, list[float]]:
    """Extract the magnitude of the coherence between two states as a time series."""
    density_matrix = dict(getattr(trajectory, "density_matrix", {}) or {})
    if not density_matrix:
        return {}
    
    snapshots = list(density_matrix.get("snapshots", []) or [])
    if not snapshots:
        return {}

    # Determine basis labels to find indices of state_a and state_b
    num_qubits = int(model_spec.system.num_qubits or 0)
    levels = (
        int(model_spec.system.transmon_levels or 2)
        if str(model_spec.system.model_type).strip().lower() in {"transmon_nlevel", "cqed_jc", "cqed_dispersive"}
        else 2
    )
    labels = _basis_labels(len(snapshots[0]), num_qubits, max(2, levels))
    
    try:
        idx_a = labels.index(state_a)
        idx_b = labels.index(state_b)
    except ValueError:
        return {}

    series: list[float] = []
    for snapshot in snapshots:
        # density_matrix snapshot is typically a list of lists (rows)
        # we want rho_{ab} = snapshot[idx_a][idx_b]
        val = _complex_scalar(snapshot[idx_a][idx_b])
        series.append(abs(val))
        
    return {f"coherence_{state_a}_{state_b}": series}

def metric_coherence_01(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    """Compute the coherence (magnitude of rho_01) between states |0> and |1>."""
    del metric_cfg, context
    # Use the helper
    coherence_data = _coherence_series(trajectory, model_spec, state_a="0", state_b="1")
    if not coherence_data:
        return {
            "payload": MetricSeries(),
            "observable_updates": {},
        }
    
    # The helper returns a dict keyed by the specific state pair
    # We expect "coherence_0_1"
    label = "coherence_0_1"
    values = coherence_data.get(label, [])
    
    # Align times
    times = list(trajectory.times[:len(values)])
    
    return {
        "payload": MetricSeries(times=times, values=values),
        "observable_updates": {
            "coherence_01": float(values[-1]) if values else 0.0,
        },
    }