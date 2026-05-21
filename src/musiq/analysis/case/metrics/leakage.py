"""Leakage-based metrics for quantum states."""

from typing import Any
from musiq.common.schemas import ModelSpec, Trajectory
from musiq.schemas.results import MetricSeries
from musiq.analysis.case.metrics.population import _population_series

def metric_leakage(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    """Compute population leakage outside the computational subspace {|0>, |1>}."""
    del metric_cfg, context
    basis_series = _population_series(trajectory, model_spec)
    if not basis_series:
        return {
            "payload": MetricSeries(),
            "observable_updates": {},
        }
    
    # Computational subspace is typically states "0" and "1"
    computational_states = {"0", "1"}
    leakage_labels = [label for label in basis_series.keys() if label not in computational_states]
    
    if not leakage_labels:
        return {
            "payload": MetricSeries(
                times=list(trajectory.times), 
                values=[0.0] * len(trajectory.times)
            ),
            "observable_updates": {"leakage": 0.0},
        }
    
    # Sum populations of all non-computational states for each time point
    time_points = list(trajectory.times)
    leakage_values: list[float] = []
    
    # Get the length of the shortest series to avoid index errors
    series_len = min((len(v) for v in basis_series.values()), default=0)
    
    for i in range(series_len):
        total = sum(basis_series[label][i] for label in leakage_labels if i < len(basis_series[label]))
        leakage_values.append(float(total))
    
    # Align times
    aligned_times = time_points[:len(leakage_values)]
    
    return {
        "payload": MetricSeries(times=aligned_times, values=leakage_values),
        "observable_updates": {
            "leakage": float(leakage_values[-1]) if leakage_values else 0.0,
        },
    }