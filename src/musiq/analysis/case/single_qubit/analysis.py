"""Analysis implementation for CASE.SingleQubit."""

from __future__ import annotations

from typing import Any
from musiq.analysis.case.metrics.coherence import metric_coherence_01
from musiq.analysis.case.metrics.leakage import metric_leakage
from musiq.analysis.case.metrics.population import metric_population
from musiq.analysis.common.observables import compute_observables
from musiq.common.schemas import Trajectory, ModelSpec, Observables, Report
from musiq.schemas.results import MetricSeries

# Local registry for SingleQubit metrics
METRIC_MAP = {
    "population": metric_population,
    "leakage": metric_leakage,
    "coherence_01": metric_coherence_01,
}

def resolve_single_qubit_metrics(
    trajectory: Trajectory,
    model_spec: ModelSpec,
    requested_metrics: list[str],
    initial_observables: dict[str, float] | None = None,
) -> tuple[dict[str, MetricSeries], dict[str, float], Report]:
    """
    Resolve metrics specific to single-qubit state analysis.
    
    Args:
        trajectory: The input trajectory.
        model_spec: The model specification for basis/system info.
        requested_metrics: List of metric names to compute.
        initial_observables: Starting observables to be updated.
        
    Returns:
        A tuple of (metric_items, updated_observables, report).
    """
    observable_values = dict(initial_observables or {})
    metric_items: dict[str, MetricSeries] = {}

    for name in requested_metrics:
        lowered = name.lower()
        if lowered in METRIC_MAP:
            func = METRIC_MAP[lowered]
            # Call the metric function. context is empty here.
            result = func(trajectory, model_spec, {}, {})
            
            payload = result.get("payload")
            if isinstance(payload, dict) and all(isinstance(v, MetricSeries) for v in payload.values()):
                metric_items.update(payload)
            elif isinstance(payload, MetricSeries):
                metric_items[name] = payload
            else:
                # Handle scalar results as single-point series
                val = _extract_terminal_value(payload)
                if val is not None:
                    metric_items[name] = MetricSeries(values=[float(val)])
            
            # Update observables
            for obs_name, obs_value in result.get("observable_updates", {}).items():
                observable_values[str(obs_name)] = float(obs_value)
        elif name in observable_values:
            metric_items[name] = MetricSeries(values=[float(observable_values[name])])

    # Generate report and error budget
    error_budget = {}
    for key, series in metric_items.items():
        if series.values:
            if isinstance(series.values, dict):
                # Use state "1" or last state as proxy
                val_series = series.values.get("1") or (
                    series.values[sorted(series.values.keys())[-1]] 
                    if series.values else []
                )
                error_budget[key] = float(val_series[-1]) if val_series else 0.0
            else:
                error_budget[key] = float(series.values[-1])

    report = Report(
        summary={
            "metrics": list(metric_items.keys()),
            "metric_mode": "time_series",
            "metric_terminal_values": error_budget,
            "metric_registry": list(METRIC_MAP.keys()),
        },
        error_budget=error_budget,
    )
    
    return metric_items, observable_values, report

def _extract_terminal_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if isinstance(value.get("values"), list) and value.get("values"):
            tail = value["values"][-1]
            if isinstance(tail, (int, float)):
                return float(tail)
    return None


def build_single_qubit_analysis(
    *,
    trajectory: Trajectory,
    model_spec: ModelSpec,
    **_: Any,
) -> dict[str, MetricSeries]:
    """Compatibility wrapper for CASE.SingleQubit dispatch."""

    observables = compute_observables(trajectory)
    requested_metrics = ["population", "leakage", "coherence_01"]
    metric_items, _observable_values, _report = resolve_single_qubit_metrics(
        trajectory,
        model_spec,
        requested_metrics,
        initial_observables=dict(observables.values or {}),
    )
    return metric_items


__all__ = ["resolve_single_qubit_metrics", "build_single_qubit_analysis"]
