"""Parametric metrics that extract terminal values from case-level analysis."""

from typing import Any
from musiq.common.schemas import ModelSpec, Trajectory
from musiq.schemas.results import MetricSeries
from musiq.analysis.case.metrics.population import metric_population
from musiq.analysis.case.metrics.leakage import metric_leakage
from musiq.analysis.case.metrics.coherence import metric_coherence_01

def _get_terminal_value(metric_fn, trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None) -> float:
    """Helper to execute a case metric and extract the final scalar value."""
    result = metric_fn(trajectory, model_spec, metric_cfg, {})
    payload = result.get("payload")
    
    if isinstance(payload, MetricSeries):
        if not payload.values:
            return 0.0
        # Handle both dict (multi-label) and list (single-label) values
        if isinstance(payload.values, dict):
            # Get the last label's last value
            last_label = sorted(payload.values.keys())[-1]
            series = payload.values[last_label]
            return float(series[-1]) if series else 0.0
        else:
            return float(payload.values[-1]) if payload.values else 0.0
    return 0.0

def metric_final_population(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    """Extract final population values. cfg can specify 'label' (e.g. '1')."""
    label = (metric_cfg or {}).get("label", "1")
    # Note: metric_population returns a dict of series, we need the specific label
    result = metric_population(trajectory, model_spec, metric_cfg, context)
    payload = result.get("payload")
    if isinstance(payload, MetricSeries) and isinstance(payload.values, dict):
        series = payload.values.get(label, [])
        val = float(series[-1]) if series else 0.0
        return {"payload": MetricSeries(values=[val]), "observable_updates": {label: val}}
    return {"payload": MetricSeries(values=[0.0]), "observable_updates": {}}

def metric_final_leakage(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the terminal leakage value."""
    val = _get_terminal_value(metric_leakage, trajectory, model_spec, metric_cfg)
    return {"payload": MetricSeries(values=[val]), "observable_updates": {"final_leakage": val}}

def metric_final_fidelity(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    """
    Extract terminal fidelity. 
    Typically defined as P1 for a target |1> state.
    """
    val = _get_terminal_value(lambda t, m, c, ctx: metric_population(t, m, {"label": "1"}, ctx), trajectory, model_spec, metric_cfg)
    return {"payload": MetricSeries(values=[val]), "observable_updates": {"final_fidelity": val}}

__all__ = ["metric_final_population", "metric_final_leakage", "metric_final_fidelity"]