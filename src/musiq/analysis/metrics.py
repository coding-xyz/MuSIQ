"""Central hub for resolving and registering analysis metrics across different levels."""

from __future__ import annotations

from typing import Any
from musiq.analysis.registry import MetricRegistry
from musiq.common.schemas import ModelSpec, Trajectory
from musiq.schemas.results import MetricSeries, Observables, Report
from musiq.analysis.definitions import collect_analysis_metrics
from musiq.analysis.common.observables import compute_observables

# Import metrics from their level-specific locations
from musiq.analysis.case.metrics.population import metric_population
from musiq.analysis.case.metrics.leakage import metric_leakage
from musiq.analysis.case.metrics.coherence import metric_coherence_01

from musiq.analysis.parametric.metrics.terminal import (
    metric_final_population,
    metric_final_leakage,
    metric_final_fidelity,
)

from musiq.analysis.comprehensive.metrics.readout import (
    metric_iq_clouds,
    metric_centroids,
    metric_confusion_matrix,
    metric_readout_fidelity,
    metric_snr,
)

def build_default_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    # Case Level
    registry.register("population", metric_population)
    registry.register("leakage", metric_leakage)
    registry.register("coherence_01", metric_coherence_01)
    
    # Parametric Level
    registry.register("final_population", metric_final_population)
    registry.register("final_leakage", metric_final_leakage)
    registry.register("final_fidelity", metric_final_fidelity)
    
    # Comprehensive Level
    registry.register("iq_clouds", metric_iq_clouds)
    registry.register("centroids", metric_centroids)
    registry.register("confusion_matrix", metric_confusion_matrix)
    registry.register("readout_fidelity", metric_readout_fidelity)
    registry.register("snr", metric_snr)
    
    return registry

DEFAULT_METRIC_REGISTRY = build_default_metric_registry()

def _required_case_metrics(cfg: dict[str, Any]) -> list[str | dict[str, Any]]:
    requested_case = list(cfg.get("case_metrics", []) or cfg.get("metrics", []) or [])
    requested_case.extend(collect_analysis_metrics(cfg, level="CASE", metric_source="registry"))
    requested_sweep = list(cfg.get("sweep_metrics", []) or cfg.get("parametric_metrics", []) or [])
    requested_sweep.extend(collect_analysis_metrics(cfg, level="PARAMETRIC"))
    normalized_case: list[str | dict[str, Any]] = []
    seen_case_names: set[str] = set()

    def _append_metric(item: str | dict[str, Any]) -> None:
        if isinstance(item, str):
            name = str(item).strip()
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            return
        if not name:
            return
        lowered = name.lower()
        if lowered.startswith("final_p"):
            canonical = "final_population"
        elif lowered == "final_leakage":
            canonical = "final_leakage"
        elif lowered == "final_coherence_01":
            canonical = "final_coherence_01"
        else:
            canonical = name
        key = canonical.lower()
        if key in seen_case_names:
            return
        normalized_case.append(canonical)
        seen_case_names.add(key)

    for item in requested_case:
        _append_metric(item)
    for item in requested_sweep:
        _append_metric(item)
    return normalized_case

def resolve_metrics_payload(
    trajectory: Trajectory,
    model_spec: ModelSpec,
    analyser_cfg: dict[str, Any] | None,
    *,
    registry: MetricRegistry | None = None,
) -> tuple[dict[str, MetricSeries], Observables, Report]:
    cfg = analyser_cfg or {}
    requested_metrics = _required_case_metrics(cfg)

    observables = compute_observables(trajectory)
    observable_values = dict(observables.values or {})
    metric_items: dict[str, MetricSeries] = {}
    metric_registry = registry or DEFAULT_METRIC_REGISTRY

    if not requested_metrics:
        for key, val in observable_values.items():
            metric_items[key] = MetricSeries(values=[float(val)])
    else:
        for item in requested_metrics:
            if isinstance(item, str):
                name = str(item).strip()
                metric_cfg = {}
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                metric_cfg = dict(item)
            else:
                continue
            if not name:
                continue
            key = name.lower()
            if metric_registry.has(key):
                entry = metric_registry.get(key)
                # Handle both per-trajectory and per-run metrics
                # For now, we pass trajectory; the comprehensive metrics 
                # will need to be called differently by the dispatcher if they need ALL trajectories.
                result = entry.callable_obj(
                    trajectory,
                    model_spec,
                    metric_cfg,
                    {"observable_values": dict(observable_values)},
                )
                payload = result.get("payload")
                if isinstance(payload, dict) and all(isinstance(v, MetricSeries) for v in payload.values()):
                    metric_items.update(payload)
                elif isinstance(payload, MetricSeries):
                    metric_items[name] = payload
                else:
                    try:
                        val = float(payload) if payload is not None else 0.0
                        metric_items[name] = MetricSeries(values=[val])
                    except (TypeError, ValueError):
                        pass
                
                for obs_name, obs_value in dict(result.get("observable_updates", {}) or {}).items():
                    observable_values[str(obs_name)] = float(obs_value)
                continue
            if key in observable_values:
                metric_items[name] = MetricSeries(values=[float(observable_values[key])])
            elif name in observable_values:
                metric_items[name] = MetricSeries(values=[float(observable_values[name])])

    error_budget = {}
    for key, series in metric_items.items():
        if series.values:
            if isinstance(series.values, dict):
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
            "metric_registry": metric_registry.names(),
        },
        error_budget=error_budget,
    )
    return metric_items, Observables(values=observable_values), report

__all__ = [
    "DEFAULT_METRIC_REGISTRY",
    "MetricRegistry",
    "build_default_metric_registry",
    "resolve_metrics_payload",
]
