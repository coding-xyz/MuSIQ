"""Central hub for resolving and registering analysis metrics across different levels."""

from __future__ import annotations

from typing import Any
import numpy as np
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
    requested_case = list(collect_analysis_metrics(cfg, level="CASE", metric_source="registry"))
    requested_sweep = list(collect_analysis_metrics(cfg, level="PARAMETRIC"))
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
            canonical = "population"
        elif lowered == "final_leakage":
            canonical = "leakage"
        elif lowered == "final_coherence_01":
            canonical = "coherence_01"
        elif lowered == "final_fidelity":
            canonical = ""
        else:
            canonical = name
        if not canonical:
            return
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
    trajectories: list[Trajectory] | None = None,
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
                trajectory_list = [traj for traj in list(trajectories or []) if traj is not None]
                if len(trajectory_list) > 1:
                    aggregated_items, observable_updates = _resolve_multi_trajectory_metric(
                        name=name,
                        entry=entry.callable_obj,
                        trajectories=trajectory_list,
                        model_spec=model_spec,
                        metric_cfg=metric_cfg,
                        observable_values=observable_values,
                    )
                    metric_items.update(aggregated_items)
                    for obs_name, obs_value in observable_updates.items():
                        observable_values[str(obs_name)] = float(obs_value)
                    continue

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


def _resolve_multi_trajectory_metric(
    *,
    name: str,
    entry,
    trajectories: list[Trajectory],
    model_spec: ModelSpec,
    metric_cfg: dict[str, Any],
    observable_values: dict[str, float],
) -> tuple[dict[str, MetricSeries], dict[str, float]]:
    payloads: list[dict[str, MetricSeries]] = []
    observable_updates_acc: dict[str, list[float]] = {}

    for trajectory in trajectories:
        result = entry(
            trajectory,
            model_spec,
            metric_cfg,
            {"observable_values": dict(observable_values)},
        )
        payload = result.get("payload")
        if isinstance(payload, dict) and all(isinstance(v, MetricSeries) for v in payload.values()):
            payloads.append(payload)
        elif isinstance(payload, MetricSeries):
            payloads.append({name: payload})
        for obs_name, obs_value in dict(result.get("observable_updates", {}) or {}).items():
            observable_updates_acc.setdefault(str(obs_name), []).append(float(obs_value))

    if not payloads:
        return {}, {}

    aggregated_items: dict[str, MetricSeries] = {}
    payload_keys = list(payloads[0].keys())
    for payload_key in payload_keys:
        series_list = [payload[payload_key] for payload in payloads if payload_key in payload]
        if not series_list:
            continue
        mean_series, std_series = _aggregate_metric_series(series_list)
        aggregated_items[f"{payload_key}_mean"] = mean_series
        aggregated_items[f"{payload_key}_std"] = std_series

    observable_updates = {
        obs_name: float(np.mean(values))
        for obs_name, values in observable_updates_acc.items()
        if values
    }
    return aggregated_items, observable_updates


def _aggregate_metric_series(series_list: list[MetricSeries]) -> tuple[MetricSeries, MetricSeries]:
    reference = series_list[0]
    mean_values, std_values = _aggregate_metric_values([series.values for series in series_list])
    return (
        MetricSeries(times=list(reference.times), values=mean_values),
        MetricSeries(times=list(reference.times), values=std_values),
    )


def _aggregate_metric_values(values_list: list[list[float] | dict[str, list[float]]]) -> tuple[list[float] | dict[str, list[float]], list[float] | dict[str, list[float]]]:
    first = values_list[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        mean_map: dict[str, list[float]] = {}
        std_map: dict[str, list[float]] = {}
        for key in keys:
            stacked = [np.asarray(list(values.get(key, []) or []), dtype=float) for values in values_list if isinstance(values, dict)]
            if not stacked:
                continue
            min_len = min(arr.size for arr in stacked)
            if min_len <= 0:
                mean_map[key] = []
                std_map[key] = []
                continue
            data = np.stack([arr[:min_len] for arr in stacked], axis=0)
            mean_map[key] = np.mean(data, axis=0).tolist()
            std_map[key] = np.std(data, axis=0).tolist()
        return mean_map, std_map

    stacked = [np.asarray(list(values or []), dtype=float) for values in values_list if isinstance(values, list)]
    if not stacked:
        return [], []
    min_len = min(arr.size for arr in stacked)
    if min_len <= 0:
        return [], []
    data = np.stack([arr[:min_len] for arr in stacked], axis=0)
    return np.mean(data, axis=0).tolist(), np.std(data, axis=0).tolist()

__all__ = [
    "DEFAULT_METRIC_REGISTRY",
    "MetricRegistry",
    "build_default_metric_registry",
    "resolve_metrics_payload",
]
