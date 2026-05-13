"""Registered analyser metrics and metric payload resolution."""

from __future__ import annotations

from typing import Any

from musiq.analysis.observables import compute_observables
from musiq.analysis.registry import MetricRegistry
from musiq.common.schemas import ModelSpec, Observables, Report, Trajectory
from musiq.schemas.results import MetricSeries


def _complex_scalar(value) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, dict) and "__musiq_complex__" in value:
        pair = list(value.get("__musiq_complex__", []) or [])
        if len(pair) >= 2:
            return complex(float(pair[0]), float(pair[1]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return complex(float(value[0]), float(value[1]))
    return complex(float(value), 0.0)


def _basis_labels(dimension: int, num_qubits: int, levels: int) -> list[str]:
    if dimension <= 0:
        return []
    if num_qubits > 0 and levels > 1:
        expected = levels**num_qubits
        if expected == dimension:
            labels: list[str] = []
            for idx in range(dimension):
                digits: list[str] = []
                rem = idx
                for _ in range(num_qubits):
                    digits.append(str(rem % levels))
                    rem //= levels
                labels.append("".join(reversed(digits)))
            return labels
    return [str(i) for i in range(dimension)]


def _label_excitation_value(label: str, *, num_qubits: int) -> float:
    digits = [int(ch) for ch in str(label) if ch.isdigit()]
    if not digits:
        return 0.0
    if num_qubits > 0 and len(digits) >= num_qubits:
        return float(sum(digits[:num_qubits])) / float(num_qubits)
    return float(sum(digits)) / float(len(digits))


def _population_series_from_quantum_state(trajectory: Trajectory, model_spec: ModelSpec) -> dict[str, list[float]]:
    density_matrix = dict(getattr(trajectory, "density_matrix", {}) or {})
    wave_function = dict(getattr(trajectory, "wave_function", {}) or {})
    if density_matrix:
        qstate = density_matrix
    elif wave_function:
        qstate = wave_function
    else:
        return {}
    snapshots = list(qstate.get("snapshots", []) or [])
    if not snapshots:
        return {}
    actual_kind = str(qstate.get("actual_kind", "")).strip().lower()
    num_qubits = int(model_spec.system.num_qubits or 0)
    levels = (
        int(model_spec.system.transmon_levels or 2)
        if str(model_spec.system.model_type).strip().lower() in {"transmon_nlevel", "cqed_jc", "cqed_dispersive"}
        else 2
    )
    series: dict[str, list[float]] = {}
    labels: list[str] = []

    for snapshot in snapshots:
        populations: list[float]
        if actual_kind == "density_matrix":
            populations = []
            for i, row in enumerate(snapshot):
                if i >= len(row):
                    populations.append(0.0)
                else:
                    populations.append(max(0.0, float(_complex_scalar(row[i]).real)))
        elif actual_kind == "wave_function":
            populations = [abs(_complex_scalar(v)) ** 2 for v in snapshot]
        else:
            return {}

        if not labels:
            labels = _basis_labels(len(populations), num_qubits, max(2, levels))
            series = {label: [] for label in labels}
        for idx, label in enumerate(labels):
            value = float(populations[idx]) if idx < len(populations) else 0.0
            series[label].append(value)
    return series


def _population_series_from_classical(trajectory: Trajectory, model_spec: ModelSpec) -> dict[str, list[float]]:
    classical = dict(getattr(trajectory, "classical", {}) or {})
    basis_payload = dict(classical.get("basis_population", {}) or {})
    basis_values = [list(row) for row in list(basis_payload.get("values", []) or [])]
    if basis_values:
        labels = list(basis_payload.get("series_labels", []) or [])
        if not labels and basis_values[0]:
            labels = [str(i) for i in range(len(basis_values[0]))]
        series = {label: [] for label in labels}
        for row in basis_values:
            for idx, label in enumerate(labels):
                series[label].append(float(row[idx]) if idx < len(row) else 0.0)
        return series
    return {}


def _population_series(trajectory: Trajectory, model_spec: ModelSpec) -> dict[str, list[float]]:
    quantum_series = _population_series_from_quantum_state(trajectory, model_spec)
    quantum_len = max((len(values) for values in quantum_series.values()), default=0)
    if quantum_series and (quantum_len > 1 or quantum_len == len(trajectory.times)):
        return quantum_series
    return _population_series_from_classical(trajectory, model_spec) or quantum_series


def _mean_excited_series_from_population(series: dict[str, list[float]], model_spec: ModelSpec) -> list[float]:
    if not series:
        return []
    num_qubits = int(model_spec.system.num_qubits or 0)
    labels = list(series.keys())
    length = max(len(values) for values in series.values())
    values: list[float] = []
    for idx in range(length):
        total = 0.0
        for label in labels:
            sample = series[label][idx] if idx < len(series[label]) else 0.0
            total += _label_excitation_value(label, num_qubits=num_qubits) * float(sample)
        values.append(float(total))
    return values


def _variance_series_from_population(series: dict[str, list[float]], model_spec: ModelSpec) -> list[float]:
    if not series:
        return []
    num_qubits = int(model_spec.system.num_qubits or 0)
    labels = list(series.keys())
    label_values = {label: _label_excitation_value(label, num_qubits=num_qubits) for label in labels}
    means = _mean_excited_series_from_population(series, model_spec)
    values: list[float] = []
    for idx, mean in enumerate(means):
        total = 0.0
        for label in labels:
            sample = series[label][idx] if idx < len(series[label]) else 0.0
            delta = label_values[label] - mean
            total += float(sample) * float(delta * delta)
        values.append(float(total))
    return values


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


def _metric_result(payload: Any, *, observable_updates: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "payload": payload,
        "observable_updates": dict(observable_updates or {}),
    }


def metric_population(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    del metric_cfg, context
    basis_series = _population_series(trajectory, model_spec)
    series_length = max((len(values) for values in basis_series.values()), default=0)

    payload = {
        "population": MetricSeries(
            times=list(trajectory.times[: series_length]),
            values=basis_series,
        )
    }

    return _metric_result(payload, observable_updates={})


def metric_mean_excited(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    del metric_cfg, context
    basis_series = _population_series(trajectory, model_spec)
    mean_series = _mean_excited_series_from_population(basis_series, model_spec)
    updates = {"mean_excited": float(mean_series[-1])} if mean_series else {}
    return _metric_result(
        MetricSeries(times=list(trajectory.times), values=list(mean_series)),
        observable_updates=updates,
    )


def metric_variance(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    del metric_cfg, context
    basis_series = _population_series(trajectory, model_spec)
    variance_series = _variance_series_from_population(basis_series, model_spec)
    updates = {"variance": float(variance_series[-1])} if variance_series else {}
    return _metric_result(
        MetricSeries(times=list(trajectory.times), values=list(variance_series)),
        observable_updates=updates,
    )


def metric_leakage(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    """Compute population leakage outside the computational subspace {|0>, |1>}."""
    del metric_cfg, context
    basis_series = _population_series(trajectory, model_spec)
    if not basis_series:
        return _metric_result(MetricSeries(), observable_updates={})
    
    # Computational subspace is typically states "0" and "1"
    computational_states = {"0", "1"}
    leakage_labels = [label for label in basis_series.keys() if label not in computational_states]
    
    if not leakage_labels:
        return _metric_result(
            MetricSeries(times=list(trajectory.times), values=[0.0] * len(trajectory.times)),
            observable_updates={"leakage": 0.0},
        )
    
    # Sum populations of all non-computational states for each time point
    time_points = list(trajectory.times)
    leakage_values: list[float] = []
    
    # Get the length of the shortest series to avoid index errors, though they should be same
    series_len = min((len(v) for v in basis_series.values()), default=0)
    
    for i in range(series_len):
        total = sum(basis_series[label][i] for label in leakage_labels if i < len(basis_series[label]))
        leakage_values.append(float(total))
    
    # Align times
    aligned_times = time_points[:len(leakage_values)]
    
    return _metric_result(
        MetricSeries(times=aligned_times, values=leakage_values),
        observable_updates={
            "leakage": float(leakage_values[-1]) if leakage_values else 0.0,
        },
    )


def metric_coherence_01(trajectory: Trajectory, model_spec: ModelSpec, metric_cfg: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    """Compute the coherence (magnitude of rho_01) between states |0> and |1>."""
    del metric_cfg, context
    # Use the helper we created
    coherence_data = _coherence_series(trajectory, model_spec, state_a="0", state_b="1")
    if not coherence_data:
        return _metric_result(MetricSeries(), observable_updates={})
    
    label = "coherence_01"
    values = coherence_data[f"coherence_0_1"]
    
    # Align times
    times = list(trajectory.times[:len(values)])
    
    return _metric_result(
        MetricSeries(times=times, values=values),
        observable_updates={
            "coherence_01": float(values[-1]) if values else 0.0,
        },
    )


def _metric_terminal_value(value: Any):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if isinstance(value.get("values"), list) and value.get("values"):
            tail = value["values"][-1]
            if isinstance(tail, (int, float)):
                return float(tail)
        if isinstance(value.get("values"), dict):
            return None
    return None


def build_default_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register("population", metric_population)
    registry.register("leakage", metric_leakage)
    registry.register("coherence_01", metric_coherence_01)
    # Legacy metrics kept for backward compatibility but removed from default list in templates
    registry.register("mean_excited", metric_mean_excited)
    registry.register("variance", metric_variance)
    return registry


DEFAULT_METRIC_REGISTRY = build_default_metric_registry()


def _required_case_metrics(cfg: dict[str, Any]) -> list[str | dict[str, Any]]:
    requested_case = list(cfg.get("case_metrics", []) or cfg.get("metrics", []) or [])
    requested_sweep = list(cfg.get("sweep_metrics", []) or cfg.get("parametric_metrics", []) or [])
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
        # Handle default observables as scalar metrics
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
                    terminal = _metric_terminal_value(payload)
                    if terminal is not None:
                        metric_items[name] = MetricSeries(values=[float(terminal)])
                
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
                # For population distributions, use state "1" or the last state as budget proxy
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
    "metric_population",
    "metric_leakage",
    "metric_coherence_01",
    "metric_mean_excited",
    "metric_variance",
    "resolve_metrics_payload",
]
