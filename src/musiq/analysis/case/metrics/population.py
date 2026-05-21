"""Population-based metrics for quantum states."""

from typing import Any
from musiq.common.schemas import ModelSpec, Trajectory
from musiq.schemas.results import MetricSeries
from musiq.analysis.common.metrics_utils import _basis_labels, _complex_scalar, _label_excitation_value

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

    return {
        "payload": payload,
        "observable_updates": {},
    }