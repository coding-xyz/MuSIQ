"""Small quantum-state helpers for notebook and workflow analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from musiq.common.schemas import Trajectory


def final_density_matrix(source: Any) -> np.ndarray:
    """Return the final density matrix from a trajectory-like object.

    ``source`` may be a ``Trajectory``, a solver-run bundle with a ``trajectory``
    attribute, or a ``Model`` containing ``runs``.
    """
    trajectory = source
    if hasattr(source, "trajectory"):
        trajectory = source.trajectory
    elif hasattr(source, "runs") and source.runs:
        # Use the first available run trajectory as representative
        first_run = next(iter(source.runs.values()))
        first_result = next(iter(first_run.results.values()), None)
        trajectory = next(iter(first_result.trajectories.values()), None) if first_result else None

    if not isinstance(trajectory, Trajectory) and not hasattr(trajectory, "density_matrix"):
        raise TypeError("source must be a Trajectory, solver-run bundle, or Model with density-matrix results")

    density_matrix = dict(getattr(trajectory, "density_matrix", {}) or {})
    snapshots = list(density_matrix.get("snapshots", []) or [])
    if not snapshots:
        raise ValueError("trajectory does not contain density_matrix snapshots")
    return np.asarray(snapshots[-1], dtype=complex)


def state_fidelity(rho: Any, psi: Any) -> float:
    """Compute pure-state fidelity ``<psi|rho|psi>``.

    The state vector is normalized defensively. If ``rho`` is larger than the
    target vector, the top-left computational subspace with matching dimension
    is used.
    """
    rho_arr = np.asarray(rho, dtype=complex)
    psi_arr = np.asarray(psi, dtype=complex).reshape(-1)
    norm = np.linalg.norm(psi_arr)
    if norm == 0.0:
        raise ValueError("psi must be nonzero")
    psi_arr = psi_arr / norm
    dim = psi_arr.shape[0]
    if rho_arr.shape[0] < dim or rho_arr.shape[1] < dim:
        raise ValueError("rho dimension is smaller than psi dimension")
    rho_sub = rho_arr[:dim, :dim]
    return float(np.real(np.vdot(psi_arr, rho_sub @ psi_arr)))


def complex_scalar(value: Any) -> complex:
    """Safely coerce a value to a complex scalar."""
    if isinstance(value, complex):
        return value
    if isinstance(value, dict) and "__musiq_complex__" in value:
        pair = list(value.get("__musiq_complex__", []) or [])
        if len(pair) >= 2:
            return complex(float(pair[0]), float(pair[1]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return complex(float(value[0]), float(value[1]))
    return complex(float(value), 0.0)


def basis_labels(dimension: int, num_qubits: int, levels: int) -> list[str]:
    """Generate basis labels for a given Hilbert space dimension."""
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


def label_excitation_value(label: str, *, num_qubits: int) -> float:
    """Compute the average excitation level from a basis label."""
    digits = [int(ch) for ch in str(label) if ch.isdigit()]
    if not digits:
        return 0.0
    if num_qubits > 0 and len(digits) >= num_qubits:
        return float(sum(digits[:num_qubits])) / float(num_qubits)
    return float(sum(digits)) / float(len(digits))


def population_series(trajectory: Trajectory, model_spec: Any) -> dict[str, list[float]]:
    """Extract population time-series for all basis states."""
    # Quantum state path
    density_matrix = dict(getattr(trajectory, "density_matrix", {}) or {})
    wave_function = dict(getattr(trajectory, "wave_function", {}) or {})
    
    if density_matrix:
        qstate = density_matrix
    elif wave_function:
        qstate = wave_function
    else:
        # Fallback to classical path
        return _population_series_from_classical(trajectory)

    snapshots = list(qstate.get("snapshots", []) or [])
    if not snapshots:
        return _population_series_from_classical(trajectory)

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
                    populations.append(max(0.0, float(complex_scalar(row[i]).real)))
        elif actual_kind == "wave_function":
            populations = [abs(complex_scalar(v)) ** 2 for v in snapshot]
        else:
            return _population_series_from_classical(trajectory)

        if not labels:
            labels = basis_labels(len(populations), num_qubits, max(2, levels))
            series = {label: [] for label in labels}
        for idx, label in enumerate(labels):
            value = float(populations[idx]) if idx < len(populations) else 0.0
            series[label].append(value)
    return series


def _population_series_from_classical(trajectory: Trajectory) -> dict[str, list[float]]:
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


def mean_excited_series(series: dict[str, list[float]], model_spec: Any) -> list[float]:
    """Compute the mean excitation level as a time series."""
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
            total += label_excitation_value(label, num_qubits=num_qubits) * float(sample)
        values.append(float(total))
    return values


def variance_series(series: dict[str, list[float]], model_spec: Any) -> list[float]:
    """Compute the variance of excitations as a time series."""
    if not series:
        return []
    num_qubits = int(model_spec.system.num_qubits or 0)
    labels = list(series.keys())
    label_values = {label: label_excitation_value(label, num_qubits=num_qubits) for label in labels}
    means = mean_excited_series(series, model_spec)
    values: list[float] = []
    for idx, mean in enumerate(means):
        total = 0.0
        for label in labels:
            sample = series[label][idx] if idx < len(series[label]) else 0.0
            delta = label_values[label] - mean
            total += float(sample) * float(delta * delta)
        values.append(float(total))
    return values


def coherence_series(trajectory: Trajectory, model_spec: Any, state_a: str = "0", state_b: str = "1") -> dict[str, list[float]]:
    """Extract the magnitude of the coherence between two states as a time series."""
    density_matrix = dict(getattr(trajectory, "density_matrix", {}) or {})
    if not density_matrix:
        return {}
    
    snapshots = list(density_matrix.get("snapshots", []) or [])
    if not snapshots:
        return {}

    num_qubits = int(model_spec.system.num_qubits or 0)
    levels = (
        int(model_spec.system.transmon_levels or 2)
        if str(model_spec.system.model_type).strip().lower() in {"transmon_nlevel", "cqed_jc", "cqed_dispersive"}
        else 2
    )
    labels = basis_labels(len(snapshots[0]), num_qubits, max(2, levels))
    
    try:
        idx_a = labels.index(state_a)
        idx_b = labels.index(state_b)
    except ValueError:
        return {}

    series: list[float] = []
    for snapshot in snapshots:
        val = complex_scalar(snapshot[idx_a][idx_b])
        series.append(abs(val))
        
    return {f"coherence_{state_a}_{state_b}": series}
