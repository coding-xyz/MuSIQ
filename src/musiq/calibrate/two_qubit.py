from __future__ import annotations

import numpy as np

from musiq.analysis.common.state_utils import basis_labels, complex_scalar, final_density_matrix, state_fidelity
from musiq.pulse.catalog import _channel_name_for_gate
from musiq.schemas.circuit import CircuitGate, CircuitIR

from .common import (
    COUPLER_ID_RE,
    CalibrationConfig,
    CalibrationTarget,
    GateCalibrationResult,
    build_circuit,
    config_resource_ids,
    optimize_parameters,
    proximity_penalty,
    prepare_target_calibration_model,
    resolved_gate_param,
    run_model,
    set_gate_param,
    two_qubit_bounds,
)


def _h_sequence(qubit: int) -> list[CircuitGate]:
    return [
        CircuitGate(name="rz", qubits=[int(qubit)], params=[-0.5 * np.pi]),
        CircuitGate(name="sx", qubits=[int(qubit)]),
        CircuitGate(name="rz", qubits=[int(qubit)], params=[-0.5 * np.pi]),
    ]


def _cz_bell_circuit(*, control: int, target: int, num_qubits: int) -> CircuitIR:
    gates: list[CircuitGate] = []
    gates.extend(_h_sequence(int(control)))
    gates.extend(_h_sequence(int(target)))
    gates.append(CircuitGate(name="cz", qubits=[int(control), int(target)]))
    gates.extend(_h_sequence(int(target)))
    return build_circuit(gates, num_qubits=num_qubits)


def _iswap_transfer_circuit(*, source: int, sink: int, num_qubits: int) -> CircuitIR:
    return build_circuit(
        [
            CircuitGate(name="x", qubits=[int(source)]),
            CircuitGate(name="iswap", qubits=[int(source), int(sink)]),
        ],
        num_qubits=num_qubits,
    )


def _computational_leakage(population: dict[str, float]) -> float:
    retained = 0.0
    for label, value in population.items():
        if set(str(label)) <= {"0", "1"}:
            retained += float(value)
    return max(0.0, 1.0 - retained)


def _label_combine(*labels: str) -> str:
    if not labels:
        return ""
    digits = [max(int(chars[idx]) for chars in labels) for idx in range(len(labels[0]))]
    return "".join(str(value) for value in digits)


def _final_state_fidelity(trajectory, model_spec, target_states: dict[str, complex]) -> float:
    try:
        rho = final_density_matrix(trajectory)
    except Exception:
        rho = None
    num_qubits = int(model_spec.system.num_qubits or 0)
    levels = int(model_spec.system.transmon_levels or 2)
    if rho is not None and len(np.asarray(rho).shape) == 2:
        labels = basis_labels(rho.shape[0], num_qubits, max(2, levels))
        target = np.zeros(rho.shape[0], dtype=complex)
        for label, amp in target_states.items():
            target[labels.index(label)] = complex(amp)
        return float(state_fidelity(rho, target))

    snapshots = list(getattr(trajectory, "wave_function", []) or [])
    if not snapshots:
        return 0.0
    psi = np.asarray([complex_scalar(value) for value in list(snapshots[-1])], dtype=complex)
    labels = basis_labels(psi.shape[0], num_qubits, max(2, levels))
    target = np.zeros(psi.shape[0], dtype=complex)
    for label, amp in target_states.items():
        target[labels.index(label)] = complex(amp)
    norm = np.linalg.norm(target)
    if norm == 0.0:
        return 0.0
    target = target / norm
    return float(abs(np.vdot(target, psi)) ** 2)


def infer_excited_labels(
    working_model,
    *,
    config: CalibrationConfig,
    scope_components_list: list[str],
) -> dict[int, str]:
    config_resource_ids(config, context="Two-qubit calibration")
    labels: dict[int, str] = {}
    for qubit_index in range(len(scope_components_list)):
        probe_target = CalibrationTarget(
            key="probe",
            kind="single",
            component_ids=tuple(scope_components_list),
            qubit_indices=tuple(range(len(scope_components_list))),
            channel_name="",
            scope_components=tuple(scope_components_list),
            scope_connections=tuple(),
        )
        probe = prepare_target_calibration_model(
            working_model,
            config=config,
            target=probe_target,
            circuit_ir=build_circuit([CircuitGate(name="x", qubits=[int(qubit_index)])], num_qubits=len(scope_components_list)),
            context="Two-qubit calibration",
        )
        _, _, terminal = run_model(probe)
        candidates = [(label, value) for label, value in terminal.items() if label != "0" * len(label)]
        best_label = max(candidates or list(terminal.items()), key=lambda item: item[1])[0]
        labels[qubit_index] = str(best_label)
    return labels


def two_qubit_target_metric(
    gate_name: str,
    *,
    trajectory,
    model_spec,
    terminal_population: dict[str, float],
    excited_labels: dict[int, str],
    qubit_indices: tuple[int, int],
) -> tuple[str, float, float]:
    q0, q1 = qubit_indices
    label_q0 = excited_labels[q0]
    label_q1 = excited_labels[q1]
    if gate_name == "cz":
        bell_label = _label_combine(label_q0, label_q1)
        fidelity = _final_state_fidelity(
            trajectory,
            model_spec,
            {"0" * len(label_q0): 1.0 / np.sqrt(2.0), bell_label: 1.0 / np.sqrt(2.0)},
        )
        leakage = _computational_leakage(terminal_population)
        loss = (1.0 - fidelity) + 0.5 * leakage
        return "bell_fidelity", float(fidelity), float(loss)
    if gate_name == "iswap":
        fidelity = _final_state_fidelity(trajectory, model_spec, {label_q1: 1.0})
        leakage = _computational_leakage(terminal_population)
        loss = (1.0 - fidelity) + 0.5 * leakage
        return "swap_fidelity", float(fidelity), float(loss)
    raise ValueError(f"Unsupported two-qubit calibration gate `{gate_name}`.")


def calibrate_two_qubit_gate(
    working_model,
    target: CalibrationTarget,
    *,
    gate_name: str,
    config: CalibrationConfig,
) -> GateCalibrationResult:
    pulse_id, _, _ = config_resource_ids(config, context="Two-qubit calibration")
    q0, q1 = target.qubit_indices
    channel_name = str(target.channel_name)
    initial_amplitude = float(resolved_gate_param(working_model, pulse_id=pulse_id, gate_name=gate_name, param_name="amplitude_Hz", channel_name=channel_name))
    initial_duration = float(
        resolved_gate_param(working_model, pulse_id=pulse_id, gate_name=gate_name, param_name="duration_ns", channel_name=channel_name)
        or resolved_gate_param(working_model, pulse_id=pulse_id, gate_name=gate_name, param_name="duration_ns")
    )

    if gate_name == "cz":
        circuit = _cz_bell_circuit(control=q0, target=q1, num_qubits=len(target.scope_components))
    elif gate_name == "iswap":
        circuit = _iswap_transfer_circuit(source=q0, sink=q1, num_qubits=len(target.scope_components))
    else:
        raise ValueError(f"Unsupported two-qubit calibration gate `{gate_name}`.")

    template = prepare_target_calibration_model(
        working_model,
        config=config,
        target=target,
        circuit_ir=circuit,
        context="Two-qubit calibration",
    )
    excited_labels = infer_excited_labels(
        working_model,
        config=config,
        scope_components_list=list(target.scope_components),
    )

    cache: dict[tuple[float, ...], float] = {}
    initial_values, bounds = two_qubit_bounds(initial_amplitude, initial_duration, relative_span=float(config.relative_span))

    def objective(params: list[float]) -> float:
        key = tuple(round(float(value), 6) for value in params)
        if key in cache:
            return cache[key]
        amplitude_hz, duration_ns = float(params[0]), float(params[1])
        trial = template.copy(include_results=False)
        set_gate_param(
            trial,
            pulse_id=pulse_id,
            gate_name=gate_name,
            param_name="amplitude_Hz",
            value=amplitude_hz,
            channel_name=channel_name,
        )
        set_gate_param(
            trial,
            pulse_id=pulse_id,
            gate_name=gate_name,
            param_name="duration_ns",
            value=duration_ns,
            channel_name=channel_name,
        )
        trajectory, model_spec, terminal = run_model(trial)
        _, _, loss = two_qubit_target_metric(
            gate_name,
            trajectory=trajectory,
            model_spec=model_spec,
            terminal_population=terminal,
            excited_labels=excited_labels,
            qubit_indices=(q0, q1),
        )
        loss += proximity_penalty(
            params,
            initial_values,
            bounds,
            weight=float(config.proximity_weight),
        )
        cache[key] = float(loss)
        return float(loss)

    best_values, best_loss = optimize_parameters(
        initial_values,
        bounds,
        objective,
        points=int(config.points),
        maxiter=int(config.maxiter),
    )

    best_model = template.copy(include_results=False)
    set_gate_param(
        best_model,
        pulse_id=pulse_id,
        gate_name=gate_name,
        param_name="amplitude_Hz",
        value=float(best_values[0]),
        channel_name=channel_name,
    )
    set_gate_param(
        best_model,
        pulse_id=pulse_id,
        gate_name=gate_name,
        param_name="duration_ns",
        value=float(best_values[1]),
        channel_name=channel_name,
    )
    trajectory, model_spec, terminal = run_model(best_model)
    metric_name, metric_value, _ = two_qubit_target_metric(
        gate_name,
        trajectory=trajectory,
        model_spec=model_spec,
        terminal_population=terminal,
        excited_labels=excited_labels,
        qubit_indices=(q0, q1),
    )
    return GateCalibrationResult(
        gate_name=gate_name,
        channel_name=channel_name,
        target_components=tuple(target.component_ids),
        amplitude_Hz=float(best_values[0]),
        initial_amplitude_Hz=float(initial_amplitude),
        duration_ns=float(best_values[1]),
        initial_duration_ns=float(initial_duration),
        loss=float(best_loss),
        terminal_population=terminal,
        target_metric_name=metric_name,
        target_metric_value=float(metric_value),
    )


def resolve_iswap_target_channel(model, *, device_id: str, target: CalibrationTarget) -> CalibrationTarget:
    if not target.channel_name.startswith("C_"):
        return target
    return CalibrationTarget(
        key=target.key,
        kind=target.kind,
        component_ids=target.component_ids,
        qubit_indices=target.qubit_indices,
        channel_name=str(
            _channel_name_for_gate(
                "iswap",
                list(target.qubit_indices),
                None,
                None,
                hw=dict(model.config.devices[device_id].device or {}),
            )
            or target.channel_name
        ),
        scope_components=target.scope_components,
        scope_connections=target.scope_connections,
    )
