from __future__ import annotations

from musiq.schemas.circuit import CircuitGate, CircuitIR

from .common import (
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
    single_bounds,
)


def _single_gate_circuit(gate_name: str, *, qubit: int, num_qubits: int) -> CircuitIR:
    return build_circuit([CircuitGate(name=str(gate_name), qubits=[int(qubit)])], num_qubits=num_qubits)


def _double_sx_circuit(*, qubit: int, num_qubits: int) -> CircuitIR:
    return build_circuit(
        [CircuitGate(name="sx", qubits=[int(qubit)]), CircuitGate(name="sx", qubits=[int(qubit)])],
        num_qubits=num_qubits,
    )


def _x_phase_circuit(*, qubit: int, num_qubits: int) -> CircuitIR:
    return build_circuit(
        [
            CircuitGate(name="sx", qubits=[int(qubit)]),
            CircuitGate(name="x", qubits=[int(qubit)]),
            CircuitGate(name="sx", qubits=[int(qubit)]),
        ],
        num_qubits=num_qubits,
    )


def _loss_sx(single_pop: dict[str, float], twice_pop: dict[str, float]) -> float:
    return (
        (single_pop.get("0", 0.0) - 0.5) ** 2
        + (single_pop.get("1", 0.0) - 0.5) ** 2
        + 5.0 * single_pop.get("2", 0.0) ** 2
        + (twice_pop.get("1", 0.0) - 1.0) ** 2
        + twice_pop.get("0", 0.0) ** 2
        + 3.0 * twice_pop.get("2", 0.0) ** 2
    )


def _loss_x(direct_pop: dict[str, float], phase_pop: dict[str, float]) -> float:
    return (
        (direct_pop.get("1", 0.0) - 1.0) ** 2
        + direct_pop.get("0", 0.0) ** 2
        + 5.0 * direct_pop.get("2", 0.0) ** 2
        + phase_pop.get("1", 0.0) ** 2
        + 3.0 * phase_pop.get("2", 0.0) ** 2
    )


def calibrate_single_gate(
    working_model,
    target: CalibrationTarget,
    *,
    gate_name: str,
    config: CalibrationConfig,
) -> GateCalibrationResult:
    pulse_id, _, _ = config_resource_ids(config, context="Single-qubit calibration")
    qubit_index = int(target.qubit_indices[0])
    channel_name = str(target.channel_name)
    initial_amplitude = float(
        resolved_gate_param(
            working_model,
            pulse_id=pulse_id,
            gate_name=gate_name,
            param_name="amplitude_Hz",
            channel_name=channel_name,
        )
    )
    raw_drag_beta = resolved_gate_param(
        working_model,
        pulse_id=pulse_id,
        gate_name=gate_name,
        param_name="drag_beta",
        channel_name=channel_name,
    )
    initial_drag_beta = float(raw_drag_beta) if raw_drag_beta is not None else None

    if gate_name == "sx":
        primary_circuit = _single_gate_circuit("sx", qubit=qubit_index, num_qubits=len(target.scope_components))
        secondary_circuit = _double_sx_circuit(qubit=qubit_index, num_qubits=len(target.scope_components))
    elif gate_name == "x":
        primary_circuit = _single_gate_circuit("x", qubit=qubit_index, num_qubits=len(target.scope_components))
        secondary_circuit = _x_phase_circuit(qubit=qubit_index, num_qubits=len(target.scope_components))
    else:
        raise ValueError(f"Unsupported single-qubit calibration gate `{gate_name}`.")

    primary_template = prepare_target_calibration_model(
        working_model,
        config=config,
        target=target,
        circuit_ir=primary_circuit,
        context="Single-qubit calibration",
    )
    secondary_template = prepare_target_calibration_model(
        working_model,
        config=config,
        target=target,
        circuit_ir=secondary_circuit,
        context="Single-qubit calibration",
    )

    cache: dict[tuple[float, ...], float] = {}
    initial_values, bounds = single_bounds(initial_amplitude, initial_drag_beta, relative_span=float(config.relative_span))

    def objective(params: list[float]) -> float:
        key = tuple(round(float(value), 6) for value in params)
        if key in cache:
            return cache[key]
        amplitude_hz = float(params[0])
        drag_beta = float(params[1]) if len(params) > 1 else initial_drag_beta

        primary_trial = primary_template.copy(include_results=False)
        secondary_trial = secondary_template.copy(include_results=False)
        for trial in (primary_trial, secondary_trial):
            set_gate_param(
                trial,
                pulse_id=pulse_id,
                gate_name=gate_name,
                param_name="amplitude_Hz",
                value=amplitude_hz,
                channel_name=channel_name,
            )
            if drag_beta is not None:
                set_gate_param(
                    trial,
                    pulse_id=pulse_id,
                    gate_name=gate_name,
                    param_name="drag_beta",
                    value=drag_beta,
                    channel_name=channel_name,
                )
        _, _, primary_pop = run_model(primary_trial)
        _, _, secondary_pop = run_model(secondary_trial)
        loss = _loss_sx(primary_pop, secondary_pop) if gate_name == "sx" else _loss_x(primary_pop, secondary_pop)
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

    best_model = primary_template.copy(include_results=False)
    set_gate_param(
        best_model,
        pulse_id=pulse_id,
        gate_name=gate_name,
        param_name="amplitude_Hz",
        value=float(best_values[0]),
        channel_name=channel_name,
    )
    best_drag_beta = float(best_values[1]) if len(best_values) > 1 else initial_drag_beta
    if best_drag_beta is not None:
        set_gate_param(
            best_model,
            pulse_id=pulse_id,
            gate_name=gate_name,
            param_name="drag_beta",
            value=float(best_drag_beta),
            channel_name=channel_name,
        )
    _, _, terminal = run_model(best_model)
    return GateCalibrationResult(
        gate_name=gate_name,
        channel_name=channel_name,
        target_components=tuple(target.component_ids),
        amplitude_Hz=float(best_values[0]),
        initial_amplitude_Hz=float(initial_amplitude),
        drag_beta=float(best_drag_beta) if best_drag_beta is not None else None,
        initial_drag_beta=float(initial_drag_beta) if initial_drag_beta is not None else None,
        loss=float(best_loss),
        terminal_population=terminal,
        target_metric_name="population",
    )
