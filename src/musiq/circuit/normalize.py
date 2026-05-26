"""Circuit normalization helpers used before lowering."""

from __future__ import annotations

from musiq.common.schemas import CircuitGate, CircuitIR


def normalize_circuit(circuit: CircuitIR) -> CircuitIR:
    """Normalize gate names and copy circuit into canonical representation."""
    schedule: dict[int, list[list[CircuitGate]]] = {}
    for tick, lanes in sorted(dict(circuit.schedule or {}).items(), key=lambda item: int(item[0])):
        schedule[int(tick)] = [
            [
                CircuitGate(
                    name=str(g.name).lower(),
                    qubits=list(g.qubits),
                    params=list(g.params),
                    clbits=list(g.clbits),
                )
                for g in list(lane or [])
            ]
            for lane in list(lanes or [])
        ]
    return CircuitIR(
        schema_version=circuit.schema_version,
        format=circuit.format,
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        schedule=schedule,
        source_qasm=circuit.source_qasm,
    )
