from musiq.circuit.normalize import normalize_circuit
from musiq.common.schemas import CircuitGate, CircuitIR
from musiq.schemas.circuit import build_serial_schedule


def test_normalize_circuit_lowercases_gate_names_and_preserves_shape():
    gates = [
        CircuitGate(name="H", qubits=[0]),
        CircuitGate(name="RZ", qubits=[0], params=[1.25]),
        CircuitGate(name="MEASURE", qubits=[0], clbits=[0]),
    ]
    circuit = CircuitIR(
        num_qubits=2,
        num_clbits=1,
        schedule=build_serial_schedule(gates, num_qubits=2),
        source_qasm="OPENQASM 3;",
    )

    normalized = normalize_circuit(circuit)

    assert [gate.name for gate in normalized.gates] == ["h", "rz", "measure"]
    assert normalized.schedule[0][0][0].name == "h"
    assert normalized.num_qubits == 2
    assert normalized.num_clbits == 1
    assert normalized.source_qasm == "OPENQASM 3;"


def test_normalize_circuit_copies_gate_payloads():
    circuit = CircuitIR(
        num_qubits=1,
        schedule=build_serial_schedule([CircuitGate(name="RX", qubits=[0], params=[0.5], clbits=[1])], num_qubits=1),
    )

    normalized = normalize_circuit(circuit)
    normalized.schedule[0][0][0].qubits.append(2)
    normalized.schedule[0][0][0].params.append(1.0)
    normalized.schedule[0][0][0].clbits.append(3)

    assert circuit.schedule[0][0][0].qubits == [0]
    assert circuit.schedule[0][0][0].params == [0.5]
    assert circuit.schedule[0][0][0].clbits == [1]


def test_schedule_flattens_mirrored_two_qubit_gate_once():
    circuit = CircuitIR(
        num_qubits=3,
        schedule={
            0: [[], [CircuitGate(name="cz", qubits=[2, 1])], [CircuitGate(name="cz", qubits=[2, 1])]],
        },
    )

    assert [gate.name for gate in circuit.gates] == ["cz"]
