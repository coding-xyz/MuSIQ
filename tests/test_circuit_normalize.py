from musiq.circuit.normalize import normalize_circuit
from musiq.common.schemas import CircuitGate, CircuitIR


def test_normalize_circuit_lowercases_gate_names_and_preserves_shape():
    circuit = CircuitIR(
        num_qubits=2,
        num_clbits=1,
        gates=[
            CircuitGate(name="H", qubits=[0]),
            CircuitGate(name="RZ", qubits=[0], params=[1.25]),
            CircuitGate(name="MEASURE", qubits=[0], clbits=[0]),
        ],
        source_qasm="OPENQASM 3;",
    )

    normalized = normalize_circuit(circuit)

    assert [gate.name for gate in normalized.gates] == ["h", "rz", "measure"]
    assert normalized.num_qubits == 2
    assert normalized.num_clbits == 1
    assert normalized.source_qasm == "OPENQASM 3;"


def test_normalize_circuit_copies_gate_payloads():
    circuit = CircuitIR(
        gates=[CircuitGate(name="RX", qubits=[0], params=[0.5], clbits=[1])],
    )

    normalized = normalize_circuit(circuit)
    normalized.gates[0].qubits.append(2)
    normalized.gates[0].params.append(1.0)
    normalized.gates[0].clbits.append(3)

    assert circuit.gates[0].qubits == [0]
    assert circuit.gates[0].params == [0.5]
    assert circuit.gates[0].clbits == [1]
