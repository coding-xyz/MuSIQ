from musiq.backend.scheduling import build_gate_schedule
from musiq.common.schemas import CircuitGate, CircuitIR


def test_parallel_schedule_aligns_lane_starts_within_tick():
    circuit = CircuitIR(
        num_qubits=4,
        schedule={
            0: [
                [CircuitGate(name="rz", qubits=[0], params=[0.1]), CircuitGate(name="sx", qubits=[0])],
                [CircuitGate(name="rz", qubits=[1], params=[0.2]), CircuitGate(name="sx", qubits=[1])],
                [CircuitGate(name="rz", qubits=[2], params=[0.3]), CircuitGate(name="sx", qubits=[2])],
                [CircuitGate(name="rz", qubits=[3], params=[0.4]), CircuitGate(name="sx", qubits=[3])],
            ]
        },
    )

    scheduled = build_gate_schedule(
        circuit,
        {
            "gate_duration_ns": 20.0,
            "single_qubit_gate_duration_ns": 20.0,
            "double_qubit_gate_duration_ns": 40.0,
            "idle_duration_ns": 20.0,
            "measure_duration_ns": 100.0,
            "reset_measure_duration_ns": 100.0,
            "reset_deplete_duration_ns": 20.0,
            "reset_latency_duration_ns": 20.0,
            "reset_pi_duration_ns": 20.0,
            "reset_apply_feedback": True,
            "schedule_policy": "parallel",
        },
    )

    sx_starts = {
        tuple(item["gate"].qubits): item["start_ns"]
        for item in scheduled
        if str(item["gate"].name).lower() == "sx"
    }
    assert sx_starts == {
        (0,): 0.0,
        (1,): 0.0,
        (2,): 0.0,
        (3,): 0.0,
    }


def test_parallel_schedule_keeps_lane_order_and_dedupes_two_qubit_gate():
    circuit = CircuitIR(
        num_qubits=3,
        schedule={
            0: [
                [CircuitGate(name="sx", qubits=[0])],
                [CircuitGate(name="cz", qubits=[2, 1]), CircuitGate(name="sx", qubits=[1])],
                [CircuitGate(name="cz", qubits=[2, 1]), CircuitGate(name="sx", qubits=[2])],
            ]
        },
    )

    scheduled = build_gate_schedule(
        circuit,
        {
            "gate_duration_ns": 20.0,
            "single_qubit_gate_duration_ns": 20.0,
            "double_qubit_gate_duration_ns": 50.0,
            "idle_duration_ns": 20.0,
            "measure_duration_ns": 100.0,
            "reset_measure_duration_ns": 100.0,
            "reset_deplete_duration_ns": 20.0,
            "reset_latency_duration_ns": 20.0,
            "reset_pi_duration_ns": 20.0,
            "reset_apply_feedback": True,
            "schedule_policy": "parallel",
        },
    )

    cz_items = [item for item in scheduled if str(item["gate"].name).lower() == "cz"]
    sx_items = [item for item in scheduled if str(item["gate"].name).lower() == "sx"]

    assert len(cz_items) == 1
    assert cz_items[0]["start_ns"] == 0.0
    trailing_sx_starts = {
        tuple(item["gate"].qubits): item["start_ns"]
        for item in sx_items
        if tuple(item["gate"].qubits) in {(1,), (2,)}
    }
    assert trailing_sx_starts == {
        (1,): 50.0,
        (2,): 50.0,
    }


def test_parallel_schedule_uses_typed_gate_recipe_durations_for_tick_cursor():
    circuit = CircuitIR(
        num_qubits=2,
        schedule={
            0: [
                [CircuitGate(name="sx", qubits=[0])],
                [CircuitGate(name="sx", qubits=[1])],
            ],
            1: [
                [CircuitGate(name="cz", qubits=[0, 1])],
                [CircuitGate(name="cz", qubits=[0, 1])],
            ],
        },
    )

    scheduled = build_gate_schedule(
        circuit,
        {
            "gate_duration_ns": 20.0,
            "single_qubit_gate_duration_ns": 20.0,
            "double_qubit_gate_duration_ns": 40.0,
            "idle_duration_ns": 20.0,
            "measure_duration_ns": 100.0,
            "reset_measure_duration_ns": 100.0,
            "reset_deplete_duration_ns": 20.0,
            "reset_latency_duration_ns": 20.0,
            "reset_pi_duration_ns": 20.0,
            "reset_apply_feedback": True,
            "schedule_policy": "parallel",
            "gates": {
                "sx": {
                    "recipe_type": "sx",
                    "duration_ns": 30.0,
                    "amplitude_Hz": 1.0e7,
                },
                "cz": {
                    "recipe_type": "cz",
                    "duration_ns": 50.0,
                    "amplitude_Hz": -6.0e6,
                },
            },
        },
    )

    sx_items = [item for item in scheduled if str(item["gate"].name).lower() == "sx"]
    cz_items = [item for item in scheduled if str(item["gate"].name).lower() == "cz"]

    assert len(sx_items) == 2
    assert {item["start_ns"] for item in sx_items} == {0.0}
    assert {item["end_ns"] for item in sx_items} == {30.0}
    assert len(cz_items) == 1
    assert cz_items[0]["start_ns"] == 30.0
    assert cz_items[0]["end_ns"] == 80.0
