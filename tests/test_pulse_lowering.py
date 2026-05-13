import math

from musiq.common.schemas import BackendConfig, CircuitGate, CircuitIR
from musiq.pulse.lowering import DefaultPulseLowering


def test_lower_applies_virtual_z_phase_to_following_xy_pulse():
    circuit = CircuitIR(
        num_qubits=1,
        gates=[
            CircuitGate(name="h", qubits=[0]),
            CircuitGate(name="x", qubits=[0]),
        ],
    )

    pulse_ir, executable = DefaultPulseLowering().lower(circuit, {}, BackendConfig())

    xy_channel = next(channel for channel in pulse_ir.channels if channel.name == "XY_0")

    assert len(xy_channel.pulses) == 2
    assert xy_channel.pulses[0].carrier is not None
    assert xy_channel.pulses[1].carrier is not None
    assert xy_channel.pulses[0].carrier.phase == math.pi / 2.0
    assert xy_channel.pulses[1].carrier.phase == math.pi
    assert executable.metadata["schedule_debug"][1]["virtual_z_phase_before_rad"] == {0: math.pi}


def test_lower_emits_measurement_for_empty_zero_qubit_circuit():
    circuit = CircuitIR(num_qubits=0, gates=[])

    pulse_ir, executable = DefaultPulseLowering().lower(circuit, {}, BackendConfig())

    ro_channel = next(channel for channel in pulse_ir.channels if channel.name == "RO_0")
    assert len(ro_channel.pulses) == 1
    assert ro_channel.pulses[0].shape == "readout"
    assert executable.metadata["schedule_debug"][0]["gate_name"] == "measure"
    assert executable.metadata["t_end_ns"] > 0.0


def test_lower_tracks_reset_events_and_metadata():
    circuit = CircuitIR(num_qubits=1, gates=[CircuitGate(name="reset", qubits=[0])])

    pulse_ir, executable = DefaultPulseLowering().lower(
        circuit,
        {"reset_feedback_policy": "serial_global", "reset_apply_feedback": True},
        BackendConfig(),
    )

    assert pulse_ir.t_end_ns > 0.0
    assert len(executable.metadata["reset_events"]) == 1
    event = executable.metadata["reset_events"][0]
    assert event["qubit"] == 0
    assert event["apply_feedback"] is True
    assert executable.metadata["reset_feedback_policy"] == "serial_global"
