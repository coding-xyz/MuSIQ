import math

import pytest

from musiq.common.schemas import BackendConfig, CircuitGate, CircuitIR
from musiq.pulse.lowering import DefaultPulseLowering
from musiq.pulse.catalog import instantiate_operation_recipe
from musiq.schemas.circuit import build_serial_schedule


def test_lower_applies_virtual_z_phase_to_following_xy_pulse():
    circuit = CircuitIR(
        num_qubits=1,
        schedule=build_serial_schedule(
            [
                CircuitGate(name="h", qubits=[0]),
                CircuitGate(name="x", qubits=[0]),
            ],
            num_qubits=1,
        ),
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
    circuit = CircuitIR(num_qubits=0, schedule={})

    pulse_ir, executable = DefaultPulseLowering().lower(circuit, {}, BackendConfig())

    ro_channel = next(channel for channel in pulse_ir.channels if channel.name == "RO_0")
    assert len(ro_channel.pulses) == 1
    assert ro_channel.pulses[0].shape == "readout"
    assert executable.metadata["schedule_debug"][0]["gate_name"] == "measure"
    assert executable.metadata["t_end_ns"] > 0.0


def test_lower_tracks_reset_events_and_metadata():
    circuit = CircuitIR(num_qubits=1, schedule=build_serial_schedule([CircuitGate(name="reset", qubits=[0])], num_qubits=1))

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


def test_gate_recipe_supports_split_single_and_double_qubit_calibration_knobs():
    hw = {
        "gate_duration_ns": 20.0,
        "single_qubit_gate_duration_ns": 24.0,
        "double_qubit_gate_duration_ns": 70.0,
        "single_qubit_gate_amp_scale": 1.5,
        "double_qubit_gate_amp_scale": 0.5,
        "xy_freq_Hz": 5.0e9,
        "single_qubit_shape": "drag",
        "single_qubit_sigma_fraction": 1.0 / 6.0,
        "single_qubit_drag_beta": 0.25,
        "rect_edge_ns": 2.0,
    }

    rx_pulses, rx_duration, _ = instantiate_operation_recipe("rx", [0], gate_params=[math.pi], start_ns=0.0, hw=hw)
    cz_pulses, cz_duration, _ = instantiate_operation_recipe("cz", [0, 1], start_ns=0.0, hw=hw)

    assert rx_duration == 24.0
    assert len(rx_pulses) == 1
    assert math.isclose(rx_pulses[0][1].t1_s - rx_pulses[0][1].t0_s, 24.0e-9)

    baseline_rx_pulses, _, _ = instantiate_operation_recipe(
        "rx",
        [0],
        gate_params=[math.pi],
        start_ns=0.0,
        hw={**hw, "single_qubit_gate_amp_scale": 1.0},
    )
    assert rx_pulses[0][1].amp == baseline_rx_pulses[0][1].amp * 1.5

    assert cz_duration == 70.0
    assert len(cz_pulses) == 1
    assert math.isclose(cz_pulses[0][1].t1_s - cz_pulses[0][1].t0_s, 70.0e-9)

    baseline_cz_pulses, _, _ = instantiate_operation_recipe(
        "cz",
        [0, 1],
        start_ns=0.0,
        hw={**hw, "double_qubit_gate_amp_scale": 1.0},
    )
    assert cz_pulses[0][1].amp == baseline_cz_pulses[0][1].amp * 0.5


def test_cz_recipe_uses_connection_max_effective_coupling_when_present():
    hw = {
        "double_qubit_gate_duration_ns": 80.0,
        "double_qubit_gate_amp_scale": 0.5,
        "rect_edge_ns": 2.0,
        "connections": [
            {
                "id": "zz_q0_q1",
                "type": "zz",
                "a": "q0",
                "b": "q1",
                "parameters": {"max_effective_coupling_Hz": 20.0e6},
            }
        ],
    }

    cz_pulses, _, _ = instantiate_operation_recipe("cz", [0, 1], start_ns=0.0, hw=hw, tc_channel="TC_q0_q1")

    assert len(cz_pulses) == 1
    assert cz_pulses[0][1].amp == pytest.approx(math.pi * 20.0e6)


def test_default_lowering_preserves_connection_strength_for_cz():
    circuit = CircuitIR(num_qubits=2, schedule=build_serial_schedule([CircuitGate(name="cz", qubits=[0, 1])], num_qubits=2))
    hw = {
        "double_qubit_gate_duration_ns": 80.0,
        "double_qubit_gate_amp_scale": 0.5,
        "rect_edge_ns": 2.0,
        "connections": [
            {
                "id": "zz_q0_q1",
                "type": "zz",
                "a": "q0",
                "b": "q1",
                "parameters": {"max_effective_coupling_Hz": 20.0e6},
            }
        ],
    }

    pulse_ir, _ = DefaultPulseLowering().lower(circuit, hw, BackendConfig())

    tc_channel = next(channel for channel in pulse_ir.channels if channel.name == "TC_q0_q1")
    assert len(tc_channel.pulses) == 1
    assert tc_channel.pulses[0].amp == pytest.approx(math.pi * 20.0e6)


def test_lower_processes_schedule_debug_by_tick():
    circuit = CircuitIR(
        num_qubits=2,
        schedule={
            0: [
                [CircuitGate(name="sx", qubits=[0])],
                [CircuitGate(name="sx", qubits=[1])],
            ],
            1: [
                [CircuitGate(name="rz", qubits=[0], params=[0.5])],
                [],
            ],
        },
    )

    _, executable = DefaultPulseLowering().lower(
        circuit,
        {"single_qubit_gate_duration_ns": 20.0, "schedule_policy": "parallel"},
        BackendConfig(),
    )

    debug = executable.metadata["schedule_debug"]
    assert [item["tick"] for item in debug] == [0, 0, 1]
    assert debug[0]["start_ns"] == pytest.approx(0.0)
    assert debug[1]["start_ns"] == pytest.approx(0.0)


def test_typed_sx_recipe_uses_explicit_duration_amplitude_and_channel_override():
    hw = {
        "defaults": {"xy_carrier_freq_Hz": 5.0e9},
        "gates": {
            "sx": {
                "recipe_type": "sx",
                "shape": "drag",
                "duration_ns": 28.0,
                "amplitude_Hz": 10.5e6,
                "sigma_fraction": 0.1,
                "drag_beta": 0.11,
            }
        },
        "channel_overrides": {
            "XY_0": {
                "sx": {
                    "duration_ns": 26.0,
                    "amplitude_Hz": 10.8e6,
                    "drag_beta": 0.09,
                }
            }
        },
    }

    pulses, duration, _ = instantiate_operation_recipe("sx", [0], start_ns=0.0, hw=hw)

    assert duration == 26.0
    assert len(pulses) == 1
    assert pulses[0][0] == "XY_0"
    assert pulses[0][1].amp == pytest.approx(2.0 * math.pi * 10.8e6)
    assert pulses[0][1].params["beta"] == pytest.approx(0.09)


def test_typed_cz_and_virtual_z_recipes_use_new_schema():
    hw = {
        "gates": {
            "virtual_z": {"recipe_type": "virtual_z"},
            "cz": {
                "recipe_type": "cz",
                "duration_ns": 52.0,
                "amplitude_Hz": 20.0e6,
                "edge_ns": 2.0,
                "target_conditional_phase_rad": math.pi,
            },
        }
    }

    z_pulses, z_duration, _ = instantiate_operation_recipe("rz", [0], gate_params=[math.pi / 4.0], start_ns=0.0, hw=hw)
    cz_pulses, cz_duration, _ = instantiate_operation_recipe("cz", [0, 1], start_ns=0.0, hw=hw)

    assert z_pulses == []
    assert z_duration == 0.0
    assert cz_duration == 52.0
    assert len(cz_pulses) == 1
    assert cz_pulses[0][1].amp == pytest.approx(2.0 * math.pi * 20.0e6)


def test_typed_measure_recipe_emits_segmented_readout():
    circuit = CircuitIR(num_qubits=1, schedule=build_serial_schedule([CircuitGate(name="measure", qubits=[0])], num_qubits=1))
    hw = {
        "gates": {
            "measure": {
                "recipe_type": "measure",
                "carrier_freq_Hz": 6.45e9,
                "segments": [
                    {"duration_ns": 40.0, "amplitude": 1400.0, "shape": "readout", "rise_ns": 10.0, "fall_ns": 0.0},
                    {"duration_ns": 180.0, "amplitude": 200.0, "shape": "readout", "rise_ns": 0.0, "fall_ns": 10.0},
                ],
            }
        }
    }

    pulse_ir, _ = DefaultPulseLowering().lower(circuit, hw, BackendConfig())

    ro_channel = next(channel for channel in pulse_ir.channels if channel.name == "RO_0")
    assert len(ro_channel.pulses) == 2
    assert ro_channel.pulses[0].amp == pytest.approx(1400.0)
    assert ro_channel.pulses[1].amp == pytest.approx(200.0)
