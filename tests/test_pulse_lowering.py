import math

import pytest

from musiq.backend.config import normalize_device_config
from musiq.backend.model.lowering import lower_sampled_channels
from musiq.common.schemas import BackendConfig, CircuitGate, CircuitIR
from musiq.pulse.catalog import instantiate_operation_recipe, resolve_typed_gate_recipe
from musiq.pulse.lowering import DefaultPulseLowering
from musiq.schemas.pulse import CouplerTwoQubitRecipe, DrivenSingleQubitRecipe, MeasureRecipe, VirtualPhaseGateRecipe
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


def test_legacy_amp_scale_fields_are_rejected():
    with pytest.raises(ValueError, match="Legacy pulse schema"):
        instantiate_operation_recipe(
            "rx",
            [0],
            gate_params=[math.pi],
            start_ns=0.0,
            hw={"single_qubit_gate_amp_scale": 1.5},
        )

    with pytest.raises(ValueError, match="Legacy pulse schema"):
        instantiate_operation_recipe(
            "cz",
            [0, 1],
            start_ns=0.0,
            hw={"double_qubit_gate_amp_scale": 0.5},
        )


def test_unknown_gate_is_rejected_instead_of_silent_idle():
    with pytest.raises(ValueError, match="Unsupported gate for pulse lowering: rxx"):
        instantiate_operation_recipe("rxx", [0, 1], gate_params=[0.25], start_ns=0.0, hw={})


def test_cz_recipe_uses_connection_max_effective_coupling_when_present():
    hw = {
        "double_qubit_gate_duration_ns": 80.0,
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

    cz_pulses, _, _ = instantiate_operation_recipe("cz", [0, 1], start_ns=0.0, hw=hw, tc_channel="TC_0_1")

    assert len(cz_pulses) == 1
    assert cz_pulses[0][1].amp == pytest.approx(2.0 * math.pi * 20.0e6)


def test_default_lowering_preserves_connection_strength_for_cz():
    circuit = CircuitIR(num_qubits=2, schedule=build_serial_schedule([CircuitGate(name="cz", qubits=[0, 1])], num_qubits=2))
    hw = {
        "double_qubit_gate_duration_ns": 80.0,
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

    tc_channel = next(channel for channel in pulse_ir.channels if channel.name == "TC_0_1")
    assert len(tc_channel.pulses) == 1
    assert tc_channel.pulses[0].amp == pytest.approx(2.0 * math.pi * 20.0e6)


def test_iswap_recipe_uses_tc_channel_and_typed_schema():
    hw = {
        "gates": {
            "iswap": {
                "recipe_type": "iswap",
                "shape": "rect",
                "duration_ns": 36.0,
                "amplitude_Hz": 28.0e6,
                "edge_ns": 2.0,
            }
        }
    }

    iswap_pulses, iswap_duration, _ = instantiate_operation_recipe("iswap", [0, 1], start_ns=0.0, hw=hw)
    iswap_recipe = resolve_typed_gate_recipe(hw, "iswap", channel_name="TC_0_1")

    assert iswap_duration == pytest.approx(36.0)
    assert len(iswap_pulses) == 1
    assert iswap_pulses[0][0] == "TC_0_1"
    assert iswap_pulses[0][1].amp == pytest.approx(2.0 * math.pi * 28.0e6)
    assert isinstance(iswap_recipe, CouplerTwoQubitRecipe)
    assert iswap_recipe.logical_gate == "iswap"


def test_two_qubit_recipe_uses_explicit_coupler_channel_when_present():
    hw = {
        "gates": {
            "cz": {
                "recipe_type": "cz",
                "shape": "rect",
                "duration_ns": 52.0,
                "amplitude_Hz": 20.0e6,
            }
        },
        "connections": [
            {"id": "q0_c0", "type": "xx+yy", "a": "q0", "b": "c0", "parameters": {"g_Hz": 45.0e6}},
            {"id": "q1_c0", "type": "xx+yy", "a": "q1", "b": "c0", "parameters": {"g_Hz": 45.0e6}},
            {"id": "q0_q1", "type": "xx+yy", "a": "q0", "b": "q1", "parameters": {"g_Hz": 6.0e6}},
        ],
    }

    pulses, duration, _ = instantiate_operation_recipe("cz", [0, 1], start_ns=0.0, hw=hw, tc_channel="TC_0_1")

    assert duration == pytest.approx(52.0)
    assert len(pulses) == 1
    assert pulses[0][0] == "C_0"


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


def test_resolve_typed_gate_recipe_returns_family_dataclasses():
    hw = {
        "gates": {
            "x": {"recipe_type": "x", "duration_ns": 28.0, "amplitude_Hz": 21.0e6},
            "sx": {"recipe_type": "sx", "duration_ns": 28.0, "amplitude_Hz": 10.5e6},
            "rx": {"recipe_type": "rx", "duration_ns": 30.0, "amplitude_Hz": 12.0e6},
            "virtual_z": {"recipe_type": "virtual_z"},
            "cz": {"recipe_type": "cz", "duration_ns": 52.0, "amplitude_Hz": 20.0e6},
            "iswap": {"recipe_type": "iswap", "duration_ns": 36.0, "amplitude_Hz": 28.0e6},
            "measure": {
                "recipe_type": "measure",
                "segments": [{"duration_ns": 40.0, "amplitude": 1400.0}],
            },
        },
        "channel_overrides": {"XY_0": {"sx": {"duration_ns": 26.0}}},
    }

    x_recipe = resolve_typed_gate_recipe(hw, "x", channel_name="XY_0")
    sx_recipe = resolve_typed_gate_recipe(hw, "sx", channel_name="XY_0")
    rx_recipe = resolve_typed_gate_recipe(hw, "rx", channel_name="XY_0")
    z_recipe = resolve_typed_gate_recipe(hw, "rz")
    cz_recipe = resolve_typed_gate_recipe(hw, "cz", channel_name="TC_0_1")
    iswap_recipe = resolve_typed_gate_recipe(hw, "iswap", channel_name="TC_0_1")
    measure_recipe = resolve_typed_gate_recipe(hw, "measure", channel_name="RO_0")

    assert isinstance(x_recipe, DrivenSingleQubitRecipe)
    assert isinstance(sx_recipe, DrivenSingleQubitRecipe)
    assert isinstance(rx_recipe, DrivenSingleQubitRecipe)
    assert x_recipe.logical_gate == "x"
    assert sx_recipe.logical_gate == "sx"
    assert rx_recipe.logical_gate == "rx"
    assert sx_recipe.duration_ns == pytest.approx(26.0)
    assert isinstance(z_recipe, VirtualPhaseGateRecipe)
    assert z_recipe.logical_gate == "rz"
    assert isinstance(cz_recipe, CouplerTwoQubitRecipe)
    assert isinstance(iswap_recipe, CouplerTwoQubitRecipe)
    assert iswap_recipe.logical_gate == "iswap"
    assert isinstance(measure_recipe, MeasureRecipe)
    assert measure_recipe.duration_ns == pytest.approx(40.0)


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


def test_coupler_channel_can_target_explicit_coupler_frequency():
    device = normalize_device_config(
        {
            "simulation_level": "nlevel",
            "components": [
                {
                    "id": "q0",
                    "type": "transmon",
                    "parameters": {"freq_Hz": 5.08e9, "anharmonicity_Hz": -2.2e8},
                },
                {
                    "id": "q1",
                    "type": "transmon",
                    "parameters": {"freq_Hz": 4.92e9, "anharmonicity_Hz": -2.0e8},
                },
                {
                    "id": "c0",
                    "type": "transmon",
                    "parameters": {"freq_Hz": 4.95e9, "anharmonicity_Hz": -2.6e8},
                },
            ],
            "connections": [
                {
                    "id": "tc_q0_q1",
                    "type": "tunable_coupler_flux",
                    "a": "q0",
                    "b": "q1",
                    "via": "c0",
                    "parameters": {"park_freq_Hz": 4.95e9},
                }
            ],
        }
    )

    sampled = lower_sampled_channels(
        device,
        {
            "C_0": {
                "t": [0.0, 1.0e-9, 2.0e-9],
                "y": [0.0, 2.0 * math.pi * 25.0e6, 0.0],
            }
        },
        num_qubits=3,
    )

    assert len(sampled.controls) == 1
    ctrl = sampled.controls[0]
    assert ctrl["channel"] == "C_0"
    assert ctrl["axis"] == "z"
    assert ctrl["target"] == 2
    assert ctrl["coupler_channel_index"] == 0


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


def test_virtual_z_rejects_duration_or_amplitude_fields():
    with pytest.raises(ValueError, match="forbidden VirtualZ pulse fields"):
        instantiate_operation_recipe(
            "rz",
            [0],
            gate_params=[math.pi / 4.0],
            start_ns=0.0,
            hw={"gates": {"virtual_z": {"recipe_type": "virtual_z", "duration_ns": 1.0}}},
        )
