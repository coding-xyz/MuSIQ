from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from musiq.backend.config import normalize_device_config
from musiq.backend.model.lowering import lower_couplings
from musiq.schemas.connections import SystemConnectionSpec
from musiq.schemas.utils import ParameterList, ParameterSweepConfig
from musiq.ui.cli import build_parser
from musiq.workflow import CircuitConfig, ProfileConfig, create_model, load_model
from musiq.workflow.contracts import filter_composite_device_for_step
from musiq.workflow.task_io import (
    load_analyser_config_file,
    load_circuit_config_file,
    load_config,
    load_device_config_file,
    load_pulse_config_file,
    load_solver_config_file,
)


def _typed_pulse_payload(*, measure_amp: float = 0.8) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "pulse": {
            "defaults": {
                "xy_carrier_freq_Hz": 5.0e9,
                "ro_carrier_freq_Hz": 6.5e9,
                "schedule_policy": "parallel",
            },
            "gates": {
                "x": {
                    "recipe_type": "x",
                    "shape": "drag",
                    "duration_ns": 20.0,
                    "amplitude_Hz": 12.5e6,
                    "carrier_freq_Hz": 5.0e9,
                    "sigma_fraction": 1.0 / 6.0,
                    "drag_beta": 0.25,
                },
                "ry": {
                    "recipe_type": "ry",
                    "shape": "drag",
                    "duration_ns": 20.0,
                    "amplitude_Hz": 12.5e6,
                    "carrier_freq_Hz": 5.0e9,
                    "phase_rad": 1.5707963267948966,
                    "sigma_fraction": 1.0 / 6.0,
                    "drag_beta": 0.25,
                },
                "measure": {
                    "recipe_type": "measure",
                    "carrier_freq_Hz": 6.5e9,
                    "duration_ns": 200.0,
                    "amplitude": measure_amp,
                    "shape": "readout",
                    "edge_ns": 10.0,
                },
                "virtual_z": {"recipe_type": "virtual_z"},
            },
            "channel_overrides": {
                "XY_0": {
                    "x": {
                        "amplitude_Hz": 13.0e6,
                    }
                }
            },
            "acquisition": {
                "measure_start_delay_ns": 20.0,
                "integration_window_ns": 160.0,
            },
        },
    }


def _write_basic_solver_device_pulse_and_analyser(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    solver_cfg = {
        "template": "qutip_default",
        "backend": {"level": "qubit", "analysis_pipeline": "default", "truncation": {}},
        "run": {"engine": "qutip", "solver_mode": "me", "seed": 7, "sweep": []},
        "frame": {"mode": "rotating", "reference": "pulse_carrier", "rwa": True},
    }
    device_cfg = {
        "template": "transmon_default",
        "device": {"simulation_level": "qubit", "qubits": [{"freq_Hz": 5.0e9, "anharmonicity_Hz": -2.0e8}]},
        "noise": {"model": "markovian_lindblad", "T1_s": 1e-5, "T2_s": 8e-6},
    }
    analyser_cfg = {
        "trajectory": {"quantum": "", "save_times": "all", "save_final_state": True},
        "metrics": ["population", "mean_excited", "variance"],
    }
    solver_path = tmp_path / "solver.json"
    device_path = tmp_path / "device.json"
    pulse_path = tmp_path / "pulse.json"
    analyser_path = tmp_path / "analyser.json"
    solver_path.write_text(json.dumps(solver_cfg, ensure_ascii=False), encoding="utf-8")
    device_path.write_text(json.dumps(device_cfg, ensure_ascii=False), encoding="utf-8")
    pulse_path.write_text(json.dumps(_typed_pulse_payload(), ensure_ascii=False), encoding="utf-8")
    analyser_path.write_text(json.dumps(analyser_cfg, ensure_ascii=False), encoding="utf-8")
    return solver_path, device_path, pulse_path, analyser_path


def test_load_circuit_config_file_reads_qasm_and_bindings(tmp_path: Path):
    qasm_path = tmp_path / "task.qasm"
    qasm_path.write_text("OPENQASM 3; qubit[1] q;", encoding="utf-8")
    cfg = {
        "schema_version": "1.0",
        "qasm_path": "task.qasm",
        "param_bindings": {"theta": 0.1},
    }
    cfg_path = tmp_path / "circuit.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    circuit = load_circuit_config_file(cfg_path)

    assert "OPENQASM 3" in circuit.qasm_text
    assert circuit.param_bindings == {"theta": 0.1}


def test_circuit_config_requires_exactly_one_qasm_source(tmp_path: Path):
    cfg_with_both = {"qasm_text": "OPENQASM 3; qubit[1] q;", "qasm_path": "task.qasm"}
    cfg_with_none = {"param_bindings": {"theta": 0.2}}
    (tmp_path / "task.qasm").write_text("OPENQASM 3; qubit[1] q;", encoding="utf-8")

    both_path = tmp_path / "both.json"
    both_path.write_text(json.dumps(cfg_with_both), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_circuit_config_file(both_path)

    none_path = tmp_path / "none.json"
    none_path.write_text(json.dumps(cfg_with_none), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_circuit_config_file(none_path)


def test_load_circuit_config_file_reads_schedule_yaml(tmp_path: Path):
    cfg = {
        "schema_version": "1.0",
        "format": "circuit_layer_yaml",
        "num_qubits": 3,
        "num_clbits": 0,
        "schedule": {
            "0": [
                [["rz", [0], 0.7853981633974474], ["sx", [0]], ["rz", [0], 1.5707963267948968]],
                [],
                [],
            ],
            "1": [
                [],
                [["cz", [2, 1]]],
                [["cz", [2, 1]]],
            ],
        },
    }
    cfg_path = tmp_path / "circuit_schedule.yaml"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    circuit = load_circuit_config_file(cfg_path)

    assert circuit.qasm_text is None
    assert circuit.circuit_ir is not None
    assert circuit.circuit_ir.schedule[0][0][1].name == "sx"
    assert circuit.circuit_ir.schedule[1][1][0].qubits == [2, 1]
    assert circuit.circuit_ir.schedule[1][2][0].name == "cz"


def test_circuit_config_helper_reads_pure_schedule_yaml(tmp_path: Path):
    schedule_only = {
        "0": [
            [["sx", [0]]],
            [],
        ],
        "1": [
            [["cz", [0, 1]]],
            [["cz", [0, 1]]],
        ],
    }
    cfg_path = tmp_path / "schedule_only.yaml"
    cfg_path.write_text(json.dumps(schedule_only, ensure_ascii=False), encoding="utf-8")

    circuit = CircuitConfig.from_schedule_file(cfg_path)

    assert circuit.qasm_text is None
    assert circuit.circuit_ir is not None
    assert circuit.circuit_ir.num_qubits == 2
    assert circuit.circuit_ir.schedule[0][0][0].name == "sx"
    assert circuit.circuit_ir.schedule[1][1][0].qubits == [0, 1]


def test_load_pulse_config_file_reads_typed_schema(tmp_path: Path):
    pulse_path = tmp_path / "pulse.yaml"
    pulse_path.write_text(json.dumps(_typed_pulse_payload(measure_amp=1.2), ensure_ascii=False), encoding="utf-8")

    pulse = load_pulse_config_file(pulse_path)

    assert pulse["defaults"]["xy_carrier_freq_Hz"] == 5.0e9
    assert pulse["gates"]["x"]["amplitude_Hz"] == 12.5e6
    assert pulse["gates"]["measure"]["amplitude"] == 1.2
    assert pulse["channel_overrides"]["XY_0"]["x"]["amplitude_Hz"] == 13.0e6
    assert pulse["acquisition"]["integration_window_ns"] == 160.0


def test_load_pulse_config_file_rejects_legacy_operation_schema(tmp_path: Path):
    pulse_path = tmp_path / "pulse.yaml"
    pulse_path.write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "pulse": {
                    "channels": [{"name": "XY_q0", "kind": "drive"}],
                    "carriers": {"XY_q0": {"freq_Hz": 5.0e9}},
                    "waveforms": {"x90": {"shape": "drag", "duration_ns": 20.0}},
                    "operations": {"sx": [{"channel": "XY_q0", "waveform": "x90"}]},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy pulse schema"):
        load_pulse_config_file(pulse_path)


def test_load_pulse_config_file_rejects_flat_legacy_pulse_fields(tmp_path: Path):
    pulse_path = tmp_path / "pulse.yaml"
    pulse_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "pulse": {
                    "gate_duration_ns": 20.0,
                    "xy_freq_Hz": 5.0e9,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="typed top-level schema"):
        load_pulse_config_file(pulse_path)


def test_report_pulse_files_use_typed_schema_and_load():
    pulse_paths = [
        Path("report/task1_single_qubit_rabi/pulse.yaml"),
        Path("report/task2_single_qubit_decoherence/pulse.yaml"),
        Path("report/task2_single_qubit_decoherence/pulse_1overf.yaml"),
        Path("report/task3_gaussian_drag_comparison/pulse_drag.yaml"),
        Path("report/task3_gaussian_drag_comparison/pulse_gaussian.yaml"),
        Path("report/task6_single_qubit_readout/pulse.yaml"),
        Path("report/task6_single_qubit_readout/pulse_cqed.yaml"),
    ]

    for pulse_path in pulse_paths:
        pulse_cfg = load_pulse_config_file(pulse_path)
        assert pulse_cfg.get("gates")
        assert "gates" in pulse_cfg
        assert isinstance(pulse_cfg["gates"], dict)
        assert pulse_cfg.get("channel_overrides", {}) == {} or isinstance(pulse_cfg.get("channel_overrides"), dict)


def test_template_and_example_pulse_files_use_typed_schema_and_load():
    pulse_paths = [
        Path("templates/pulses/single_qubit.yaml"),
        Path("src/musiq/workflow/templates/pulses/single_qubit_default.yaml"),
        Path("examples/noise_simulation_tests/task1/pulse.yaml"),
        Path("examples/noise_simulation_tests/task2/pulse.yaml"),
        Path("examples/noise_simulation_tests/task3/pulse.yaml"),
        Path("examples/noise_simulation_tests/task4/pulse.yaml"),
        Path("examples/noise_simulation_tests/task5/pulse_drag.yaml"),
        Path("examples/noise_simulation_tests/task5/pulse_square.yaml"),
        Path("examples/noise_simulation_tests/task6/pulse.yaml"),
        Path("examples/noise_simulation_tests/task7/pulse.yaml"),
        Path("report/compile_test/pulse.yaml"),
    ]

    for pulse_path in pulse_paths:
        pulse_cfg = load_pulse_config_file(pulse_path)
        assert isinstance(pulse_cfg, dict)
        assert "gates" in pulse_cfg


def test_solver_config_engine_dependency_validation(tmp_path: Path):
    cfg = {"backend": {"level": "qubit"}, "run": {"engine": "qutip", "julia_bin": "julia"}}
    p = tmp_path / "solver_bad.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="not supported by selected engine"):
        load_solver_config_file(p)


def test_solver_config_loads_frame_options(tmp_path: Path):
    cfg = {
        "backend": {"level": "qubit"},
        "run": {"engine": "qutip", "solver_mode": "me"},
        "frame": {"mode": "rotating", "reference": "explicit", "rwa": True, "qubit_reference_freqs_Hz": [5.0e9]},
    }
    p = tmp_path / "solver_frame.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    solver = load_solver_config_file(p)

    assert solver.frame.mode == "rotating"
    assert solver.frame.reference == "explicit"
    assert solver.frame.rwa is True
    assert solver.frame.qubit_reference_freqs_Hz == [5.0e9]


def test_device_config_loads_with_template(tmp_path: Path):
    p = tmp_path / "device.yaml"
    p.write_text(
        "template: transmon_default\ndevice:\n  simulation_level: qubit\nnoise:\n  model: markovian_lindblad\n  T1_s: 1.0e-5\n",
        encoding="utf-8",
    )
    cfg = load_device_config_file(p)
    assert cfg.device is None or isinstance(cfg.device, dict)
    assert isinstance(cfg.noise, dict)


def test_device_config_rejects_component_representation_basis_and_role(tmp_path: Path):
    p = tmp_path / "device.yaml"
    p.write_text(
        json.dumps(
            {
                "device": {
                    "components": [
                        {
                            "id": "q0",
                            "type": "transmon",
                            "role": "qubit",
                            "representation": "quantum",
                            "basis": {"kind": "nlevel", "levels": 3},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no longer supported"):
        load_device_config_file(p)


def test_filter_composite_device_for_step_keeps_only_explicit_active_components():
    device = {
        "components": [
            {"id": "q0", "type": "transmon"},
            {"id": "q1", "type": "transmon"},
            {"id": "q2", "type": "transmon"},
        ],
        "connections": [
            {"id": "zz_q0_q1", "type": "zz", "a": "q0", "b": "q1"},
            {"id": "zz_q1_q2", "type": "zz", "a": "q1", "b": "q2"},
        ],
    }

    filtered = filter_composite_device_for_step(
        device,
        {"active_components": ["q0"], "active_connections": []},
    )

    assert [comp["id"] for comp in filtered["components"]] == ["q0"]
    assert filtered["connections"] == []


def test_analyser_config_loads_metrics_and_trajectory(tmp_path: Path):
    p = tmp_path / "analyser.yaml"
    p.write_text(
        "trajectory:\n  quantum: density_matrix\n  save_times: all\nmetrics:\n  - population\n  - mean_excited\n",
        encoding="utf-8",
    )
    analyser = load_analyser_config_file(p)
    assert analyser.trajectory.extras["quantum"] == "density_matrix"
    assert analyser.metrics == ["population", "mean_excited"]


def test_sweep_config_accepts_numpy_array_shorthand():
    sweep = load_config({"pulse.defaults.idle_duration_ns": np.linspace(0.0, 100.0, 5)}, "sweep")
    parameter = sweep.parameters["pulse.defaults.idle_duration_ns"]

    assert parameter.target == "pulse.defaults.idle_duration_ns"
    assert parameter.values == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_cli_parser_supports_circuit_and_optional_overrides():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-model",
            "--circuit-config",
            "circuits/demo.yaml",
            "--solver-config",
            "solvers/qutip.yaml",
            "--device-config",
            "device/default.yaml",
            "--pulse-config",
            "pulses/default.yaml",
            "--analyser-config",
            "analysers/default.yaml",
        ]
    )
    assert args.cmd == "run-model"
    assert args.circuit_config == "circuits/demo.yaml"
    assert args.solver_config == "solvers/qutip.yaml"
    assert args.device_config == "device/default.yaml"
    assert args.pulse_config == "pulses/default.yaml"
    assert args.analyser_config == "analysers/default.yaml"


def test_cli_parser_rejects_removed_run_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--qasm", "a", "--backend", "b", "--out", "c"])


def test_create_model_accepts_direct_resource_configs(tmp_path: Path):
    qasm_path = tmp_path / "task.qasm"
    qasm_path.write_text("OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nmeasure q[0] -> c[0];\n", encoding="utf-8")
    solver_path, device_path, pulse_path, analyser_path = _write_basic_solver_device_pulse_and_analyser(tmp_path)
    circuit_path = tmp_path / "circuit.json"
    circuit_path.write_text(json.dumps({"qasm_path": "task.qasm"}, ensure_ascii=False), encoding="utf-8")

    model = create_model(
        circuits=circuit_path,
        solvers=solver_path,
        devices=device_path,
        pulses=pulse_path,
        analysers=analyser_path,
    )
    model.run()

    assert "default" in model.config.circuits
    assert "default" in model.config.devices
    assert "default" in model.config.pulses
    assert "solver_0" in model.config.solvers
    assert "analyser_0" in model.config.analysers
    assert sorted(model.config.profiles.keys()) == ["default"]
    assert model.config.profiles["default"].solver_id == "solver_0"
    assert "solver_0" in model.runs


def test_create_model_accepts_circuit_config_from_schedule_helper(tmp_path: Path):
    solver_path, device_path, pulse_path, analyser_path = _write_basic_solver_device_pulse_and_analyser(tmp_path)
    schedule_path = tmp_path / "schedule_only.yaml"
    schedule_path.write_text(json.dumps({"0": [[["x", [0]]]]}, ensure_ascii=False), encoding="utf-8")

    model = create_model(
        circuits=CircuitConfig.from_schedule_file(schedule_path),
        solvers=solver_path,
        devices=device_path,
        pulses=pulse_path,
        analysers=analyser_path,
    )
    model.run()

    assert "default" in model.config.circuits
    assert model.config.circuits["default"].circuit_ir is not None
    assert model.config.circuits["default"].circuit_ir.schedule[0][0][0].name == "x"
    assert "solver_0" in model.runs


def test_create_model_accepts_named_resource_dicts_and_builds_profiles(tmp_path: Path):
    (tmp_path / "task_a.qasm").write_text("OPENQASM 3; qubit[1] q;", encoding="utf-8")
    (tmp_path / "task_b.qasm").write_text("OPENQASM 3; qubit[1] q; x q[0];", encoding="utf-8")
    solver_path, device_path, pulse_path, analyser_path = _write_basic_solver_device_pulse_and_analyser(tmp_path)

    circuit_a = tmp_path / "circuit_a.json"
    circuit_b = tmp_path / "circuit_b.json"
    circuit_a.write_text(json.dumps({"qasm_path": "task_a.qasm"}), encoding="utf-8")
    circuit_b.write_text(json.dumps({"qasm_path": "task_b.qasm"}), encoding="utf-8")

    model = create_model(
        circuits={"ground": circuit_a, "excited": circuit_b},
        solvers={"solver_main": solver_path},
        devices={"device_main": device_path},
        pulses={"pulse_main": pulse_path},
        analysers={"analyser_main": analyser_path},
    )

    assert sorted(model.config.circuits.keys()) == ["excited", "ground"]
    assert sorted(model.config.solvers.keys()) == ["solver_main"]
    assert sorted(model.config.devices.keys()) == ["device_main"]
    assert sorted(model.config.pulses.keys()) == ["pulse_main"]
    assert sorted(model.config.analysers.keys()) == ["analyser_main"]
    assert sorted(model.config.profiles.keys()) == [
        "excited_device_main_pulse_main_solver_main",
        "ground_device_main_pulse_main_solver_main",
    ]


def test_zz_connection_uses_new_effective_and_residual_fields():
    device = normalize_device_config(
        {
            "components": [
                {"id": "q0", "type": "transmon", "parameters": {"freq_Hz": 5.0e9, "anharmonicity_Hz": -2.0e8}},
                {"id": "q1", "type": "transmon", "parameters": {"freq_Hz": 5.1e9, "anharmonicity_Hz": -2.1e8}},
            ],
            "connections": [
                {
                    "id": "zz_q0_q1",
                    "type": "zz",
                    "a": "q0",
                    "b": "q1",
                    "parameters": {"max_effective_coupling_Hz": 18.0e6},
                    "noise": {"residual_zz_Hz": 1.2e5},
                }
            ],
            "simulation_level": "nlevel",
        }
    )

    static_couplings = lower_couplings(device, num_qubits=2)
    conn_spec = SystemConnectionSpec.from_dict(device.connection_dicts[0])

    assert len(static_couplings) == 1
    assert static_couplings[0].kind == "zz"
    assert static_couplings[0].coefficient_Hz == pytest.approx(1.2e5)
    assert conn_spec.to_device_dict()["parameters"]["max_effective_coupling_Hz"] == pytest.approx(18.0e6)
    assert conn_spec.to_device_dict()["noise"]["residual_zz_Hz"] == pytest.approx(1.2e5)


def test_save_and_load_model_round_trips_profiles_and_resource_pools(tmp_path: Path):
    (tmp_path / "task_a.qasm").write_text("OPENQASM 3; qubit[1] q;", encoding="utf-8")
    (tmp_path / "task_b.qasm").write_text("OPENQASM 3; qubit[1] q; x q[0];", encoding="utf-8")
    solver_path, device_path, pulse_path, analyser_path = _write_basic_solver_device_pulse_and_analyser(tmp_path)

    circuit_a = tmp_path / "circuit_a.json"
    circuit_b = tmp_path / "circuit_b.json"
    circuit_a.write_text(json.dumps({"qasm_path": "task_a.qasm"}), encoding="utf-8")
    circuit_b.write_text(json.dumps({"qasm_path": "task_b.qasm"}), encoding="utf-8")

    model = create_model(
        circuits={"ground": circuit_a, "excited": circuit_b},
        solvers={"solver_0": solver_path},
        devices={"device_a": device_path},
        pulses={"pulse_a": pulse_path},
        analysers={"analyser_0": analyser_path},
    )
    model.config.profiles = {
        "ground_profile": ProfileConfig(
            circuit_id="ground",
            device_id="device_a",
            pulse_id="pulse_a",
            solver_id="solver_0",
            analyser_id="analyser_0",
        ),
        "excited_profile": ProfileConfig(
            circuit_id="excited",
            device_id="device_a",
            pulse_id="pulse_a",
            solver_id="solver_0",
            analyser_id="analyser_0",
        ),
    }
    model.config.parameter_sweep = ParameterSweepConfig(
        parameters={
            "theta": ParameterList(
                target="circuit.param_bindings.theta",
                values=[0.0, 1.0],
                unit="rad",
                description="rotation angle",
            )
        },
        mode="zip",
        metadata={"suite": "roundtrip"},
    )

    out_dir = tmp_path / "saved_model"
    model.save(out_dir)
    restored = load_model(out_dir)

    assert sorted(restored.config.circuits.keys()) == ["excited", "ground"]
    assert sorted(restored.config.devices.keys()) == ["device_a"]
    assert sorted(restored.config.pulses.keys()) == ["pulse_a"]
    assert sorted(restored.config.solvers.keys()) == ["solver_0"]
    assert sorted(restored.config.analysers.keys()) == ["analyser_0"]
    assert sorted(restored.config.profiles.keys()) == ["excited_profile", "ground_profile"]
    assert restored.config.profiles["ground_profile"].circuit_id == "ground"
    assert restored.config.parameter_sweep is not None
    assert restored.config.parameter_sweep.mode == "zip"
    assert restored.config.parameter_sweep.parameters["theta"].values == [0.0, 1.0]


def test_model_copy_can_drop_results_but_keep_detached_configs(tmp_path: Path):
    qasm_path = tmp_path / "task.qasm"
    qasm_path.write_text("OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nmeasure q[0] -> c[0];\n", encoding="utf-8")
    solver_path, device_path, pulse_path, analyser_path = _write_basic_solver_device_pulse_and_analyser(tmp_path)
    circuit_path = tmp_path / "circuit.json"
    circuit_path.write_text(json.dumps({"qasm_path": "task.qasm"}, ensure_ascii=False), encoding="utf-8")

    model = create_model(
        circuits=circuit_path,
        solvers=solver_path,
        devices=device_path,
        pulses=pulse_path,
        analysers=analyser_path,
    )
    model.run()

    copied = model.copy(include_results=False)

    assert copied.runs == {}
    assert copied.analyses == {}
    assert copied.out_dir is None
    assert copied.config is not model.config
    assert copied.config.pulses["default"] is not model.config.pulses["default"]

    copied.config.tags.append("copied")
    assert "copied" not in model.config.tags
