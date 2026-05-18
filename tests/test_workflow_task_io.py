from __future__ import annotations

import json
from pathlib import Path

import pytest

from musiq.ui.cli import build_parser
from musiq.workflow import create_model, load_model
from musiq.workflow.task_io import (
    load_analyser_config_file,
    load_circuit_config_file,
    load_device_config_file,
    load_pulse_config_file,
    load_solver_config_file,
)
from musiq.schemas.utils import ParameterList, ParameterSweepConfig
from musiq.workflow import ProfileConfig


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
    pulse_cfg = {"template": "single_qubit_default"}
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
    pulse_path.write_text(json.dumps(pulse_cfg, ensure_ascii=False), encoding="utf-8")
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


def test_solver_config_loads_timing_controls(tmp_path: Path):
    cfg = {
        "backend": {"level": "qubit"},
        "run": {"engine": "qutip", "solver_mode": "me", "dt_s": 1.0e-9, "t_end_s": 3.0e-7, "t_padding_s": 2.0e-8},
    }
    p = tmp_path / "solver_timing.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    solver = load_solver_config_file(p)

    assert solver.run.dt_s == 1.0e-9
    assert solver.run.t_end_s == 3.0e-7
    assert solver.run.t_padding_s == 2.0e-8


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


def test_v3_solver_rejects_legacy_study_parameters(tmp_path: Path):
    p = tmp_path / "solver.yaml"
    p.write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "solver": {
                    "engine": "qutip",
                    "study": [{"name": "bad", "solver_mode": "me", "parameters": {"prep_label": "|0>", "prep_sequence": []}}],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="study\\[\\]\\.parameters is no longer supported"):
        load_solver_config_file(p)


def test_pulse_config_loads_with_template(tmp_path: Path):
    p = tmp_path / "pulse.yaml"
    p.write_text("template: single_qubit_default\npulse:\n  gate_duration_ns: 24.0\n", encoding="utf-8")
    pulse = load_pulse_config_file(p)
    assert pulse["gate_duration_ns"] == 24.0
    assert pulse["xy_freq_Hz"] == 5.0e9


def test_v3_pulse_config_maps_measure_scale_into_measure_amp(tmp_path: Path):
    p = tmp_path / "pulse_v3.yaml"
    p.write_text(
        """
schema_version: "3.0"
pulse:
  channels:
    - name: RO_q0
      kind: readout_drive
      target: r0
      port: ro_in
  carriers:
    RO_q0:
      freq_Hz: 6.45e9
      phase_rad: 0.0
  waveforms:
    readout_probe:
      shape: readout
      duration_ns: 160.0
      edge_ns: 20.0
  operations:
    measure:
      - channel: RO_q0
        waveform: readout_probe
        scale: 4000.0
""",
        encoding="utf-8",
    )
    pulse = load_pulse_config_file(p)
    assert pulse["measure_duration_ns"] == 160.0
    assert pulse["readout_edge_ns"] == 20.0
    assert pulse["measure_amp"] == 3200.0


def test_v3_pulse_config_maps_multi_segment_measure_into_measure_segments(tmp_path: Path):
    p = tmp_path / "pulse_v3_segments.yaml"
    p.write_text(
        """
schema_version: "3.0"
pulse:
  channels:
    - name: RO_q0
      kind: readout_drive
      target: r0
      port: ro_in
  carriers:
    RO_q0:
      freq_Hz: 6.45e9
      phase_rad: 0.0
  waveforms:
    readout_kick:
      shape: readout
      duration_ns: 120.0
      edge_ns: 20.0
    readout_hold:
      shape: readout
      duration_ns: 680.0
      edge_ns: 20.0
  operations:
    measure:
      - channel: RO_q0
        waveform: readout_kick
        scale: 3000.0
      - channel: RO_q0
        waveform: readout_hold
        scale: 1000.0
""",
        encoding="utf-8",
    )
    pulse = load_pulse_config_file(p)
    assert pulse["measure_duration_ns"] == 800.0
    assert pulse["measure_amp"] == 2400.0
    assert pulse["measure_segments"] == [
        {"duration_ns": 120.0, "amp": 2400.0, "edge_ns": 20.0, "rise_ns": 20.0, "fall_ns": 20.0, "shape": "readout"},
        {"duration_ns": 680.0, "amp": 800.0, "edge_ns": 20.0, "rise_ns": 20.0, "fall_ns": 20.0, "shape": "readout"},
    ]


def test_analyser_config_loads_metrics_and_trajectory(tmp_path: Path):
    p = tmp_path / "analyser.yaml"
    p.write_text(
        "trajectory:\n  quantum: density_matrix\n  save_times: all\nmetrics:\n  - population\n  - mean_excited\n",
        encoding="utf-8",
    )
    analyser = load_analyser_config_file(p)
    assert analyser.trajectory.extras["quantum"] == "density_matrix"
    assert analyser.metrics == ["population", "mean_excited"]


def test_analyser_config_loads_case_and_sweep_metrics(tmp_path: Path):
    p = tmp_path / "analyser.yaml"
    p.write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                "case_metrics:",
                "  - population",
                "sweep_metrics:",
                "  - final_P0",
                "  - final_leakage",
            ]
        ),
        encoding="utf-8",
    )
    analyser = load_analyser_config_file(p)
    assert analyser.case_metrics == ["population"]
    assert analyser.sweep_metrics == ["final_P0", "final_leakage"]


def test_device_config_with_qubits_is_accepted(tmp_path: Path):
    p = tmp_path / "device.json"
    p.write_text(
        json.dumps(
            {
                "device": {
                    "simulation_level": "qubit",
                    "qubits": [{"freq_Hz": 5.0e9, "anharmonicity_Hz": -2.0e8, "T1_s": 1.2e-4, "T2_s": 9.0e-5}]
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_device_config_file(p)

    assert cfg.device is not None
    assert cfg.device["qubits"][0]["freq_Hz"] == 5.0e9


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
    circuit_cfg = {
        "qasm_path": "task.qasm",
    }
    circuit_path = tmp_path / "circuit.json"
    circuit_path.write_text(json.dumps(circuit_cfg, ensure_ascii=False), encoding="utf-8")

    model = create_model(
        circuit_config=circuit_path,
        solver_config=solver_path,
        device_config=device_path,
        pulse_config=pulse_path,
        analyser_config=analyser_path,
    )
    model.run()
    assert "default" in model.config.circuits
    assert "default" in model.config.devices
    assert "default" in model.config.pulses
    assert "solver_0" in model.config.solvers
    assert "analyser_0" in model.config.analysers
    assert model.config.profiles["default"].solver_id == "solver_0"
    assert "run_0" in model.runs
    assert model.out_dir is not None


def test_create_model_accepts_named_resource_dicts(tmp_path: Path):
    (tmp_path / "task_a.qasm").write_text("OPENQASM 3; qubit[1] q;", encoding="utf-8")
    (tmp_path / "task_b.qasm").write_text("OPENQASM 3; qubit[1] q; x q[0];", encoding="utf-8")
    solver_path, device_path, pulse_path, analyser_path = _write_basic_solver_device_pulse_and_analyser(tmp_path)

    circuit_a = tmp_path / "circuit_a.json"
    circuit_b = tmp_path / "circuit_b.json"
    circuit_a.write_text(json.dumps({"qasm_path": "task_a.qasm"}), encoding="utf-8")
    circuit_b.write_text(json.dumps({"qasm_path": "task_b.qasm"}), encoding="utf-8")

    model = create_model(
        circuit_config={"ground": circuit_a, "excited": circuit_b},
        solver_config={"solver_main": solver_path},
        device_config={"device_main": device_path},
        pulse_config={"pulse_main": pulse_path},
        analyser_config={"analyser_main": analyser_path},
    )

    assert sorted(model.config.circuits.keys()) == ["excited", "ground"]
    assert sorted(model.config.solvers.keys()) == ["solver_main"]
    assert sorted(model.config.devices.keys()) == ["device_main"]
    assert sorted(model.config.pulses.keys()) == ["pulse_main"]
    assert sorted(model.config.analysers.keys()) == ["analyser_main"]
    assert model.config.profiles == {}


def test_save_and_load_model_round_trips_profiles_and_resource_pools(tmp_path: Path):
    (tmp_path / "task_a.qasm").write_text("OPENQASM 3; qubit[1] q;", encoding="utf-8")
    (tmp_path / "task_b.qasm").write_text("OPENQASM 3; qubit[1] q; x q[0];", encoding="utf-8")
    solver_path, device_path, pulse_path, analyser_path = _write_basic_solver_device_pulse_and_analyser(tmp_path)

    circuit_a = tmp_path / "circuit_a.json"
    circuit_b = tmp_path / "circuit_b.json"
    circuit_a.write_text(json.dumps({"qasm_path": "task_a.qasm"}), encoding="utf-8")
    circuit_b.write_text(json.dumps({"qasm_path": "task_b.qasm"}), encoding="utf-8")

    model = create_model(
        circuit_config={"ground": circuit_a, "excited": circuit_b},
        solver_config={"solver_0": solver_path},
        device_config={"device_a": device_path},
        pulse_config={"pulse_a": pulse_path},
        analyser_config={"analyser_0": analyser_path},
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
    model.config.parameter_list = ParameterSweepConfig(
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
    assert restored.config.parameter_list is not None
    assert restored.config.parameter_list.mode == "zip"
    assert restored.config.parameter_list.parameters["theta"].values == [0.0, 1.0]
