from pathlib import Path

from musiq.calibrate import CalibrationConfig
from musiq.workflow import create_model


ROOT = Path(__file__).resolve().parents[1]
TASK8_DIR = ROOT / "report" / "task8_dynamical_decoupling"
TASK10_DIR = ROOT / "report" / "task10_two_qubit_gates"


def _build_task8_model():
    return create_model(
        circuits=TASK8_DIR / "circuits" / "calibrate_x.yaml",
        devices=TASK8_DIR / "device.yaml",
        pulses=TASK8_DIR / "pulses.yaml",
        solvers=TASK8_DIR / "solver.yaml",
        analysers=TASK8_DIR / "analyser.yaml",
    )


def _build_task10_model():
    return create_model(
        circuits=TASK10_DIR / "circuits" / "cz.yaml",
        devices=TASK10_DIR / "device.yaml",
        pulses=TASK10_DIR / "pulse.yaml",
        solvers=TASK10_DIR / "solver.yaml",
        analysers=TASK10_DIR / "analyser.yaml",
    )


def test_model_calibrate_updates_supported_single_qubit_gates():
    model = _build_task8_model()

    result = model.calibrate(CalibrationConfig(gates=("sx", "x"), disable_noise=True, print_results=False))

    assert result.component_id == "q0"
    assert result.calibration_solver_mode == "me"
    assert sorted(result.results.keys()) == ["sx", "x"]
    assert result.results["sx"].amplitude_Hz > 0.0
    assert result.results["x"].amplitude_Hz > 0.0
    assert result.results["sx"].drag_beta is not None
    assert result.results["x"].drag_beta is not None
    assert abs(float(model.config.pulses["default"].extras["channel_overrides"]["XY_0"]["sx"]["amplitude_Hz"]) - result.results["sx"].amplitude_Hz) < 1.0
    assert abs(float(model.config.pulses["default"].extras["channel_overrides"]["XY_0"]["x"]["amplitude_Hz"]) - result.results["x"].amplitude_Hz) < 1.0
    assert result.results["sx"].terminal_population.get("1", 0.0) > 0.45
    assert result.results["x"].terminal_population.get("1", 0.0) > 0.94


def test_model_calibrate_supports_keyword_arguments_form():
    model = _build_task8_model()

    result = model.calibrate(component_id="q0", gates=["x", "sx"], disable_noise=True, print_results=False)

    assert sorted(result.results.keys()) == ["sx", "x"]
    assert result.component_id == "q0"


def test_model_calibrate_can_leave_original_pulse_unchanged():
    model = _build_task8_model()
    original_sx = float(model.config.pulses["default"].extras["gates"]["sx"]["amplitude_Hz"])

    result = model.calibrate(CalibrationConfig(gates=("sx",), disable_noise=True, update_model=False, points=5, rounds=1, maxiter=8, print_results=False))

    assert result.results["sx"].amplitude_Hz > 0.0
    assert float(model.config.pulses["default"].extras["gates"]["sx"]["amplitude_Hz"]) == original_sx


def test_model_calibrate_preserves_original_solver_mode_after_temporary_override():
    model = _build_task8_model()
    solver_cfg = next(iter(model.config.solvers.values()))

    assert solver_cfg.run.solver_mode is None
    assert solver_cfg.study[0]["solver_mode"] == "mcwf"

    result = model.calibrate(CalibrationConfig(gates=("sx",), disable_noise=False, points=5, rounds=1, maxiter=8, print_results=False))

    assert result.calibration_solver_mode == "me"
    assert solver_cfg.run.solver_mode is None
    assert solver_cfg.study[0]["solver_mode"] == "mcwf"


def test_model_calibrate_can_keep_original_solver_mode_when_requested():
    model = _build_task8_model()
    solver_cfg = next(iter(model.config.solvers.values()))

    result = model.calibrate(CalibrationConfig(gates=("sx",), disable_noise=False, calibration_solver_mode=None, points=5, rounds=1, maxiter=8, print_results=False))

    assert result.calibration_solver_mode is None
    assert solver_cfg.run.solver_mode is None
    assert solver_cfg.study[0]["solver_mode"] == "mcwf"


def test_model_calibrate_supports_multiple_single_qubit_targets():
    model = _build_task10_model()

    result = model.calibrate(CalibrationConfig(gates=("sx",), component_ids=("q0", "q1"), disable_noise=True, points=5, rounds=1, maxiter=8, print_results=False))

    assert result.results == {}
    assert sorted(result.target_results.keys()) == ["q0", "q1"]
    assert result.target_results["q0"]["sx"].channel_name == "XY_0"
    assert result.target_results["q1"]["sx"].channel_name == "XY_1"
    assert "XY_0" in model.config.pulses["default"].extras["channel_overrides"]
    assert "XY_1" in model.config.pulses["default"].extras["channel_overrides"]


def test_model_calibrate_supports_config_form():
    model = _build_task10_model()

    config = CalibrationConfig(
        gates=("x",),
        component_ids=("q0", "q1"),
        points=5,
        rounds=1,
        maxiter=8,
        disable_noise=True,
        print_results=False,
    )
    result = model.calibrate(config)

    assert sorted(result.target_results.keys()) == ["q0", "q1"]
    assert sorted(result.target_results["q0"].keys()) == ["x"]
    assert sorted(result.target_results["q1"].keys()) == ["x"]


def test_model_calibrate_keyword_overrides_take_precedence_over_config():
    model = _build_task10_model()

    config = CalibrationConfig(
        gates=("sx",),
        component_ids=("q0", "q1"),
        disable_noise=True,
        points=5,
        rounds=1,
        maxiter=8,
        print_results=False,
    )
    result = model.calibrate(config, gates=["x"], component_ids=["q0"], print_results=False)

    assert sorted(result.target_results.keys()) == ["q0"]
    assert sorted(result.target_results["q0"].keys()) == ["x"]


def test_model_calibrate_supports_two_qubit_target():
    model = _build_task10_model()

    result = model.calibrate(CalibrationConfig(gates=("cz",), pair_component_ids=(("q0", "q1"),), disable_noise=True, points=5, rounds=1, maxiter=8, print_results=False))

    assert sorted(result.results.keys()) == ["cz"]
    cz_result = result.results["cz"]
    assert cz_result.channel_name == "C_0"
    assert cz_result.duration_ns is not None
    assert cz_result.target_metric_name == "bell_fidelity"
    assert cz_result.target_metric_value is not None


def test_model_calibrate_prints_summary(capsys):
    model = _build_task8_model()

    result = model.calibrate(CalibrationConfig(gates=("sx",), disable_noise=True, points=5, rounds=1, maxiter=8, print_results=True))
    captured = capsys.readouterr()

    assert "Calibration complete:" in captured.out
    assert "q0:sx" in captured.out
    assert result.format_summary().strip() in captured.out
