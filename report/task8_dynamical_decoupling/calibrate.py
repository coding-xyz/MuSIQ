from __future__ import annotations

import json
from pathlib import Path

from musiq.calibrate import CalibrationConfig
from musiq.workflow import create_model


MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = MODEL_DIR / "calibration_results.json"


def build_model(circuit_name: str):
    model = create_model(
        circuits=MODEL_DIR / "circuits" / circuit_name,
        devices=MODEL_DIR / "device.yaml",
        pulses=MODEL_DIR / "pulses.yaml",
        solvers=MODEL_DIR / "solver.yaml",
        analysers=MODEL_DIR / "analyser.yaml",
    )
    return model


if __name__ == "__main__":
    model = build_model("calibrate_x.yaml")
    calibration = model.calibrate(CalibrationConfig(gates=("sx", "x"), disable_noise=True, calibration_solver_mode="me"))
    sx = calibration.results["sx"]
    x = calibration.results["x"]

    payload = {
        "x": {
            "amplitude_Hz": x.amplitude_Hz,
            "drag_beta": x.drag_beta,
            "terminal_population": x.terminal_population,
        },
        "sx": {
            "amplitude_Hz": sx.amplitude_Hz,
            "drag_beta": sx.drag_beta,
            "terminal_population": sx.terminal_population,
        },
        "calibration_mode": {
            "noise": "disabled",
            "solver_mode": calibration.calibration_solver_mode,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"best x  amplitude_Hz = {x.amplitude_Hz:.1f}  drag_beta = {float(x.drag_beta or 0.0):.4f}  terminal population = {x.terminal_population}"
    )
    print(
        f"best sx amplitude_Hz = {sx.amplitude_Hz:.1f} drag_beta = {float(sx.drag_beta or 0.0):.4f} terminal population = {sx.terminal_population}"
    )
