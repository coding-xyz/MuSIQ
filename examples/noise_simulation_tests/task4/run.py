# -*- coding: utf-8 -*-
from pathlib import Path

from workflow import create_model


def _terminal_metric(analysis, name: str):
    metric = (analysis.output.metrics or {}).get(name) if analysis is not None else None
    values = getattr(metric, "values", None)
    return values[-1] if isinstance(values, list) and values else None


BASE = Path(__file__).resolve().parent
model = create_model(
    circuit_config=BASE / "circuit.yaml",
    solver_config=BASE / "solver.yaml",
    device_config=BASE / "device.yaml",
    pulse_config=BASE / "pulse.yaml",
    analyser_config=BASE / "analyser.yaml",
)
model.config.output.out_dir = str((BASE.parent / "runs" / "task4_parametric_state_prep").resolve())
model.run_all()
analysis = model.get_analysis()
trajectory = model.get_trajectory()
theta = dict(model.circuit.param_bindings or {}).get("theta")
control_scale = model.device.device.get("control_scale", 1.0)

print(model.out_dir)
print(
    {
        "engine": trajectory.engine if trajectory else "",
        "theta": theta,
        "control_scale": control_scale,
        "mean_excited": _terminal_metric(analysis, "mean_excited"),
        "variance": _terminal_metric(analysis, "variance"),
    }
)
