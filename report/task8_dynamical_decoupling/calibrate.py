from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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
    configure_ideal_calibration(model)
    return model


def configure_ideal_calibration(model) -> None:
    device_cfg = model.config.devices["default"]
    device_payload = dict(device_cfg.device or {})
    components = []
    for comp in list(device_payload.get("components", []) or []):
        comp_payload = dict(comp or {})
        comp_payload["noise"] = []
        components.append(comp_payload)
    device_payload["components"] = components
    device_cfg.device = device_payload
    device_cfg.noise = None

    solver_cfg = model.config.solvers["solver_0"]
    solver_cfg.run.solver_mode = "me"
    solver_cfg.run.mcwf_ntraj = 1
    for step in list(solver_cfg.study or []):
        if isinstance(step, dict):
            step["solver_mode"] = "me"


def set_gate_amplitude(model, gate_name: str, amplitude_hz: float) -> None:
    model.config.pulses["default"].extras["gates"][gate_name]["amplitude_Hz"] = float(amplitude_hz)


def set_gate_drag_beta(model, gate_name: str, drag_beta: float) -> None:
    model.config.pulses["default"].extras["gates"][gate_name]["drag_beta"] = float(drag_beta)


def terminal_populations(model) -> dict[str, float]:
    metrics = model.analyses["case_0"].metrics
    pop_metric = metrics.get("population")
    if pop_metric is None:
        pop_metric = metrics.get("population_mean")
    if pop_metric is None:
        raise KeyError("population metric not found in case analysis output")
    pop = pop_metric.values
    return {label: float(series[-1]) for label, series in pop.items()}


def leakage_penalty(pop: dict[str, float], *, weight: float = 5.0) -> float:
    return float(weight) * pop.get("2", 0.0) ** 2


def loss_x(params: np.ndarray, x_model, phase_model) -> float:
    direct_trial = x_model.copy(include_results=False)
    set_gate_amplitude(direct_trial, "x", float(params[0]))
    set_gate_drag_beta(direct_trial, "x", float(params[1]))
    direct_trial.run_all()
    pop = terminal_populations(direct_trial)
    p0 = pop.get("0", 0.0)
    p1 = pop.get("1", 0.0)
    direct_loss = (p1 - 1.0) ** 2 + p0**2 + leakage_penalty(pop)

    phase_trial = phase_model.copy(include_results=False)
    set_gate_amplitude(phase_trial, "x", float(params[0]))
    set_gate_drag_beta(phase_trial, "x", float(params[1]))
    phase_trial.run_all()
    phase_pop = terminal_populations(phase_trial)
    phase_loss = phase_pop.get("1", 0.0) ** 2 + leakage_penalty(phase_pop, weight=2.0)

    return direct_loss + phase_loss


def loss_sx(params: np.ndarray, sx_model, compose_model) -> float:
    half_trial = sx_model.copy(include_results=False)
    set_gate_amplitude(half_trial, "sx", float(params[0]))
    set_gate_drag_beta(half_trial, "sx", float(params[1]))
    half_trial.run_all()
    pop = terminal_populations(half_trial)
    p0 = pop.get("0", 0.0)
    p1 = pop.get("1", 0.0)
    half_loss = (p0 - 0.5) ** 2 + (p1 - 0.5) ** 2 + leakage_penalty(pop)

    compose_trial = compose_model.copy(include_results=False)
    set_gate_amplitude(compose_trial, "sx", float(params[0]))
    set_gate_drag_beta(compose_trial, "sx", float(params[1]))
    compose_trial.run_all()
    compose_pop = terminal_populations(compose_trial)
    compose_loss = (compose_pop.get("1", 0.0) - 1.0) ** 2 + compose_pop.get("0", 0.0) ** 2 + leakage_penalty(
        compose_pop,
        weight=2.0,
    )

    return half_loss + compose_loss


def optimize_gate(
    *,
    models: tuple,
    gate_name: str,
    x0_amplitude: float,
    x0_drag_beta: float,
    loss_fn,
) -> tuple[float, float, dict[str, float]]:
    primary_model = models[0]
    result = minimize(
        lambda x: loss_fn(x, *models),
        x0=np.array([x0_amplitude, x0_drag_beta], dtype=float),
        method="Nelder-Mead",
        tol=1e-8,
        options={
            # "xatol": 1.0e-3,
            # "fatol": 1.0e-8,
            "maxiter": 500,
            "disp": True,
        },
    )

    best_amplitude = float(result.x[0])
    best_drag_beta = float(result.x[1])
    best_model = primary_model.copy(include_results=False)
    set_gate_amplitude(best_model, gate_name, best_amplitude)
    set_gate_drag_beta(best_model, gate_name, best_drag_beta)
    best_model.run_all()
    return best_amplitude, best_drag_beta, terminal_populations(best_model)


if __name__ == "__main__":
    sx_models = (
        build_model("calibrate_sx.yaml"),
        build_model("calibrate_sx_twice.yaml"),
    )
    best_sx_amp, best_sx_beta, sx_pop = optimize_gate(
        models=sx_models,
        gate_name="sx",
        x0_amplitude=8.5e6,
        x0_drag_beta=0.1,
        loss_fn=loss_sx,
    )

    x_models = (
        build_model("calibrate_x.yaml"),
        build_model("calibrate_x_phase.yaml"),
    )
    for model in x_models:
        set_gate_amplitude(model, "sx", best_sx_amp)
        set_gate_drag_beta(model, "sx", best_sx_beta)
    best_x_amp, best_x_beta, x_pop = optimize_gate(
        models=x_models,
        gate_name="x",
        x0_amplitude=17.0e6,
        x0_drag_beta=0.1,
        loss_fn=loss_x,
    )

    payload = {
        "x": {
            "amplitude_Hz": best_x_amp,
            "drag_beta": best_x_beta,
            "terminal_population": x_pop,
        },
        "sx": {
            "amplitude_Hz": best_sx_amp,
            "drag_beta": best_sx_beta,
            "terminal_population": sx_pop,
        },
        "calibration_mode": {
            "noise": "disabled",
            "solver_mode": "me",
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"best x  amplitude_Hz = {best_x_amp:.1f}  drag_beta = {best_x_beta:.4f}  terminal population = {x_pop}")
    print(f"best sx amplitude_Hz = {best_sx_amp:.1f} drag_beta = {best_sx_beta:.4f} terminal population = {sx_pop}")
