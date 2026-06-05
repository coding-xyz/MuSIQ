from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from musiq.analysis.common.state_utils import population_series
from musiq.pulse.catalog import _channel_name_for_gate
from musiq.schemas.circuit import CircuitGate, CircuitIR, build_serial_schedule
from musiq.workflow.contracts import CircuitConfig, ProfileConfig


SUPPORTED_SINGLE_QUBIT_RECIPES = {"sx", "x"}
SUPPORTED_TWO_QUBIT_RECIPES = {"cz", "iswap"}
ORDERED_CALIBRATION_GATES = ("sx", "x", "cz", "iswap")
COUPLER_ID_RE = re.compile(r"^c(\d+)$", re.IGNORECASE)


@dataclass(slots=True)
class GateCalibrationResult:
    gate_name: str
    channel_name: str
    target_components: tuple[str, ...]
    amplitude_Hz: float
    initial_amplitude_Hz: float
    drag_beta: float | None = None
    initial_drag_beta: float | None = None
    duration_ns: float | None = None
    initial_duration_ns: float | None = None
    loss: float = 0.0
    terminal_population: dict[str, float] = field(default_factory=dict)
    target_metric_name: str | None = None
    target_metric_value: float | None = None


@dataclass(slots=True)
class CalibrationResult:
    pulse_id: str
    solver_id: str
    device_id: str
    component_id: str | None
    disable_noise: bool
    calibration_solver_mode: str | None
    model_updated: bool
    results: dict[str, GateCalibrationResult] = field(default_factory=dict)
    target_results: dict[str, dict[str, GateCalibrationResult]] = field(default_factory=dict)

    def iter_results(self) -> list[tuple[str, GateCalibrationResult]]:
        rows: list[tuple[str, GateCalibrationResult]] = []
        for target_key in sorted(self.target_results.keys()):
            for gate_name in sorted(self.target_results[target_key].keys()):
                rows.append((target_key, self.target_results[target_key][gate_name]))
        if rows:
            return rows
        if self.results:
            for gate_name in sorted(self.results.keys()):
                rows.append((self.component_id or "default", self.results[gate_name]))
        return rows

    def format_summary(self) -> str:
        mode = self.calibration_solver_mode or "original"
        update_status = "updated" if self.model_updated else "not-updated"
        noise_status = "disabled" if self.disable_noise else "enabled"
        lines = [
            (
                f"Calibration complete: pulse={self.pulse_id} solver={self.solver_id} "
                f"device={self.device_id} noise={noise_status} solver_mode={mode} model={update_status}"
            )
        ]
        for target_key, result in self.iter_results():
            parts = [
                f"{target_key}:{result.gate_name}",
                f"channel={result.channel_name}",
                f"amp={result.amplitude_Hz:.6g}Hz",
            ]
            if result.drag_beta is not None:
                parts.append(f"drag_beta={result.drag_beta:.6g}")
            if result.duration_ns is not None:
                parts.append(f"duration={result.duration_ns:.6g}ns")
            if result.target_metric_name and result.target_metric_value is not None:
                parts.append(f"{result.target_metric_name}={result.target_metric_value:.6g}")
            parts.append(f"loss={result.loss:.6g}")
            lines.append("  " + "  ".join(parts))
        return "\n".join(lines)


@dataclass(slots=True)
class CalibrationConfig:
    gates: tuple[str, ...] | None = None
    pulse_id: str | None = None
    solver_id: str | None = None
    device_id: str | None = None
    component_id: str | None = None
    component_ids: tuple[str, ...] | None = None
    pair_component_ids: tuple[tuple[str, str], ...] | None = None
    disable_noise: bool = False
    calibration_solver_mode: str | None = "me"
    update_model: bool = True
    points: int = 17
    rounds: int = 3
    relative_span: float = 0.35
    maxiter: int = 80
    proximity_weight: float = 1.0e-4
    print_results: bool = True


@dataclass(slots=True)
class CalibrationTarget:
    key: str
    kind: str
    component_ids: tuple[str, ...]
    qubit_indices: tuple[int, ...]
    channel_name: str
    scope_components: tuple[str, ...]
    scope_connections: tuple[str, ...]


def config_resource_ids(config: CalibrationConfig, *, context: str) -> tuple[str, str, str]:
    pulse_id = str(config.pulse_id or "")
    solver_id = str(config.solver_id or "")
    device_id = str(config.device_id or "")
    if not pulse_id or not solver_id or not device_id:
        raise ValueError(f"{context} requires resolved pulse_id, solver_id, and device_id.")
    return pulse_id, solver_id, device_id


def first_run_and_spec(model) -> tuple[Any, Any]:
    run_obj = next(iter(model.runs.values()))
    result = next(iter(run_obj.results.values()))
    trajectory = next(iter(result.trajectories.values()))
    model_spec = run_obj.artifacts.model_spec
    return trajectory, model_spec


def run_model(model) -> tuple[Any, Any, dict[str, float]]:
    model.run()
    trajectory, model_spec = first_run_and_spec(model)
    series = population_series(trajectory, model_spec)
    terminal = {label: float((values or [0.0])[-1]) for label, values in series.items()}
    return trajectory, model_spec, terminal


def raw_pulse_extras(model, *, pulse_id: str) -> dict[str, Any]:
    pulse_cfg = model.config.pulses[pulse_id]
    if pulse_cfg.extras is None:
        pulse_cfg.extras = {}
    return pulse_cfg.extras


def gate_recipe_payload(model, *, pulse_id: str, gate_name: str, channel_name: str | None = None, create: bool = False) -> dict[str, Any]:
    extras = raw_pulse_extras(model, pulse_id=pulse_id)
    gates = extras.setdefault("gates", {})
    if create:
        gates.setdefault(gate_name, {})
    base_recipe = dict(gates.get(gate_name, {}) or {})
    if channel_name is None:
        if create and gate_name not in gates:
            gates[gate_name] = {}
        return gates.get(gate_name, {}) if create else base_recipe
    overrides = extras.setdefault("channel_overrides", {}) if create else dict(extras.get("channel_overrides", {}) or {})
    if create:
        channel_patch = overrides.setdefault(channel_name, {})
        return channel_patch.setdefault(gate_name, {})
    channel_patch = dict(overrides.get(channel_name, {}) or {})
    gate_patch = dict(channel_patch.get(gate_name, {}) or {})
    return {**base_recipe, **gate_patch}


def resolved_gate_param(
    model,
    *,
    pulse_id: str,
    gate_name: str,
    param_name: str,
    channel_name: str | None = None,
) -> Any:
    recipe = gate_recipe_payload(model, pulse_id=pulse_id, gate_name=gate_name, channel_name=channel_name, create=False)
    return recipe.get(param_name)


def set_gate_param(
    model,
    *,
    pulse_id: str,
    gate_name: str,
    param_name: str,
    value: float,
    channel_name: str | None = None,
) -> None:
    recipe = gate_recipe_payload(model, pulse_id=pulse_id, gate_name=gate_name, channel_name=channel_name, create=channel_name is not None)
    if channel_name is None:
        extras = raw_pulse_extras(model, pulse_id=pulse_id)
        extras.setdefault("gates", {}).setdefault(gate_name, {})[param_name] = float(value)
        return
    recipe[param_name] = float(value)


def strip_noise_from_device(model, *, device_id: str) -> None:
    device_cfg = model.config.devices[device_id]
    device_payload = dict(device_cfg.device or {})
    cleaned_components: list[dict[str, Any]] = []
    for component in list(device_payload.get("components", []) or []):
        component_payload = dict(component or {})
        component_payload.pop("noise", None)
        cleaned_components.append(component_payload)
    device_payload["components"] = cleaned_components
    device_cfg.device = device_payload
    device_cfg.noise = None


def apply_calibration_solver_mode(model, *, solver_id: str, calibration_solver_mode: str | None) -> None:
    if calibration_solver_mode is None:
        return
    solver_mode = str(calibration_solver_mode).strip()
    if not solver_mode:
        return
    solver_cfg = model.config.solvers[solver_id]
    solver_cfg.run.solver_mode = solver_mode
    if solver_mode.lower() == "me":
        solver_cfg.run.mcwf_ntraj = 1
    for step in list(solver_cfg.study or []):
        if isinstance(step, dict):
            step["solver_mode"] = solver_mode


def device_components(model, *, device_id: str) -> list[dict[str, Any]]:
    return [dict(component or {}) for component in list((model.config.devices[device_id].device or {}).get("components", []) or [])]


def component_level_map(model, *, device_id: str) -> dict[str, int]:
    level_map: dict[str, int] = {}
    for component in device_components(model, device_id=device_id):
        comp_id = str(component.get("id", "")).strip()
        if not comp_id:
            continue
        if component.get("levels") is not None:
            level_map[comp_id] = max(2, int(component.get("levels") or 2))
            continue
        basis = dict(component.get("basis", {}) or {})
        if basis.get("levels") is not None:
            level_map[comp_id] = max(2, int(basis.get("levels") or 2))
            continue
        level_map[comp_id] = 3 if str(component.get("type", "")).strip().lower() == "transmon" else 2
    return level_map


def device_component_order(model, *, device_id: str) -> list[str]:
    return [str(component.get("id", "")).strip() for component in device_components(model, device_id=device_id) if str(component.get("id", "")).strip()]


def primary_step(model, *, solver_id: str) -> dict[str, Any]:
    solver_cfg = model.config.solvers[solver_id]
    for step in list(solver_cfg.study or []):
        if isinstance(step, dict):
            return step
    return {}


def scope_components(model, *, solver_id: str, device_id: str) -> list[str]:
    step = primary_step(model, solver_id=solver_id)
    active = [str(item).strip() for item in list(step.get("active_components", []) or []) if str(item).strip()]
    return active or device_component_order(model, device_id=device_id)


def scope_connections(model, *, solver_id: str) -> list[str]:
    step = primary_step(model, solver_id=solver_id)
    return [str(item).strip() for item in list(step.get("active_connections", []) or []) if str(item).strip()]


def set_scope(
    model,
    *,
    solver_id: str,
    scope_components_list: list[str],
    scope_connections_list: list[str],
    component_levels: dict[str, int],
) -> None:
    solver_cfg = model.config.solvers[solver_id]
    for step in list(solver_cfg.study or []):
        if not isinstance(step, dict):
            continue
        step["active_components"] = list(scope_components_list)
        step["active_connections"] = list(scope_connections_list)
        step["representations"] = {component_id: "quantum" for component_id in scope_components_list}
        step["bases"] = {
            component_id: {"kind": "nlevel", "levels": int(component_levels.get(component_id, 2))}
            for component_id in scope_components_list
        }


def build_circuit(gates: list[CircuitGate], *, num_qubits: int) -> CircuitIR:
    return CircuitIR(num_qubits=int(num_qubits), schedule=build_serial_schedule(gates, num_qubits=int(num_qubits)))


def prepare_calibration_model(
    source_model,
    *,
    pulse_id: str,
    solver_id: str,
    device_id: str,
    circuit_ir: CircuitIR,
    scope_components_list: list[str],
    scope_connections_list: list[str],
    disable_noise: bool,
    calibration_solver_mode: str | None,
) -> Any:
    trial = source_model.copy(include_results=False)
    trial.config.circuits = {"calibration": CircuitConfig(circuit_ir=circuit_ir)}
    levels = component_level_map(trial, device_id=device_id)
    set_scope(
        trial,
        solver_id=solver_id,
        scope_components_list=scope_components_list,
        scope_connections_list=scope_connections_list,
        component_levels=levels,
    )
    if disable_noise:
        strip_noise_from_device(trial, device_id=device_id)
    apply_calibration_solver_mode(trial, solver_id=solver_id, calibration_solver_mode=calibration_solver_mode)
    base_profile = next(iter(trial.config.profiles.values()))
    trial.config.profiles = {
        "default": ProfileConfig(
            circuit_id="calibration",
            device_id=device_id,
            pulse_id=pulse_id,
            solver_id=solver_id,
            analyser_id=base_profile.analyser_id,
        )
    }
    return trial


def prepare_target_calibration_model(
    source_model,
    *,
    config: CalibrationConfig,
    target: CalibrationTarget,
    circuit_ir: CircuitIR,
    context: str,
) -> Any:
    pulse_id, solver_id, device_id = config_resource_ids(config, context=context)
    return prepare_calibration_model(
        source_model,
        pulse_id=pulse_id,
        solver_id=solver_id,
        device_id=device_id,
        circuit_ir=circuit_ir,
        scope_components_list=list(target.scope_components),
        scope_connections_list=list(target.scope_connections),
        disable_noise=bool(config.disable_noise),
        calibration_solver_mode=config.calibration_solver_mode,
    )


def resolve_resource_ids(model, *, pulse_id: str | None, solver_id: str | None, device_id: str | None) -> tuple[str, str, str]:
    return (
        str(pulse_id or next(iter(model.config.pulses.keys()))),
        str(solver_id or next(iter(model.config.solvers.keys()))),
        str(device_id or next(iter(model.config.devices.keys()))),
    )


def supported_gate_names(model, *, pulse_id: str) -> list[str]:
    pulse_cfg = model.config.pulses[pulse_id]
    gates = dict((pulse_cfg.extras or {}).get("gates", {}) or {})
    supported: list[str] = []
    for name, payload in gates.items():
        if not isinstance(payload, dict):
            continue
        recipe_type = str(payload.get("recipe_type", "")).strip().lower()
        if recipe_type in SUPPORTED_SINGLE_QUBIT_RECIPES or recipe_type in SUPPORTED_TWO_QUBIT_RECIPES:
            supported.append(str(name))
    order = {name: idx for idx, name in enumerate(ORDERED_CALIBRATION_GATES)}
    return sorted(supported, key=lambda name: (order.get(name, 99), name))


def validate_gate_selection(model, *, pulse_id: str, gates: list[str] | None) -> list[str]:
    supported = supported_gate_names(model, pulse_id=pulse_id)
    if gates is None:
        if not supported:
            raise ValueError("No supported calibration gates found in this model pulse configuration.")
        return supported
    requested = [str(name).strip() for name in list(gates or []) if str(name).strip()]
    unsupported = [name for name in requested if name not in supported]
    if unsupported:
        raise ValueError(f"Unsupported calibration gates: {unsupported}. Supported gates are {supported}.")
    order = {name: idx for idx, name in enumerate(ORDERED_CALIBRATION_GATES)}
    return sorted(requested, key=lambda name: (order.get(name, 99), name))


def default_single_targets(model, *, solver_id: str, device_id: str) -> list[str]:
    scoped = scope_components(model, solver_id=solver_id, device_id=device_id)
    return [scoped[0]] if scoped else []


def default_pair_targets(model, *, solver_id: str, device_id: str) -> list[tuple[str, str]]:
    scoped = scope_components(model, solver_id=solver_id, device_id=device_id)
    qubits_only = [component_id for component_id in scoped if COUPLER_ID_RE.match(component_id) is None]
    if len(qubits_only) < 2:
        return []
    return [(qubits_only[0], qubits_only[1])]


def single_targets(
    model,
    *,
    solver_id: str,
    device_id: str,
    component_ids: list[str] | None,
) -> list[CalibrationTarget]:
    scoped = scope_components(model, solver_id=solver_id, device_id=device_id)
    wanted = default_single_targets(model, solver_id=solver_id, device_id=device_id) if component_ids is None else [str(component_id).strip() for component_id in list(component_ids or []) if str(component_id).strip()]
    if not wanted:
        return []
    component_index = {component_id: idx for idx, component_id in enumerate(scoped)}
    targets: list[CalibrationTarget] = []
    for component_id in wanted:
        if component_id not in component_index:
            raise ValueError(f"Calibration component `{component_id}` is not in the active component scope {scoped}.")
        qubit_index = int(component_index[component_id])
        targets.append(
            CalibrationTarget(
                key=component_id,
                kind="single",
                component_ids=(component_id,),
                qubit_indices=(qubit_index,),
                channel_name=f"XY_{qubit_index}",
                scope_components=tuple(scoped),
                scope_connections=tuple(),
            )
        )
    return targets


def pair_targets(
    model,
    *,
    solver_id: str,
    device_id: str,
    pair_component_ids: list[tuple[str, str]] | None,
    gate_name: str,
) -> list[CalibrationTarget]:
    scoped = scope_components(model, solver_id=solver_id, device_id=device_id)
    scoped_connections = scope_connections(model, solver_id=solver_id)
    wanted = default_pair_targets(model, solver_id=solver_id, device_id=device_id) if pair_component_ids is None else [(str(a).strip(), str(b).strip()) for a, b in list(pair_component_ids or [])]
    if not wanted:
        return []
    component_index = {component_id: idx for idx, component_id in enumerate(scoped)}
    hw = dict(model.config.devices[device_id].device or {})
    targets: list[CalibrationTarget] = []
    for comp_a, comp_b in wanted:
        if comp_a not in component_index or comp_b not in component_index:
            raise ValueError(f"Calibration pair `({comp_a}, {comp_b})` must lie within active components {scoped}.")
        q0, q1 = sorted((int(component_index[comp_a]), int(component_index[comp_b])))
        channel_name = str(_channel_name_for_gate(gate_name, [q0, q1], None, None, hw=hw) or "")
        if not channel_name:
            raise ValueError(f"Unable to resolve a physical control channel for gate `{gate_name}` on pair `{comp_a}, {comp_b}`.")
        targets.append(
            CalibrationTarget(
                key=f"{comp_a}-{comp_b}",
                kind="two_qubit",
                component_ids=(comp_a, comp_b),
                qubit_indices=(q0, q1),
                channel_name=channel_name,
                scope_components=tuple(scoped),
                scope_connections=tuple(scoped_connections),
            )
        )
    return targets


def single_bounds(
    initial_amplitude: float,
    initial_drag_beta: float | None,
    *,
    relative_span: float,
) -> tuple[list[float], list[tuple[float, float]]]:
    amp_span = max(abs(initial_amplitude) * max(0.5, relative_span * 2.0), 5.0e5)
    values = [float(initial_amplitude)]
    bounds = [(float(initial_amplitude - amp_span), float(initial_amplitude + amp_span))]
    if initial_drag_beta is not None:
        beta_span = max(abs(initial_drag_beta) * max(1.0, relative_span * 2.0), 0.2)
        values.append(float(initial_drag_beta))
        bounds.append((float(initial_drag_beta - beta_span), float(initial_drag_beta + beta_span)))
    return values, bounds


def two_qubit_bounds(
    initial_amplitude: float,
    initial_duration_ns: float,
    *,
    relative_span: float,
) -> tuple[list[float], list[tuple[float, float]]]:
    amp_span = max(abs(initial_amplitude) * max(0.5, relative_span * 2.0), 5.0e5)
    duration_span = max(abs(initial_duration_ns) * max(0.35, relative_span), 4.0)
    return (
        [float(initial_amplitude), float(initial_duration_ns)],
        [
            (float(initial_amplitude - amp_span), float(initial_amplitude + amp_span)),
            (float(max(1.0, initial_duration_ns - duration_span)), float(initial_duration_ns + duration_span)),
        ],
    )


def optimize_parameters(
    initial_values: list[float],
    bounds: list[tuple[float, float]],
    objective,
    *,
    points: int,
    maxiter: int,
) -> tuple[list[float], float]:
    if len(initial_values) == 1:
        center = float(initial_values[0])
        lo, hi = bounds[0]
        coarse_grid = np.linspace(lo, hi, max(9, int(points)))
        coarse_losses = [float(objective([float(value)])) for value in coarse_grid]
        ranked_indices = sorted(range(len(coarse_grid)), key=lambda idx: coarse_losses[idx])[: min(3, len(coarse_grid))]
        best_value = float(coarse_grid[int(np.argmin(coarse_losses))])
        best_loss = float(min(coarse_losses))
        for idx in ranked_indices:
            left_idx = max(0, idx - 1)
            right_idx = min(len(coarse_grid) - 1, idx + 1)
            seg_lo = float(coarse_grid[left_idx])
            seg_hi = float(coarse_grid[right_idx])
            if seg_lo == seg_hi:
                seg_lo, seg_hi = float(lo), float(hi)
            if seg_lo > seg_hi:
                seg_lo, seg_hi = seg_hi, seg_lo
            result = minimize_scalar(
                lambda value: float(objective([float(value)])),
                bounds=(seg_lo, seg_hi),
                method="bounded",
                options={"xatol": max(abs(center) * 1.0e-5, 1.0), "maxiter": max(50, int(maxiter))},
            )
            if result.success and float(result.fun) < best_loss:
                best_value = float(result.x)
                best_loss = float(result.fun)
        return [best_value], best_loss

    result = minimize(
        lambda params: float(objective([float(value) for value in params])),
        x0=np.asarray(initial_values, dtype=float),
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": max(200, int(maxiter)), "maxfev": max(40, int(maxiter) * 8), "xatol": 1.0e-6, "fatol": 1.0e-8},
    )
    if result.success:
        return [float(value) for value in list(result.x)], float(result.fun)
    return [float(value) for value in list(initial_values)], float(objective(initial_values))


def proximity_penalty(
    params: list[float],
    initial_values: list[float],
    bounds: list[tuple[float, float]],
    *,
    weight: float,
) -> float:
    penalty_weight = float(weight)
    if penalty_weight <= 0.0:
        return 0.0
    total = 0.0
    for value, initial, (lo, hi) in zip(params, initial_values, bounds, strict=False):
        span = max(abs(float(hi) - float(lo)), abs(float(initial)), 1.0)
        total += ((float(value) - float(initial)) / span) ** 2
    return penalty_weight * total


def apply_result_to_model(model, *, pulse_id: str, result: GateCalibrationResult) -> None:
    set_gate_param(
        model,
        pulse_id=pulse_id,
        gate_name=result.gate_name,
        param_name="amplitude_Hz",
        value=result.amplitude_Hz,
        channel_name=result.channel_name,
    )
    if result.drag_beta is not None:
        set_gate_param(
            model,
            pulse_id=pulse_id,
            gate_name=result.gate_name,
            param_name="drag_beta",
            value=result.drag_beta,
            channel_name=result.channel_name,
        )
    if result.duration_ns is not None:
        set_gate_param(
            model,
            pulse_id=pulse_id,
            gate_name=result.gate_name,
            param_name="duration_ns",
            value=result.duration_ns,
            channel_name=result.channel_name,
        )


def flatten_target_results(target_results: dict[str, dict[str, GateCalibrationResult]]) -> tuple[dict[str, GateCalibrationResult], str | None]:
    flattened_results: dict[str, GateCalibrationResult] = {}
    resolved_component_id: str | None = None
    if len(target_results) == 1:
        only_key, only_value = next(iter(target_results.items()))
        flattened_results = deepcopy(only_value)
        if len(next(iter(only_value.values())).target_components) == 1:
            resolved_component_id = next(iter(only_value.values())).target_components[0]
        else:
            resolved_component_id = only_key
    return flattened_results, resolved_component_id
