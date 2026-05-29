"""Catalog and instantiation helpers for gate-to-pulse mappings."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from musiq.common.unit_schema import MODEL_HARDWARE_KEYS, reject_unknown_keys
from musiq.common.schemas import Carrier, PulseSpec
from musiq.schemas.pulse import (
    CouplerTwoQubitRecipe,
    DrivenSingleQubitRecipe,
    GateRecipe,
    IdleGateRecipe,
    MeasureRecipe,
    MeasureSegmentRecipe,
    VirtualPhaseGateRecipe,
)
from musiq.pulse.shapes import make_shape

NS_TO_S = 1e-9
DEFAULT_BREAK_KEEP_HEAD_S = 60.0 * NS_TO_S
DEFAULT_BREAK_KEEP_TAIL_S = 60.0 * NS_TO_S
DEFAULT_RESET_DEPL_BREAK_KEEP_HEAD_S = 30.0 * NS_TO_S
DEFAULT_RESET_DEPL_BREAK_KEEP_TAIL_S = 30.0 * NS_TO_S

# Backward-compatible aliases for display-layer code still phrased in ns.
DEFAULT_BREAK_KEEP_HEAD_NS = DEFAULT_BREAK_KEEP_HEAD_S * 1e9
DEFAULT_BREAK_KEEP_TAIL_NS = DEFAULT_BREAK_KEEP_TAIL_S * 1e9
DEFAULT_RESET_DEPL_BREAK_KEEP_HEAD_NS = DEFAULT_RESET_DEPL_BREAK_KEEP_HEAD_S * 1e9
DEFAULT_RESET_DEPL_BREAK_KEEP_TAIL_NS = DEFAULT_RESET_DEPL_BREAK_KEEP_TAIL_S * 1e9

_LEGACY_PULSE_SCHEMA_KEYS = {
    "channels",
    "carriers",
    "waveforms",
    "operations",
    "single_qubit_gate_amp_scale",
    "double_qubit_gate_amp_scale",
}

_DRIVEN_SINGLE_QUBIT_SPECS: dict[str, dict[str, Any]] = {
    "x": {"rotation_axis": "x", "fixed_rotation_rad": math.pi, "parametric_rotation": False},
    "sx": {"rotation_axis": "x", "fixed_rotation_rad": 0.5 * math.pi, "parametric_rotation": False},
    "rx": {"rotation_axis": "x", "fixed_rotation_rad": None, "parametric_rotation": True},
    "ry": {"rotation_axis": "y", "fixed_rotation_rad": None, "parametric_rotation": True},
    "h": {"rotation_axis": "y", "fixed_rotation_rad": 0.5 * math.pi, "parametric_rotation": False},
}


@dataclass(frozen=True, slots=True)
class PlannedPulse:
    """Normalized pulse plan entry before materialization into ``PulseSpec``."""

    channel: str
    t0_ns: float
    t1_ns: float
    amp: float
    shape: str
    params: dict[str, Any] = field(default_factory=dict)
    carrier: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class OperationPlan:
    """Normalized operation plan shared by typed and fallback lowering paths."""

    duration_ns: float
    pulses: tuple[PlannedPulse, ...] = ()
    events: tuple[dict[str, Any], ...] = ()


def breakable_params(
    *,
    keep_head_s: float,
    keep_tail_s: float,
    break_kind: str,
    break_stage: str | None = None,
) -> dict[str, Any]:
    """Return standard breakability metadata stored on pulse params."""
    out: dict[str, Any] = {
        "breakable": True,
        "break_keep_head_s": float(keep_head_s),
        "break_keep_tail_s": float(keep_tail_s),
        "break_kind": str(break_kind),
    }
    if break_stage is not None:
        out["break_stage"] = str(break_stage)
    return out


def pulse_break_window(channel_name: str, pulse: PulseSpec) -> tuple[float, float] | None:
    """Return breakable middle window for one pulse if explicitly allowed."""
    params = dict(getattr(pulse, "params", {}) or {})
    if not bool(params.get("breakable", False)):
        return None
    keep_head_s = float(params.get("break_keep_head_s", DEFAULT_BREAK_KEEP_HEAD_S))
    keep_tail_s = float(params.get("break_keep_tail_s", DEFAULT_BREAK_KEEP_TAIL_S))
    t0 = float(pulse.t0_s)
    t1 = float(pulse.t1_s)
    b0 = t0 + max(0.0, keep_head_s)
    b1 = t1 - max(0.0, keep_tail_s)
    return (b0, b1) if b1 > b0 else None


def _normalized_pulse_area_s(shape: str, duration_s: float, params: dict[str, Any]) -> float:
    sampler = make_shape(shape, params)
    n = 257
    if duration_s <= 0.0:
        return 0.0
    dt = duration_s / (n - 1)
    values = [sampler.sample(i * dt, 0.0, duration_s, 1.0) for i in range(n)]
    area = 0.0
    for i in range(n - 1):
        area += 0.5 * (values[i] + values[i + 1]) * dt
    return max(area, 1e-18)


def _xy_rotation_amp_rad_s(*, shape: str, duration_s: float, params: dict[str, Any], rotation_rad: float) -> float:
    area_s = _normalized_pulse_area_s(shape, duration_s, params)
    return float(rotation_rad) / (2.0 * area_s)


def _single_qubit_shape(cfg: dict[str, Any]) -> str:
    shape = str(cfg.get("single_qubit_shape", "gaussian")).strip().lower()
    if shape not in {"gaussian", "drag", "rect"}:
        return "gaussian"
    return shape


def _single_qubit_shape_params(
    cfg: dict[str, Any],
    *,
    rotation_rad: float,
    rotation_axis: str,
) -> tuple[str, dict[str, Any]]:
    gate_dur_s = float(cfg["single_qubit_gate_duration_ns"]) * NS_TO_S
    shape = _single_qubit_shape(cfg)
    params: dict[str, Any] = {
        "rotation_rad": float(rotation_rad),
        "rotation_axis": str(rotation_axis),
    }
    if shape in {"gaussian", "drag"}:
        sigma_fraction = max(float(cfg.get("single_qubit_sigma_fraction", 1.0 / 6.0)), 1e-6)
        params["sigma_s"] = max(gate_dur_s * sigma_fraction, 1e-18)
    if shape == "drag":
        params["beta"] = float(cfg.get("single_qubit_drag_beta", 0.35))
    if shape == "rect":
        edge_s = max(float(cfg.get("single_qubit_rect_edge_ns", 0.0)), 0.0) * NS_TO_S
        params["rise_s"] = edge_s
        params["fall_s"] = edge_s
    return shape, params


def resolve_lowering_hardware(hw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize lowering device/pulse knobs into one resolved config."""
    hw = hw or {}
    validate_typed_pulse_schema(hw)
    reject_unknown_keys("device", hw, MODEL_HARDWARE_KEYS)
    typed_defaults = _typed_defaults(hw)
    gate_dur = float(hw.get("gate_duration_ns", 20.0))
    if "gate_duration_ns" in typed_defaults:
        gate_dur = float(typed_defaults.get("gate_duration_ns", gate_dur))
    single_gate_dur = float(hw.get("single_qubit_gate_duration_ns", gate_dur))
    if "single_qubit_gate_duration_ns" in typed_defaults:
        single_gate_dur = float(typed_defaults.get("single_qubit_gate_duration_ns", single_gate_dur))
    double_gate_dur = float(hw.get("double_qubit_gate_duration_ns", 2.0 * single_gate_dur))
    if "double_qubit_gate_duration_ns" in typed_defaults:
        double_gate_dur = float(typed_defaults.get("double_qubit_gate_duration_ns", double_gate_dur))
    idle_dur = float(hw.get("idle_duration_ns", gate_dur))
    if "idle_duration_ns" in typed_defaults:
        idle_dur = float(typed_defaults.get("idle_duration_ns", idle_dur))
    measure_dur = float(hw.get("measure_duration_ns", 200.0))
    if "measure_duration_ns" in typed_defaults:
        measure_dur = float(typed_defaults.get("measure_duration_ns", measure_dur))
    edge_ns = float(hw.get("rect_edge_ns", 2.0))
    if "rect_edge_ns" in typed_defaults:
        edge_ns = float(typed_defaults.get("rect_edge_ns", edge_ns))
    schedule_value = hw.get("schedule", hw.get("schedule_policy", "serial"))
    if "schedule_policy" in typed_defaults:
        schedule_value = typed_defaults.get("schedule_policy", schedule_value)
    resolved = {
        "xy_freq_Hz": float(typed_defaults.get("xy_carrier_freq_Hz", typed_defaults.get("xy_freq_Hz", hw.get("xy_freq_Hz", 5.0e9)))),
        "ro_freq_Hz": float(typed_defaults.get("ro_carrier_freq_Hz", typed_defaults.get("ro_freq_Hz", hw.get("ro_freq_Hz", 6.5e9)))),
        "schedule_policy": str(schedule_value).strip().lower() or "serial",
        "gate_duration_ns": gate_dur,
        "single_qubit_gate_duration_ns": single_gate_dur,
        "double_qubit_gate_duration_ns": double_gate_dur,
        "idle_duration_ns": idle_dur,
        "measure_duration_ns": measure_dur,
        "measure_amp": float(hw.get("measure_amp", 0.8)),
        "rect_edge_ns": edge_ns,
        "readout_edge_ns": float(hw.get("readout_edge_ns", edge_ns)),
        "single_qubit_shape": _single_qubit_shape(hw),
        "single_qubit_sigma_fraction": float(hw.get("single_qubit_sigma_fraction", 1.0 / 6.0)),
        "single_qubit_drag_beta": float(hw.get("single_qubit_drag_beta", 0.35)),
        "single_qubit_rect_edge_ns": float(hw.get("single_qubit_rect_edge_ns", hw.get("rect_edge_ns", 0.0))),
        "reset_measure_duration_ns": float(hw.get("reset_measure_duration_ns", max(measure_dur, 400.0))),
        "reset_deplete_duration_ns": float(hw.get("reset_deplete_duration_ns", 150.0)),
        "reset_latency_duration_ns": float(hw.get("reset_latency_duration_ns", 120.0)),
        "reset_pi_duration_ns": float(hw.get("reset_pi_duration_ns", gate_dur)),
        "reset_measure_amp": float(hw.get("reset_measure_amp", 0.8)),
        "reset_deplete_amp": float(hw.get("reset_deplete_amp", 0.15)),
        "reset_pi_amp": float(hw.get("reset_pi_amp", 1.0)),
        "reset_cond_on": int(hw.get("reset_cond_on", 1)),
        "reset_apply_feedback": bool(hw.get("reset_apply_feedback", True)),
        "reset_feedback_policy": str(hw.get("reset_feedback_policy", "parallel")).strip().lower() or "parallel",
        "defaults": typed_defaults,
        "gates": dict(hw.get("gates", {}) or {}),
        "channel_overrides": dict(hw.get("channel_overrides", {}) or {}),
    }
    measure_segments = list(hw.get("measure_segments", []) or [])
    if measure_segments:
        resolved["measure_segments"] = [
            {
                "duration_ns": float(seg.get("duration_ns", 0.0) or 0.0),
                "amp": float(seg.get("amp", resolved["measure_amp"]) or resolved["measure_amp"]),
                "edge_ns": float(seg.get("edge_ns", resolved["readout_edge_ns"]) or resolved["readout_edge_ns"]),
                "rise_ns": float(seg.get("rise_ns", seg.get("edge_ns", resolved["readout_edge_ns"])) or 0.0),
                "fall_ns": float(seg.get("fall_ns", seg.get("edge_ns", resolved["readout_edge_ns"])) or 0.0),
                "shape": str(seg.get("shape", "readout") or "readout"),
            }
            for seg in measure_segments
            if float(seg.get("duration_ns", 0.0) or 0.0) > 0.0
        ]
        if resolved["measure_segments"]:
            resolved["measure_duration_ns"] = float(sum(seg["duration_ns"] for seg in resolved["measure_segments"]))
            resolved["measure_amp"] = float(resolved["measure_segments"][0]["amp"])
    resolved["measure_start_delay_ns"] = float(hw.get("measure_start_delay_ns", 0.0) or 0.0)
    # Preserve connection metadata so two-qubit lowering can resolve per-pair
    # coupler strengths such as max_effective_coupling_Hz during full runs.
    resolved["connections"] = [dict(item) for item in list(hw.get("connections", []) or []) if isinstance(item, dict)]
    return resolved


def validate_typed_pulse_schema(hw: dict[str, Any] | None, *, strict_user_payload: bool = False) -> None:
    """Validate the external typed pulse schema used by lowering.

    The supported user-facing schema is schedule-first and gate-recipe-first:
    ``defaults`` provides non-gate-specific fallback values, ``gates`` provides
    canonical logical recipes, and ``channel_overrides`` applies per-channel
    patches over those recipes.
    """
    raw = dict(hw or {})
    legacy_keys = sorted(set(raw) & _LEGACY_PULSE_SCHEMA_KEYS)
    if legacy_keys:
        raise ValueError(
            "Legacy pulse schema is no longer supported. "
            f"Replace {legacy_keys} with typed `defaults`/`gates`/`channel_overrides` entries."
        )
    if strict_user_payload:
        allowed_top_level = {"defaults", "gates", "channel_overrides", "acquisition"}
        unsupported = sorted(set(raw) - allowed_top_level)
        if unsupported:
            raise ValueError(
                "Pulse config must use the typed top-level schema "
                "(`defaults`, `gates`, `channel_overrides`, optional `acquisition`). "
                f"Unsupported keys: {unsupported}"
            )

    defaults = raw.get("defaults", {})
    gates = raw.get("gates", {})
    channel_overrides = raw.get("channel_overrides", {})
    if defaults and not isinstance(defaults, dict):
        raise ValueError("Pulse `defaults` must be a mapping when provided.")
    if gates and not isinstance(gates, dict):
        raise ValueError("Pulse `gates` must be a mapping when provided.")
    if channel_overrides and not isinstance(channel_overrides, dict):
        raise ValueError("Pulse `channel_overrides` must be a mapping when provided.")

    def _reject_legacy_amp_scale(section: str, payload: dict[str, Any]) -> None:
        bad = sorted(key for key in payload if "amp_scale" in str(key).lower())
        if bad:
            raise ValueError(
                f"Legacy amplitude scaling fields are no longer supported in {section}: {bad}. "
                "Use explicit `amplitude_Hz` in typed gate recipes."
            )

    _reject_legacy_amp_scale("pulse.defaults", dict(defaults or {}))
    for gate_name, recipe in dict(gates or {}).items():
        if not isinstance(recipe, dict):
            raise ValueError(f"Pulse recipe `gates.{gate_name}` must be a mapping.")
        _reject_legacy_amp_scale(f"pulse.gates.{gate_name}", recipe)
        recipe_type = str(recipe.get("recipe_type", gate_name)).strip().lower()
        if recipe_type in {"sx", "cz"}:
            missing = [field for field in ("duration_ns", "amplitude_Hz") if field not in recipe]
            if missing:
                raise ValueError(f"Pulse recipe `gates.{gate_name}` is missing required fields: {missing}")
        if recipe_type == "virtual_z":
            forbidden = [field for field in ("duration_ns", "amplitude_Hz") if field in recipe]
            if forbidden:
                raise ValueError(
                    f"Pulse recipe `gates.{gate_name}` uses forbidden VirtualZ pulse fields: {forbidden}"
                )
        _build_typed_gate_recipe(str(gate_name), dict(recipe), section=f"gates.{gate_name}")
    for channel_name, overrides in dict(channel_overrides or {}).items():
        if not isinstance(overrides, dict):
            raise ValueError(f"Pulse channel override `channel_overrides.{channel_name}` must be a mapping.")
        for gate_name, patch in overrides.items():
            if not isinstance(patch, dict):
                raise ValueError(
                    f"Pulse channel override `channel_overrides.{channel_name}.{gate_name}` must be a mapping."
                )
            _reject_legacy_amp_scale(f"pulse.channel_overrides.{channel_name}.{gate_name}", patch)


def _typed_defaults(hw: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict((hw or {}).get("defaults", {}) or {})
    if not isinstance(raw, dict):
        return {}
    return raw


def _tc_channel_name(qubits: list[int]) -> str:
    qs = [int(q) for q in list(qubits or [0, 1])]
    i, j = min(qs), max(qs)
    return f"TC_{i}_{j}"


def _channel_name_for_gate(gate_name: str, qubits: list[int], tc_index: int | None, tc_channel: str | None = None) -> str | None:
    gate = str(gate_name).strip().lower()
    if gate in {"x", "sx", "rx", "ry", "h"}:
        if not qubits:
            return None
        return f"XY_{int(qubits[0])}"
    if gate == "measure":
        if not qubits:
            return None
        return f"RO_{int(qubits[0])}"
    if gate in {"cz", "cx"}:
        return str(tc_channel or _tc_channel_name(qubits))
    return None


def _recipe_aliases_for_gate(gate_name: str) -> list[str]:
    gate = str(gate_name).strip().lower()
    aliases = [gate]
    if gate in {"z", "rz"}:
        aliases.append("virtual_z")
    return aliases


def _build_typed_gate_recipe(
    gate_name: str,
    recipe: dict[str, Any],
    *,
    section: str,
) -> GateRecipe:
    logical_gate = str(gate_name).strip().lower()
    recipe_type = str(recipe.get("recipe_type", gate_name)).strip().lower()
    if recipe_type == "virtual_z":
        return VirtualPhaseGateRecipe(
            logical_gate=logical_gate,
            recipe_type="virtual_z",
            phase_rad=float(recipe["phase_rad"]) if "phase_rad" in recipe else None,
        )
    if recipe_type in _DRIVEN_SINGLE_QUBIT_SPECS:
        gate_spec = _DRIVEN_SINGLE_QUBIT_SPECS[recipe_type]
        return DrivenSingleQubitRecipe(
            logical_gate=logical_gate,
            recipe_type=recipe_type,
            duration_ns=float(recipe["duration_ns"]),
            amplitude_Hz=float(recipe.get("amplitude_Hz", 0.0) or 0.0),
            shape=str(recipe["shape"]) if "shape" in recipe and recipe.get("shape") is not None else None,
            sigma_fraction=float(recipe["sigma_fraction"]) if "sigma_fraction" in recipe else None,
            drag_beta=float(recipe["drag_beta"]) if "drag_beta" in recipe else None,
            edge_ns=float(recipe["edge_ns"]) if "edge_ns" in recipe else None,
            rect_edge_ns=float(recipe["rect_edge_ns"]) if "rect_edge_ns" in recipe else None,
            carrier_freq_Hz=float(recipe["carrier_freq_Hz"]) if "carrier_freq_Hz" in recipe else None,
            phase_rad=float(recipe["phase_rad"]) if "phase_rad" in recipe else None,
            rotation_axis=str(gate_spec["rotation_axis"]),
            fixed_rotation_rad=gate_spec["fixed_rotation_rad"],
            parametric_rotation=bool(gate_spec["parametric_rotation"]),
        )
    if recipe_type == "cz":
        return CouplerTwoQubitRecipe(
            logical_gate=logical_gate,
            recipe_type="cz",
            duration_ns=float(recipe["duration_ns"]),
            amplitude_Hz=float(recipe["amplitude_Hz"]),
            shape=str(recipe["shape"]) if "shape" in recipe and recipe.get("shape") is not None else None,
            edge_ns=float(recipe["edge_ns"]) if "edge_ns" in recipe else None,
            rect_edge_ns=float(recipe["rect_edge_ns"]) if "rect_edge_ns" in recipe else None,
            target_conditional_phase_rad=(
                float(recipe["target_conditional_phase_rad"])
                if "target_conditional_phase_rad" in recipe
                else None
            ),
        )
    if recipe_type == "id":
        return IdleGateRecipe(duration_ns=float(recipe.get("duration_ns", 0.0) or 0.0))
    if recipe_type == "measure":
        raw_segments = list(recipe.get("segments", []) or [])
        segments = tuple(
            MeasureSegmentRecipe(
                duration_ns=float(seg.get("duration_ns", 0.0) or 0.0),
                amplitude=float(seg.get("amplitude", seg.get("amp", 0.0)) or 0.0),
                shape=str(seg.get("shape", "readout") or "readout"),
                rise_ns=float(seg.get("rise_ns", seg.get("edge_ns", 0.0)) or 0.0),
                fall_ns=float(seg.get("fall_ns", seg.get("edge_ns", 0.0)) or 0.0),
            )
            for seg in raw_segments
            if isinstance(seg, dict) and float(seg.get("duration_ns", 0.0) or 0.0) > 0.0
        )
        duration_ns = float(recipe.get("duration_ns", 0.0) or 0.0)
        if segments:
            duration_ns = float(sum(segment.duration_ns for segment in segments))
        return MeasureRecipe(
            duration_ns=duration_ns,
            carrier_freq_Hz=float(recipe["carrier_freq_Hz"]) if "carrier_freq_Hz" in recipe else None,
            phase_rad=float(recipe["phase_rad"]) if "phase_rad" in recipe else None,
            amplitude=float(recipe["amplitude"]) if "amplitude" in recipe else None,
            shape=str(recipe["shape"]) if "shape" in recipe and recipe.get("shape") is not None else None,
            rise_ns=float(recipe["rise_ns"]) if "rise_ns" in recipe else None,
            fall_ns=float(recipe["fall_ns"]) if "fall_ns" in recipe else None,
            edge_ns=float(recipe["edge_ns"]) if "edge_ns" in recipe else None,
            segments=segments,
        )
    raise ValueError(
        f"Pulse recipe `{section}` uses unsupported recipe_type `{recipe_type}`."
    )


def resolve_typed_gate_recipe(
    hw: dict[str, Any] | None,
    gate_name: str,
    *,
    channel_name: str | None = None,
) -> GateRecipe | None:
    raw_hw = hw or {}
    gates = dict(raw_hw.get("gates", {}) or {})
    if not gates:
        return None
    gate_aliases = _recipe_aliases_for_gate(gate_name)
    recipe: dict[str, Any] | None = None
    for candidate in gate_aliases:
        raw_recipe = gates.get(candidate)
        if isinstance(raw_recipe, dict):
            recipe = dict(raw_recipe)
            break
    if recipe is None:
        return None
    if channel_name:
        raw_overrides = dict(raw_hw.get("channel_overrides", {}) or {})
        channel_overrides = raw_overrides.get(channel_name)
        if isinstance(channel_overrides, dict):
            for candidate in gate_aliases:
                override_recipe = channel_overrides.get(candidate)
                if isinstance(override_recipe, dict):
                    recipe = {**recipe, **dict(override_recipe)}
                    break
    return _build_typed_gate_recipe(gate_aliases[0], recipe, section=f"gates.{gate_aliases[0]}")


def _single_qubit_shape_from_recipe(recipe: DrivenSingleQubitRecipe, cfg: dict[str, Any]) -> str:
    shape = str(recipe.shape or "").strip().lower()
    if shape in {"gaussian", "drag", "rect"}:
        return shape
    return _single_qubit_shape(cfg)


def _single_qubit_shape_params_from_recipe(
    recipe: DrivenSingleQubitRecipe,
    cfg: dict[str, Any],
    *,
    duration_ns: float,
    rotation_rad: float,
    rotation_axis: str,
) -> tuple[str, dict[str, Any]]:
    shape = _single_qubit_shape_from_recipe(recipe, cfg)
    params: dict[str, Any] = {
        "rotation_rad": float(rotation_rad),
        "rotation_axis": str(rotation_axis),
    }
    if shape in {"gaussian", "drag"}:
        sigma_fraction = max(
            float(recipe.sigma_fraction if recipe.sigma_fraction is not None else cfg.get("single_qubit_sigma_fraction", 1.0 / 6.0)),
            1e-6,
        )
        params["sigma_s"] = max(float(duration_ns) * NS_TO_S * sigma_fraction, 1e-18)
    if shape == "drag":
        params["beta"] = float(recipe.drag_beta if recipe.drag_beta is not None else cfg.get("single_qubit_drag_beta", 0.35))
    if shape == "rect":
        edge_ns = max(
            float(
                recipe.edge_ns
                if recipe.edge_ns is not None
                else recipe.rect_edge_ns
                if recipe.rect_edge_ns is not None
                else cfg.get("single_qubit_rect_edge_ns", 0.0)
            ),
            0.0,
        )
        params["rise_s"] = edge_ns * NS_TO_S
        params["fall_s"] = edge_ns * NS_TO_S
    return shape, params


def _xy_carrier(cfg: dict[str, Any], phase: float = 0.0) -> dict[str, float]:
    return {"freq": float(cfg["xy_freq_Hz"]), "phase": float(phase)}


def _ro_carrier(cfg: dict[str, Any], phase: float = 0.0) -> dict[str, float]:
    return {"freq": float(cfg["ro_freq_Hz"]), "phase": float(phase)}


def _tc_connection_parameters(hw: dict[str, Any] | None, tc_index: int | None, tc_channel: str | None = None) -> dict[str, Any]:
    if hw is None:
        return {}
    connections = [dict(item) for item in list(hw.get("connections", []) or []) if isinstance(item, dict)]
    if tc_channel:
        import re

        match = re.match(r"^TC_(\d+)_(\d+)$", str(tc_channel), re.IGNORECASE)
        if match:
            qa, qb = int(match.group(1)), int(match.group(2))
            want = {f"q{qa}", f"q{qb}"}
            for item in connections:
                endpoints = {str(item.get("a", "")).strip(), str(item.get("b", "")).strip()}
                if endpoints == want:
                    return dict(item.get("parameters", {}) or {})
    if tc_index is None:
        return {}
    if 0 <= int(tc_index) < len(connections):
        return dict(connections[int(tc_index)].get("parameters", {}) or {})
    return {}


def _double_qubit_effective_amp_rad_s(
    *,
    cfg: dict[str, Any],
    hw: dict[str, Any] | None,
    tc_index: int | None,
    tc_channel: str | None,
    duration_s: float,
) -> float:
    conn_params = _tc_connection_parameters(hw, tc_index, tc_channel)
    max_effective_hz = float(conn_params.get("max_effective_coupling_Hz", 0.0) or 0.0)
    if max_effective_hz > 0.0:
        return 2.0 * math.pi * max_effective_hz
    return -math.pi / max(duration_s, 1e-18)


def _materialize_operation_plan(plan: OperationPlan) -> list[tuple[str, PulseSpec]]:
    """Convert a normalized operation plan into concrete ``PulseSpec`` entries."""
    pulses: list[tuple[str, PulseSpec]] = []
    for pulse in plan.pulses:
        pulses.append(
            (
                pulse.channel,
                PulseSpec(
                    t0_s=float(pulse.t0_ns) * NS_TO_S,
                    t1_s=float(pulse.t1_ns) * NS_TO_S,
                    amp=float(pulse.amp),
                    shape=str(pulse.shape),
                    params=dict(pulse.params),
                    carrier=(
                        Carrier(
                            freq=float(pulse.carrier["freq"]),
                            phase=float(pulse.carrier.get("phase", 0.0)),
                        )
                        if pulse.carrier is not None
                        else None
                    ),
                ),
            )
        )
    return pulses


def _typed_measure_segments(recipe: MeasureRecipe, cfg: dict[str, Any]) -> list[MeasureSegmentRecipe]:
    segments = list(recipe.segments)
    if segments:
        return segments
    default_edge_ns = recipe.edge_ns if recipe.edge_ns is not None else cfg["readout_edge_ns"]
    return [
        MeasureSegmentRecipe(
            duration_ns=float(recipe.duration_ns or cfg["measure_duration_ns"]),
            amplitude=float(recipe.amplitude if recipe.amplitude is not None else cfg["measure_amp"]),
            shape=str(recipe.shape or "readout"),
            rise_ns=float(recipe.rise_ns if recipe.rise_ns is not None else default_edge_ns),
            fall_ns=float(recipe.fall_ns if recipe.fall_ns is not None else default_edge_ns),
        )
    ]


def _default_measure_segments(cfg: dict[str, Any]) -> list[MeasureSegmentRecipe]:
    segments = list(cfg.get("measure_segments", []) or [])
    if segments:
        return [
            MeasureSegmentRecipe(
                duration_ns=float(seg.get("duration_ns", 0.0) or 0.0),
                amplitude=float(seg.get("amp", cfg["measure_amp"]) or cfg["measure_amp"]),
                shape=str(seg.get("shape", "readout") or "readout"),
                rise_ns=float(seg.get("rise_ns", seg.get("edge_ns", cfg["readout_edge_ns"])) or 0.0),
                fall_ns=float(seg.get("fall_ns", seg.get("edge_ns", cfg["readout_edge_ns"])) or 0.0),
            )
            for seg in segments
            if float(seg.get("duration_ns", 0.0) or 0.0) > 0.0
        ]
    return [
        MeasureSegmentRecipe(
            duration_ns=float(cfg["measure_duration_ns"]),
            amplitude=float(cfg["measure_amp"]),
            shape="readout",
            rise_ns=float(cfg["readout_edge_ns"]),
            fall_ns=float(cfg["readout_edge_ns"]),
        )
    ]


def _fallback_recipe_for_gate(
    gate: str,
    *,
    cfg: dict[str, Any],
    hw: dict[str, Any] | None,
    tc_index: int | None,
    tc_channel: str | None,
) -> GateRecipe | None:
    if gate in _DRIVEN_SINGLE_QUBIT_SPECS:
        gate_spec = _DRIVEN_SINGLE_QUBIT_SPECS[gate]
        return DrivenSingleQubitRecipe(
            logical_gate=gate,
            recipe_type=gate,
            duration_ns=float(cfg["single_qubit_gate_duration_ns"]),
            carrier_freq_Hz=float(cfg["xy_freq_Hz"]),
            rotation_axis=str(gate_spec["rotation_axis"]),
            fixed_rotation_rad=gate_spec["fixed_rotation_rad"],
            parametric_rotation=bool(gate_spec["parametric_rotation"]),
        )
    if gate in {"rz", "z"}:
        return VirtualPhaseGateRecipe(logical_gate=gate, recipe_type="virtual_z", duration_ns=0.0)
    if gate == "id":
        return IdleGateRecipe(duration_ns=float(cfg["idle_duration_ns"]))
    if gate == "measure":
        return MeasureRecipe(
            duration_ns=float(sum(seg.duration_ns for seg in _default_measure_segments(cfg))),
            carrier_freq_Hz=float(cfg["ro_freq_Hz"]),
            phase_rad=0.0,
            segments=tuple(_default_measure_segments(cfg)),
        )
    if gate == "cz":
        duration = float(cfg["double_qubit_gate_duration_ns"])
        return CouplerTwoQubitRecipe(
            logical_gate="cz",
            recipe_type="cz",
            duration_ns=duration,
            amplitude_Hz=_double_qubit_effective_amp_rad_s(
                cfg=cfg,
                hw=hw,
                tc_index=tc_index,
                tc_channel=tc_channel,
                duration_s=duration * NS_TO_S,
            )
            / (2.0 * math.pi),
            shape="rect",
            edge_ns=float(cfg["rect_edge_ns"]),
            target_conditional_phase_rad=math.pi,
        )
    return None


def _plan_single_qubit_drive(
    *,
    qubits: list[int],
    start_ns: float,
    duration_ns: float,
    amp: float,
    shape: str,
    params: dict[str, Any],
    carrier_freq_hz: float,
    phase_rad: float,
) -> OperationPlan:
    return OperationPlan(
        duration_ns=float(duration_ns),
        pulses=tuple(
            PlannedPulse(
                channel=f"XY_{q}",
                t0_ns=float(start_ns),
                t1_ns=float(start_ns + duration_ns),
                amp=float(amp),
                shape=str(shape),
                params=dict(params),
                carrier={"freq": float(carrier_freq_hz), "phase": float(phase_rad)},
            )
            for q in qubits
        ),
    )


def _plan_driven_single_qubit_recipe(
    recipe: DrivenSingleQubitRecipe,
    *,
    qubits: list[int],
    gate_params: list[float] | None,
    start_ns: float,
    cfg: dict[str, Any],
) -> OperationPlan:
    gate_dur = float(cfg["single_qubit_gate_duration_ns"])
    rotation_rad = float(recipe.rotation_rad(gate_params))
    duration = float(recipe.duration_ns or gate_dur)
    shape, params = _single_qubit_shape_params_from_recipe(
        recipe,
        cfg,
        duration_ns=duration,
        rotation_rad=rotation_rad,
        rotation_axis=str(recipe.rotation_axis),
    )
    amp_hz = float(recipe.amplitude_Hz or 0.0)
    if amp_hz > 0.0:
        amp = 2.0 * math.pi * amp_hz
        if bool(recipe.parametric_rotation):
            amp *= float(rotation_rad) / math.pi
    else:
        amp = _xy_rotation_amp_rad_s(
            shape=shape,
            duration_s=duration * NS_TO_S,
            params=params,
            rotation_rad=float(params["rotation_rad"]),
        )
    carrier_freq_hz = float(recipe.carrier_freq_Hz if recipe.carrier_freq_Hz is not None else cfg["xy_freq_Hz"])
    phase_rad = float(recipe.resolved_phase_rad())
    return _plan_single_qubit_drive(
        qubits=qubits,
        start_ns=start_ns,
        duration_ns=duration,
        amp=amp,
        shape=shape,
        params=params,
        carrier_freq_hz=carrier_freq_hz,
        phase_rad=phase_rad,
    )


def _plan_coupler_two_qubit_recipe(
    recipe: CouplerTwoQubitRecipe,
    *,
    qubits: list[int],
    start_ns: float,
    cfg: dict[str, Any],
    tc_channel: str | None,
) -> OperationPlan:
    duration = float(recipe.duration_ns)
    edge_ns = float(
        recipe.edge_ns
        if recipe.edge_ns is not None
        else recipe.rect_edge_ns
        if recipe.rect_edge_ns is not None
        else cfg["rect_edge_ns"]
    )
    return OperationPlan(
        duration_ns=float(duration),
        pulses=(
            PlannedPulse(
                channel=str(tc_channel or _tc_channel_name(qubits)),
                t0_ns=float(start_ns),
                t1_ns=float(start_ns + duration),
                amp=2.0 * math.pi * float(recipe.amplitude_Hz),
                shape=str(recipe.shape or "rect"),
                params={
                    "rise_s": edge_ns * NS_TO_S,
                    "fall_s": edge_ns * NS_TO_S,
                    "target_conditional_phase_rad": float(
                        recipe.target_conditional_phase_rad if recipe.target_conditional_phase_rad is not None else math.pi
                    ),
                },
                carrier=None,
            ),
        ),
    )


def _plan_measure_recipe(
    *,
    qubits: list[int],
    start_ns: float,
    segments: list[MeasureSegmentRecipe],
    carrier_freq_hz: float,
    phase_rad: float,
) -> OperationPlan:
    pulses: list[PlannedPulse] = []
    duration = 0.0
    for q in qubits:
        offset_ns = 0.0
        for idx, seg in enumerate(segments):
            if seg.duration_ns <= 0.0:
                continue
            pulses.append(
                PlannedPulse(
                    channel=f"RO_{q}",
                    t0_ns=float(start_ns + offset_ns),
                    t1_ns=float(start_ns + offset_ns + float(seg.duration_ns)),
                    amp=float(seg.amplitude),
                    shape=str(seg.shape or "readout"),
                    params={
                        "rise_s": float(seg.rise_ns) * NS_TO_S,
                        "fall_s": float(seg.fall_ns) * NS_TO_S,
                        "measure_segment_index": idx,
                        "measure_segment_count": len(segments),
                        **breakable_params(
                            keep_head_s=DEFAULT_BREAK_KEEP_HEAD_S,
                            keep_tail_s=DEFAULT_BREAK_KEEP_TAIL_S,
                            break_kind="readout",
                            break_stage="measure",
                        ),
                    },
                    carrier={"freq": float(carrier_freq_hz), "phase": float(phase_rad)},
                )
            )
            offset_ns += float(seg.duration_ns)
        duration = max(duration, offset_ns)
    return OperationPlan(duration_ns=float(duration), pulses=tuple(pulses))


def _plan_typed_measure_recipe(
    recipe: MeasureRecipe,
    *,
    qubits: list[int],
    start_ns: float,
    cfg: dict[str, Any],
) -> OperationPlan:
    segments = _typed_measure_segments(recipe, cfg)
    carrier_freq_hz = float(recipe.carrier_freq_Hz if recipe.carrier_freq_Hz is not None else cfg["ro_freq_Hz"])
    phase_rad = float(recipe.phase_rad if recipe.phase_rad is not None else 0.0)
    return _plan_measure_recipe(
        qubits=qubits,
        start_ns=start_ns,
        segments=segments,
        carrier_freq_hz=carrier_freq_hz,
        phase_rad=phase_rad,
    )


def _plan_typed_recipe(
    recipe: GateRecipe,
    *,
    qubits: list[int],
    gate_params: list[float] | None,
    start_ns: float,
    cfg: dict[str, Any],
    tc_channel: str | None,
) -> OperationPlan | None:
    if isinstance(recipe, VirtualPhaseGateRecipe):
        return OperationPlan(duration_ns=0.0)
    if isinstance(recipe, DrivenSingleQubitRecipe):
        return _plan_driven_single_qubit_recipe(
            recipe,
            qubits=qubits,
            gate_params=gate_params,
            start_ns=start_ns,
            cfg=cfg,
        )
    if isinstance(recipe, CouplerTwoQubitRecipe):
        return _plan_coupler_two_qubit_recipe(
            recipe,
            qubits=qubits,
            start_ns=start_ns,
            cfg=cfg,
            tc_channel=tc_channel,
        )
    if isinstance(recipe, IdleGateRecipe):
        return OperationPlan(duration_ns=float(recipe.duration_ns or cfg["idle_duration_ns"]))
    if isinstance(recipe, MeasureRecipe):
        return _plan_typed_measure_recipe(recipe, qubits=qubits, start_ns=start_ns, cfg=cfg)
    return None


def _plan_default_cx_recipe(
    *,
    qubits: list[int],
    start_ns: float,
    cfg: dict[str, Any],
    hw: dict[str, Any] | None,
    tc_index: int | None,
    tc_channel: str | None,
) -> OperationPlan:
    qs = qubits or [0, 1]
    duration = float(cfg["double_qubit_gate_duration_ns"])
    gate_sigma_s = float(cfg["single_qubit_gate_duration_ns"]) * NS_TO_S / 4.0
    edge_s = float(cfg["rect_edge_ns"]) * NS_TO_S
    tc_amp = _double_qubit_effective_amp_rad_s(
        cfg=cfg,
        hw=hw,
        tc_index=tc_index,
        tc_channel=tc_channel,
        duration_s=duration * NS_TO_S,
    )
    return OperationPlan(
        duration_ns=float(duration),
        pulses=(
            PlannedPulse(
                channel=f"XY_{qs[0]}",
                t0_ns=float(start_ns),
                t1_ns=float(start_ns + duration),
                amp=1.2,
                shape="drag",
                params={"beta": 0.35, "sigma_s": gate_sigma_s},
                carrier=_xy_carrier(cfg, phase=0.0),
            ),
            PlannedPulse(
                channel=f"XY_{qs[-1]}",
                t0_ns=float(start_ns),
                t1_ns=float(start_ns + duration),
                amp=1.2,
                shape="drag",
                params={"beta": 0.35, "sigma_s": gate_sigma_s},
                carrier=_xy_carrier(cfg, phase=0.2),
            ),
            PlannedPulse(
                channel=str(tc_channel or _tc_channel_name(qubits)),
                t0_ns=float(start_ns),
                t1_ns=float(start_ns + duration),
                amp=tc_amp,
                shape="rect",
                params={"rise_s": edge_s, "fall_s": edge_s},
                carrier=None,
            ),
        ),
    )


def _plan_default_reset_recipe(
    *,
    qubits: list[int],
    start_ns: float,
    cfg: dict[str, Any],
    reset_feedback_offset_ns: float,
) -> OperationPlan:
    qs = qubits or [0]
    t0 = start_ns
    t1 = t0 + float(cfg["reset_measure_duration_ns"])
    t2 = t1 + float(cfg["reset_deplete_duration_ns"])
    t3 = t2 + float(cfg["reset_latency_duration_ns"]) + max(0.0, float(reset_feedback_offset_ns))
    t4 = t3 + (float(cfg["reset_pi_duration_ns"]) if bool(cfg["reset_apply_feedback"]) else 0.0)
    edge_s = float(cfg["readout_edge_ns"]) * NS_TO_S
    pulses: list[PlannedPulse] = []
    events: list[dict[str, Any]] = []
    for q in qs:
        pulses.append(
            PlannedPulse(
                channel=f"RO_{q}",
                t0_ns=float(t0),
                t1_ns=float(t1),
                amp=float(cfg["reset_measure_amp"]),
                shape="readout",
                params={
                    "stage": "reset_measure",
                    "rise_s": edge_s,
                    "fall_s": edge_s,
                    **breakable_params(
                        keep_head_s=DEFAULT_BREAK_KEEP_HEAD_S,
                        keep_tail_s=DEFAULT_BREAK_KEEP_TAIL_S,
                        break_kind="reset",
                        break_stage="reset_measure",
                    ),
                },
                carrier=_ro_carrier(cfg),
            )
        )
        pulses.append(
            PlannedPulse(
                channel=f"RO_{q}",
                t0_ns=float(t1),
                t1_ns=float(t2),
                amp=float(cfg["reset_deplete_amp"]),
                shape="rect",
                params={
                    "stage": "reset_deplete",
                    "rise_s": edge_s,
                    "fall_s": edge_s,
                    **breakable_params(
                        keep_head_s=DEFAULT_RESET_DEPL_BREAK_KEEP_HEAD_S,
                        keep_tail_s=DEFAULT_RESET_DEPL_BREAK_KEEP_TAIL_S,
                        break_kind="reset",
                        break_stage="reset_deplete",
                    ),
                },
                carrier=_ro_carrier(cfg),
            )
        )
        if bool(cfg["reset_apply_feedback"]) and t4 > t3:
            reset_pi_duration_s = float(cfg["reset_pi_duration_ns"]) * NS_TO_S
            params = {
                "stage": "reset_conditional_pi",
                "sigma_s": max(reset_pi_duration_s / 6.0, 1e-18),
                "conditional": True,
                "cond_on": int(cfg["reset_cond_on"]),
                "rotation_rad": math.pi,
            }
            amp = _xy_rotation_amp_rad_s(
                shape="gaussian",
                duration_s=max(t4 - t3, 0.0) * NS_TO_S,
                params=params,
                rotation_rad=float(params["rotation_rad"]),
            )
            pulses.append(
                PlannedPulse(
                    channel=f"XY_{q}",
                    t0_ns=float(t3),
                    t1_ns=float(t4),
                    amp=amp,
                    shape="gaussian",
                    params=params,
                    carrier=_xy_carrier(cfg),
                )
            )
        events.append(
            {
                "qubit": int(q),
                "t0": float(t0),
                "t_meas_end": float(t1),
                "t_deplete_end": float(t2),
                "t_feedback_end": float(t3),
                "t1": float(t4),
                "conditional_on": int(cfg["reset_cond_on"]),
                "apply_feedback": bool(cfg["reset_apply_feedback"]),
                "feedback_offset_ns": float(max(0.0, float(reset_feedback_offset_ns))),
            }
        )
    return OperationPlan(duration_ns=float(t4 - t0), pulses=tuple(pulses), events=tuple(events))


def instantiate_operation_recipe(
    gate_name: str,
    qubits: list[int],
    *,
    gate_params: list[float] | None = None,
    start_ns: float,
    hw: dict[str, Any] | None = None,
    tc_index: int | None = None,
    tc_channel: str | None = None,
    reset_feedback_offset_ns: float = 0.0,
) -> tuple[list[tuple[str, PulseSpec]], float, list[dict[str, Any]]]:
    """Instantiate one operation into scheduled pulses and events."""
    cfg = resolve_lowering_hardware(hw)
    gate = str(gate_name).lower()
    typed_channel = _channel_name_for_gate(gate, qubits, tc_index, tc_channel)
    typed_recipe = resolve_typed_gate_recipe(hw, gate, channel_name=typed_channel)
    plan = _plan_typed_recipe(
        typed_recipe if typed_recipe is not None else _fallback_recipe_for_gate(
            gate,
            cfg=cfg,
            hw=hw,
            tc_index=tc_index,
            tc_channel=tc_channel,
        ),
        qubits=qubits,
        gate_params=gate_params,
        start_ns=start_ns,
        cfg=cfg,
        tc_channel=tc_channel,
    ) if (typed_recipe is not None or gate not in {"cx", "reset", "barrier"}) else None
    if plan is None and gate == "cx":
        plan = _plan_default_cx_recipe(
            qubits=qubits,
            start_ns=start_ns,
            cfg=cfg,
            hw=hw,
            tc_index=tc_index,
            tc_channel=tc_channel,
        )
    elif plan is None and gate == "reset":
        plan = _plan_default_reset_recipe(
            qubits=qubits,
            start_ns=start_ns,
            cfg=cfg,
            reset_feedback_offset_ns=reset_feedback_offset_ns,
        )
    elif plan is None and gate == "barrier":
        plan = OperationPlan(duration_ns=0.0)
    elif plan is None:
        supported = sorted([*_DRIVEN_SINGLE_QUBIT_SPECS.keys(), "barrier", "cz", "cx", "id", "measure", "reset", "rz", "z"])
        raise ValueError(
            f"Unsupported gate for pulse lowering: {gate}. "
            f"Supported gates: {', '.join(supported)}"
        )

    return _materialize_operation_plan(plan), float(plan.duration_ns), [dict(item) for item in plan.events]
