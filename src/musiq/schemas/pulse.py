"""Pulse and executable-model intermediate representations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from musiq.schemas.utils import SCHEMA_VERSION


@dataclass
class Carrier:
    """Carrier tone parameters for pulse modulation."""

    freq: float
    phase: float = 0.0


@dataclass
class PulseSpec:
    """Single pulse segment scheduled on a channel."""

    t0_s: float
    t1_s: float
    amp: float
    shape: str
    params: dict[str, Any] = field(default_factory=dict)
    carrier: Carrier | None = None

    @property
    def duration_s(self) -> float:
        """Pulse duration in seconds."""
        return float(self.t1_s) - float(self.t0_s)

    @property
    def t0_ns(self) -> float:
        """Pulse start time in nanoseconds."""
        return round(float(self.t0_s) * 1e9, 12)

    @property
    def t1_ns(self) -> float:
        """Pulse end time in nanoseconds."""
        return round(float(self.t1_s) * 1e9, 12)

    @property
    def duration_ns(self) -> float:
        """Pulse duration in nanoseconds."""
        return round(self.duration_s * 1e9, 12)


@dataclass
class ChannelSpec:
    """Collection of pulses for one hardware channel."""

    name: str
    pulses: list[PulseSpec] = field(default_factory=list)


@dataclass
class PulseIR:
    """Pulse-level intermediate representation for one schedule."""

    schema_version: str = SCHEMA_VERSION
    t_end_s: float = 0.0
    channels: list[ChannelSpec] = field(default_factory=list)

    @property
    def t_end_ns(self) -> float:
        """Schedule end time in nanoseconds."""
        return round(float(self.t_end_s) * 1e9, 12)


@dataclass
class ExecutableModel:
    """Lowered executable model before numeric model construction."""

    schema_version: str = SCHEMA_VERSION
    level: str = "qubit"
    solver: str = "se"
    h_terms: list[dict[str, Any]] = field(default_factory=list)
    noise_terms: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateRecipe:
    """Base typed lowering recipe shared by all logical gate families."""

    logical_gate: str
    recipe_type: str
    duration_ns: float


@dataclass(frozen=True, slots=True)
class DrivenSingleQubitRecipe(GateRecipe):
    """Family-level typed lowering recipe for driven single-qubit gates."""

    amplitude_Hz: float = 0.0
    shape: str | None = None
    sigma_fraction: float | None = None
    drag_beta: float | None = None
    edge_ns: float | None = None
    rect_edge_ns: float | None = None
    carrier_freq_Hz: float | None = None
    phase_rad: float | None = None
    rotation_axis: str = "x"
    fixed_rotation_rad: float | None = None
    parametric_rotation: bool = False

    def rotation_rad(self, gate_params: list[float] | None = None) -> float:
        """Resolve the target logical rotation for one gate instance."""
        if self.parametric_rotation:
            return float(list(gate_params or [0.0])[0])
        if self.fixed_rotation_rad is None:
            raise ValueError(f"Driven recipe `{self.logical_gate}` is missing a fixed rotation.")
        return float(self.fixed_rotation_rad)

    def resolved_phase_rad(self) -> float:
        """Return the carrier phase used when the recipe omits an explicit override."""
        if self.phase_rad is not None:
            return float(self.phase_rad)
        return 0.5 * math.pi if self.logical_gate in {"ry", "h"} else 0.0


@dataclass(frozen=True, slots=True)
class TwoQubitGateRecipe(GateRecipe):
    """Base typed lowering recipe for two-qubit entangling gates."""


@dataclass(frozen=True, slots=True)
class CouplerTwoQubitRecipe(TwoQubitGateRecipe):
    """Family-level typed lowering recipe for coupler-driven two-qubit gates."""

    amplitude_Hz: float
    shape: str | None = None
    edge_ns: float | None = None
    rect_edge_ns: float | None = None
    target_conditional_phase_rad: float | None = None


@dataclass(frozen=True, slots=True)
class VirtualPhaseGateRecipe(GateRecipe):
    """Family-level typed lowering recipe for zero-duration virtual phase updates."""

    duration_ns: float = 0.0
    phase_rad: float | None = None


@dataclass(frozen=True, slots=True)
class IdleGateRecipe(GateRecipe):
    """Typed lowering recipe for an idle window."""

    logical_gate: str = field(default="id", init=False)
    recipe_type: str = field(default="id", init=False)
    duration_ns: float = 0.0


@dataclass(frozen=True, slots=True)
class MeasureSegmentRecipe:
    """Typed readout segment for a measure recipe."""

    duration_ns: float
    amplitude: float
    shape: str = "readout"
    rise_ns: float = 0.0
    fall_ns: float = 0.0


@dataclass(frozen=True, slots=True)
class MeasureRecipe(GateRecipe):
    """Family-level typed lowering recipe for readout pulses."""

    logical_gate: str = field(default="measure", init=False)
    recipe_type: str = field(default="measure", init=False)
    carrier_freq_Hz: float | None = None
    phase_rad: float | None = None
    amplitude: float | None = None
    shape: str | None = None
    rise_ns: float | None = None
    fall_ns: float | None = None
    edge_ns: float | None = None
    segments: tuple[MeasureSegmentRecipe, ...] = ()


