"""Circuit-level intermediate representations.

``CircuitIR`` is schedule-first: the execution structure is carried by
``schedule`` and not by a flat logical gate list. Code that needs a flattened
view should call :func:`flatten_schedule` explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from musiq.schemas.utils import SCHEMA_VERSION


@dataclass
class CircuitGate:
    """One logical gate operation in circuit IR.

    Attributes:
        name: Name of the gate operation (e.g., "rx", "cx").
        qubits: List of target/control qubit indices.
        params: Numerical parameters for the gate (e.g., rotation angle).
        clbits: List of classical bit indices involved.
    """

    name: str
    qubits: list[int] = field(default_factory=list)
    params: list[float] = field(default_factory=list)
    clbits: list[int] = field(default_factory=list)


def _copy_gate(gate: CircuitGate | dict[str, Any]) -> CircuitGate:
    raw = gate if isinstance(gate, CircuitGate) else CircuitGate(**dict(gate))
    return CircuitGate(
        name=str(raw.name),
        qubits=[int(q) for q in list(raw.qubits or [])],
        params=[float(p) for p in list(raw.params or [])],
        clbits=[int(c) for c in list(raw.clbits or [])],
    )


def _infer_schedule_dimensions(
    schedule: dict[int, list[list[CircuitGate]]] | None,
) -> tuple[int, int]:
    """Infer minimal qubit/classical-bit counts from a scheduled circuit."""
    max_lane_count = 0
    max_qubit_index = -1
    max_clbit_index = -1
    for raw_lanes in dict(schedule or {}).values():
        lanes = list(raw_lanes or [])
        max_lane_count = max(max_lane_count, len(lanes))
        for lane in lanes:
            for gate in list(lane or []):
                copied = _copy_gate(gate)
                if copied.qubits:
                    max_qubit_index = max(max_qubit_index, max(copied.qubits))
                if copied.clbits:
                    max_clbit_index = max(max_clbit_index, max(copied.clbits))
    inferred_qubits = max(max_lane_count, max_qubit_index + 1 if max_qubit_index >= 0 else 0)
    inferred_clbits = max_clbit_index + 1 if max_clbit_index >= 0 else 0
    return inferred_qubits, inferred_clbits


def build_serial_schedule(
    gates: list[CircuitGate] | None,
    *,
    num_qubits: int,
) -> dict[int, list[list[CircuitGate]]]:
    """Build a schedule with one serial layer per logical gate."""
    schedule: dict[int, list[list[CircuitGate]]] = {}
    for tick, gate in enumerate(list(gates or [])):
        lane_count = max(int(num_qubits), max([*list(gate.qubits or []), -1]) + 1, 1)
        lanes: list[list[CircuitGate]] = [[] for _ in range(max(0, lane_count))]
        targets = [int(q) for q in list(gate.qubits or [])]
        if targets:
            for q in targets:
                lanes[int(q)].append(_copy_gate(gate))
        else:
            lanes[0].append(_copy_gate(gate))
        schedule[int(tick)] = lanes
    return schedule


def flatten_schedule(schedule: dict[int, list[list[CircuitGate]]] | None) -> list[CircuitGate]:
    """Return one logical gate stream per schedule layer.

    Multi-qubit gates are mirrored across participating qubit lanes in the
    schedule representation. This helper deduplicates those mirrored entries
    while preserving within-layer order as observed from lower qubit lanes first.
    """
    out: list[CircuitGate] = []
    for tick in sorted((schedule or {}).keys()):
        lanes = list((schedule or {}).get(tick, []) or [])
        consumed: set[tuple[int, int]] = set()
        for lane_idx, lane in enumerate(lanes):
            for gate_idx, gate in enumerate(list(lane or [])):
                pos = (int(lane_idx), int(gate_idx))
                if pos in consumed:
                    continue
                copied = _copy_gate(gate)
                out.append(copied)
                if len(copied.qubits) <= 1:
                    consumed.add(pos)
                    continue
                matched_positions = [pos]
                for partner in copied.qubits:
                    if int(partner) == int(lane_idx):
                        continue
                    partner_lane = lanes[int(partner)] if 0 <= int(partner) < len(lanes) else []
                    found: tuple[int, int] | None = None
                    for partner_idx, partner_gate in enumerate(list(partner_lane or [])):
                        partner_pos = (int(partner), int(partner_idx))
                        if partner_pos in consumed:
                            continue
                        partner_copy = _copy_gate(partner_gate)
                        if (
                            partner_copy.name == copied.name
                            and partner_copy.qubits == copied.qubits
                            and partner_copy.params == copied.params
                            and partner_copy.clbits == copied.clbits
                        ):
                            found = partner_pos
                            break
                    if found is not None:
                        matched_positions.append(found)
                consumed.update(matched_positions)
    return out


@dataclass
class CircuitIR:
    """Normalized circuit representation used by compile pipeline.

    Attributes:
        schema_version: Version of the circuit IR schema.
        format: Format of the circuit (e.g., "openqasm3"). Defaults to "openqasm3".
        num_qubits: Number of qubits in the circuit. Defaults to 0.
        num_clbits: Number of classical bits in the circuit. Defaults to 0.
        schedule: Parallel-aware circuit schedule keyed by integer layer.
        source_qasm: Original QASM source code string.
    """

    schema_version: str = SCHEMA_VERSION
    format: str = "openqasm3"
    num_qubits: int = 0
    num_clbits: int = 0
    schedule: dict[int, list[list[CircuitGate]]] = field(default_factory=dict)
    source_qasm: str = ""

    def __post_init__(self) -> None:
        inferred_qubits, inferred_clbits = _infer_schedule_dimensions(self.schedule)
        self.num_qubits = max(int(self.num_qubits or 0), int(inferred_qubits))
        self.num_clbits = max(int(self.num_clbits or 0), int(inferred_clbits))
        normalized: dict[int, list[list[CircuitGate]]] = {}
        for raw_tick, raw_lanes in sorted(dict(self.schedule or {}).items(), key=lambda item: int(item[0])):
            tick = int(raw_tick)
            lanes = list(raw_lanes or [])
            copied_lanes: list[list[CircuitGate]] = []
            for lane in lanes:
                copied_lanes.append([_copy_gate(gate) for gate in list(lane or [])])
            while len(copied_lanes) < int(self.num_qubits):
                copied_lanes.append([])
            normalized[tick] = copied_lanes
        self.schedule = normalized


@dataclass
class CircuitSpec:
    """Circuit snapshot kept with ``ModelSpec`` for engines that need gate context.

    Attributes:
        schema_version: Version of the circuit spec schema.
        format: Format of the circuit (e.g., "openqasm3"). Defaults to "openqasm3".
        num_qubits: Number of qubits in the circuit. Defaults to 0.
        num_clbits: Number of classical bits in the circuit. Defaults to 0.
        schedule: Parallel-aware circuit schedule keyed by integer layer.
        source_qasm: Original QASM source code string.
        stage: Compilation stage of the snapshot (e.g., "normalized"). Defaults to "normalized".
    """

    schema_version: str = SCHEMA_VERSION
    format: str = "openqasm3"
    num_qubits: int = 0
    num_clbits: int = 0
    schedule: dict[int, list[list[CircuitGate]]] = field(default_factory=dict)
    source_qasm: str = ""
    stage: str = "normalized"

    @classmethod
    def from_circuit_ir(cls, circuit: CircuitIR, *, stage: str = "normalized") -> "CircuitSpec":
        """Create a ``CircuitSpec`` snapshot from normalized ``CircuitIR``.

        Args:
            circuit (CircuitIR): The source circuit intermediate representation.
            stage (str): The compilation stage name. Defaults to "normalized".

        Returns:
            CircuitSpec: A snapshot of the circuit for model specification.
        """
        return cls(
            schema_version=str(circuit.schema_version),
            format=str(circuit.format),
            num_qubits=int(circuit.num_qubits),
            num_clbits=int(circuit.num_clbits),
            schedule={
                int(tick): [[_copy_gate(gate) for gate in list(lane or [])] for lane in list(lanes or [])]
                for tick, lanes in sorted(dict(circuit.schedule or {}).items(), key=lambda item: int(item[0]))
            },
            source_qasm=str(circuit.source_qasm or ""),
            stage=str(stage),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CircuitSpec":
        """Create a ``CircuitSpec`` from a JSON-style mapping.

        Args:
            data (dict[str, Any] | None): Input dictionary containing circuit fields.

        Returns:
            CircuitSpec: A typed circuit specification.
        """
        raw = dict(data or {})
        return cls(
            schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
            format=str(raw.get("format", "openqasm3")),
            num_qubits=int(raw.get("num_qubits", 0) or 0),
            num_clbits=int(raw.get("num_clbits", 0) or 0),
            schedule={
                int(tick): [[_copy_gate(gate) for gate in list(lane or [])] for lane in list(lanes or [])]
                for tick, lanes in dict(raw.get("schedule", {}) or {}).items()
            },
            source_qasm=str(raw.get("source_qasm", "") or ""),
            stage=str(raw.get("stage", "normalized") or "normalized"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the circuit snapshot to a JSON-safe dictionary.

        Returns:
            dict[str, Any]: A JSON-serializable representation of the circuit spec.
        """
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "schedule": {
                int(tick): [[asdict(_copy_gate(gate)) for gate in list(lane or [])] for lane in list(lanes or [])]
                for tick, lanes in sorted(dict(self.schedule or {}).items(), key=lambda item: int(item[0]))
            },
            "source_qasm": self.source_qasm,
            "stage": self.stage,
        }


