"""Gate scheduling policies for pulse lowering."""

from __future__ import annotations

from typing import Any

from musiq.common.schemas import CircuitGate, CircuitIR


def _gate_family(name: str) -> str:
    gate = str(name).lower()
    if gate in {"x", "sx", "h", "rx", "ry", "z", "rz"}:
        return "single_qubit"
    if gate in {"cz", "cx"}:
        return "two_qubit"
    if gate == "measure":
        return "measure"
    if gate == "reset":
        return "reset"
    if gate == "barrier":
        return "barrier"
    return "other"


def _typed_gate_recipe_duration_ns(gate: CircuitGate, hw: dict[str, Any]) -> float | None:
    gates = dict(hw.get("gates", {}) or {})
    if not gates:
        return None

    gate_name = str(gate.name).strip().lower()
    qubits = [int(q) for q in list(gate.qubits or [])]
    if gate_name in {"z", "rz"}:
        gate_aliases = ["virtual_z", gate_name]
    else:
        gate_aliases = [gate_name]

    channel_name: str | None = None
    if gate_name in {"x", "sx", "rx", "ry", "h"} and qubits:
        channel_name = f"XY_{qubits[0]}"
    elif gate_name == "measure" and qubits:
        channel_name = f"RO_{qubits[0]}"
    elif gate_name in {"cz", "cx"} and qubits:
        channel_name = _tc_channel_name(qubits)

    recipe: dict[str, Any] | None = None
    for candidate in gate_aliases:
        raw_recipe = gates.get(candidate)
        if isinstance(raw_recipe, dict):
            recipe = dict(raw_recipe)
            break
    if recipe is None:
        return None

    if channel_name:
        raw_overrides = dict(hw.get("channel_overrides", {}) or {})
        channel_overrides = raw_overrides.get(channel_name)
        if isinstance(channel_overrides, dict):
            for candidate in gate_aliases:
                override_recipe = channel_overrides.get(candidate)
                if isinstance(override_recipe, dict):
                    recipe = {**recipe, **dict(override_recipe)}
                    break

    recipe_type = str(recipe.get("recipe_type", gate_aliases[0] if gate_aliases else gate_name)).strip().lower()
    if recipe_type == "virtual_z":
        return 0.0
    if recipe_type == "measure":
        segments = list(recipe.get("segments", []) or [])
        if segments:
            return float(sum(float(seg.get("duration_ns", 0.0) or 0.0) for seg in segments))
    if "duration_ns" in recipe:
        return float(recipe.get("duration_ns", 0.0) or 0.0)
    return None


def _gate_duration_ns(gate: CircuitGate, hw: dict[str, Any]) -> float:
    name = str(gate.name).lower()
    typed_duration = _typed_gate_recipe_duration_ns(gate, hw)
    if typed_duration is not None:
        return float(typed_duration)
    gate_dur = float(hw.get("single_qubit_gate_duration_ns", hw["gate_duration_ns"]))
    two_qubit_dur = float(hw.get("double_qubit_gate_duration_ns", 2.0 * gate_dur))
    idle_dur = float(hw.get("idle_duration_ns", gate_dur))
    if name in {"z", "rz"}:
        return 0.0
    if name == "id":
        return idle_dur
    if name in {"x", "sx", "h", "rx", "ry"}:
        return gate_dur
    if name in {"cz", "cx"}:
        return two_qubit_dur
    if name == "measure":
        return float(hw["measure_duration_ns"])
    if name == "reset":
        return (
            float(hw["reset_measure_duration_ns"])
            + float(hw["reset_deplete_duration_ns"])
            + float(hw["reset_latency_duration_ns"])
            + (float(hw["reset_pi_duration_ns"]) if bool(hw["reset_apply_feedback"]) else 0.0)
        )
    if name == "barrier":
        return 0.0
    return gate_dur


def _reset_prefix_duration_ns(hw: dict[str, Any]) -> float:
    return (
        float(hw["reset_measure_duration_ns"])
        + float(hw["reset_deplete_duration_ns"])
        + float(hw["reset_latency_duration_ns"])
    )


def _reset_feedback_duration_ns(hw: dict[str, Any]) -> float:
    return float(hw["reset_pi_duration_ns"]) if bool(hw["reset_apply_feedback"]) else 0.0


def _pair_key(qubits: list[int]) -> tuple[int, int]:
    qs = qubits or [0, 1]
    return int(min(qs)), int(max(qs))


def _tc_channel_name(qubits: list[int]) -> str:
    i, j = _pair_key(qubits)
    return f"TC_q{i}_q{j}"


def _gate_resources(gate: CircuitGate, *, tc_channel: str | None = None) -> set[str]:
    name = str(gate.name).lower()
    qs = [int(q) for q in (gate.qubits or [])]
    resources = {f"Q{q}" for q in qs}
    if name in {"measure", "reset"}:
        resources.update(f"RO{q}" for q in qs)
    if name in {"cz", "cx"}:
        resources.add(str(tc_channel or _tc_channel_name(qs)))
    return resources


def _copy_gate(gate: CircuitGate) -> CircuitGate:
    return CircuitGate(
        name=str(gate.name),
        qubits=[int(q) for q in list(gate.qubits or [])],
        params=[float(p) for p in list(gate.params or [])],
        clbits=[int(c) for c in list(gate.clbits or [])],
    )


def _layer_logical_gates(lanes: list[list[CircuitGate]]) -> list[CircuitGate]:
    """Return unique logical gates from one schedule layer."""
    logical: list[CircuitGate] = []
    consumed: set[tuple[int, int]] = set()
    for lane_idx, lane in enumerate(list(lanes or [])):
        for gate_idx, gate in enumerate(list(lane or [])):
            pos = (int(lane_idx), int(gate_idx))
            if pos in consumed:
                continue
            copied = _copy_gate(gate)
            logical.append(copied)
            if len(copied.qubits) <= 1:
                consumed.add(pos)
                continue
            matched = [pos]
            for partner in copied.qubits:
                if int(partner) == int(lane_idx):
                    continue
                partner_lane = lanes[int(partner)] if 0 <= int(partner) < len(lanes) else []
                for partner_idx, partner_gate in enumerate(list(partner_lane or [])):
                    partner_pos = (int(partner), int(partner_idx))
                    if partner_pos in consumed:
                        continue
                    if (
                        str(partner_gate.name) == copied.name
                        and [int(q) for q in list(partner_gate.qubits or [])] == copied.qubits
                        and [float(p) for p in list(partner_gate.params or [])] == copied.params
                        and [int(c) for c in list(partner_gate.clbits or [])] == copied.clbits
                    ):
                        matched.append(partner_pos)
                        break
            consumed.update(matched)
    return logical


def _partner_lane_matches(
    lanes: list[list[CircuitGate]],
    lane_index: dict[int, int],
    gate: CircuitGate,
) -> bool:
    qubits = [int(q) for q in list(gate.qubits or [])]
    if len(qubits) != 2:
        return False
    for q in qubits:
        lane = list(lanes[q] or []) if 0 <= q < len(lanes) else []
        idx = int(lane_index.get(q, 0))
        if idx >= len(lane):
            return False
        partner = lane[idx]
        if (
            str(partner.name) != str(gate.name)
            or [int(x) for x in list(partner.qubits or [])] != qubits
            or [float(x) for x in list(partner.params or [])] != [float(x) for x in list(gate.params or [])]
            or [int(x) for x in list(partner.clbits or [])] != [int(x) for x in list(gate.clbits or [])]
        ):
            return False
    return True


def _segment_gates(gates: list[dict[str, Any]], policy: str) -> list[list[dict[str, Any]]]:
    if policy == "parallel":
        segments: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for item in gates:
            if item["family"] == "barrier":
                if current:
                    segments.append(current)
                    current = []
                segments.append([item])
                continue
            current.append(item)
        if current:
            segments.append(current)
        return segments

    segments = []
    current = []
    current_family: str | None = None
    for item in gates:
        family = item["family"]
        if family == "barrier":
            if current:
                segments.append(current)
                current = []
                current_family = None
            segments.append([item])
            continue
        if current and family != current_family:
            segments.append(current)
            current = []
        current.append(item)
        current_family = family
    if current:
        segments.append(current)
    return segments


def _schedule_reset_segment(
    segment: list[dict[str, Any]],
    *,
    segment_start: float,
    hw: dict[str, Any],
    layer_id: int,
) -> tuple[list[dict[str, Any]], float]:
    feedback_policy = str(hw.get("reset_feedback_policy", "parallel")).strip().lower() or "parallel"
    if feedback_policy not in {"parallel", "serial_global"}:
        raise ValueError(f"Unsupported reset_feedback_policy: {feedback_policy}")

    prefix = _reset_prefix_duration_ns(hw)
    feedback = _reset_feedback_duration_ns(hw)
    scheduled: list[dict[str, Any]] = []
    segment_end = segment_start
    for i, item in enumerate(segment):
        feedback_offset = (i * feedback) if (feedback_policy == "serial_global" and feedback > 0.0) else 0.0
        duration = prefix + feedback_offset + feedback
        start_ns = segment_start
        end_ns = start_ns + duration
        segment_end = max(segment_end, end_ns)
        scheduled.append(
            {
                **item,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": duration,
                "reset_feedback_offset_ns": feedback_offset,
                "layer_id": layer_id,
                "blocked_by_resources": [],
                "reset_feedback_mode": feedback_policy,
            }
        )
    return scheduled, segment_end


def build_gate_schedule(schedule_or_circuit: CircuitIR, hw: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a scheduled gate list according to the selected policy."""
    policy = str(hw.get("schedule_policy", "serial")).strip().lower() or "serial"
    if policy not in {"serial", "parallel", "hybrid"}:
        raise ValueError(f"Unsupported schedule_policy: {policy}")

    raw_gates: list[tuple[int, CircuitGate]] = []
    for tick, lanes in sorted(dict(schedule_or_circuit.schedule or {}).items(), key=lambda item: int(item[0])):
        for gate in _layer_logical_gates(list(lanes or [])):
            raw_gates.append((int(tick), gate))
    gates: list[dict[str, Any]] = []
    for idx, (tick, gate) in enumerate(raw_gates):
        tc_channel = _tc_channel_name(gate.qubits) if str(gate.name).lower() in {"cz", "cx"} else None
        gates.append(
            {
                "index": idx,
                "tick": int(tick),
                "gate": gate,
                "family": _gate_family(gate.name),
                "duration_ns": _gate_duration_ns(gate, hw),
                "tc_channel": tc_channel,
            }
        )

    if policy == "serial":
        scheduled: list[dict[str, Any]] = []
        cursor = 0.0
        n = len(gates)
        layer_id = 0
        i = 0
        while i < n:
            item = gates[i]
            family = item["family"]
            if family == "barrier":
                i += 1
                continue
            if family == "reset":
                group = [item]
                j = i + 1
                while j < n and gates[j]["family"] == "reset":
                    group.append(gates[j])
                    j += 1
                group_scheduled, group_end = _schedule_reset_segment(group, segment_start=cursor, hw=hw, layer_id=layer_id)
                scheduled.extend(group_scheduled)
                cursor = group_end
                layer_id += 1
                i = j
                continue
            duration = float(item["duration_ns"])
            scheduled.append(
                {
                    **item,
                    "start_ns": cursor,
                    "end_ns": cursor + duration,
                    "reset_feedback_offset_ns": 0.0,
                    "layer_id": layer_id,
                    "blocked_by_resources": [],
                    "reset_feedback_mode": None,
                }
            )
            if family == "measure":
                next_same = (i + 1 < n) and (gates[i + 1]["family"] == "measure")
                if not next_same:
                    cursor += duration
                    layer_id += 1
            else:
                cursor += duration
                layer_id += 1
            i += 1
        return scheduled

    scheduled: list[dict[str, Any]] = []
    tick_start = 0.0
    next_index = 0
    sorted_ticks = sorted(dict(schedule_or_circuit.schedule or {}).items(), key=lambda item: int(item[0]))
    for layer_id, (tick, raw_lanes) in enumerate(sorted_ticks):
        lanes = [[_copy_gate(gate) for gate in list(lane or [])] for lane in list(raw_lanes or [])]
        if not lanes:
            continue
        lane_cursor = {q: tick_start for q in range(len(lanes))}
        lane_index = {q: 0 for q in range(len(lanes))}
        resource_busy_until: dict[str, float] = {}

        while True:
            made_progress = False
            for q, lane in enumerate(lanes):
                gate_idx = int(lane_index[q])
                if gate_idx >= len(lane):
                    continue

                gate = lane[gate_idx]
                family = _gate_family(gate.name)
                if family == "barrier":
                    lane_index[q] += 1
                    made_progress = True
                    continue

                if family == "reset":
                    group = [
                        {
                            "index": next_index + offset,
                            "tick": int(tick),
                            "gate": _copy_gate(gate),
                            "family": family,
                            "duration_ns": _gate_duration_ns(gate, hw),
                            "tc_channel": None,
                        }
                    ]
                    group_scheduled, group_end = _schedule_reset_segment(group, segment_start=lane_cursor[q], hw=hw, layer_id=layer_id)
                    scheduled.extend(group_scheduled)
                    next_index += len(group)
                    lane_cursor[q] = group_end
                    lane_index[q] += 1
                    made_progress = True
                    continue

                if family == "two_qubit":
                    pair = _pair_key(gate.qubits)
                    primary_lane = int(pair[0])
                    if q != primary_lane:
                        continue
                    if not _partner_lane_matches(lanes, lane_index, gate):
                        continue
                    tc_channel = _tc_channel_name(gate.qubits)
                    resources = _gate_resources(gate, tc_channel=tc_channel)
                    qubits = [int(qubit) for qubit in list(gate.qubits or [])]
                    prior_resource_times = [resource_busy_until.get(r, tick_start) for r in resources]
                    start_ns = max(
                        [tick_start, *[lane_cursor[qubit] for qubit in qubits], *prior_resource_times]
                    )
                    duration = float(_gate_duration_ns(gate, hw))
                    end_ns = start_ns + duration
                    blocking = sorted([r for r in resources if resource_busy_until.get(r, tick_start) > max(lane_cursor[qubit] for qubit in qubits)])
                    for resource in resources:
                        resource_busy_until[resource] = end_ns
                    for qubit in qubits:
                        lane_cursor[qubit] = end_ns
                        lane_index[qubit] += 1
                    scheduled.append(
                        {
                            "index": next_index,
                            "tick": int(tick),
                            "gate": _copy_gate(gate),
                            "family": family,
                            "duration_ns": duration,
                            "tc_channel": tc_channel,
                            "start_ns": start_ns,
                            "end_ns": end_ns,
                            "reset_feedback_offset_ns": 0.0,
                            "layer_id": layer_id,
                            "blocked_by_resources": blocking,
                            "reset_feedback_mode": None,
                        }
                    )
                    next_index += 1
                    made_progress = True
                    continue

                resources = _gate_resources(gate, tc_channel=None)
                prior_resource_times = [resource_busy_until.get(r, tick_start) for r in resources]
                start_ns = max([tick_start, lane_cursor[q], *prior_resource_times])
                duration = float(_gate_duration_ns(gate, hw))
                end_ns = start_ns + duration
                blocking = sorted([r for r in resources if resource_busy_until.get(r, tick_start) > lane_cursor[q]])
                for resource in resources:
                    resource_busy_until[resource] = end_ns
                lane_cursor[q] = end_ns
                lane_index[q] += 1
                scheduled.append(
                    {
                        "index": next_index,
                        "tick": int(tick),
                        "gate": _copy_gate(gate),
                        "family": family,
                        "duration_ns": duration,
                        "tc_channel": None,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "reset_feedback_offset_ns": 0.0,
                        "layer_id": layer_id,
                        "blocked_by_resources": blocking,
                        "reset_feedback_mode": None,
                    }
                )
                next_index += 1
                made_progress = True

            if not made_progress:
                remaining = [
                    (q, lane_index[q], len(lane))
                    for q, lane in enumerate(lanes)
                    if int(lane_index[q]) < len(lane)
                ]
                if remaining:
                    raise ValueError(f"Unable to schedule tick {tick}: lane state stalled at {remaining}")
                break

        tick_start = max(list(lane_cursor.values()) or [tick_start])
    return scheduled
