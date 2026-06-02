"""Canonical unit-bearing field names for hardware and noise configs."""

from __future__ import annotations

from typing import Any

PULSE_KEYS = {
    "defaults",
    "gates",
    "channel_overrides",
    "gate_duration_ns",
    "single_qubit_gate_duration_ns",
    "double_qubit_gate_duration_ns",
    "idle_duration_ns",
    "measure_duration_ns",
    "measure_amp",
    "measure_segments",
    "measure_start_delay_ns",
    "rect_edge_ns",
    "readout_edge_ns",
    "single_qubit_shape",
    "single_qubit_sigma_fraction",
    "single_qubit_drag_beta",
    "single_qubit_rect_edge_ns",
    "reset_measure_duration_ns",
    "reset_deplete_duration_ns",
    "reset_latency_duration_ns",
    "reset_pi_duration_ns",
    "reset_measure_amp",
    "reset_deplete_amp",
    "reset_pi_amp",
    "reset_cond_on",
    "reset_apply_feedback",
    "xy_freq_Hz",
    "ro_freq_Hz",
}

LOWERING_HARDWARE_KEYS = PULSE_KEYS | {
    "schedule_policy",
    "schedule",
    "reset_feedback_policy",
}

MODEL_HARDWARE_KEYS = LOWERING_HARDWARE_KEYS | {
    "acquisition",
    "qubits",
    "simulation_level",
    "dimension",
    "control_scale",
    "transmon_levels",
    "cavity_nmax",
    "qubit_freq_Hz",
    "qubit_freqs_Hz",
    "anharmonicity_Hz",
    "cavity_freq_Hz",
    "g_cavity_Hz",
    "couplings",
    "components",
    "connections",
    "parameters",
    "shared_noise",
    "control_crosstalk",
    "readout_crosstalk",
}

COUPLING_KEYS = {"i", "j", "g_Hz", "kind"}

NOISE_KEYS = {
    "model",
    "type",
    "one_over_f",
    "readout_error",
    "one_over_f_amp_Hz",
    "one_over_f_fmin_Hz",
    "one_over_f_fmax_Hz",
    "one_over_f_exponent",
    "ou_sigma_Hz",
    "ou_tau_s",
    "sources",
    "enabled_sources",
    "disabled_sources",
    "overrides",
}

NS_TO_S = 1e-9


def reject_unknown_keys(section: str, payload: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unsupported keys in {section}: {unknown}")


def reject_unknown_coupling_keys(couplings: list[Any]) -> None:
    for idx, coupling in enumerate(couplings):
        if not isinstance(coupling, dict):
            continue
        unknown = sorted(set(coupling) - COUPLING_KEYS)
        if unknown:
            raise ValueError(f"Unsupported keys in device.couplings[{idx}]: {unknown}")
