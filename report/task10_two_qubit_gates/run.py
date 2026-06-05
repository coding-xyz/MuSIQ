from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
SRC_ROOT = ROOT.parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from musiq.analysis.common.state_utils import basis_labels, complex_scalar, final_density_matrix, population_series, state_fidelity
from musiq.workflow import create_model


def _first_run_and_spec(model):
    run_obj = next(iter(model.runs.values()))
    result = next(iter(run_obj.results.values()))
    trajectory = next(iter(result.trajectories.values()))
    model_spec = run_obj.artifacts.model_spec
    return trajectory, model_spec


def _build_model(circuit_name: str):
    return create_model(
        circuits=ROOT / "circuits" / f"{circuit_name}.yaml",
        solvers=ROOT / "solver.yaml",
        devices=ROOT / "device.yaml",
        pulses=ROOT / "pulse.yaml",
        analysers=ROOT / "analyser.yaml",
    )


def _set_active_scope(model, *, components: list[str], connections: list[str]) -> None:
    solver_cfg = next(iter(model.config.solvers.values()))
    for step in list(solver_cfg.study or []):
        if not isinstance(step, dict):
            continue
        step["active_components"] = list(components)
        step["active_connections"] = list(connections)
        step["representations"] = {
            key: value for key, value in dict(step.get("representations", {}) or {}).items() if key in components
        }
        step["bases"] = {key: value for key, value in dict(step.get("bases", {}) or {}).items() if key in components}


def _set_gate_amplitude(model, gate_name: str, amplitude_hz: float) -> None:
    model.config.pulses["default"].extras["gates"][gate_name]["amplitude_Hz"] = float(amplitude_hz)


def _set_gate_duration(model, gate_name: str, duration_ns: float) -> None:
    model.config.pulses["default"].extras["gates"][gate_name]["duration_ns"] = float(duration_ns)


def _apply_calibrated_single_qubit_pulses(model, *, sx_amp_hz: float, x_amp_hz: float) -> None:
    _set_gate_amplitude(model, "sx", sx_amp_hz)
    _set_gate_amplitude(model, "x", x_amp_hz)


def _run_model(model):
    model.run_all()
    trajectory, model_spec = _first_run_and_spec(model)
    times_ns = np.asarray(list(trajectory.times or []), dtype=float) * 1.0e9
    series = population_series(trajectory, model_spec)
    return trajectory, model_spec, times_ns, series


def _run_case(circuit_name: str):
    return _run_model(_build_model(circuit_name))


def _bell_fidelity(trajectory, model_spec) -> float:
    rho = None
    try:
        rho = final_density_matrix(trajectory)
    except Exception:
        rho = None
    if rho is None or len(np.asarray(rho).shape) != 2:
        snapshots = list(getattr(trajectory, "wave_function", []) or [])
        if not snapshots:
            return 0.0
        psi = np.asarray([complex_scalar(value) for value in list(snapshots[-1])], dtype=complex)
        labels = basis_labels(psi.shape[0], int(model_spec.system.num_qubits or 0), int(model_spec.system.transmon_levels or 3))
        idx_000 = labels.index("000")
        idx_011 = labels.index("011")
        target = np.zeros(psi.shape[0], dtype=complex)
        target[idx_000] = 1.0 / np.sqrt(2.0)
        target[idx_011] = 1.0 / np.sqrt(2.0)
        return float(abs(np.vdot(target, psi)) ** 2)
    labels = basis_labels(rho.shape[0], int(model_spec.system.num_qubits or 0), int(model_spec.system.transmon_levels or 3))
    idx_000 = labels.index("000")
    idx_011 = labels.index("011")
    target = np.zeros(rho.shape[0], dtype=complex)
    target[idx_000] = 1.0 / np.sqrt(2.0)
    target[idx_011] = 1.0 / np.sqrt(2.0)
    return float(state_fidelity(rho, target))


def _single_qutrit_terminal_pop(series: dict[str, list[float]]) -> dict[str, float]:
    return {label: float((values or [0.0])[-1]) for label, values in series.items()}


def _single_qubit_loss_sx(single_pop: dict[str, float], twice_pop: dict[str, float]) -> float:
    return (
        (single_pop.get("0", 0.0) - 0.5) ** 2
        + (single_pop.get("1", 0.0) - 0.5) ** 2
        + 4.0 * single_pop.get("2", 0.0) ** 2
        + (twice_pop.get("1", 0.0) - 1.0) ** 2
        + twice_pop.get("0", 0.0) ** 2
        + 2.0 * twice_pop.get("2", 0.0) ** 2
    )


def _single_qubit_loss_x(pop: dict[str, float]) -> float:
    return (pop.get("1", 0.0) - 1.0) ** 2 + pop.get("0", 0.0) ** 2 + 4.0 * pop.get("2", 0.0) ** 2


def _calibrate_sx_q0() -> dict[str, float | list[float]]:
    amplitudes = np.linspace(7.0e6, 15.0e6, 33)
    loss_values: list[float] = []
    final_p0: list[float] = []
    final_p1: list[float] = []
    final_p2: list[float] = []
    twice_p1: list[float] = []

    for amplitude_hz in amplitudes:
        single_model = _build_model("sx_q0")
        _set_active_scope(single_model, components=["q0"], connections=[])
        _set_gate_amplitude(single_model, "sx", float(amplitude_hz))
        _, _, _, single_series = _run_model(single_model)
        single_pop = _single_qutrit_terminal_pop(single_series)

        twice_model = _build_model("sx_q0_twice")
        _set_active_scope(twice_model, components=["q0"], connections=[])
        _set_gate_amplitude(twice_model, "sx", float(amplitude_hz))
        _, _, _, twice_series = _run_model(twice_model)
        twice_pop = _single_qutrit_terminal_pop(twice_series)

        final_p0.append(single_pop.get("0", 0.0))
        final_p1.append(single_pop.get("1", 0.0))
        final_p2.append(single_pop.get("2", 0.0))
        twice_p1.append(twice_pop.get("1", 0.0))
        loss_values.append(_single_qubit_loss_sx(single_pop, twice_pop))

    best_idx = int(np.argmin(loss_values))
    best_amplitude = float(amplitudes[best_idx])

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(amplitudes * 1.0e-6, final_p0, label="single sx: P0")
    ax.plot(amplitudes * 1.0e-6, final_p1, label="single sx: P1")
    ax.plot(amplitudes * 1.0e-6, final_p2, label="single sx: P2")
    ax.plot(amplitudes * 1.0e-6, twice_p1, label="sx then sx: P1")
    ax.axvline(best_amplitude * 1.0e-6, color="black", linestyle="--", linewidth=1.0, label="best amplitude")
    ax.set_title("Q0 SX Calibration")
    ax.set_xlabel("SX amplitude (MHz)")
    ax.set_ylabel("final population")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sx_calibration_q0.png", dpi=160)
    plt.close(fig)

    return {
        "amplitude_Hz": best_amplitude,
        "loss": float(loss_values[best_idx]),
        "single_sx_final_P0": float(final_p0[best_idx]),
        "single_sx_final_P1": float(final_p1[best_idx]),
        "single_sx_final_P2": float(final_p2[best_idx]),
        "double_sx_final_P1": float(twice_p1[best_idx]),
    }


def _calibrate_x_q0() -> dict[str, float | list[float]]:
    amplitudes = np.linspace(15.0e6, 30.0e6, 33)
    loss_values: list[float] = []
    final_p0: list[float] = []
    final_p1: list[float] = []
    final_p2: list[float] = []

    for amplitude_hz in amplitudes:
        model = _build_model("x_q0")
        _set_active_scope(model, components=["q0"], connections=[])
        _set_gate_amplitude(model, "x", float(amplitude_hz))
        _, _, _, series = _run_model(model)
        pop = _single_qutrit_terminal_pop(series)
        final_p0.append(pop.get("0", 0.0))
        final_p1.append(pop.get("1", 0.0))
        final_p2.append(pop.get("2", 0.0))
        loss_values.append(_single_qubit_loss_x(pop))

    best_idx = int(np.argmin(loss_values))
    best_amplitude = float(amplitudes[best_idx])

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(amplitudes * 1.0e-6, final_p0, label="single x: P0")
    ax.plot(amplitudes * 1.0e-6, final_p1, label="single x: P1")
    ax.plot(amplitudes * 1.0e-6, final_p2, label="single x: P2")
    ax.axvline(best_amplitude * 1.0e-6, color="black", linestyle="--", linewidth=1.0, label="best amplitude")
    ax.set_title("Q0 X Calibration")
    ax.set_xlabel("X amplitude (MHz)")
    ax.set_ylabel("final population")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "x_calibration_q0.png", dpi=160)
    plt.close(fig)

    return {
        "amplitude_Hz": best_amplitude,
        "loss": float(loss_values[best_idx]),
        "single_x_final_P0": float(final_p0[best_idx]),
        "single_x_final_P1": float(final_p1[best_idx]),
        "single_x_final_P2": float(final_p2[best_idx]),
    }


def _logical_population_map(series: dict[str, list[float]]) -> dict[str, float]:
    def final(label: str) -> float:
        values = list(series.get(label, []) or [])
        return float(values[-1]) if values else 0.0

    coupler_excited = 0.0
    for label, values in series.items():
        if len(label) >= 3 and label[0] != "0":
            coupler_excited += float((values or [0.0])[-1])
    return {
        "P00": final("000"),
        "P10": final("001"),
        "P01": final("010"),
        "P11": final("011"),
        "Pcoupler_excited": coupler_excited,
    }


def _scan_cz_gate_time(*, sx_amp_hz: float, x_amp_hz: float) -> dict[str, float | list[float]]:
    durations_ns = np.linspace(24.0, 144.0, 25)
    final_p00: list[float] = []
    final_p10: list[float] = []
    final_p01: list[float] = []
    final_p11: list[float] = []
    final_pc: list[float] = []
    bell_fidelities: list[float] = []

    for duration_ns in durations_ns:
        model = _build_model("cz")
        _apply_calibrated_single_qubit_pulses(model, sx_amp_hz=sx_amp_hz, x_amp_hz=x_amp_hz)
        _set_gate_duration(model, "cz", float(duration_ns))
        trajectory, model_spec, _, series = _run_model(model)
        logical_pop = _logical_population_map(series)
        final_p00.append(logical_pop["P00"])
        final_p10.append(logical_pop["P10"])
        final_p01.append(logical_pop["P01"])
        final_p11.append(logical_pop["P11"])
        final_pc.append(logical_pop["Pcoupler_excited"])
        bell_fidelities.append(_bell_fidelity(trajectory, model_spec))

    best_idx = int(np.argmax(bell_fidelities))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(durations_ns, final_p00, label="final P00")
    ax.plot(durations_ns, final_p10, label="final P10")
    ax.plot(durations_ns, final_p01, label="final P01")
    ax.plot(durations_ns, final_p11, label="final P11")
    ax.plot(durations_ns, final_pc, label="final coupler excited")
    ax.set_title("CZ Final Population vs Gate Time")
    ax.set_xlabel("CZ gate time (ns)")
    ax.set_ylabel("final population")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    ax2 = ax.twinx()
    ax2.plot(durations_ns, bell_fidelities, color="black", linestyle="--", label="Bell fidelity")
    ax2.set_ylabel("Bell fidelity")
    ax2.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "cz_final_population_vs_gate_time.png", dpi=160)
    plt.close(fig)

    return {
        "best_duration_ns": float(durations_ns[best_idx]),
        "best_bell_fidelity": float(bell_fidelities[best_idx]),
        "best_final_P00": float(final_p00[best_idx]),
        "best_final_P11": float(final_p11[best_idx]),
    }


def _scan_iswap_gate_time(*, sx_amp_hz: float, x_amp_hz: float) -> dict[str, float | list[float]]:
    durations_ns = np.linspace(24.0, 168.0, 25)
    final_p00: list[float] = []
    final_p10: list[float] = []
    final_p01: list[float] = []
    final_p11: list[float] = []
    final_pc: list[float] = []

    for duration_ns in durations_ns:
        model = _build_model("iswap")
        _apply_calibrated_single_qubit_pulses(model, sx_amp_hz=sx_amp_hz, x_amp_hz=x_amp_hz)
        _set_gate_duration(model, "iswap", float(duration_ns))
        _, _, _, series = _run_model(model)
        logical_pop = _logical_population_map(series)
        final_p00.append(logical_pop["P00"])
        final_p10.append(logical_pop["P10"])
        final_p01.append(logical_pop["P01"])
        final_p11.append(logical_pop["P11"])
        final_pc.append(logical_pop["Pcoupler_excited"])

    best_idx = int(np.argmax(final_p01))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(durations_ns, final_p00, label="final P00")
    ax.plot(durations_ns, final_p10, label="final P10")
    ax.plot(durations_ns, final_p01, label="final P01")
    ax.plot(durations_ns, final_p11, label="final P11")
    ax.plot(durations_ns, final_pc, label="final coupler excited")
    ax.set_title("iSWAP Final Population vs Gate Time")
    ax.set_xlabel("iSWAP gate time (ns)")
    ax.set_ylabel("final population")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "iswap_final_population_vs_gate_time.png", dpi=160)
    plt.close(fig)

    return {
        "best_duration_ns": float(durations_ns[best_idx]),
        "best_final_P01": float(final_p01[best_idx]),
        "best_final_P10": float(final_p10[best_idx]),
        "best_final_coupler_excited": float(final_pc[best_idx]),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sx_calibration = _calibrate_sx_q0()
    x_calibration = _calibrate_x_q0()

    summary = {
        "single_qubit_calibration": {
            "sx_q0": sx_calibration,
            "x_q0": x_calibration,
        },
        "cz_gate_time_scan": _scan_cz_gate_time(
            sx_amp_hz=float(sx_calibration["amplitude_Hz"]),
            x_amp_hz=float(x_calibration["amplitude_Hz"]),
        ),
        "iswap_gate_time_scan": _scan_iswap_gate_time(
            sx_amp_hz=float(sx_calibration["amplitude_Hz"]),
            x_amp_hz=float(x_calibration["amplitude_Hz"]),
        ),
    }
    (FIG_DIR / "task10_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Single-qubit calibration:", json.dumps(summary["single_qubit_calibration"], indent=2))
    print("CZ gate-time scan:", json.dumps(summary["cz_gate_time_scan"], indent=2))
    print("iSWAP gate-time scan:", json.dumps(summary["iswap_gate_time_scan"], indent=2))


if __name__ == "__main__":
    main()
