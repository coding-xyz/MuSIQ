"""Readout-chain postprocessing for cqed task flows."""

from __future__ import annotations

import math
from typing import Any
from dataclasses import asdict

import numpy as np
from musiq.schemas.results import IQAnalysis, ReadoutAnalysis, ShotData

from musiq.backend.config import normalize_device_config
from musiq.backend.model.lowering import (
    infer_classical_readout_chain,
    readout_coupling_prefactor,
    readout_topology_input,
)
from musiq.common.channels import safe_float
from musiq.pulse.shapes import make_shape


def _complex_pairs(values: np.ndarray) -> list[list[float]]:
    arr = np.asarray(values, dtype=complex).reshape(-1)
    return [[float(v.real), float(v.imag)] for v in arr]


def _complex_from_pairs(values: list[list[float]] | list[float] | None) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=complex)
    if isinstance(values[0], complex):
        return np.asarray(values, dtype=complex).reshape(-1)
    if isinstance(values[0], dict) and "__musiq_complex__" in values[0]:
        return np.asarray(
            [
                complex(float(item["__musiq_complex__"][0]), float(item["__musiq_complex__"][1]))
                for item in values
                if isinstance(item, dict) and "__musiq_complex__" in item
            ],
            dtype=complex,
        ).reshape(-1)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.astype(complex)
    if arr.shape[-1] < 2:
        return arr.reshape(-1).astype(complex)
    return arr[..., 0].reshape(-1) + 1j * arr[..., 1].reshape(-1)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return safe_float(value, default)


def _readout_coupling_prefactor(kappa_ext_hz: float) -> float:
    return readout_coupling_prefactor(kappa_ext_hz)


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _real_list(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return [float(x) for x in arr]


def _infer_channel_carrier(pulse_ir, channel_name: str) -> tuple[float, float]:
    target = str(channel_name or "").strip()
    if not target:
        return 0.0, 0.0
    for channel in list(getattr(pulse_ir, "channels", []) or []):
        if str(getattr(channel, "name", "")).strip() != target:
            continue
        for pulse in list(getattr(channel, "pulses", []) or []):
            carrier = getattr(pulse, "carrier", None)
            if carrier is None:
                continue
            return (
                _safe_float(getattr(carrier, "freq", 0.0), 0.0),
                _safe_float(getattr(carrier, "phase", 0.0), 0.0),
            )
    return 0.0, 0.0


def _resample_complex(times: np.ndarray, values: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    if sample_times.size <= 0:
        return np.asarray([], dtype=complex)
    src_t = np.asarray(times, dtype=float).reshape(-1)
    src_v = np.asarray(values, dtype=complex).reshape(-1)
    dst_t = np.asarray(sample_times, dtype=float).reshape(-1)
    if src_t.size <= 0 or src_v.size <= 0:
        return np.zeros(dst_t.size, dtype=complex)
    if src_t.size == dst_t.size and np.allclose(src_t, dst_t):
        return src_v.astype(complex)
    re = np.interp(dst_t, src_t, np.real(src_v), left=0.0, right=0.0)
    im = np.interp(dst_t, src_t, np.imag(src_v), left=0.0, right=0.0)
    return re.astype(complex) + 1j * im.astype(complex)


def _build_sample_times(times: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    src_t = np.asarray(times, dtype=float).reshape(-1)
    if src_t.size <= 1 or sample_rate_hz <= 0.0:
        return src_t
    dt = 1.0 / float(sample_rate_hz)
    t0 = float(src_t[0])
    t1 = float(src_t[-1])
    sample_times = np.arange(t0, t1 + 0.5 * dt, dt, dtype=float)
    if sample_times.size <= 0:
        return np.asarray([t0, t1], dtype=float)
    if sample_times[-1] < t1 - 1.0e-18:
        sample_times = np.append(sample_times, t1)
    return sample_times


def _signed_alias_frequency(freq_hz: float, sample_rate_hz: float) -> float:
    if sample_rate_hz <= 0.0:
        return 0.0
    fs = float(sample_rate_hz)
    return float(((float(freq_hz) + 0.5 * fs) % fs) - 0.5 * fs)


def _receiver_traces(
    *,
    times: np.ndarray,
    complex_envelope: np.ndarray,
    pulse_ir,
    readout_cfg: dict[str, Any],
    pulse_cfg: dict[str, Any],
    chain: dict[str, float | str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    receiver_cfg = dict(readout_cfg.get("receiver", {}) or {})
    channels_cfg = dict(readout_cfg.get("channels", {}) or {})
    mode = str(receiver_cfg.get("mode", "direct_adc") or "direct_adc").strip().lower()
    if mode not in {"direct_adc", "downconversion"}:
        mode = "direct_adc"

    demod_cfg = dict((pulse_cfg.get("acquisition", {}) or {}).get("demodulation", {}) or {})
    cavity_drive_channel = str(channels_cfg.get("cavity_drive", "")) or str(demod_cfg.get("drive_channel", ""))
    lo_channel = str(receiver_cfg.get("lo_channel", channels_cfg.get("local_oscillator", demod_cfg.get("lo_channel", ""))) or "")

    carrier_freq_hz, carrier_phase = _infer_channel_carrier(pulse_ir, cavity_drive_channel)
    lo_freq_hz, lo_phase = _infer_channel_carrier(pulse_ir, lo_channel)
    carrier_freq_hz = _safe_float(
        receiver_cfg.get("carrier_frequency_Hz", carrier_freq_hz if carrier_freq_hz > 0.0 else chain.get("center_freq_Hz", 0.0)),
        _safe_float(chain.get("center_freq_Hz", 0.0), 0.0),
    )
    lo_freq_hz = _safe_float(receiver_cfg.get("lo_frequency_Hz", lo_freq_hz), lo_freq_hz)
    rf_phase = _safe_float(receiver_cfg.get("rf_phase_rad", carrier_phase), carrier_phase)
    if_phase = _safe_float(receiver_cfg.get("if_phase_rad", carrier_phase - lo_phase), carrier_phase - lo_phase)
    digital_lo_phase = _safe_float(receiver_cfg.get("digital_lo_phase_rad", demod_cfg.get("phase_rad", 0.0)), _safe_float(demod_cfg.get("phase_rad", 0.0), 0.0))

    if times.size > 1:
        inferred_rate = 1.0 / max(float(np.mean(np.diff(times))), 1.0e-18)
    else:
        inferred_rate = 0.0
    adc_sample_rate_hz = _safe_float(receiver_cfg.get("adc_sample_rate_Hz", inferred_rate), inferred_rate)
    adc_times = _build_sample_times(times, adc_sample_rate_hz)
    envelope_adc = _resample_complex(times, complex_envelope, adc_times)

    rf_signal = np.real(complex_envelope * np.exp(1j * (2.0 * math.pi * carrier_freq_hz * times + rf_phase)))
    rf_signal_adc = np.real(envelope_adc * np.exp(1j * (2.0 * math.pi * carrier_freq_hz * adc_times + rf_phase)))

    if_frequency_hz = _safe_float(
        receiver_cfg.get("if_frequency_Hz", demod_cfg.get("if_Hz", abs(carrier_freq_hz - lo_freq_hz))),
        abs(carrier_freq_hz - lo_freq_hz),
    )
    if_signal = np.real(complex_envelope * np.exp(1j * (2.0 * math.pi * if_frequency_hz * times + if_phase)))
    if_signal_adc = np.real(envelope_adc * np.exp(1j * (2.0 * math.pi * if_frequency_hz * adc_times + if_phase)))

    sampled_frequency_hz = carrier_freq_hz if mode == "direct_adc" else if_frequency_hz
    sampled_signal = rf_signal_adc if mode == "direct_adc" else if_signal_adc
    rf_noise_sigma = max(0.0, _safe_float(receiver_cfg.get("rf_noise_sigma", 0.0), 0.0))
    adc_noise_sigma = max(0.0, _safe_float(receiver_cfg.get("adc_noise_sigma", 0.0), 0.0))
    if rf_noise_sigma > 0.0:
        sampled_signal = sampled_signal + rng.normal(0.0, rf_noise_sigma, size=adc_times.size)
    adc_signal = sampled_signal + (rng.normal(0.0, adc_noise_sigma, size=adc_times.size) if adc_noise_sigma > 0.0 else 0.0)

    adc_alias_signed_hz = _signed_alias_frequency(sampled_frequency_hz, adc_sample_rate_hz)
    rf_alias_signed_hz = _signed_alias_frequency(carrier_freq_hz, adc_sample_rate_hz)
    digital_baseband = 2.0 * adc_signal.astype(complex) * np.exp(
        -1j * (2.0 * math.pi * adc_alias_signed_hz * adc_times + digital_lo_phase)
    )

    return {
        "mode": mode,
        "adc_times": adc_times.astype(float),
        "adc_sample_rate_Hz": float(adc_sample_rate_hz),
        "complex_envelope": np.asarray(complex_envelope, dtype=complex),
        "complex_envelope_adc": envelope_adc,
        "rf_signal": rf_signal.astype(float),
        "rf_signal_adc": rf_signal_adc.astype(float),
        "if_signal": if_signal.astype(float),
        "if_signal_adc": if_signal_adc.astype(float),
        "adc_signal": np.asarray(adc_signal, dtype=float),
        "digital_baseband": np.asarray(digital_baseband, dtype=complex),
        "carrier_frequency_Hz": float(carrier_freq_hz),
        "lo_frequency_Hz": float(lo_freq_hz),
        "if_frequency_Hz": float(if_frequency_hz),
        "sampled_frequency_Hz": float(sampled_frequency_hz),
        "alias_frequency_Hz": abs(float(adc_alias_signed_hz)),
        "alias_frequency_signed_Hz": float(adc_alias_signed_hz),
        "rf_alias_frequency_Hz": abs(float(rf_alias_signed_hz)),
        "rf_alias_frequency_signed_Hz": float(rf_alias_signed_hz),
        "rf_phase_rad": float(rf_phase),
        "if_phase_rad": float(if_phase),
        "digital_lo_phase_rad": float(digital_lo_phase),
        "adc_noise_sigma": float(adc_noise_sigma),
        "rf_noise_sigma": float(rf_noise_sigma),
        "adc_source": "rf_signal" if mode == "direct_adc" else "if_signal",
    }


def _extract_readout_windows(pulse_ir) -> list[dict[str, float | str]]:
    windows: list[dict[str, float | str]] = []
    for channel in list(getattr(pulse_ir, "channels", []) or []):
        channel_name = str(getattr(channel, "name", ""))
        if not channel_name.upper().startswith("RO_"):
            continue
        pending: dict[str, float | str] | None = None
        for pulse in list(getattr(channel, "pulses", []) or []):
            shape = str(getattr(pulse, "shape", "")).lower()
            params = dict(getattr(pulse, "params", {}) or {})
            if shape != "readout":
                continue
            if str(params.get("break_stage", "")).strip().lower() != "measure":
                continue
            t0 = float(getattr(pulse, "t0_s", 0.0))
            t1 = float(getattr(pulse, "t1_s", 0.0))
            if pending is None:
                pending = {"channel": channel_name, "t0_s": t0, "t1_s": t1}
                continue
            if abs(float(pending["t1_s"]) - t0) <= 1.0e-18:
                pending["t1_s"] = t1
                continue
            windows.append(pending)
            pending = {"channel": channel_name, "t0_s": t0, "t1_s": t1}
        if pending is not None:
            windows.append(pending)
    windows.sort(key=lambda item: float(item["t0_s"]))
    return windows


def _sample_readout_drive(pulse_ir, times: np.ndarray) -> np.ndarray:
    drive = np.zeros_like(times, dtype=complex)
    if times.size <= 0:
        return drive
    for channel in list(getattr(pulse_ir, "channels", []) or []):
        channel_name = str(getattr(channel, "name", ""))
        if not channel_name.upper().startswith("RO_"):
            continue
        for pulse in list(getattr(channel, "pulses", []) or []):
            params = dict(getattr(pulse, "params", {}) or {})
            sampler = make_shape(str(getattr(pulse, "shape", "rect")), params)
            amp = float(getattr(pulse, "amp", 0.0))
            t0 = float(getattr(pulse, "t0_s", 0.0))
            t1 = float(getattr(pulse, "t1_s", 0.0))
            carrier = getattr(pulse, "carrier", None)
            phase = float(getattr(carrier, "phase", 0.0)) if carrier is not None else 0.0
            phase_factor = complex(math.cos(phase), math.sin(phase))
            env = np.asarray([sampler.sample(float(t), t0, t1, amp) for t in times], dtype=float)
            drive = drive + env.astype(complex) * phase_factor
    return drive


def _nearest_centroid(point: complex, centroids: dict[str, complex]) -> str:
    if not centroids:
        return ""
    return min(centroids, key=lambda label: abs(point - centroids[label]))


def _integrate_window(times: np.ndarray, i_trace: np.ndarray, q_trace: np.ndarray, t0: float, t1: float) -> complex | None:
    mask = (times >= t0) & (times <= t1)
    if not np.any(mask):
        return None
    t_sel = times[mask]
    i_sel = i_trace[mask]
    q_sel = q_trace[mask]
    if t_sel.size == 1:
        return complex(float(i_sel[0]), float(q_sel[0]))
    span = max(float(t_sel[-1] - t_sel[0]), 1.0e-18)
    i_int = _integrate_trapezoid(i_sel, t_sel) / span
    q_int = _integrate_trapezoid(q_sel, t_sel) / span
    return complex(i_int, q_int)


def _shot_integrated_iq(shot_payload: dict[str, Any], integrated_iq: complex | None) -> complex | None:
    raw_point = shot_payload.get("integrated_iq")
    if isinstance(raw_point, list) and len(raw_point) >= 2:
        return complex(float(raw_point[0]), float(raw_point[1]))
    return integrated_iq


def _shot_trace_integrated_iq(
    shot_payload: dict[str, Any],
    times: np.ndarray,
    t0: float,
    t1: float,
) -> complex | None:
    measured = _complex_from_pairs(list(shot_payload.get("measured_voltage", []) or []))
    if measured.size > 0:
        return _integrate_window(times, np.real(measured), np.imag(measured), t0, t1)
    heterodyne = _complex_from_pairs(list(shot_payload.get("heterodyne_current", []) or []))
    if heterodyne.size > 0:
        return _integrate_window(times, np.real(heterodyne), np.imag(heterodyne), t0, t1)
    return None


def build_readout_analysis(
    *,
    trajectory,
    model_spec,
    pulse_ir,
    pulse_cfg: Any,
    device_cfg: Any,
    seed: int,
) -> dict[str, Any]:
    """
    Build Case-level readout analysis.
    Responsibility: Signal chain reconstruction and IQ point integration for a single trajectory.
    """
    # Ensure configs are dictionaries
    p_cfg = asdict(pulse_cfg) if hasattr(pulse_cfg, "__dataclass_fields__") else dict(pulse_cfg or {})
    d_cfg = asdict(device_cfg) if hasattr(device_cfg, "__dataclass_fields__") else dict(device_cfg or {})
    
    pulse_cfg = p_cfg
    device_cfg = d_cfg

    # ``run_analysis`` passes the typed workflow ``DeviceConfig`` dataclass, whose
    # actual component graph lives under the nested ``device`` key. Older callers
    # pass the component graph directly. Support both so we can always recover the
    # ``ro0`` receiver parameters used for RF/ADC reconstruction.
    device_payload = dict(device_cfg.get("device", {}) or {}) if isinstance(device_cfg.get("device"), dict) else dict(device_cfg)
    
    # Get RO line parameters from device components (Flat structure as agreed)
    ro0_params = {}
    for comp in list(device_payload.get("components", []) or []):
        if str(comp.get("id", "")).strip() == "ro0":
            ro0_params = dict(comp.get("parameters", {}) or {})
            break
    if not ro0_params:
        # Fallback to legacy single-device payloads that expose parameters directly.
        ro0_params = dict(device_payload.get("parameters", {}) or {})

    obs = dict((getattr(trajectory, "classical", {}) or {}).get("readout", {}) or {})
    times = np.asarray(list(getattr(trajectory, "times", []) or []), dtype=float)
    
    if times.size <= 0:
        return {}

    # 1. Physical Signal Reconstruction
    a_in_obs = _complex_from_pairs(list(obs.get("a_in", []) or []))
    cavity_a = _complex_from_pairs(list(obs.get("cavity_a", []) or []))
    a_out_obs = _complex_from_pairs(list(obs.get("a_out", []) or []))
    
    # Drive signal if not observed
    drive = a_in_obs if a_in_obs.size > 0 else _sample_readout_drive(pulse_ir, times)
    
    # Coupling prefactor from RO line
    kappa_ext = _safe_float(ro0_params.get("kappa_ext_Hz", 0.0), 0.0)
    coupling_scale = _readout_coupling_prefactor(kappa_ext)
    
    # Actual output field
    a_out = a_out_obs if a_out_obs.size > 0 else (drive - coupling_scale * cavity_a)
    
    # 2. Receiver Processing
    # Use agreed-upon flat parameters from ro0
    carrier_freq = _safe_float(ro0_params.get("carrier_frequency_Hz", 0.0), 0.0)
    lo_freq = _safe_float(ro0_params.get("lo_frequency_Hz", 0.0), 0.0)
    if_freq = _safe_float(ro0_params.get("if_frequency_Hz", 0.0), 0.0)
    rf_phase = _safe_float(ro0_params.get("rf_phase_rad", 0.0), 0.0)
    if_phase = _safe_float(ro0_params.get("if_phase_rad", 0.0), 0.0)
    digital_lo_phase = _safe_float(ro0_params.get("digital_lo_phase_rad", 0.0), 0.0)
    adc_rate = _safe_float(ro0_params.get("adc_sample_rate_Hz", 0.0), 0.0)
    
    rng = np.random.default_rng(int(seed))
    
    # Inject flattened params into the 'receiver' sub-dict for compatibility with _receiver_traces
    receiver_fake_cfg = {"receiver": ro0_params}
    
    receiver = _receiver_traces(
        times=times,
        complex_envelope=a_out,
        pulse_ir=pulse_ir,
        readout_cfg=receiver_fake_cfg,
        pulse_cfg=pulse_cfg,
        chain=ro0_params,
        rng=rng,
    )

    # 3. IQ Integration
    adc_times = np.asarray(receiver.get("adc_times", times), dtype=float)
    digital_baseband = np.asarray(receiver.get("digital_baseband", []), dtype=complex)
    
    measure_windows = _extract_readout_windows(pulse_ir)
    integration_window_s = _safe_float((pulse_cfg.get("acquisition", {}) or {}).get("integration_window_ns", 0.0), 0.0) * 1.0e-9
    start_delay_s = _safe_float((pulse_cfg.get("acquisition", {}) or {}).get("start_delay_ns", 0.0), 0.0) * 1.0e-9
    
    if integration_window_s <= 0.0:
        integration_window_s = _safe_float(pulse_cfg.get("measure_duration_ns", 0.0), 0.0) * 1.0e-9

    # For Case level, we integrate the first window as the primary IQ point
    integrated_iq = None
    if measure_windows and digital_baseband.size > 0:
        window = measure_windows[0]
        t0 = float(window["t0_s"]) + start_delay_s
        t1 = min(float(window["t1_s"]), t0 + integration_window_s)
        integrated_iq = _integrate_window(adc_times, np.real(digital_baseband), np.imag(digital_baseband), t0, t1)

    shot_window_t0 = None
    shot_window_t1 = None
    if measure_windows:
        shot_window_t0 = float(measure_windows[0]["t0_s"]) + start_delay_s
        shot_window_t1 = min(float(measure_windows[0]["t1_s"]), shot_window_t0 + integration_window_s)

    shot_payloads = list(obs.get("shots", []) or [])
    shots: list[ShotData] = []
    integrated_points: list[complex] = []
    for shot_payload in shot_payloads:
        shot_dict = dict(shot_payload or {})
        shot_point = None
        if shot_window_t0 is not None and shot_window_t1 is not None:
            shot_point = _shot_trace_integrated_iq(shot_dict, times, shot_window_t0, shot_window_t1)
        if shot_point is None:
            shot_point = _shot_integrated_iq(shot_dict, integrated_iq)
        if shot_point is not None:
            integrated_points.append(shot_point)
        shots.append(
            ShotData(
                timestamp=float(times[0]) if times.size > 0 else 0.0,
                a_out=shot_dict.get("a_out"),
                integrated_iq=[float(shot_point.real), float(shot_point.imag)] if shot_point is not None else None,
                metadata={
                    k: v
                    for k, v in shot_dict.items()
                    if k not in {"a_out", "integrated_iq"}
                },
            )
        )
    if not integrated_points and integrated_iq is not None:
        integrated_points.append(integrated_iq)

    return ReadoutAnalysis(
        sim_times=list(times),
        adc_times=list(adc_times),
        chain_params={
            "carrier_freq": carrier_freq,
            "adc_rate": adc_rate,
            **ro0_params,
        },
        signals={
            "intracavity_field": cavity_a,
            "outgoing_field": a_out,
            "complex_envelope": digital_baseband,
            "rf_signal": receiver.get("rf_signal"),
            "adc_signal": receiver.get("adc_signal"),
        },
        demodulation=receiver,
        shots=shots,
        integrated_points=integrated_points,
    )

def build_iq_analysis(
    *,
    case_results: list[dict[str, Any]],
    labels: list[str],
    seed: int,
) -> IQAnalysis:
    """
    Build Comprehensive-level IQ analysis.
    Responsibility: Aggregate case results to compute centroids, clouds, and fidelity.
    """
    rng = np.random.default_rng(int(seed))
    
    # 1. Aggregate IQ Clouds
    clouds: dict[str, list[complex]] = {label: [] for label in labels}
    
    for i, res in enumerate(case_results):
        if i >= len(labels): break
        label = labels[i]
        iq_values = res.get("integrated_iq")
        if iq_values is None:
            continue
        if isinstance(iq_values, list):
            for item in iq_values:
                if item is not None:
                    clouds[label].append(complex(item))
            continue
        clouds[label].append(complex(iq_values))
    
    # 2. Calculate Centroids
    centroids: dict[str, complex] = {}
    for label, points in clouds.items():
        if points:
            centroids[label] = sum(points) / len(points)
        else:
            centroids[label] = 0.0 + 0.0j

    # 3. Confusion Matrix & Fidelity
    labels_list = list(centroids.keys())
    n_labels = len(labels_list)
    confusion = np.zeros((n_labels, n_labels), dtype=int)
    
    for i, label in enumerate(labels_list):
        for point in clouds[label]:
            pred = _nearest_centroid(point, centroids)
            if pred in labels_list:
                confusion[i, labels_list.index(pred)] += 1
    
    fidelity = float(np.trace(confusion) / max(1, confusion.sum())) if confusion.size else 0.0
    
    # 4. SNR and separation
    pairwise_dist = []
    for i, l1 in enumerate(labels_list):
        for l2 in labels_list[i+1:]:
            pairwise_dist.append(abs(centroids[l1] - centroids[l2]))
    
    first_label = labels_list[0] if labels_list else None
    noise_sigma = 0.0
    if first_label and clouds[first_label]:
        pts = np.asarray(clouds[first_label])
        noise_sigma = float(np.std(np.abs(pts - centroids[first_label])))
    
    snr = float(np.mean(pairwise_dist) / (2 * noise_sigma) if pairwise_dist and noise_sigma > 0 else 0.0)

    # 5. Discrimination Line (Simple bisector for 2 states)
    discrimination_line = None
    if n_labels == 2:
        c0, c1 = centroids[labels_list[0]], centroids[labels_list[1]]
        diff = c1 - c0
        dist = abs(diff)
        if dist > 1e-15:
            mid = (c0 + c1) / 2
            slope = diff / dist # Direction vector
            discrimination_line = {
                "midpoint": [mid.real, mid.imag],
                "normal": [slope.real, slope.imag],
            }

    return IQAnalysis(
        centroids={l: v for l, v in centroids.items()},
        confusion_matrix={
            "labels": labels_list,
            "values": confusion.astype(int).tolist(),
        },
        assignment_fidelity=fidelity,
        noise_sigma=noise_sigma,
        snr=snr,
        iq_clouds={l: [[p.real, p.imag] for p in pts] for l, pts in clouds.items()},
        discrimination_line=discrimination_line,
    )

__all__ = ["build_readout_analysis", "build_iq_analysis"]
