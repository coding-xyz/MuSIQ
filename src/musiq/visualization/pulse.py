"""Pulse plotting helpers extracted from report use-cases."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from musiq.pulse.drawer_adapter import EngineeringDrawer
from musiq.pulse.sequence import PulseCompiler
from musiq.schemas.pulse import PulseIR

S_TO_NS = 1e9


def _coerce_sample_map(pulse_data: PulseIR | Mapping[str, Mapping[str, Any]], sample_rate: float) -> dict[str, dict[str, Any]]:
    if isinstance(pulse_data, PulseIR):
        compiled = PulseCompiler.compile(pulse_data, sample_rate_Hz=sample_rate)
        sample_map = {str(name): dict(payload) for name, payload in compiled.items()}
        return _ensure_iq_for_carrier_channels(sample_map, pulse_data)
    sample_map = {str(name): dict(payload) for name, payload in pulse_data.items()}
    return _ensure_iq_for_carrier_channels(sample_map)


def _payload_has_carrier(payload: Mapping[str, Any]) -> bool:
    carrier_freq = np.asarray(payload.get("carrier_freq_Hz", []), dtype=float)
    carrier_phase = np.asarray(payload.get("carrier_phase_rad", []), dtype=float)
    return bool(np.any(np.abs(carrier_freq) > 0.0) or np.any(np.abs(carrier_phase) > 0.0))


def _ensure_iq_for_carrier_channels(
    sample_map: Mapping[str, Mapping[str, Any]],
    pulse_ir: PulseIR | None = None,
) -> dict[str, dict[str, Any]]:
    carrier_channels: set[str] = set()
    if pulse_ir is not None:
        for channel in pulse_ir.channels:
            if any(pulse.carrier is not None for pulse in channel.pulses):
                carrier_channels.add(str(channel.name))

    normalized: dict[str, dict[str, Any]] = {}
    for channel_name, payload in sample_map.items():
        coerced = dict(payload)
        has_carrier = channel_name in carrier_channels or _payload_has_carrier(coerced)
        if has_carrier and "y_quadrature" not in coerced:
            coerced["y_quadrature"] = np.zeros_like(np.asarray(coerced.get("y", []), dtype=float))
        normalized[str(channel_name)] = coerced
    return normalized


def _resolve_run(model, run_id: str | None = None, *, study_name: str | None = None, solver_id: str | None = None):
    if run_id is not None:
        try:
            return model.runs[run_id]
        except KeyError as exc:
            raise KeyError(f"Run `{run_id}` not found. Available runs: {sorted(model.runs.keys())}") from exc
    for candidate_id, run_obj in getattr(model, "runs", {}).items():
        identity = getattr(run_obj, "identity", None)
        if study_name is not None and str(getattr(identity, "study_name", "") or "") != str(study_name):
            continue
        if solver_id is not None and str(getattr(identity, "solver_id", "") or "") != str(solver_id):
            continue
        return run_obj
    raise KeyError("No matching run found in model.")


def _first_result(run_obj):
    return next(iter(getattr(run_obj, "results", {}).values()), None)


def _first_trajectory(run_obj):
    result = _first_result(run_obj)
    if result is None:
        return None
    return next(iter(getattr(result, "trajectories", {}).values()), None)


def _max_abs(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def plot_pulse_envelope(
    ax,
    times_ns,
    i_values,
    q_values=None,
    *,
    title: str | None = None,
    xlabel: str = "time (ns)",
    ylabel: str = "pulse amplitude",
    i_label: str = "I",
    q_label: str = "Q",
) -> None:
    """Plot a single-channel pulse envelope."""
    x = np.asarray(times_ns, dtype=float)
    ax.plot(x, np.asarray(i_values, dtype=float), label=i_label)
    if q_values is not None:
        ax.plot(x, np.asarray(q_values, dtype=float), linestyle="--", label=q_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()


def plot_pulse_channels(
    ax,
    pulse_data: PulseIR | Mapping[str, Mapping[str, Any]],
    *,
    sample_rate: float = 1.0e9,
    title: str | None = None,
    xlabel: str = "time (ns)",
) -> None:
    """Plot compiled multi-channel pulses in stacked rows."""
    samples = _coerce_sample_map(pulse_data, sample_rate)
    if not samples:
        ax.set_title(title or "Pulse channels")
        ax.set_xlabel(xlabel)
        return

    peak = 0.0
    for payload in samples.values():
        peak = max(peak, _max_abs(payload.get("y", [])))
        if "y_quadrature" in payload:
            peak = max(peak, _max_abs(payload.get("y_quadrature", [])))
    scale = peak if peak > 0.0 else 1.0
    row_gap = 1.8

    labels = []
    ticks = []
    for idx, (channel, payload) in enumerate(samples.items()):
        row = row_gap * idx
        times_ns = np.asarray(payload.get("t", []), dtype=float) * S_TO_NS
        i_vals = np.asarray(payload.get("y", []), dtype=float) / scale + row
        q_vals = np.asarray(payload.get("y_quadrature", []), dtype=float) / scale + row if "y_quadrature" in payload else None
        ax.plot(times_ns, i_vals, linewidth=1.5, label=f"{channel} I" if idx == 0 else None)
        if q_vals is not None:
            ax.plot(times_ns, q_vals, linestyle="--", linewidth=1.2, alpha=0.8, label=f"{channel} Q" if idx == 0 else None)
        labels.append(channel)
        ticks.append(row)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", linestyle="-", linewidth=0.5, color="0.85")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("channel")
    ax.set_title(title or "Pulse channels")
    if len(samples) == 1:
        ax.legend()


def plot_pulse(
    ax,
    model,
    run_id: str | None = None,
    *,
    study_name: str | None = None,
    solver_id: str | None = None,
    sample_rate: float = 1.0e9,
    timing_layout: bool = True,
    extend_to_trajectory_end: bool = False,
    title: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> None:
    """Plot one model run's pulse schedule directly onto a provided axis."""
    run_obj = _resolve_run(model, run_id, study_name=study_name, solver_id=solver_id)
    pulse_ir = getattr(getattr(run_obj, "artifacts", None), "pulse_ir", None)
    if pulse_ir is None:
        raise KeyError("Selected run does not contain pulse_ir artifacts.")
    if extend_to_trajectory_end:
        trajectory = _first_trajectory(run_obj)
        if trajectory is not None and getattr(trajectory, "times", None):
            pulse_ir = replace(pulse_ir, t_end_s=float(trajectory.times[-1]))

    plot_style = dict(style or {})
    if not timing_layout:
        plot_pulse_channels(
            ax,
            pulse_ir,
            sample_rate=sample_rate,
            title=title or "Pulse channels",
            xlabel=str(plot_style.pop("xlabel", "time (ns)")),
        )
        return

    samples = _coerce_sample_map(pulse_ir, sample_rate)
    row_gap = float(plot_style.pop("row_gap", 1.0))
    amplitude_scale = float(plot_style.pop("amplitude_scale", 0.7e-8))
    linewidth = float(plot_style.pop("linewidth", 1.5))
    quadrature_linewidth = float(plot_style.pop("quadrature_linewidth", linewidth))
    quadrature_alpha = float(plot_style.pop("quadrature_alpha", 0.7))
    quadrature_linestyle = str(plot_style.pop("quadrature_linestyle", "--"))
    xlabel = str(plot_style.pop("xlabel", "time (ns)"))
    ylabel = str(plot_style.pop("ylabel", "channel"))
    grid_color = str(plot_style.pop("grid_color", "gray"))
    grid_linestyle = str(plot_style.pop("grid_linestyle", "-"))
    grid_linewidth = float(plot_style.pop("grid_linewidth", 0.5))
    channel_colors = dict(plot_style.pop("channel_colors", {}))

    ticklabels = []
    ticks = []
    for idx, (channel, waveform) in enumerate(samples.items()):
        y0 = row_gap * idx
        ticklabels.append(channel)
        ticks.append(y0)
        colors = channel_colors.get(channel)
        if colors is None:
            if str(channel).startswith("XY"):
                colors = ("tab:blue", "tab:green")
            elif str(channel).startswith("TC"):
                colors = ("tab:orange", "tab:gold")
            else:
                colors = ("tab:red", "tab:pink")
        ax.plot(
            np.asarray(waveform["t"], dtype=float) * S_TO_NS,
            np.asarray(waveform["y"], dtype=float) * amplitude_scale + y0,
            color=colors[0],
            linewidth=linewidth,
        )
        if "y_quadrature" in waveform:
            ax.plot(
                np.asarray(waveform["t"], dtype=float) * S_TO_NS,
                np.asarray(waveform["y_quadrature"], dtype=float) * amplitude_scale + y0,
                quadrature_linestyle,
                color=colors[1],
                linewidth=quadrature_linewidth,
                alpha=quadrature_alpha,
            )

    ax.set_yticks(np.asarray(ticks, dtype=float))
    ax.set_yticklabels(ticklabels)
    ax.grid(color=grid_color, linestyle=grid_linestyle, linewidth=grid_linewidth, axis="y", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Pulse sequences: {getattr(getattr(run_obj, 'identity', None), 'study_name', run_id or 'run')}")


def make_pulse_figure(
    pulse_ir: PulseIR,
    *,
    sample_rate: float = 1.0e9,
    timing_layout: bool = False,
    title: str | None = None,
    show_clock: bool = False,
    png_path: str | Path | None = None,
    dxf_path: str | Path | None = None,
    **_: Any,
):
    """Create a pulse figure without routing through ``musiq.pulse.visualize``."""
    import matplotlib.pyplot as plt

    compiled = _coerce_sample_map(pulse_ir, sample_rate)
    channel_count = max(len(compiled), 1)
    fig_h = max(3.5, 1.0 + 0.9 * channel_count)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    if not timing_layout and len(compiled) == 1:
        _, payload = next(iter(compiled.items()))
        plot_pulse_envelope(
            ax,
            np.asarray(payload.get("t", []), dtype=float) * S_TO_NS,
            payload.get("y", []),
            payload.get("y_quadrature"),
            title=title or "Pulse waveform",
            i_label="I",
            q_label="Q",
        )
    else:
        clock_title = " with clock" if show_clock else ""
        plot_pulse_channels(ax, compiled, title=(title or f"Pulse channels{clock_title}"), sample_rate=sample_rate)

    fig.tight_layout()
    if png_path is not None:
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=180)
    if dxf_path is not None:
        style = {
            "title": title or "Pulse channels",
            "clk_mhz": 100.0 if show_clock else None,
            "breaks": [],
        }
        EngineeringDrawer.export_dxf(pulse_ir, dxf_path, style=style)
    return fig
