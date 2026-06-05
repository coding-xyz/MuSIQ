"""Reusable report-oriented visualization helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


_TIME_UNIT_SCALE: dict[str, float] = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
}


def _as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _resolve_mapping_key(mapping: Mapping[str, Any], name: str) -> str:
    if name in mapping:
        return name
    wanted = _normalize_key(name)
    for key in mapping:
        if _normalize_key(key) == wanted:
            return key
    raise KeyError(f"Key `{name}` not found. Available keys: {sorted(mapping.keys())}")


def _first_result(run_obj):
    return next(iter(getattr(run_obj, "results", {}).values()), None)


def _first_trajectory(run_obj):
    result = _first_result(run_obj)
    if result is None:
        return None
    return next(iter(getattr(result, "trajectories", {}).values()), None)


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


def _resolve_analysis(model, analysis_id: str | None = None, *, scope: str | None = None):
    analyses = getattr(model, "analyses", {})
    if analysis_id is not None:
        try:
            return analyses[analysis_id]
        except KeyError as exc:
            raise KeyError(f"Analysis `{analysis_id}` not found. Available analyses: {sorted(analyses.keys())}") from exc
    for analysis in analyses.values():
        analysis_scope = getattr(getattr(analysis, "scope", None), "value", getattr(analysis, "scope", None))
        if scope is not None and str(analysis_scope) != str(scope):
            continue
        return analysis
    raise KeyError(f"No analysis found for scope `{scope}`.")


def _resolve_metric(metric_map: Mapping[str, Any], metric_name: str):
    key = _resolve_mapping_key(metric_map, metric_name)
    return key, metric_map[key]


def _resolve_case_result(model, analysis) -> Any:
    refs = list(getattr(analysis, "input_results", []) or [])
    if not refs:
        raise KeyError("Case analysis has no input results.")
    ref = refs[0]
    run_obj = model.runs[str(ref.run_id)]
    try:
        return run_obj.results[str(ref.parameter_id)]
    except KeyError as exc:
        raise KeyError(
            f"Result `{ref.parameter_id}` not found in run `{ref.run_id}`. "
            f"Available results: {sorted(run_obj.results.keys())}"
        ) from exc


def _resolve_case_trajectory(model, analysis):
    result = _resolve_case_result(model, analysis)
    try:
        return next(iter(result.trajectories.values()))
    except StopIteration as exc:
        raise KeyError("Case analysis does not reference a trajectory.") from exc


def _default_ylabel(metric_name: str) -> str:
    normalized = _normalize_key(metric_name)
    if normalized == "population":
        return "population"
    if normalized.startswith("finalp"):
        return "final population"
    return metric_name


def _normalize_time_unit(unit: str | None) -> str:
    if unit is None:
        return "s"
    normalized = str(unit).strip().lower()
    if normalized not in _TIME_UNIT_SCALE:
        supported = ", ".join(sorted(_TIME_UNIT_SCALE))
        raise ValueError(f"Unsupported time_unit `{unit}`. Supported units: {supported}.")
    return normalized


def _scale_times(values: Sequence[float] | np.ndarray, *, time_unit: str | None) -> np.ndarray:
    unit = _normalize_time_unit(time_unit)
    scale = _TIME_UNIT_SCALE[unit]
    return _as_float_array(values) / scale


def _merge_series_style(
    series_name: str,
    *,
    base_style: Mapping[str, Any] | None = None,
    series_styles: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge default and per-series matplotlib kwargs, defaulting label to the series name."""
    kwargs: dict[str, Any] = {"label": str(series_name)}
    kwargs.update(dict(base_style or {}))
    if series_styles:
        key = str(series_name)
        if key not in series_styles:
            wanted = _normalize_key(key)
            key = next((candidate for candidate in series_styles if _normalize_key(candidate) == wanted), "")
        if key:
            kwargs.update(dict(series_styles.get(key, {}) or {}))
    return kwargs


def plot_population_series(
    ax,
    times,
    series: Mapping[str, Sequence[float] | np.ndarray],
    *,
    xlabel: str = "time (ns)",
    ylabel: str = "population",
    title: str | None = None,
    style: Mapping[str, Any] | None = None,
    styles: Mapping[str, dict[str, Any]] | None = None,
    legend: bool = True,
) -> None:
    """Plot one or more population traces on a provided axis."""
    x = _as_float_array(times)
    for label, values in series.items():
        kwargs = _merge_series_style(label, base_style=style, series_styles=styles)
        ax.plot(x, _as_float_array(values), **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and series:
        ax.legend()


def plot_metric_series(
    ax,
    x_values,
    series: Mapping[str, Sequence[float] | np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    logy: bool = False,
    style: Mapping[str, Any] | None = None,
    styles: Mapping[str, dict[str, Any]] | None = None,
    legend: bool = True,
) -> None:
    """Plot one or more metric curves against a common x-axis."""
    x = _as_float_array(x_values)
    for label, values in series.items():
        kwargs = _merge_series_style(label, base_style=style, series_styles=styles)
        y = _as_float_array(values)
        if logy:
            ax.semilogy(x, y, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and series:
        ax.legend()


def plot_grouped_bars(
    ax,
    categories: Sequence[str],
    groups: Mapping[str, Sequence[float] | np.ndarray],
    *,
    ylabel: str,
    xlabel: str | None = None,
    title: str | None = None,
    annotate: bool = False,
    style: Mapping[str, Any] | None = None,
    group_styles: Mapping[str, Mapping[str, Any]] | None = None,
    legend: bool = True,
) -> None:
    """Plot grouped bar charts with one bar series per group label."""
    labels = list(categories)
    x = np.arange(len(labels), dtype=float)
    group_items = list(groups.items())
    if not group_items:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    width = 0.8 / max(len(group_items), 1)
    center = (len(group_items) - 1) / 2.0
    for idx, (group_label, values) in enumerate(group_items):
        offset = (idx - center) * width
        kwargs = _merge_series_style(group_label, base_style=style, series_styles=group_styles)
        rects = ax.bar(x + offset, _as_float_array(values), width=width, **kwargs)
        if annotate:
            for rect in rects:
                height = rect.get_height()
                if np.isnan(height):
                    continue
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2.0, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if legend and group_items:
        ax.legend()


def plot_error_budget(ax, report: Mapping[str, Any], *, title: str = "Error Budget") -> None:
    """Plot an analysis report error budget as a bar chart."""
    error_budget = dict(report.get("error_budget", {}) or {})
    ax.bar(list(error_budget.keys()), list(error_budget.values()))
    ax.set_title(title)
    ax.set_ylabel("value")


def plot_case_metrics(
    ax,
    model,
    analysis_id: str = "case_0",
    metric_name: str = "population",
    *,
    series_keys: Sequence[str] | None = None,
    style: Mapping[str, Any] | None = None,
    series_styles: Mapping[str, Mapping[str, Any]] | None = None,
    time_unit: str | None = "s",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
) -> None:
    """Plot one case-analysis metric directly from a ``model``."""
    analysis = _resolve_analysis(model, analysis_id, scope="case")
    metric_map = dict(getattr(getattr(analysis, "output", None), "metrics", {}) or {})
    metric_key, metric = _resolve_metric(metric_map, metric_name)
    resolved_time_unit = _normalize_time_unit(time_unit)
    x = _scale_times(getattr(metric, "times", []) or [], time_unit=resolved_time_unit)
    values = getattr(metric, "values", [])
    style = dict(style or {})
    resolved_xlabel = xlabel or f"time ({resolved_time_unit})"

    if isinstance(values, Mapping):
        ordered_keys = list(series_keys or values.keys())
        plotted = {}
        for raw_key in ordered_keys:
            key = _resolve_mapping_key(values, str(raw_key))
            plotted[str(raw_key)] = _as_float_array(values[key])
        plot_population_series(
            ax,
            x,
            plotted,
            xlabel=resolved_xlabel,
            ylabel=ylabel or _default_ylabel(metric_key),
            title=title or f"{analysis_id}: {metric_key}",
            style=style,
            styles=series_styles,
            legend=legend,
        )
        return

    ax.plot(x, _as_float_array(values), **style)
    ax.set_xlabel(resolved_xlabel)
    ax.set_ylabel(ylabel or _default_ylabel(metric_key))
    ax.set_title(title or f"{analysis_id}: {metric_key}")


def plot_sweep_metrics(
    ax,
    model,
    analysis_id: str = "sweep_0",
    metric_name: str = "final_P0",
    *,
    parameter_name: str | None = None,
    style: Mapping[str, Any] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    logy: bool = False,
) -> None:
    """Plot one parametric-analysis metric directly from a ``model``."""
    analysis = _resolve_analysis(model, analysis_id, scope="parametric")
    parameters = dict(getattr(getattr(analysis, "output", None), "parameters", {}) or {})
    metric_map = dict(getattr(getattr(analysis, "output", None), "metrics", {}) or {})
    metric_key, metric = _resolve_metric(metric_map, metric_name)

    if not parameters:
        raise KeyError(f"Sweep analysis `{analysis_id}` has no parameter axes.")
    if parameter_name is None:
        axis_key = next(iter(parameters.keys()))
    else:
        axis_key = _resolve_mapping_key(parameters, parameter_name)
    axis = parameters[axis_key]
    x = _as_float_array(getattr(axis, "values", []) or [])
    y = _as_float_array(getattr(metric, "values", []) or [])
    style = dict(style or {})

    if logy:
        ax.semilogy(x, y, **style)
    else:
        ax.plot(x, y, **style)
    ax.set_xlabel(xlabel or getattr(axis, "parameter_name", axis_key))
    ax.set_ylabel(ylabel or _default_ylabel(metric_key))
    ax.set_title(title or f"{analysis_id}: {metric_key}")


def plot_case_final_population(
    ax,
    model,
    analysis_id: str = "case_0",
    metric_name: str = "population",
    *,
    series_keys: Sequence[str] | None = None,
    title: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> None:
    """Plot final-time metric values from one case analysis as a bar chart."""
    analysis = _resolve_analysis(model, analysis_id, scope="case")
    metric_map = dict(getattr(getattr(analysis, "output", None), "metrics", {}) or {})
    metric_key, metric = _resolve_metric(metric_map, metric_name)
    values = getattr(metric, "values", [])
    if not isinstance(values, Mapping):
        raise TypeError(f"Metric `{metric_key}` is not multi-series and cannot be plotted as grouped bars.")
    ordered_keys = list(series_keys or values.keys())
    labels = []
    finals = []
    for raw_key in ordered_keys:
        key = _resolve_mapping_key(values, str(raw_key))
        series = _as_float_array(values[key])
        labels.append(str(raw_key))
        finals.append(float(series[-1]) if series.size else np.nan)
    ax.bar(labels, finals, **dict(style or {}))
    ax.set_xlabel("state")
    ax.set_ylabel("final population")
    ax.set_title(title or f"{analysis_id}: final {metric_key}")


def plot_case_iq_cloud(
    ax,
    model,
    analysis_id: str = "case_0",
    *,
    title: str | None = None,
    style: Mapping[str, Any] | None = None,
    mean_style: Mapping[str, Any] | None = None,
) -> None:
    """Plot one case's integrated IQ cloud directly from a ``model``."""
    analysis = _resolve_analysis(model, analysis_id, scope="case")
    trajectory = _resolve_case_trajectory(model, analysis)
    points = integrated_heterodyne_iq({"trajectory": trajectory, "label": analysis_id})
    point_style = {"s": 24, "alpha": 0.72}
    point_style.update(dict(style or {}))
    centroid_style = {"marker": "x", "s": 90, "color": "black"}
    centroid_style.update(dict(mean_style or {}))
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], **point_style)
        ax.scatter([points[:, 0].mean()], [points[:, 1].mean()], **centroid_style)
    ax.set_title(title or analysis_id)
    ax.set_xlabel("integrated I")
    ax.set_ylabel("integrated Q")
    ax.axis("equal")


def density_snapshots(source: Any) -> np.ndarray:
    """Return density-matrix snapshots from a musiq case or trajectory."""
    trajectory = source.get("trajectory") if isinstance(source, dict) else source
    density_matrix = getattr(trajectory, "density_matrix", None)
    if isinstance(density_matrix, dict):
        return np.asarray(density_matrix.get("snapshots", []) or [], dtype=complex)
    return np.asarray(density_matrix or [], dtype=complex)


def _case_model_spec(case: Mapping[str, Any]):
    run_obj = next(iter(case["model"].runs.values()))
    return run_obj.artifacts.model_spec


def qubit_level_populations(case: Mapping[str, Any], *, normalize: bool = True) -> np.ndarray:
    """Return transmon-level populations from density matrices in a musiq case."""
    rho = density_snapshots(case)
    if rho.size == 0:
        return np.zeros((0, 0), dtype=float)
    model_spec = _case_model_spec(case)
    levels = int(model_spec.system.transmon_levels or 2)
    cavity_dim = int(model_spec.system.cavity_nmax or 0) + 1
    pops = np.zeros((rho.shape[0], levels), dtype=float)
    for t_idx, mat in enumerate(rho):
        for cavity in range(cavity_dim):
            for level in range(levels):
                idx = cavity * levels + level
                if idx < mat.shape[0]:
                    pops[t_idx, level] += float(np.real(mat[idx, idx]))
    pops = np.clip(pops, 0.0, None)
    if normalize:
        norm = pops.sum(axis=1, keepdims=True)
        pops = np.divide(pops, norm, out=np.zeros_like(pops), where=norm > 0.0)
    return pops


def final_level_population_table(cases: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return one row of final level populations for each musiq case."""
    rows = []
    width = 0
    for case in cases:
        pops = qubit_level_populations(case)
        row = pops[-1] if pops.size else np.asarray([], dtype=float)
        width = max(width, int(row.size))
        rows.append(row)
    if width == 0:
        return np.zeros((len(rows), 0), dtype=float)
    return np.asarray([np.pad(row, (0, width - row.size), constant_values=np.nan) for row in rows], dtype=float)


def integrated_heterodyne_iq(case: Mapping[str, Any]) -> np.ndarray:
    """Integrate per-shot heterodyne I/Q records over the readout window."""
    trajectory = case["trajectory"]
    times = np.asarray(trajectory.times, dtype=float)
    records = list((trajectory.measurements or {}).get("records", []) or [])
    if times.size == 0 or not records:
        return np.zeros((0, 2), dtype=float)

    readout = dict((trajectory.classical or {}).get("readout", {}) or {})
    windows = list(readout.get("measurement_windows", []) or readout.get("readout_windows", []) or [])
    if windows:
        t0 = float(windows[-1].get("t0_s", times[0]))
        t1 = float(windows[-1].get("t1_s", times[-1]))
    else:
        t0 = float(times[int(0.55 * max(len(times) - 1, 0))])
        t1 = float(times[-1])
    mask = (times >= t0) & (times <= t1)

    points = []
    for record in records:
        i_vals = np.asarray(record.get("heterodyne_I", []), dtype=float)
        q_vals = np.asarray(record.get("heterodyne_Q", []), dtype=float)
        if i_vals.size != times.size or q_vals.size != times.size or not np.any(mask):
            continue
        duration = max(float(times[mask][-1] - times[mask][0]), np.finfo(float).eps)
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:
            integrate = getattr(np, "trapz")
        points.append([integrate(i_vals[mask], times[mask]) / duration, integrate(q_vals[mask], times[mask]) / duration])
    return np.asarray(points, dtype=float)


def integrated_iq_mean_error(cases: Sequence[Mapping[str, Any]], *, error: str = "sem") -> dict[str, np.ndarray]:
    """Return per-case integrated I/Q mean and error bars from shot records."""
    means = []
    errors = []
    stds = []
    counts = []
    for case in cases:
        points = integrated_heterodyne_iq(case)
        counts.append(int(points.shape[0]))
        if points.size == 0:
            means.append([np.nan, np.nan])
            errors.append([np.nan, np.nan])
            stds.append([np.nan, np.nan])
            continue
        mean = np.nanmean(points, axis=0)
        std = np.nanstd(points, axis=0, ddof=1) if points.shape[0] > 1 else np.zeros(2, dtype=float)
        if error == "std":
            err = std
        elif error == "sem":
            err = std / max(float(np.sqrt(points.shape[0])), 1.0)
        else:
            raise ValueError("error must be 'sem' or 'std'")
        means.append(mean)
        errors.append(err)
        stds.append(std)
    return {
        "mean": np.asarray(means, dtype=float),
        "error": np.asarray(errors, dtype=float),
        "std": np.asarray(stds, dtype=float),
        "n": np.asarray(counts, dtype=int),
    }


def plot_iq_cloud(ax, case: Mapping[str, Any], title: str | None = None) -> None:
    """Plot integrated heterodyne I/Q points for one musiq case."""
    points = integrated_heterodyne_iq(case)
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=24, alpha=0.72)
        ax.scatter([points[:, 0].mean()], [points[:, 1].mean()], marker="x", s=90, color="black")
    ax.set_title(title or str(case.get("label", "")))
    ax.set_xlabel("integrated I")
    ax.set_ylabel("integrated Q")
    ax.axis("equal")


def plot_iq_clouds(ax, cases: Sequence[Mapping[str, Any]], title: str | None = None) -> None:
    """Plot multiple integrated heterodyne I/Q clouds on one axis."""
    for case in cases:
        points = integrated_heterodyne_iq(case)
        label = str(case.get("label", ""))
        if not points.size:
            continue
        if hasattr(ax._get_lines, "get_next_color"):
            color = ax._get_lines.get_next_color()
        else:
            color = None
        ax.scatter(points[:, 0], points[:, 1], s=20, alpha=0.45, color=color, label=label)
        ax.scatter([points[:, 0].mean()], [points[:, 1].mean()], marker="x", s=90, color=color)
    ax.set_title(title or "Integrated I/Q clouds")
    ax.set_xlabel("integrated I")
    ax.set_ylabel("integrated Q")
    ax.axis("equal")
    ax.legend()


def make_report_figure(report: Mapping[str, Any]):
    """Create a figure for the analysis report error budget."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_error_budget(ax, report)
    fig.tight_layout()
    return fig
