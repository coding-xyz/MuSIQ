"""Central analysis-step definitions and metric request helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AnalysisStepDefinition:
    """Declarative description of one supported analysis step."""

    canonical_name: str
    level: str
    default_metrics: tuple[str, ...] = ()
    metric_source: str = "typed"
    aliases: tuple[str, ...] = ()


def _build_step_registry() -> dict[str, AnalysisStepDefinition]:
    definitions = [
        AnalysisStepDefinition(
            canonical_name="single_qubit_analysis",
            level="CASE",
            default_metrics=("population", "leakage", "coherence_01"),
            metric_source="registry",
            aliases=("state_analysis",),
        ),
        AnalysisStepDefinition(
            canonical_name="readout_analysis",
            level="CASE",
            default_metrics=(
                "integrated_iq",
                "intracavity_field",
                "outgoing_field",
                "complex_envelope",
                "rf_signal",
                "adc_signal",
                "trajectory_id",
            ),
            metric_source="typed",
        ),
        AnalysisStepDefinition(
            canonical_name="rabi_sweep_analysis",
            level="PARAMETRIC",
            default_metrics=("final_P0", "final_P1", "final_fidelity", "final_coherence_01"),
            metric_source="derived",
        ),
        AnalysisStepDefinition(
            canonical_name="iq_analysis",
            level="COMPREHENSIVE",
            default_metrics=(
                "iq_clouds",
                "discrimination_line",
                "centroids",
                "confusion_matrix",
                "readout_fidelity",
                "snr",
            ),
            metric_source="typed",
        ),
    ]
    registry: dict[str, AnalysisStepDefinition] = {}
    for definition in definitions:
        registry[definition.canonical_name.lower()] = definition
        for alias in definition.aliases:
            registry[str(alias).strip().lower()] = definition
    return registry


ANALYSIS_STEP_REGISTRY = _build_step_registry()


def get_analysis_step_definition(name: str | None) -> AnalysisStepDefinition | None:
    """Return the normalized analysis-step definition for one step name."""

    key = str(name or "").strip().lower()
    if not key:
        return None
    return ANALYSIS_STEP_REGISTRY.get(key)


def collect_analysis_metrics(
    cfg: dict[str, Any] | None,
    *,
    level: str,
    metric_source: str | None = None,
) -> list[str | dict[str, Any]]:
    """Collect requested metrics from hierarchical analysis steps."""

    payload = dict(cfg or {})
    wanted_level = str(level or "").strip().upper()
    if not wanted_level:
        return []

    requested: list[str | dict[str, Any]] = []
    seen: set[str] = set()

    def _append_metric(item: str | dict[str, Any]) -> None:
        if isinstance(item, str):
            name = str(item).strip()
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            return
        if not name:
            return
        lowered = name.lower()
        if lowered in seen:
            return
        requested.append(item)
        seen.add(lowered)

    for raw_step in list(payload.get("analysis", []) or []):
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        definition = get_analysis_step_definition(step.get("name"))
        step_level = str(step.get("level") or (definition.level if definition else "")).strip().upper()
        if step_level != wanted_level:
            continue
        if metric_source is not None and definition is not None and definition.metric_source != metric_source:
            continue
        step_metrics = list(step.get("metrics", []) or [])
        if not step_metrics and definition is not None:
            step_metrics = list(definition.default_metrics)
        for metric in step_metrics:
            _append_metric(metric)
    return requested


__all__ = [
    "ANALYSIS_STEP_REGISTRY",
    "AnalysisStepDefinition",
    "collect_analysis_metrics",
    "get_analysis_step_definition",
]
