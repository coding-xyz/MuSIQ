"""Analysis pass and metric registry orchestration."""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any, Callable, Protocol

from musiq.common.schemas import ModelSpec, Trajectory

class AnalysisLevel(Enum):
    CASE = auto()
    COMPREHENSIVE = auto()

class AnalysisKind(Enum):
    # CASE level kinds
    SINGLE_QUBIT = auto()
    READOUT = auto()
    # COMPREHENSIVE level kinds
    IQ = auto()

class AnalysisHandler(Protocol):
    """Protocol for a handler that can execute a specific kind of analysis."""
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

def _build_rev(name: str, *, kind: str) -> str:
    stamp = datetime.now(timezone.utc).isoformat()
    return hashlib.sha256(f"{kind}:{name}:{stamp}".encode("utf-8")).hexdigest()[:12]

@dataclass
class KindEntry:
    level: AnalysisLevel
    kind: AnalysisKind
    handler: AnalysisHandler
    revision: str

class AnalysisRegistry:
    """Registry for hierarchical analysis kinds."""

    def __init__(self):
        self._kinds: dict[tuple[AnalysisLevel, AnalysisKind], KindEntry] = {}

    def register_kind(
        self, 
        level: AnalysisLevel, 
        kind: AnalysisKind, 
        handler: AnalysisHandler
    ) -> str:
        """Register an analysis kind and return its revision ID."""
        rev = _build_rev(kind.name, kind="analysis_kind")
        self._kinds[(level, kind)] = KindEntry(
            level=level,
            kind=kind,
            handler=handler,
            revision=rev,
        )
        return rev

    def get_handler(self, level: AnalysisLevel, kind: AnalysisKind) -> AnalysisHandler:
        """Fetch the handler for a specific level and kind."""
        if (level, kind) not in self._kinds:
            raise KeyError(f"No handler registered for {level.name}.{kind.name}")
        return self._kinds[(level, kind)].handler

    def get_revision(self, level: AnalysisLevel, kind: AnalysisKind) -> str:
        """Fetch the revision ID for a specific kind."""
        if (level, kind) not in self._kinds:
            raise KeyError(f"No handler registered for {level.name}.{kind.name}")
        return self._kinds[(level, kind)].revision


class AnalysisRunner:
    """Compatibility wrapper around the hierarchical registry.

    Older callers expect a runner object with a ``run`` method. Keep a minimal
    implementation here so imports and simple call sites continue to work while
    analysis dispatch migrates to ``dispatcher.dispatch_analysis``.
    """

    def __init__(self, registry: AnalysisRegistry):
        self.registry = registry

    def run(self, *args: Any, **kwargs: Any) -> Any:
        level = kwargs.pop("level", None)
        kind = kwargs.pop("kind", None)
        if level is None or kind is None:
            raise ValueError("AnalysisRunner.run requires `level` and `kind`.")
        if isinstance(level, str):
            level = AnalysisLevel[level.upper()]
        if isinstance(kind, str):
            kind = AnalysisKind[kind.upper()]
        handler = self.registry.get_handler(level, kind)
        return handler(*args, **kwargs)

class MetricRegistry:
    """Registry for named analyser metrics."""

    def __init__(self):
        self._entries: dict[str, _MetricEntry] = {}

    def register(
        self,
        name: str,
        callable_obj: Callable[[Trajectory, ModelSpec, dict[str, Any] | None, dict[str, Any] | None], dict[str, Any]],
        schema_out: str = "Metric@1.0",
    ) -> str:
        metric_name = str(name).strip().lower()
        if not metric_name:
            raise ValueError("Metric name must be non-empty.")
        metric_rev = _build_rev(metric_name, kind="metric")
        self._entries[metric_name] = _MetricEntry(
            name=metric_name,
            callable_obj=callable_obj,
            schema_out=schema_out,
            metric_rev=metric_rev,
        )
        return metric_rev

    def get(self, name: str) -> _MetricEntry:
        metric_name = str(name).strip().lower()
        if metric_name not in self._entries:
            raise KeyError(f"Unknown metric: {name}")
        return self._entries[metric_name]

    def has(self, name: str) -> bool:
        return str(name).strip().lower() in self._entries

    def names(self) -> list[str]:
        return sorted(self._entries.keys())

@dataclass
class _MetricEntry:
    name: str
    callable_obj: Callable[[Trajectory, ModelSpec, dict[str, Any] | None, dict[str, Any] | None], dict[str, Any]]
    schema_out: str
    metric_rev: str

__all__ = [
    "AnalysisLevel",
    "AnalysisKind",
    "AnalysisRegistry",
    "MetricRegistry",
    "AnalysisHandler",
    "AnalysisRunner",
]
