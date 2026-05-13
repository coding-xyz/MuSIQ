"""Shared IR JSON, hashing, and serialization helpers."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
import hashlib
import json

T = TypeVar("T")

@dataclass(slots=True)
class ParametricValue(Generic[T]):
    """
    A value that can either be a static constant or a reference to a 
    dimension in a sweep space.
    """
    value: T
    dim_name: str | None = None

    def resolve(self, sweep_space: dict[str, Any] | None = None) -> T:
        """Resolve the value based on the provided sweep space."""
        if self.dim_name is None or sweep_space is None:
            return self.value
        
        # This is a simplified resolution. In the actual StudyPlanner, 
        # this will be handled by the expansion logic.
        return self.value

@dataclass(slots=True)
class ParameterList:
    """
    Configuration of a single parameter to be swept.
    The key in ParameterSweepConfig.parameters serves as the identifier.
    """
    target: str
    values: list[Any]
    unit: str | None = None
    description: str | None = None

@dataclass(slots=True)
class ParameterSweepConfig:
    """
    Overall configuration for a parameter sweep.
    """
    parameters: dict[str, ParameterList] = field(default_factory=dict)
    mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class ParameterValues:
    """
    The actual values bound to a specific parameter point in a run.
    """
    parameter_id: str
    values: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class SweepDimension:
    """
    Definition of a single dimension in a parameter sweep.
    """
    values: list[Any] = field(default_factory=list)
    range: tuple[float, float] | None = None
    step: float | None = None
    type: str = "list"  # "list", "linear", "log"

    def expand(self) -> list[Any]:
        """Expand the dimension definition into a list of concrete values."""
        if self.type == "list":
            return self.values
        if self.type == "linear" and self.range and self.step:
            import numpy as np
            return np.arange(self.range[0], self.range[1] + self.step, self.step).tolist()
        return self.values

SCHEMA_VERSION = "1.0"
COMPLEX_JSON_TAG = "__musiq_complex__"


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 digest for a file."""
    p = Path(path)
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def to_json_dict(obj: Any) -> dict[str, Any]:
    """Convert dataclass object to JSON-serializable dictionary."""
    return asdict(obj)


def json_safe(value: Any) -> Any:
    """Convert nested values into a JSON-safe representation."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, complex):
        return {COMPLEX_JSON_TAG: [float(value.real), float(value.imag)]}
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def json_restore(value: Any) -> Any:
    """Restore nested values previously converted by ``json_safe``."""
    if isinstance(value, dict):
        if set(value.keys()) == {COMPLEX_JSON_TAG}:
            pair = list(value.get(COMPLEX_JSON_TAG, []) or [])
            if len(pair) >= 2:
                return complex(float(pair[0]), float(pair[1]))
        return {str(k): json_restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_restore(item) for item in value]
    return value


def make_series_payload(
    values: list[list[float]] | list[float],
    *,
    quantity: str,
    description: str,
    series_labels: list[str] | None = None,
    unit: str = "",
) -> dict[str, Any]:
    """Build a named time-series payload for classical trajectory channels."""
    if values and isinstance(values[0], (int, float)):
        rows = [[float(v)] for v in list(values)]  # type: ignore[index]
    else:
        rows = [[float(v) for v in row] for row in list(values)]  # type: ignore[arg-type]
    if series_labels is None and rows:
        series_labels = [f"s{i}" for i in range(len(rows[0]))]
    return {
        "quantity": str(quantity),
        "description": str(description),
        "unit": str(unit or ""),
        "series_labels": list(series_labels or []),
        "values": rows,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write UTF-8 pretty JSON file and return output path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


