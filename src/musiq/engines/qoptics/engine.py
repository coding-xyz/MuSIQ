"""QuantumOptics.jl engine implementation exposed through the musiq engine API."""

from __future__ import annotations

from musiq.common.schemas import ModelSpec
from musiq.engines.base import Engine
from musiq.engines.julia_runtime import JuliaRuntimeRunner


class QOpticsEngine(Engine):
    """QuantumOptics.jl-backed dynamics engine."""

    name = "qoptics"
    _runtime = JuliaRuntimeRunner(engine_package="quantumoptics")

    def run(self, model_spec: ModelSpec, run_options: dict | None = None):
        return self._runtime.run(model_spec, run_options=dict(run_options or {}))


__all__ = ["QOpticsEngine"]
