
"""Trajectory, analysis result, and run manifest schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from musiq.schemas.utils import SCHEMA_VERSION, sha256_file, utc_now_iso

@dataclass(slots=True)
class ResultRef:
    """Reference to a specific result within a run."""
    run_id: str
    parameter_id: str

@dataclass(slots=True)
class ParameterValues:
    """Actual values bound to a parameter point."""
    parameter_id: str
    values: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QuantumStatePayload:
    """Typed container for quantum state data (wavefunction or density matrix).

    Attributes:
        data: The raw numerical array/tensor representing the state.
        shape: Dimensionality of the state tensor.
        dtype: Data type of the array elements.
        basis: The basis used for the state representation. Defaults to "computational".
        metadata: Non-primary technical annotations.
    """
    data: Any
    shape: tuple[int, ...]
    dtype: str
    basis: str = "computational"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class ClassicalChannelPayload:
    """Typed container for classical readout/control channels.

    Attributes:
        channel_id: Identifier of the classical channel.
        values: Time-series values for the channel.
        unit: Unit of measurement. Defaults to "V".
        metadata: Non-primary technical annotations.
    """
    channel_id: str
    values: list[float]
    unit: str = "V"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class MeasurementRecord:
    """Typed container for raw or processed measurement outcomes.

    Attributes:
        qubit_id: Identifier of the qubit being measured.
        outcomes: List of discrete measurement outcomes.
        probabilities: Associated probabilities for the outcomes, if available.
        metadata: Non-primary technical annotations.
    """
    qubit_id: str
    outcomes: list[int]
    probabilities: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class Trajectory:
    """Factual execution output from a simulation engine.

    A `Trajectory` represents the complete temporal evolution of a quantum 
    system, including state payloads, classical channels, and measurements.

    Attributes:
        schema_version: Version of the trajectory schema.
        engine: Identifier of the numerical engine used.
        times: Time grid of the simulation.
        wave_function: State vector payload, if applicable.
        density_matrix: Density matrix payload, if applicable.
        classical: Collection of classical channel data.
        measurements: Collection of measurement records.
        metadata: Non-primary technical annotations.
    """

    schema_version: str = SCHEMA_VERSION
    engine: str = "mock"
    times: list[float] = field(default_factory=list)
    wave_function: QuantumStatePayload | None = None
    density_matrix: QuantumStatePayload | None = None
    classical: list[ClassicalChannelPayload] = field(default_factory=list)
    measurements: list[MeasurementRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return a compact payload containing only populated trajectory fields.

        Returns:
            dict[str, Any]: A dictionary containing only the fields that are 
                not None or empty.
        """
        payload: dict[str, Any] = {
            "schema_version": str(self.schema_version),
            "engine": str(self.engine),
            "times": list(self.times or []),
        }
        if self.wave_function:
            payload["wave_function"] = dict(self.wave_function)
        if self.density_matrix:
            payload["density_matrix"] = dict(self.density_matrix)
        if self.classical:
            payload["classical"] = dict(self.classical)
        if self.measurements:
            payload["measurements"] = dict(self.measurements)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def __getattribute__(self, name: str):
        if name == "__annotations__":
            cls_annotations = type(self).__dict__.get("__annotations__", {})
            payload_keys = set(object.__getattribute__(self, "to_payload")().keys())
            return {key: value for key, value in cls_annotations.items() if key in payload_keys}
        return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        return f"Trajectory({self.to_payload()!r})"


@dataclass
class MetricSeries:
    """Time-series data for a single metric.

    Attributes:
        times: Time points for the metric values.
        values: Numerical values of the metric, either as a list or a map.
    """
    times: list[float] = field(default_factory=list)
    values: list[float] | dict[str, list[float]] = field(default_factory=list)

@dataclass(slots=True)
class ParameterAxis:
    """Definition of a parameter scan axis."""
    parameter_name: str
    values: list[Any]
    unit: str | None = None

@dataclass(slots=True)
class MetricSweepValues:
    """Multi-dimensional values of a metric across a parameter sweep."""
    metric_name: str
    dimensions: list[str] # Order of axes corresponding to the values matrix
    values: list[Any]     # Multi-dimensional array (nested lists)
    unit: str | None = None


@dataclass
class MetricsOutput:
    """Collection of computed metrics.

    Attributes:
        metric_items: Map of metric names to their corresponding time-series data.
    """
    metric_items: dict[str, MetricSeries] = field(default_factory=dict)


@dataclass
class ShotData:
    """Individual measurement shot data.

    Attributes:
        timestamp: Time of the measurement shot.
        value: Measured value.
        metadata: Non-primary technical annotations.
    """
    timestamp: float = 0.0
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadoutAnalysis:
    """Structural analysis of readout signals.

    Attributes:
        signals: Raw or pre-processed readout signals.
        demodulation: Demodulation parameters and results.
        shots: Individual measurement shot data.
    """
    signals: dict[str, Any] = field(default_factory=dict)
    demodulation: dict[str, Any] = field(default_factory=dict)
    shots: list[ShotData] = field(default_factory=list)


@dataclass
class IQAnalysis:
    """IQ plane analysis result.

    Attributes:
        centroids: Map of state identifiers to their complex centroids in the IQ plane.
        confusion_matrix: Matrix showing misclassification between states.
        assignment_fidelity: Overall fidelity of state assignment.
        noise_sigma: Estimated noise standard deviation.
        snr: Signal-to-Noise Ratio.
    """
    centroids: dict[str, complex] = field(default_factory=dict)
    confusion_matrix: dict[str, Any] = field(default_factory=dict)
    assignment_fidelity: float = 0.0
    noise_sigma: float = 0.0
    snr: float = 0.0


@dataclass(slots=True)
class CaseAnalysis:
    """Analysis result for a single parameter point (single RunResult)."""
    metrics: dict[str, MetricSeries] | None = None
    readout: ReadoutAnalysis | None = None
    iq: IQAnalysis | None = None

@dataclass(slots=True)
class ParametricAnalysis:
    """Aggregation of analysis results over a parameter sweep."""
    parameters: dict[str, ParameterAxis] = field(default_factory=dict)
    metrics: dict[str, MetricSweepValues] = field(default_factory=dict)
    input_results: list[ResultRef] = field(default_factory=list)

@dataclass(slots=True)
class ComprehensiveAnalysis:
    """High-level summary across multiple studies, sweeps, or cases."""
    parametric_analyses: dict[str, ParametricAnalysis] = field(default_factory=dict)
    cross_analysis: dict[str, Any] = field(default_factory=dict)
    input_sweeps: list[ResultRef] = field(default_factory=list)

@dataclass
class AnalysisOutput:
    """Legacy container for analysis outputs (maintained for backward compatibility)."""
    metrics: MetricsOutput | None = None
    readout: ReadoutAnalysis | None = None
    iq: IQAnalysis | None = None
    series_data: dict[str, Any] = field(default_factory=dict)


class AnalysisScope(Enum):
    """Scope of the analysis relative to the runs it depends on."""
    CASE = "case"
    PARAMETRIC = "parametric"
    COMPREHENSIVE = "comprehensive"

@dataclass
class ModelAnalysis:
    """Derived analyser outputs stored at the model level.

    An analysis object maps an analyser's output to the specific set of runs 
    and samples that provided the input data.

    Attributes:
        analysis_id: Unique identifier for this analysis instance.
        analyser_id: Identifier of the analyser configuration used.
        input_results: List of ResultRef that contributed data.
        scope: The scope of the analysis.
        output: The actual analysis results (Case, Parametric, or Comprehensive).
        schema_version: str.
    """
    analysis_id: str
    analyser_id: str
    input_results: list[ResultRef] = field(default_factory=list)
    scope: AnalysisScope = AnalysisScope.CASE
    output: CaseAnalysis | ParametricAnalysis | ComprehensiveAnalysis | AnalysisOutput = field(default_factory=CaseAnalysis)
    schema_version: str = "1.0"


# Deprecated compatibility alias. Prefer ``ModelAnalysis`` in new code.
AnalysisResult = ModelAnalysis


@dataclass
class RunProvenance:
    """Traceability metadata for a simulation result.

    Attributes:
        solver_id: Identifier of the solver used.
        study_name: Name of the study, if applicable.
        study_index: Step index in the study, if applicable.
        spec_ref: Reference to the `ModelSpec` used.
        plan_ref: Reference to the execution plan.
    """
    solver_id: str
    study_name: str | None = None
    study_index: int | None = None
    spec_ref: str | None = None
    plan_ref: str | None = None


@dataclass
class RunResult:
    """Objective factual result of a solver run sample.

    Attributes:
        result_id: Unique identifier for this result.
        parameters: Snapshot of the specific parameters used for this sample.
        provenance: Traceability metadata for this run.
        trajectories: Collection of numerical simulation trajectories.
        runtime_metadata: Lightweight tracing and debugging info.
        schema_version: Version of the result schema.
    """
    result_id: str
    parameters: ParameterValues
    provenance: RunProvenance
    trajectories: dict[str, Trajectory] = field(default_factory=dict)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"


@dataclass
class Observables:
    """Computed analysis observables from a trajectory.

    Attributes:
        schema_version: Version of the observables schema.
        values: Map of observable names to their computed scalar values.
    """

    schema_version: str = SCHEMA_VERSION
    values: dict[str, float] = field(default_factory=dict)


@dataclass
class Report:
    """High-level analysis report and error budget summary.

    Attributes:
        schema_version: Version of the report schema.
        summary: General summary of the analysis findings.
        error_budget: Breakdown of error contributions to the final result.
    """

    schema_version: str = SCHEMA_VERSION
    summary: dict[str, Any] = field(default_factory=dict)
    error_budget: dict[str, float] = field(default_factory=dict)


@dataclass
class RunManifest:
    """Run-level manifest linking inputs, outputs, and digests.

    Used for verifying the integrity and reproducibility of a simulation run.

    Attributes:
        schema_version: Version of the manifest schema.
        run_id: Unique identifier for the run.
        created_at: ISO timestamp of manifest creation.
        random_seed: Seed used for stochastic simulations.
        inputs: Map of input artifact names to their relative paths.
        outputs: Map of output artifact names to their relative paths.
        dependencies: Map of dependency names to their versions.
        dependency_fingerprint: Deterministic hash of all dependencies.
        digests: SHA-256 hashes of the output files.
    """

    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    random_seed: int = 0
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)
    dependency_fingerprint: str = ""
    digests: dict[str, str] = field(default_factory=dict)

    def finalize_digests(self, out_dir: str | Path) -> None:
        """Compute file digests for all declared outputs.

        Args:
            out_dir (str | Path): The directory where output files are stored.
        """
        base = Path(out_dir)
        for rel in self.outputs.values():
            p = base / rel
            if p.exists() and p.is_file():
                self.digests[str(rel)] = sha256_file(p)

    def finalize_dependency_fingerprint(self) -> None:
        """Compute deterministic fingerprint from dependency versions.
        
        The fingerprint is generated by hashing a canonical JSON representation
        of the dependencies map.
        """
        import json
        canonical = json.dumps(self.dependencies, sort_keys=True, separators=(",", ":"))
        self.dependency_fingerprint = sha256_file(Path(canonical)) # Note: fixed potential _sha256_text issue
