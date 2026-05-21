"""Top-level engine-neutral simulation model schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from musiq.common.channels import canonical_readout_protocol
from musiq.schemas.circuit import CircuitSpec
from musiq.schemas.hamiltonian import HamiltonianSpec, control_dict_to_hamiltonian_term
from musiq.schemas.noise import NoiseSpec
from musiq.schemas.readout import ReadoutSpec
from musiq.schemas.solver import FrameSpec, SolverSpec, TimeSpec
from musiq.schemas.study import AnalysisRequestSpec, StudySpec
from musiq.schemas.system import (
    ModelStructureSpec,
    SystemCavitySpec,
    SystemCouplingSummarySpec,
    SystemQubitSpec,
    SystemSpec,
)

# --- Run-Scoped Containers ---

from enum import Enum, auto

class RunStatus(Enum):
    """Execution status of a model run.

    Attributes:
        PENDING: Run is queued and waiting for execution.
        RUNNING: Run is currently being processed by an engine.
        COMPLETED: Run finished successfully.
        FAILED: Run terminated with an error.
    """
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass(slots=True)
class RunIdentity:
    """Unique identifier for a specific execution run.

    Attributes:
        run_id: Unique UUID or string identifying this specific run.
        solver_id: Identifier of the solver configuration used.
        study_name: Name of the study if this run is part of a study.
        study_index: Index of the step within the study.
    """
    run_id: str
    solver_id: str
    study_name: str | None = None
    study_index: int | None = None

@dataclass(slots=True)
class RunArtifacts:
    """Compiled and intermediate artifacts for a run.

    This container holds all non-factual outputs produced during the 
    compilation and lowering phase, before the numerical engine is invoked.

    Attributes:
        circuit: The original parsed circuit specification.
        normalized_circuit: The circuit after normalization and optimization.
        model_spec: The engine-neutral domain model (authoritative truth).
        pulse_ir: Intermediate representation of pulses for the hardware.
        executable_model: The lowered model ready for engine consumption.
        compile_report: Metadata and logs from the compilation process.
        decoder_outputs: Results from QEC decoding if applicable.
        timings: Calculated time offsets and durations.
    """
    circuit: CircuitSpec | None = None
    normalized_circuit: CircuitSpec | None = None
    model_spec: ModelSpec | None = None
    pulse_ir: "PulseIR | None" = None
    executable_model: "ExecutableModel | None" = None
    compile_report: dict[str, Any] = field(default_factory=dict)
    decoder_outputs: "DecoderOutputs | None" = None
    timings: dict[str, float] = field(default_factory=dict)

@dataclass(slots=True)
class ModelRun:
    """Authoritative home for one compilation unit of a solver/study combination.

    A `ModelRun` encapsulates a specific set of compiled artifacts (e.g., ModelSpec, 
    executable model) that can be reused across multiple numerical execution 
    samples (RunResults) that differ only by parameter values.

    Attributes:
        identity: The unique identity of this compilation unit.
        runtime_task: The runtime contract (input) for this execution.
        artifacts: Compiled and intermediate products (IR) shared by all samples.
        results: A collection of factual numerical outputs (samples).
        status: Current execution status of the compilation/run process.
        started_at: Epoch timestamp of start.
        finished_at: Epoch timestamp of completion.
        error: Error message if the run failed.
    """
    identity: RunIdentity
    runtime_task: "Task"
    artifacts: RunArtifacts = field(default_factory=RunArtifacts)
    results: dict[str, "RunResult"] = field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

@dataclass(slots=True)
class ModelManifest:
    """Version and layout metadata for the persisted model.

    Used to ensure compatibility and provenance when loading models from disk.

    Attributes:
        schema_version: Version of the model schema.
        created_at: ISO timestamp of model creation.
        config_layout: Map of configuration keys to their versions/sources.
        state_snapshot: Snapshot of session state at the time of manifest creation.
        provenance: Traceability data regarding the model's origin.
    """
    schema_version: str = "3.0"
    created_at: str = ""
    config_layout: dict[str, str] = field(default_factory=dict)
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

# --- Domain Model ---

@dataclass
class ModelSpec:
    """Engine-neutral simulation model specification.

    ``ModelSpec`` is the structured boundary between backend lowering and
    numerical engines. It describes the circuit context, solver request, time
    grid, frame, physical system, Hamiltonian, noise, readout, and analysis
    request without depending on a backend-private runtime representation.

    Attributes:
        circuit: The circuit specification associated with the simulation.
        solver: Solver specification (e.g., SE, ME, SME).
        time: Time grid specification (dt, t_end).
        frame: Reference frame and RWA settings.
        system: Physical system description (qubits, resonators).
        hamiltonian: System Hamiltonian including controls and couplings.
        noise: Noise model and dissipation channels.
        readout: Readout protocol and chain specification.
        analysis_request: Requested post-processing analysis.
        study: Study context if this is part of a parameter sweep.
        metadata: Non-primary technical annotations and debug notes.
    """

    circuit: "CircuitSpec | None" = None
    solver: "SolverSpec | str" = "se"
    time: "TimeSpec" = field(default_factory=lambda: TimeSpec())
    frame: "FrameSpec" = field(default_factory=lambda: FrameSpec())
    system: "SystemSpec" = field(default_factory=lambda: SystemSpec())
    hamiltonian: "HamiltonianSpec" = field(default_factory=lambda: HamiltonianSpec())
    noise: "NoiseSpec" = field(default_factory=lambda: NoiseSpec())
    readout: "ReadoutSpec | None" = None
    analysis_request: "AnalysisRequestSpec | None" = None
    study: "StudySpec | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize string solver shorthands into ``SolverSpec`` objects."""
        if isinstance(self.solver, str):
            self.solver = SolverSpec(mode=str(self.solver))

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension requested by the system model."""
        return int(self.system.dimension)

    @property
    def dt(self) -> float:
        """Simulation timestep in seconds."""
        return float(self.time.dt_s)

    @property
    def t_end(self) -> float:
        """Simulation end time in seconds."""
        return float(self.time.t_end_s)

    @property
    def solver_mode(self) -> str:
        """Normalized solver mode token such as ``se``, ``me``, or ``sme``."""
        return str(self.solver.mode if isinstance(self.solver, SolverSpec) else self.solver).strip().lower()

def model_spec_from_runtime_dict(
    solver: str = "se",
    dimension: int = 2,
    t_end: float = 1.0,
    dt: float = 0.1,
    model: dict[str, Any] | None = None,
    **kwargs,
) -> ModelSpec:
    """Legacy helper to create a ModelSpec from a flat runtime dictionary.
    
    Used primarily by tests to avoid constructing the full nested hierarchy.
    """
    from musiq.schemas.solver import TimeSpec
    
    spec = ModelSpec(
        solver=solver,
        time=TimeSpec(dt_s=dt, t_end_s=t_end),
        metadata=model or {},
        **kwargs
    )
    # Dimension is usually a property of the system; we store it in metadata 
    # for this shim to avoid forcing a complex SystemSpec construction.
    spec.metadata["dimension"] = dimension
    return spec
