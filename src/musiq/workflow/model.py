"""Model-first workflow API."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from musiq.analysis import MetricRegistry, build_default_metric_registry
from musiq.schemas.model import ModelRun, ModelSpec, RunArtifacts, RunIdentity, ModelManifest
from musiq.schemas.results import AnalysisScope, ModelAnalysis, RunResult, Trajectory
from musiq.workflow.contracts import (
    AnalyserConfig,
    CircuitConfig,
    DeviceConfig,
    ProfileConfig,
    PulseAcquisitionConfig,
    PulseChannelConfig,
    PulseConfig,
    PulseTimingConfig,
    SolverConfig,
    Task,
    WorkflowFeatureFlags,
    WorkflowOutputOptions,
)
from musiq.workflow.task_io import (
    load_analyser_config_file,
    load_circuit_config_file,
    load_device_config_file,
    load_pulse_config_file,
    load_solver_config_file,
    load_config,
)

from musiq.workflow.model_utils import (
    _UNSET,
    bind_loaded_analyser,
)
from musiq.schemas.utils import ParameterList, ParameterSweepConfig
from musiq.workflow.model_execution import (
    build_solver,
    build_study,
    find_run_id,
    run,
    run_all,
    run_analysis,
    run_engine,
    run_profile,
    run_solver,
    run_study,
)
from musiq.workflow.model_persistence import load_model as _load_model_impl, save_model as _save_model_impl

@dataclass(slots=True)
class ModelConfig:
    """Aggregated configuration for the quantum simulation workflow."""
    circuits: dict[str, CircuitConfig]
    devices: dict[str, DeviceConfig]
    pulses: dict[str, PulseConfig]
    solvers: dict[str, SolverConfig]
    analysers: dict[str, AnalyserConfig] = field(default_factory=dict)
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)
    parameter_sweep: ParameterSweepConfig | None = None
    target: str | list[str] = "trajectory"
    features: WorkflowFeatureFlags = field(default_factory=WorkflowFeatureFlags)
    output: WorkflowOutputOptions = field(default_factory=WorkflowOutputOptions)
    tags: list[str] = field(default_factory=list)

@dataclass(slots=True)
class ModelState:
    """Lightweight session state for the workflow."""
    last_out_dir: str | None = None
    last_run_id: str | None = None

@dataclass(slots=True)
class ModelRegistry:
    """Central registry for metrics and other shared resources."""
    metrics: MetricRegistry = field(default_factory=build_default_metric_registry)


@dataclass(slots=True)
class Solver:
    """Wrapper for solver configuration and execution."""
    model: Any
    solver_id: str
    config: SolverConfig

    def build_study(self, *, study_name: str | None = None, study_index: int | None = None, tag: str | None = None) -> list[str]:
        """Compile one specific study step into ``model.runs`` without running the engine."""
        return build_study(self.model, solver_id=self.solver_id, study_name_val=study_name, study_index=study_index, tag=tag)

    def build(self, tag: str | None = None) -> list[str]:
        """Compile every study step for this solver without running the engine."""
        return build_solver(self.model, solver_id=self.solver_id, tag=tag)

    def run_study(self, *, study_name: str | None = None, study_index: int | None = None) -> str:
        """Compile and solve one specific study step into ``model.runs``."""
        return run_study(self.model, solver_id=self.solver_id, study_name_val=study_name, study_index=study_index)

    def run_engine(self, tag: str | None = None) -> list[str]:
        """Run the numerical engine for every study step of this solver."""
        return run_engine(self.model, solver_id=self.solver_id, tag=tag)

@dataclass(slots=True)
class Profile:
    """Wrapper for profile configuration and execution."""
    model: Any
    profile_id: str
    config: ProfileConfig

    def build_solver(self, solver_id: str | None = None, tag: str | None = None) -> list[str]:
        """Compile one configured solver without running the engine."""
        original_profiles = dict(self.model.config.profiles)
        try:
            self.model.config.profiles = {self.profile_id: self.config}
            return build_solver(self.model, solver_id=solver_id or self.config.solver_id, tag=tag)
        finally:
            self.model.config.profiles = original_profiles

    def run_solver(self, solver_id: str | None = None, tag: str | None = None) -> list[str]:
        """Compile and solve one configured solver, running every study step by default."""
        original_profiles = dict(self.model.config.profiles)
        try:
            self.model.config.profiles = {self.profile_id: self.config}
            return run_solver(self.model, solver_id=solver_id or self.config.solver_id, tag=tag)
        finally:
            self.model.config.profiles = original_profiles

    def run_engine(self, solver_id: str | None = None, tag: str | None = None) -> list[str]:
        """Run the numerical engine for one configured solver."""
        original_profiles = dict(self.model.config.profiles)
        try:
            self.model.config.profiles = {self.profile_id: self.config}
            return run_engine(self.model, solver_id=solver_id or self.config.solver_id, tag=tag)
        finally:
            self.model.config.profiles = original_profiles

    def run_analysis(self, analyser_id: str | None = None, study_name: str | None = None, tag: str | None = None, run_ids: list[str] | None = None) -> None:
        """Run one analyser against every matching study trajectory into ``model.analyses``."""
        run_analysis(self.model, analyser_id=analyser_id or self.config.analyser_id, study_name_val=study_name, tag=tag, run_ids=run_ids)

@dataclass(slots=True)
class Model:
    """Top-down editable model object."""

    config: ModelConfig
    state: ModelState = field(default_factory=ModelState)
    registry: ModelRegistry = field(default_factory=ModelRegistry)
    manifest: ModelManifest = field(default_factory=ModelManifest)
    runs: dict[str, ModelRun] = field(default_factory=dict)
    analyses: dict[str, ModelAnalysis] = field(default_factory=dict)

    @property
    def circuit(self) -> CircuitConfig:
        return next(iter(self.config.circuits.values()))
    @property
    def device(self) -> DeviceConfig: 
        return next(iter(self.config.devices.values()))
    @property
    def solvers(self) -> dict[str, Solver]: 
        return {sid: Solver(self, sid, cfg) for sid, cfg in self.config.solvers.items()}
    @property
    def pulse(self) -> PulseConfig: 
        return next(iter(self.config.pulses.values()))
    @property
    def analysers(self) -> dict[str, AnalyserConfig]: 
        return self.config.analysers
    @property
    def metric_registry(self) -> MetricRegistry: return self.registry.metrics
    @property
    def out_dir(self) -> str | None: return self.state.last_out_dir
    @out_dir.setter
    def out_dir(self, value: str | None): self.state.last_out_dir = value

    def __repr__(self) -> str:
        run_summary = sorted(self.runs.keys())
        analysis_ids = sorted(self.analyses.keys())
        return (
            'Model('
            f'solvers={sorted(self.config.solvers.keys())}, '
            f'analysers={[(analyser_id, cfg.solver_id) for analyser_id, cfg in sorted(self.analysers.items())]}, '
            f'runs={run_summary}, '
            f'analyses={analysis_ids}'
            ')'
        )

    def _clear_solver_results(self, solver_id: str) -> None:
        self.runs = {
            run_id: run_obj
            for run_id, run_obj in self.runs.items()
            if str(run_obj.identity.solver_id) != str(solver_id)
        }

    def get_trajectory(self, solver_id: str | None = None, *, study_name: str | None = None) -> Trajectory | None:
        from musiq.workflow.model_utils import require_solver_id
        selected_solver_id = require_solver_id(self, solver_id)

        matching_runs = [
            run_obj
            for run_obj in self.runs.values()
            if str(run_obj.identity.solver_id) == selected_solver_id
            and (study_name is None or str(run_obj.identity.study_name or "").strip() == str(study_name).strip())
        ]
        if not matching_runs:
            return None
        run_obj = matching_runs[0]

        first_res = next(iter(run_obj.results.values()), None)
        return next(iter(first_res.trajectories.values()), None) if first_res else None

    def get_analysis(self, *, analyser_id: str | None = None, study_name: str | None = None) -> ModelAnalysis | None:
        from musiq.workflow.model_utils import require_analyser_id
        selected_analyser_id = require_analyser_id(self, analyser_id)
        matching = [
            analysis
            for analysis in self.analyses.values()
            if str(analysis.analyser_id) == selected_analyser_id
        ]
        if study_name is None:
            if not matching:
                return None
            for scope in (AnalysisScope.COMPREHENSIVE, AnalysisScope.PARAMETRIC):
                preferred = next((analysis for analysis in matching if analysis.scope == scope), None)
                if preferred is not None:
                    return preferred
            return matching[0] if len(matching) == 1 else None
        wanted = str(study_name).strip()
        for analysis in matching:
            for ref in analysis.input_results:
                run_obj = self.runs.get(str(ref.run_id))
                if run_obj and str(run_obj.identity.study_name or "").strip() == wanted:
                    return analysis
        return None

    def find_analysis_for_run(self, run_id: str) -> ModelAnalysis | None:
        """Find the first analysis that depends on the given run ID."""
        return next((a for a in self.analyses.values() if any(ref.run_id == run_id for ref in a.input_results)), None)

    def add_solver(self, solver_id: str, solver_cfg: SolverConfig) -> None:
        self.config.solvers[str(solver_id)] = solver_cfg

    def add_analyser(self, analyser_id: str, solver_id: str, analyser_cfg: AnalyserConfig) -> None:
        from musiq.workflow.model_utils import require_solver_id
        bound_cfg = AnalyserConfig(**analyser_cfg.to_payload())
        bound_cfg.solver_id = require_solver_id(self, solver_id)
        self.analysers[str(analyser_id)] = bound_cfg

    def register_metric(self, name: str, callable_obj, schema_out: str = 'Metric@1.0') -> str:
        return self.metric_registry.register(name, callable_obj, schema_out=schema_out)

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the current model state to a directory."""
        return _save_model_impl(self, path)

    def copy(self, *, include_results: bool = True) -> Model:
        """Return a detached copy of this model.

        When ``include_results`` is ``False``, only configuration and registry
        state are copied; run results, analyses, and session state are reset.
        """
        return Model(
            config=deepcopy(self.config),
            state=deepcopy(self.state) if include_results else ModelState(),
            registry=deepcopy(self.registry),
            manifest=deepcopy(self.manifest) if include_results else ModelManifest(),
            runs=deepcopy(self.runs) if include_results else {},
            analyses=deepcopy(self.analyses) if include_results else {},
        )

    def profile(self, profile_id: str) -> Profile:
        """Get a profile wrapper for the specified profile ID."""
        if profile_id not in self.config.profiles:
            raise KeyError(f"Profile `{profile_id}` not found in model configuration.")
        return Profile(self, str(profile_id), self.config.profiles[profile_id])

    def run_all(self) -> None:
        """Run every configured solver and then every configured analyser."""
        run_all(self)

    def build(self, solver_id: str | None = None, *, study_name: str | None = None, study_index: int | None = None, tag: str | None = None) -> list[str]:
        """Compile workflow artifacts without running the numerical engine."""
        if study_name is not None or study_index is not None:
            return build_study(self, solver_id=solver_id, study_name_val=study_name, study_index=study_index, tag=tag)
        if solver_id is not None:
            return build_solver(self, solver_id=solver_id, tag=tag)
        built: list[str] = []
        for profile_id in sorted(self.config.profiles.keys()):
            built.extend(self.profile(profile_id).build_solver(tag=tag))
        return built

    def run_engine(self, solver_id: str | None = None, *, tag: str | None = None) -> list[str]:
        """Run the numerical engine using built artifacts, auto-building when needed."""
        return run_engine(self, solver_id=solver_id, tag=tag)

    def run(self) -> None:
        """Run all configured solvers and analysers."""
        run(self)

    def run_profile(self, profile_id: str, tag: str | None = None) -> None:
        """Run one configured profile end-to-end."""
        run_profile(self, profile_id, tag=tag)

    def calibrate(
        self,
        config=None,
        **overrides: Any,
    ):
        """Auto-calibrate supported pulse parameters on this model."""
        from musiq.calibrate import CalibrationConfig, calibrate_model

        if config is not None and not isinstance(config, CalibrationConfig):
            raise TypeError("Model.calibrate() expects a CalibrationConfig instance or None.")
        return calibrate_model(self, config, **overrides)


def _ensure_config(value: Any, config_type: str, base_dir: Path | None = None) -> Any:
    """Ensure the value is a proper Config object, converting from path or dict if necessary."""
    type_map = {
        "circuit": CircuitConfig,
        "solver": SolverConfig,
        "device": DeviceConfig,
        "pulse": PulseConfig,
        "analyser": AnalyserConfig,
        "profile": ProfileConfig,
    }
    cls = type_map.get(config_type)
    if cls and isinstance(value, cls):
        return value
    
    # For pulse, it might be a dict that's already 'processed' but not a PulseConfig object
    # PulseConfig is a dataclass, so we check that.
    
    return load_config(value, config_type, base_dir=base_dir)


def _normalize_resources(
    value: Any | None,
    *,
    config_type: str,
    resource_name: str,
    required: bool,
    singleton_id: str = "default",
) -> dict[str, Any]:
    if value is None:
        if required:
            raise ValueError(f"{resource_name}_config is required.")
        return {}
    
    if isinstance(value, dict):
        # We need to distinguish between a config payload (dict) and a mapping of ids (dict[str, Any])
        # A mapping of ids typically has keys that aren't the top-level keys of the config.
        # However, the safest way is to check if the keys match any known config top-level keys
        # or if it's being passed as a single resource.
        
        # In create_model, if it's a dict, we treat it as {id: config_value}
        # unless it's the only value and looks like a payload.
        # But to keep it simple and consistent with previous behavior:
        # if it's a dict, we treat it as a mapping of IDs.
        
        normalized: dict[str, Any] = {}
        for key, val in value.items():
            key_str = str(key).strip()
            if not key_str:
                raise ValueError(f"{resource_name} config IDs must be non-empty strings.")
            normalized[key_str] = _ensure_config(val, config_type)
        if required and not normalized:
            raise ValueError(f"{resource_name}_config is required.")
        return normalized
    
    # Singleton case: value is a path, dict payload, or config object
    return {singleton_id: _ensure_config(value, config_type)}


def _load_named_configs(
    sources: dict[str, str],
    *,
    loader,
) -> dict[str, Any]:
    return {
        config_id: loader(path)
        for config_id, path in sources.items()
    }


def _normalize_analyser_paths(
    analyser_config: Any | None | object,
) -> dict[str, Any]:
    if analyser_config is _UNSET:
        return {}
    return _normalize_resources(
        analyser_config,
        config_type="analyser",
        resource_name="analyser",
        required=False,
        singleton_id="analyser_0",
    )


def _normalize_pulse_config(raw_pulse: Any) -> PulseConfig:
    if isinstance(raw_pulse, PulseConfig):
        return raw_pulse
    if not isinstance(raw_pulse, dict):
        raise TypeError(f"Unsupported pulse config payload type: {type(raw_pulse).__name__}")

    def _split_payload(raw: dict[str, Any], known: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
        known_items = {k: v for k, v in raw.items() if k in known}
        extras = {k: v for k, v in raw.items() if k not in known}
        return known_items, extras

    known_fields = {"acquisition", "timing", "channels", "extras"}
    known_args = {k: v for k, v in raw_pulse.items() if k in known_fields}
    extra_args = {k: v for k, v in raw_pulse.items() if k not in known_fields}
    extras = known_args.get("extras") or {}
    if isinstance(extras, dict):
        extras.update(extra_args)
    else:
        extras = extra_args

    acquisition_known, acquisition_extras = _split_payload(
        dict(known_args.get("acquisition", {}) or {}),
        {"shots", "averaging", "trigger_source", "extras"},
    )
    timing_known, timing_extras = _split_payload(
        dict(known_args.get("timing", {}) or {}),
        {"clock_rate_Hz", "sample_rate_Hz", "precision_s", "extras"},
    )
    acquisition_known_extras = dict(acquisition_known.pop("extras", {}) or {})
    timing_known_extras = dict(timing_known.pop("extras", {}) or {})

    return PulseConfig(
        acquisition=PulseAcquisitionConfig(
            **acquisition_known,
            extras={
                **acquisition_known_extras,
                **acquisition_extras,
            },
        ),
        timing=PulseTimingConfig(
            **timing_known,
            extras={
                **timing_known_extras,
                **timing_known_extras,
            },
        ),
        channels={
            str(channel_id): (
                channel_cfg if isinstance(channel_cfg, PulseChannelConfig)
                else PulseChannelConfig(
                    **{
                        k: v
                        for k, v in dict(channel_cfg or {}).items()
                        if k in {"type", "amplitude", "duration_ns", "phase", "frequency_Hz"}
                    },
                    extras={
                        k: v
                        for k, v in dict(channel_cfg or {}).items()
                        if k not in {"type", "amplitude", "duration_ns", "phase", "frequency_Hz"}
                    },
                )
            )
            for channel_id, channel_cfg in dict(known_args.get("channels", {}) or {}).items()
        },
        extras=extras or None,
    )


import itertools

def _build_cartesian_profiles(
    *,
    circuits: dict[str, CircuitConfig],
    devices: dict[str, DeviceConfig],
    pulses: dict[str, PulseConfig],
    solvers: dict[str, SolverConfig],
    analysers: dict[str, AnalyserConfig],
) -> dict[str, ProfileConfig]:
    """Generate all possible combinations of provided resources."""
    profiles: dict[str, ProfileConfig] = {}
    
    # We only need one analyser if available, or None
    analyser_ids = list(analysers.keys()) if analysers else [None]
    
    for c_id, d_id, p_id, s_id, a_id in itertools.product(
        circuits.keys(), devices.keys(), pulses.keys(), solvers.keys(), analyser_ids
    ):
        profile_id = f"{c_id}_{d_id}_{p_id}_{s_id}"
        profiles[profile_id] = ProfileConfig(
            circuit_id=c_id,
            device_id=d_id,
            pulse_id=p_id,
            solver_id=s_id,
            analyser_id=a_id,
        )
    
    # Collapse the singleton case to a single stable profile ID so run_all()
    # does not execute the exact same resource combination twice.
    if len(profiles) == 1:
        return {"default": next(iter(profiles.values()))}
        
    return profiles


def _resolve_profiles(
    profile_input: Any | None,
    circuits: dict[str, CircuitConfig],
    devices: dict[str, DeviceConfig],
    pulses: dict[str, PulseConfig],
    solvers: dict[str, SolverConfig],
    analysers: dict[str, AnalyserConfig],
) -> dict[str, ProfileConfig]:
    """Resolve profiles, applying Cartesian product as default or restriction."""
    if profile_input is None:
        return _build_cartesian_profiles(
            circuits=circuits, devices=devices, pulses=pulses, solvers=solvers, analysers=analysers
        )
    
    # Convert profile_input to dict of ProfileConfigs using existing normalization
    # We temporarily use a dummy name for resource_name to reuse _normalize_resources
    user_profiles = _normalize_resources(
        profile_input,
        config_type="profile",
        resource_name="profile",
        required=False,
        singleton_id="default",
    )
    
    resolved_profiles: dict[str, ProfileConfig] = {}
    
    for pid, p_cfg in user_profiles.items():
        # Cartesian expansion for any missing/None fields in a profile
        c_options = [p_cfg.circuit_id] if p_cfg.circuit_id else list(circuits.keys())
        d_options = [p_cfg.device_id] if p_cfg.device_id else list(devices.keys())
        p_options = [p_cfg.pulse_id] if p_cfg.pulse_id else list(pulses.keys())
        s_options = [p_cfg.solver_id] if p_cfg.solver_id else list(solvers.keys())
        a_options = [p_cfg.analyser_id] if p_cfg.analyser_id else (list(analysers.keys()) if analysers else [None])
        
        for c, d, p, s, a in itertools.product(c_options, d_options, p_options, s_options, a_options):
            # Create a specific ID for expanded profiles
            expanded_id = pid if (len(c_options)==1 and len(d_options)==1 and len(p_options)==1 and len(s_options)==1 and len(a_options)==1) \
                          else f"{pid}_{c}_{d}_{p}_{s}"
            resolved_profiles[expanded_id] = ProfileConfig(
                circuit_id=c, device_id=d, pulse_id=p, solver_id=s, analyser_id=a
            )
            
    return resolved_profiles


def create_model(
    *,
    circuits: Any = None,
    solvers: Any | None = None,
    devices: Any | None = None,
    pulses: Any | None = None,
    analysers: Any | None | object = _UNSET,
    profiles: Any | None = None,
    parameter_sweep: Any | None = None,
) -> Model:
    """Build a top-down editable model object from config files, dicts, or config objects."""
    target: str | list[str] = "trajectory"
    features = WorkflowFeatureFlags()
    output = WorkflowOutputOptions()
    tags: list[str] = []

    circuits_res = _normalize_resources(
        circuits,
        config_type="circuit",
        resource_name="circuit",
        required=True,
        singleton_id="default",
    )
    solvers_res = _normalize_resources(
        solvers,
        config_type="solver",
        resource_name="solver",
        required=True,
        singleton_id="solver_0",
    )
    devices_res = _normalize_resources(
        devices,
        config_type="device",
        resource_name="device",
        required=True,
        singleton_id="default",
    )
    pulses_raw = _normalize_resources(
        pulses,
        config_type="pulse",
        resource_name="pulse",
        required=False,
        singleton_id="default",
    )
    
    # Analysers need special binding to solvers
    analyser_raw = _normalize_analyser_paths(analysers)
    # Convert analyser paths to objects
    analysers_objs = {
        aid: _ensure_config(val, "analyser")
        for aid, val in analyser_raw.items()
    }
    
    solver_ids = sorted(solvers_res.keys())
    analysers_res = {
        analyser_id: bind_loaded_analyser(
            analyser_id=analyser_id,
            analyser_cfg=cfg,
            solver_ids=solver_ids,
        )
        for analyser_id, cfg in analysers_objs.items()
    }
    
    if pulses_raw:
        pulses_res = {
            pid: _normalize_pulse_config(val)
            for pid, val in pulses_raw.items()
        }
    else:
        pulses_res = {"default": PulseConfig()}

    # Normalize parameter sweep config
    param_sweep = load_config(parameter_sweep, "sweep") if parameter_sweep is not None else None

    # Resolve profiles (Cartesian product by default, or restricted by user input)
    resolved_profiles = _resolve_profiles(
        profiles,
        circuits=circuits_res,
        devices=devices_res,
        pulses=pulses_res,
        solvers=solvers_res,
        analysers=analysers_res,
    )

    config = ModelConfig(
        circuits=circuits_res,
        devices=devices_res,
        pulses=pulses_res,
        solvers=solvers_res,
        analysers=analysers_res,
        profiles=resolved_profiles,
        parameter_sweep=param_sweep,
        target=target,
        features=features,
        output=output,
        tags=tags,
    )
    return Model(config=config)


def load_model(path: str | Path) -> Model:
    """Load a persisted model directory created by ``Model.save``."""
    return _load_model_impl(Model, create_model, path)


__all__ = ['Model', 'create_model', 'load_model']
