"""Task/Solver/Hardware config loading, template merge, and validation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from musiq.backend.config import validate_backend_config
from musiq.common.schemas import CircuitIR, CircuitGate
from musiq.schemas.utils import ParameterSweepConfig, ParameterList
from musiq.workflow.contracts import (
    AnalyserTrajectoryConfig,
    AnalyserConfig,
    CircuitConfig,
    IQDiscriminationConfig,
    NoiseAnalysisConfig,
    PulseAcquisitionConfig,
    PulseChannelConfig,
    PulseTimingConfig,
    ReadoutModelConfig,
    ReportConfig,
    SolverBackendConfig,
    DeviceConfig,
    WorkflowFeatureFlags,
    WorkflowFrameOptions,
    WorkflowOutputOptions,
    PulseConfig,
    ProfileConfig,
    WorkflowRunOptions,
    SolverConfig,
    Task,
    compose_workflow_task,
)

_SOLVER_TOP_KEYS = {"schema_version", "template", "backend", "run", "frame", "study", "solver"}
_SOLVER_BACKEND_KEYS = {"level", "analysis_pipeline", "analysis", "truncation"}
_SOLVER_FRAME_KEYS = {"mode", "reference", "rwa", "qubit_reference_freqs_Hz"}
_SOLVER_RUN_COMMON_KEYS = {
    "engine",
    "solver_mode",
    "sweep",
    "seed",
    "dt_s",
    "t_end_s",
    "t_padding_s",
    "schedule_policy",
    "schedule",
    "reset_feedback_policy",
    "compare_engines",
    "allow_mock_fallback",
    "mcwf_ntraj",
    "prior_backend",
    "decoder",
    "decoder_options",
    "qec_engine",
    "qutip_options",
    "native_options",
    "backend_options",
    "one_over_f_components",
}
_SOLVER_RUN_JULIA_KEYS = {"julia_bin", "julia_depot_path", "julia_timeout_s"}

_DEVICE_TOP_KEYS = {"schema_version", "template", "device", "noise"}
_PULSE_TOP_KEYS = {"schema_version", "template", "pulse"}
_ANALYSER_TOP_KEYS = {
    "schema_version",
    "template",
    "solver_id",
    "trajectory",
    "case_metrics",
    "sweep_metrics",
    "metrics",
    "parametric_metrics",
    "readout_model",
    "iq_discrimination",
    "noise_analysis",
    "report",
    "analysis",
}
_CIRCUIT_TOP_KEYS = {"schema_version", "format", "qasm_text", "qasm_path", "param_bindings", "num_qubits", "num_clbits", "schedule", "circuit"}
def _is_v1_circuit_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("circuit"), dict)


def _is_v3_solver_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("solver"), dict)


def _is_v3_pulse_payload(payload: dict[str, Any]) -> bool:
    raw_pulse = payload.get("pulse", {}) or {}
    return isinstance(raw_pulse, dict) and any(k in raw_pulse for k in {"channels", "carriers", "waveforms", "operations"})
def _map_circuit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if _is_v1_circuit_payload(payload):
        circuit = dict(payload.get("circuit", {}) or {})
        return {
            "schema_version": str(payload.get("schema_version", "1.0")),
            "format": str(circuit.get("format", payload.get("format", "openqasm3")) or "openqasm3"),
            "qasm_text": circuit.get("qasm_text"),
            "qasm_path": circuit.get("qasm_path"),
            "num_qubits": circuit.get("num_qubits"),
            "num_clbits": circuit.get("num_clbits"),
            "schedule": circuit.get("schedule"),
            "param_bindings": dict(circuit.get("param_bindings", {}) or {}) or None,
        }
    return payload


def _build_circuit_ir_from_schedule_payload(payload: dict[str, Any]) -> CircuitIR:
    raw_schedule = dict(payload.get("schedule", {}) or {})
    num_qubits = int(payload.get("num_qubits", 0) or 0)
    num_clbits = int(payload.get("num_clbits", 0) or 0)
    schedule: dict[int, list[list[CircuitGate]]] = {}
    for raw_tick, raw_lanes in raw_schedule.items():
        tick = int(raw_tick)
        lanes: list[list[CircuitGate]] = []
        for raw_lane in list(raw_lanes or []):
            lane_gates: list[CircuitGate] = []
            for raw_gate in list(raw_lane or []):
                if isinstance(raw_gate, dict):
                    lane_gates.append(
                        CircuitGate(
                            name=str(raw_gate.get("name", "")).strip(),
                            qubits=[int(q) for q in list(raw_gate.get("qubits", []) or [])],
                            params=[float(p) for p in list(raw_gate.get("params", []) or [])],
                            clbits=[int(c) for c in list(raw_gate.get("clbits", []) or [])],
                        )
                    )
                    continue
                if not isinstance(raw_gate, list) or len(raw_gate) < 2:
                    raise ValueError(f"Invalid schedule gate entry at tick {tick}: {raw_gate!r}")
                gate_name = str(raw_gate[0]).strip()
                gate_qubits = [int(q) for q in list(raw_gate[1] or [])]
                gate_params = [float(raw_gate[2])] if len(raw_gate) >= 3 and raw_gate[2] is not None else []
                lane_gates.append(CircuitGate(name=gate_name, qubits=gate_qubits, params=gate_params))
            lanes.append(lane_gates)
        schedule[tick] = lanes
    return CircuitIR(
        schema_version=str(payload.get("schema_version", "1.0")),
        format=str(payload.get("format", "circuit_layer_yaml") or "circuit_layer_yaml"),
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        schedule=schedule,
    )


def _map_v3_pulse_payload(raw_pulse: dict[str, Any]) -> dict[str, Any]:
    channels = list(raw_pulse.get("channels", []) or [])
    carriers = dict(raw_pulse.get("carriers", {}) or {})
    waveforms = dict(raw_pulse.get("waveforms", {}) or {})
    operations = dict(raw_pulse.get("operations", {}) or {})
    acquisition = dict(raw_pulse.get("acquisition", {}) or {})

    def _carrier_freq_for_kind(kind: str, default: float) -> float:
        for ch in channels:
            if not isinstance(ch, dict):
                continue
            if str(ch.get("kind", "")).strip().lower() != kind:
                continue
            name = str(ch.get("name", ""))
            if isinstance(carriers.get(name), dict) and "freq_Hz" in carriers[name]:
                return float(carriers[name]["freq_Hz"])
        return float(default)

    def _waveform_from_operation(name: str, fallback_shapes: set[str]) -> dict[str, Any]:
        steps = list(operations.get(name, []) or [])
        for step in steps:
            if not isinstance(step, dict):
                continue
            wf_name = str(step.get("waveform", ""))
            if wf_name and isinstance(waveforms.get(wf_name), dict):
                return dict(waveforms[wf_name])
        for wf in waveforms.values():
            if isinstance(wf, dict) and str(wf.get("shape", "")).strip().lower() in fallback_shapes:
                return dict(wf)
        return {}

    def _operation_scale(name: str, default: float = 1.0) -> float:
        steps = list(operations.get(name, []) or [])
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "scale" in step:
                return float(step.get("scale", default))
        return float(default)

    def _measure_segments() -> list[dict[str, Any]]:
        steps = list(operations.get("measure", []) or [])
        segments: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            wf_name = str(step.get("waveform", ""))
            wf = dict(waveforms.get(wf_name, {}) or {}) if wf_name and isinstance(waveforms.get(wf_name), dict) else {}
            if not wf:
                continue
            if "duration_ns" not in wf:
                continue
            segments.append(
                {
                    "duration_ns": float(wf["duration_ns"]),
                    "amp": 0.8 * float(step.get("scale", 1.0)),
                    "edge_ns": float(wf.get("edge_ns", 0.0) or 0.0),
                    "rise_ns": float(wf.get("rise_ns", wf.get("edge_ns", 0.0)) or 0.0),
                    "fall_ns": float(wf.get("fall_ns", wf.get("edge_ns", 0.0)) or 0.0),
                    "shape": str(wf.get("shape", "readout") or "readout"),
                }
            )
        return segments

    gate_wf = _waveform_from_operation("x", {"drag", "gaussian", "rect"})
    measure_wf = _waveform_from_operation("measure", {"readout", "rect"})
    measure_segments = _measure_segments()

    mapped: dict[str, Any] = {}
    mapped["xy_freq_Hz"] = _carrier_freq_for_kind("drive", 5.0e9)
    mapped["ro_freq_Hz"] = _carrier_freq_for_kind("readout_drive", mapped["xy_freq_Hz"])
    if "duration_ns" in gate_wf:
        mapped["gate_duration_ns"] = float(gate_wf["duration_ns"])
    if measure_segments:
        mapped["measure_segments"] = measure_segments
        mapped["measure_duration_ns"] = float(sum(float(seg.get("duration_ns", 0.0) or 0.0) for seg in measure_segments))
        mapped["measure_amp"] = float(measure_segments[0].get("amp", 0.8))
    elif "duration_ns" in measure_wf:
        mapped["measure_duration_ns"] = float(measure_wf["duration_ns"])
        mapped["measure_amp"] = 0.8 * _operation_scale("measure", 1.0)
    else:
        mapped["measure_amp"] = 0.8 * _operation_scale("measure", 1.0)
    if "edge_ns" in measure_wf:
        mapped["readout_edge_ns"] = float(measure_wf["edge_ns"])
    if "measure_start_delay_ns" in acquisition:
        mapped["measure_start_delay_ns"] = float(acquisition["measure_start_delay_ns"])
    if "integration_window_ns" in acquisition:
        mapped["measure_duration_ns"] = float(acquisition["integration_window_ns"])
    if acquisition:
        mapped["acquisition"] = acquisition
    schedule_cfg = dict(raw_pulse.get("schedule", {}) or {})
    if schedule_cfg.get("policy"):
        mapped["schedule_policy"] = str(schedule_cfg.get("policy"))
    return mapped


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return value
    p = Path(value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _load_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    p = Path(path).resolve()
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        payload = json.loads(text)
    elif p.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}. Use .json/.yaml/.yml")
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must be a mapping object: {p}")
    return p, dict(payload)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _template_file(kind: str, template_name: str) -> Path:
    root = Path(__file__).resolve().parent / "templates" / kind
    stem = str(template_name).strip()
    candidates = [root / f"{stem}.yaml", root / f"{stem}.yml", root / f"{stem}.json"]
    for c in candidates:
        if c.exists():
            return c
    raise ValueError(f"Unknown {kind} template: {template_name!r}")


def _apply_template(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    template_name = payload.get("template")
    if not template_name:
        return payload
    template_path = _template_file(kind, str(template_name))
    _, template_payload = _load_mapping(template_path)
    merged = _deep_merge(template_payload, {k: v for k, v in payload.items() if k != "template"})
    return merged


def _reject_unknown(section: str, keys: set[str], allowed: set[str]) -> None:
    unknown = sorted(keys - allowed)
    if unknown:
        raise ValueError(f"Unsupported keys in {section}: {unknown}")
def _validate_solver_payload(payload: dict[str, Any]) -> str:
    _reject_unknown("solver top-level", set(payload), _SOLVER_TOP_KEYS)
    raw_backend = payload.get("backend", {}) or {}
    raw_run = payload.get("run", {}) or {}
    raw_frame = payload.get("frame", {}) or {}

    if not isinstance(raw_backend, dict):
        raise ValueError("Solver config `backend` must be a mapping.")
    if not isinstance(raw_run, dict):
        raise ValueError("Solver config `run` must be a mapping.")
    if not isinstance(raw_frame, dict):
        raise ValueError("Solver config `frame` must be a mapping.")

    _reject_unknown("solver.backend", set(raw_backend), _SOLVER_BACKEND_KEYS)
    _reject_unknown("solver.run", set(raw_run), _SOLVER_RUN_COMMON_KEYS | _SOLVER_RUN_JULIA_KEYS)
    _reject_unknown("solver.frame", set(raw_frame), _SOLVER_FRAME_KEYS)

    engine = str(raw_run.get("engine", "qutip")).strip().lower()
    if engine not in {"qutip", "qoptics", "qtoolbox"}:
        raise ValueError(f"Unsupported solver.run.engine: {engine!r}. Supported engines: qutip, qoptics, qtoolbox.")
    allowed_run = set(_SOLVER_RUN_COMMON_KEYS)
    is_julia = engine in {"qoptics", "qtoolbox"}
    if is_julia:
        allowed_run.update(_SOLVER_RUN_JULIA_KEYS)

    disallowed_run = sorted(set(raw_run) - allowed_run)
    if disallowed_run:
        raise ValueError(
            "Solver `run` contains keys not supported by selected engine "
            f"{engine!r}: {disallowed_run}"
        )
    return engine


def _validate_v3_solver_study(study: list[dict[str, Any]] | None) -> None:
    allowed_step_keys = {
        "name",
        "description",
        "active_components",
        "active_connections",
        "representations",
        "bases",
        "solver_mode",
        "time",
        "frame",
        "options",
        "prep_state",
        "schedule",
    }
    for idx, step in enumerate(list(study or [])):
        if not isinstance(step, dict):
            raise ValueError(f"solver.study[{idx}] must be a mapping.")
        if "parameters" in step:
            raise ValueError("solver.study[].parameters is no longer supported; use prep_state, representations, and bases.")
        unknown = sorted(set(step) - allowed_step_keys)
        if unknown:
            raise ValueError(f"Unsupported keys in solver.study[{idx}]: {unknown}")
        if "representations" in step and not isinstance(step.get("representations"), dict):
            raise ValueError(f"solver.study[{idx}].representations must be a mapping.")
        if "bases" in step and not isinstance(step.get("bases"), dict):
            raise ValueError(f"solver.study[{idx}].bases must be a mapping.")
        if "prep_state" in step:
            prep_state = step.get("prep_state")
            if not isinstance(prep_state, dict):
                raise ValueError(f"solver.study[{idx}].prep_state must be a mapping.")
            prep_unknown = sorted(set(prep_state) - {"label", "sequence"})
            if prep_unknown:
                raise ValueError(f"Unsupported keys in solver.study[{idx}].prep_state: {prep_unknown}")


def _validate_composite_device_schema(raw_device: dict[str, Any]) -> None:
    components = list(raw_device.get("components", []) or [])
    for idx, comp in enumerate(components):
        if not isinstance(comp, dict):
            raise ValueError(f"device.components[{idx}] must be a mapping.")
        if "role" in comp:
            raise ValueError("device.components[].role is no longer supported; use device.components[].description instead.")
        moved_keys = [key for key in ("representation", "basis") if key in comp]
        if moved_keys:
            raise ValueError(
                "device.components[] no longer accepts "
                f"{moved_keys}; move them into solver.study[].representations / solver.study[].bases."
            )


def _validate_device_payload(payload: dict[str, Any]) -> None:
    _reject_unknown("device top-level", set(payload), _DEVICE_TOP_KEYS)
    raw_device = payload.get("device", {}) or {}
    raw_noise = payload.get("noise", {}) or {}
    if not isinstance(raw_device, dict):
        raise ValueError("Device config `device` must be a mapping.")
    if not isinstance(raw_noise, dict):
        raise ValueError("Device config `noise` must be a mapping.")
    if "components" in raw_device:
        _validate_composite_device_schema(raw_device)


def _validate_pulse_payload(payload: dict[str, Any]) -> None:
    _reject_unknown("pulse top-level", set(payload), _PULSE_TOP_KEYS)
    raw_pulse = payload.get("pulse", {}) or {}
    if not isinstance(raw_pulse, dict):
        raise ValueError("Pulse config `pulse` must be a mapping.")


def circuit_from_payload(payload: dict[str, Any], base_dir: Path | None = None) -> CircuitConfig:
    """Convert a circuit payload dictionary into a ``CircuitConfig`` object."""
    payload = _map_circuit_payload(payload)
    _reject_unknown("circuit top-level", set(payload), _CIRCUIT_TOP_KEYS - {"circuit"})

    if payload.get("schedule") is not None:
        if payload.get("qasm_text") or payload.get("qasm_path"):
            raise ValueError("Circuit config must not mix schedule input with qasm_text/qasm_path.")
        return CircuitConfig(
            qasm_text=None,
            circuit_ir=_build_circuit_ir_from_schedule_payload(payload),
            param_bindings=dict(payload.get("param_bindings", {}) or {}) or None,
        )

    qasm_text = payload.get("qasm_text")
    qasm_path = payload.get("qasm_path")
    if bool(qasm_text) == bool(qasm_path):
        raise ValueError("Circuit config must provide exactly one of qasm_text or qasm_path.")
    if qasm_path:
        if base_dir is None:
            raise ValueError("base_dir is required to resolve qasm_path in circuit payload.")
        qasm_full = Path(_resolve_path(base_dir, str(qasm_path)))
        qasm_text = qasm_full.read_text(encoding="utf-8")

    return CircuitConfig(
        qasm_text=str(qasm_text),
        circuit_ir=None,
        param_bindings=dict(payload.get("param_bindings", {}) or {}) or None,
    )


def load_circuit_config_file(path: str | Path) -> CircuitConfig:
    """Load a circuit config file into ``CircuitConfig``."""
    cfg_path, payload = _load_mapping(path)
    return circuit_from_payload(payload, base_dir=cfg_path.parent)
def solver_from_payload(payload: dict[str, Any], base_dir: Path | None = None) -> SolverConfig:
    """Convert a solver payload dictionary into a ``SolverConfig`` object."""
    payload = _apply_template("solvers", payload)
    if _is_v3_solver_payload(payload):
        solver = dict(payload.get("solver", {}) or {})
        engine = str(solver.get("engine", "qutip")).strip().lower()
        if engine not in {"qutip", "qoptics", "qtoolbox"}:
            raise ValueError(f"Unsupported solver.engine: {engine!r}. Supported engines: qutip, qoptics, qtoolbox.")
        raw_study = [dict(step) for step in list(solver.get("study", []) or []) if isinstance(step, dict)] or None
        _validate_v3_solver_study(raw_study)
        raw_schedule = dict(solver.get("schedule", {}) or {})
        raw_run = {
            "engine": engine,
            "seed": int(solver.get("seed", 12345)),
            "schedule_policy": raw_schedule.get("policy"),
            "mcwf_ntraj": int(solver.get("mcwf_ntraj", 128) or 128),
        }
        solver_cfg = SolverConfig(
            backend=SolverBackendConfig(level="qubit", analysis_pipeline="structured", truncation={}),
            run=WorkflowRunOptions(**raw_run),
            frame=WorkflowFrameOptions(),
            study=raw_study,
        )
        if raw_run.get("julia_bin") and base_dir:
            solver_cfg.run.julia_bin = _resolve_path(base_dir, str(raw_run["julia_bin"]))
        if raw_run.get("julia_depot_path") and base_dir:
            solver_cfg.run.julia_depot_path = _resolve_path(base_dir, str(raw_run["julia_depot_path"]))
        validate_backend_config(solver_cfg.to_backend_config())
        return solver_cfg

    _validate_solver_payload(payload)
    raw_backend = dict(payload.get("backend", {}) or {})
    raw_run = dict(payload.get("run", {}) or {})
    raw_frame = dict(payload.get("frame", {}) or {})

    if "analysis" in raw_backend and "analysis_pipeline" not in raw_backend:
        raw_backend["analysis_pipeline"] = raw_backend["analysis"]
    if "schedule" in raw_run and "schedule_policy" not in raw_run:
        raw_run["schedule_policy"] = raw_run["schedule"]

    if raw_run.get("julia_bin") and base_dir:
        raw_run["julia_bin"] = _resolve_path(base_dir, str(raw_run["julia_bin"]))
    if raw_run.get("julia_depot_path") and base_dir:
        raw_run["julia_depot_path"] = _resolve_path(base_dir, str(raw_run["julia_depot_path"]))

    solver_cfg = SolverConfig(
        backend=SolverBackendConfig(**raw_backend),
        run=WorkflowRunOptions(**raw_run),
        frame=WorkflowFrameOptions(**raw_frame),
        study=[dict(step) for step in list(payload.get("study", []) or []) if isinstance(step, dict)] or None,
    )
    validate_backend_config(solver_cfg.to_backend_config())
    return solver_cfg


def load_solver_config_file(path: str | Path) -> SolverConfig:
    """Load a solver config file into ``WorkflowSolverConfig``.

    The solver config controls the backend model level, runtime engine
    selection, solver options, and reference-frame settings.
    """
    cfg_path, payload = _load_mapping(path)
    return solver_from_payload(payload, base_dir=cfg_path.parent)


def device_from_payload(payload: dict[str, Any]) -> DeviceConfig:
    """Convert a device payload dictionary into a ``DeviceConfig`` object."""
    payload = _apply_template("device", payload)

    _validate_device_payload(payload)
    raw_device = dict(payload.get("device", {}) or {})
    nested_noise = dict(raw_device.get("noise", {}) or {}) if isinstance(raw_device.get("noise"), dict) else {}
    raw_device = {k: v for k, v in raw_device.items() if k != "noise"}
    top_noise = dict(payload.get("noise", {}) or {})
    return DeviceConfig(
        device=raw_device or None,
        noise=top_noise or nested_noise or None,
    )


def load_device_config_file(path: str | Path) -> DeviceConfig:
    """Load a device config file into ``WorkflowDeviceConfig``.

    Device configs contain device-level parameters and noise model settings.
    """
    _, payload = _load_mapping(path)
    return device_from_payload(payload)


def pulse_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a pulse payload dictionary into a pulse configuration mapping."""
    payload = _apply_template("pulses", payload)
    _validate_pulse_payload(payload)
    raw_pulse = dict(payload.get("pulse", {}) or {})
    if _is_v3_pulse_payload(payload):
        return _map_v3_pulse_payload(raw_pulse)
    return raw_pulse


def load_pulse_config_file(path: str | Path) -> dict[str, Any]:
    """Load a pulse config file.

    Pulse configs contain gate duration, carrier frequency, readout, and reset
    pulse parameters. The returned mapping is later merged into
    ``WorkflowDeviceConfig.pulse``.
    """
    _, payload = _load_mapping(path)
    return pulse_from_payload(payload)


def analyser_from_payload(payload: dict[str, Any]) -> AnalyserConfig:
    """Convert an analyser payload dictionary into an ``AnalyserConfig`` object."""
    _reject_unknown("analyser top-level", set(payload), _ANALYSER_TOP_KEYS)

    # Parse hierarchical analysis steps
    analysis_steps = list(payload.get("analysis", []) or [])
    if not isinstance(analysis_steps, list):
        raise ValueError("Analyser config `analysis` must be a list of analysis steps.")

    trajectory_raw = dict(payload.get("trajectory", {}) or {})
    trajectory_known = {k: v for k, v in trajectory_raw.items() if k in {"window_start", "window_end", "stride"}}
    trajectory_extras = {k: v for k, v in trajectory_raw.items() if k not in trajectory_known}

    readout_raw = dict(payload.get("readout_model", {}) or {})
    readout_known = {
        k: v for k, v in readout_raw.items() if k in {"model_type", "integration_time", "demodulation_freq_Hz"}
    }
    readout_extras = {k: v for k, v in readout_raw.items() if k not in readout_known}

    iq_raw = dict(payload.get("iq_discrimination", {}) or {})
    iq_known = {k: v for k, v in iq_raw.items() if k in {"method", "num_clusters", "prior_centroids"}}
    iq_extras = {k: v for k, v in iq_raw.items() if k not in iq_known}

    noise_raw = dict(payload.get("noise_analysis", {}) or {})
    noise_known = {k: v for k, v in noise_raw.items() if k in {"method", "resolution_Hz"}}
    noise_extras = {k: v for k, v in noise_raw.items() if k not in noise_known}

    report_raw = dict(payload.get("report", {}) or {})
    report_known = {k: v for k, v in report_raw.items() if k in {"include_plots", "format"}}
    report_extras = {k: v for k, v in report_raw.items() if k not in report_known}

    return AnalyserConfig(
        solver_id=str(payload.get("solver_id")).strip() or None if payload.get("solver_id") is not None else None,
        trajectory=AnalyserTrajectoryConfig(**trajectory_known, extras=trajectory_extras),
        analysis=analysis_steps,
        case_metrics=list(payload.get("case_metrics", []) or payload.get("metrics", []) or []) or None,
        sweep_metrics=list(payload.get("sweep_metrics", []) or payload.get("parametric_metrics", []) or []) or None,
        metrics=list(payload.get("metrics", []) or []) or None,
        parametric_metrics=list(payload.get("parametric_metrics", []) or []) or None,
        readout_model=ReadoutModelConfig(**readout_known, extras=readout_extras),
        iq_discrimination=IQDiscriminationConfig(**iq_known, extras=iq_extras),
        noise_analysis=NoiseAnalysisConfig(**noise_known, extras=noise_extras),
        report=ReportConfig(**report_known, extras=report_extras),
    )


def profile_from_payload(payload: dict[str, Any]) -> ProfileConfig:
    """Convert a profile payload dictionary into a ``ProfileConfig`` object."""
    return ProfileConfig(**payload)

def sweep_from_payload(payload: Any) -> ParameterSweepConfig:
    """Convert payload to typed ParameterSweepConfig."""
    if not isinstance(payload, dict):
        raise TypeError(f"Sweep payload must be a dict, got {type(payload)}")

    def _looks_like_value_list(value: Any) -> bool:
        if isinstance(value, (str, bytes, bytearray, Mapping)):
            return False
        if isinstance(value, Iterable):
            try:
                iter(value)
            except TypeError:
                return False
            return True
        return False

    # Check if it's the simplified format: {'theta': [0, 0.1, ...]}
    if "parameters" not in payload and any(_looks_like_value_list(v) for v in payload.values()):
        # Simplified format: convert to full format
        payload = {
            "parameters": {
                k: {"target": k, "values": list(v)} for k, v in payload.items()
            },
            "mode": None,
            "metadata": {}
        }

    params_raw = payload.get("parameters", {})
    parameters = {}
    for p_id, p_val in params_raw.items():
        if not isinstance(p_val, dict):
            parameters[p_id] = ParameterList(target=p_id, values=list(p_val))
        else:
            parameters[p_id] = ParameterList(
                target=str(p_val.get("target", p_id)).strip(),
                values=list(p_val.get("values", [])),
                unit=str(p_val.get("unit", "")).strip() or None,
                description=str(p_val.get("description", "")).strip() or None,
            )

    return ParameterSweepConfig(
        parameters=parameters,
        mode=str(payload.get("mode", "")).strip().lower() or None,
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def load_analyser_config_file(path: str | Path) -> AnalyserConfig:
    """Load an analyser config file into ``DefaultAnalyserConfig``."""
    _, payload = _load_mapping(path)
    return analyser_from_payload(payload)


def load_config(source: str | Path | dict[str, Any], config_type: str, base_dir: Path | None = None) -> Any:
    """
    Unified dispatcher to load a configuration from a file or a dictionary payload.

    Args:
        source: Path to the config file or the config dictionary itself.
        config_type: Type of the config ("circuit", "solver", "device", "pulse", "analyser").
        base_dir: Optional base directory for resolving relative paths within the config.
    """
    payload = source
    if isinstance(source, (str, Path)):
        # If it's a path, load the mapping first
        p, payload = _load_mapping(source)
        # Use the file's directory as base_dir if none was provided
        if base_dir is None:
            base_dir = p.parent

    type_map = {
        "circuit": (circuit_from_payload, True),
        "solver": (solver_from_payload, True),
        "device": (device_from_payload, False),
        "pulse": (pulse_from_payload, False),
        "analyser": (analyser_from_payload, False),
        "profile": (profile_from_payload, False),
        "sweep": (sweep_from_payload, False),
    }

    if config_type not in type_map:
        raise ValueError(f"Unsupported config_type: {config_type}. Supported: {list(type_map.keys())}")

    converter, needs_base_dir = type_map[config_type]

    if needs_base_dir:
        return converter(payload, base_dir=base_dir)
    return converter(payload)
def load_config_bundle_files(
    *,
    circuit_config: str | Path,
    solver_config: str | Path,
    device_config: str | Path,
    pulse_config: str | Path | None = None,
    analyser_config: str | Path | None = None,
) -> Task:
    """Load and compose a resource-first config bundle into ``WorkflowTask``.

    This is the main file-driven composition helper used by the model API.
    """
    circuit_cfg = load_circuit_config_file(circuit_config)
    solver_path = str(solver_config)
    device_path = str(device_config)
    pulse_path = str(pulse_config) if pulse_config is not None else None
    analyser_path = str(analyser_config) if analyser_config is not None else None
    solver_cfg = load_solver_config_file(solver_path)
    device_cfg = load_device_config_file(device_path)
    analyser_cfg = load_analyser_config_file(analyser_path) if analyser_path else None
    if pulse_path:
        pulse_payload = load_pulse_config_file(pulse_path)
        def _split_payload(raw: dict[str, Any], known: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
            known_items = {k: v for k, v in raw.items() if k in known}
            extras_map = {k: v for k, v in raw.items() if k not in known}
            return known_items, extras_map

        known_fields = {"acquisition", "timing", "channels", "extras"}
        known_args = {k: v for k, v in pulse_payload.items() if k in known_fields}
        extra_args = {k: v for k, v in pulse_payload.items() if k not in known_fields}
        extras = dict(known_args.get("extras") or {})
        extras.update(extra_args)
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
        device_cfg.pulse = PulseConfig(
            acquisition=PulseAcquisitionConfig(
                **acquisition_known,
                extras={**acquisition_known_extras, **acquisition_extras},
            ),
            timing=PulseTimingConfig(
                **timing_known,
                extras={**timing_known_extras, **timing_extras},
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
    return compose_workflow_task(
        target="trajectory",
        features=WorkflowFeatureFlags(),
        output=WorkflowOutputOptions(),
        tags=[],
        circuit_cfg=circuit_cfg,
        solver_cfg=solver_cfg,
        device_cfg=device_cfg,
        analyser_cfg=analyser_cfg,
        model_pulse=device_cfg.pulse,
        backend_source=str(Path(solver_path).resolve()),
    )


__all__ = [
    "load_config_bundle_files",
    "load_analyser_config_file",
    "load_circuit_config_file",
    "load_device_config_file",
    "load_pulse_config_file",
    "load_solver_config_file",
    "load_config",
    "circuit_from_payload",
    "solver_from_payload",
    "device_from_payload",
    "pulse_from_payload",
    "analyser_from_payload",
    "profile_from_payload",
    "sweep_from_payload",
]
