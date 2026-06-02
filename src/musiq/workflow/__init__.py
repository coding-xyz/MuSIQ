"""Workflow public API."""

from musiq.workflow.contracts import (
    AnalyserConfig,
    CircuitConfig,
    DeviceConfig,
    ProfileConfig,
    SolverBackendConfig,
    WorkflowFeatureFlags,
    WorkflowFrameOptions,
    WorkflowInput,
    WorkflowOutputOptions,
    WorkflowRunOptions,
    SolverConfig,
    Task,
    compose_workflow_task,
    normalize_device_payload,
)
from musiq.schemas.results import ModelAnalysis
from musiq.workflow.model import Model, create_model, load_model
from musiq.workflow.model_execution import build_solver, build_study, run_solver, run_engine, run_analysis
from musiq.workflow.planner import ExecutionPlan, build_execution_plan
from musiq.workflow.session_adapter import commit_result_to_session
from musiq.workflow.task_io import (
    load_analyser_config_file,
    load_circuit_config_file,
    load_device_config_file,
    load_pulse_config_file,
    load_solver_config_file,
    load_config,
    circuit_from_payload,
    solver_from_payload,
    device_from_payload,
    pulse_from_payload,
    analyser_from_payload,
    profile_from_payload,
)

__all__ = [
    "SolverBackendConfig",
    "AnalyserConfig",
    "CircuitConfig",
    "ProfileConfig",
    "WorkflowFeatureFlags",
    "WorkflowFrameOptions",
    "DeviceConfig",
    "WorkflowInput",
    "WorkflowOutputOptions",
    "WorkflowRunOptions",
    "SolverConfig",
    "Task",
    "compose_workflow_task",
    "normalize_device_payload",
    "ModelAnalysis",
    "Model",
    "create_model",
    "load_model",
    "build_solver",
    "build_study",
    "run_solver",
    "run_engine",
    "run_analysis",
    "ExecutionPlan",
    "build_execution_plan",
    "commit_result_to_session",
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
]

# Deprecated compatibility alias. Prefer ``ModelAnalysis`` in new code.
AnalysisResult = ModelAnalysis
__all__.append("AnalysisResult")
