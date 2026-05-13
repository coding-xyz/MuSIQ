"""Workflow public API."""

from musiq.workflow.contracts import (
    AnalyserConfig,
    SolverBackendConfig,
    TaskInputConfig,
    DeviceConfig,
    WorkflowFeatureFlags,
    WorkflowFrameOptions,
    WorkflowInput,
    WorkflowOutputOptions,
    WorkflowRunOptions,
    SolverConfig,
    Task,
    TaskConfig,
    compose_workflow_task,
    normalize_device_payload,
)
from musiq.schemas.results import ModelAnalysis
from musiq.workflow.model import Model, create_model, load_model
from musiq.workflow.model_execution import run_solver, run_analysis
from musiq.workflow.planner import ExecutionPlan, build_execution_plan
from musiq.workflow.session_adapter import commit_result_to_session
from musiq.workflow.task_io import (
    load_config_bundle_files,
    load_analyser_config_file,
    load_device_config_file,
    load_pulse_config_file,
    load_solver_config_file,
    load_task_config_file,
    load_task_file,
)

__all__ = [
    "SolverBackendConfig",
    "AnalyserConfig",
    "TaskInputConfig",
    "WorkflowFeatureFlags",
    "WorkflowFrameOptions",
    "DeviceConfig",
    "WorkflowInput",
    "WorkflowOutputOptions",
    "WorkflowRunOptions",
    "SolverConfig",
    "Task",
    "TaskConfig",
    "compose_workflow_task",
    "normalize_device_payload",
    "ModelAnalysis",
    "Model",
    "create_model",
    "load_model",
    "run_solver",
    "run_analysis",
    "ExecutionPlan",
    "build_execution_plan",
    "commit_result_to_session",
    "load_config_bundle_files",
    "load_analyser_config_file",
    "load_device_config_file",
    "load_pulse_config_file",
    "load_solver_config_file",
    "load_task_config_file",
    "load_task_file",
]

# Deprecated compatibility alias. Prefer ``ModelAnalysis`` in new code.
AnalysisResult = ModelAnalysis
__all__.append("AnalysisResult")
