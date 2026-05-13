"""Study orchestration and parameter sweep expansion logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

from musiq.schemas.utils import ParameterSweepConfig, ParameterList, ParametricValue

@dataclass(slots=True)
class StudySample:
    """A single concrete execution point in the parameter space."""
    task_id: str
    device_id: str
    pulse_id: str
    solver_id: str
    params: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class StudyPlan:
    """The resolved expansion of a model configuration into a set of runs and samples."""
    # Maps run_id (compilation unit) to its constituent samples
    run_groups: dict[str, list[StudySample]]
    # The full list of samples to be executed
    all_samples: list[StudySample]

class StudyPlanner:
    """Expands a Model's parametric configuration into a concrete execution plan."""

    @staticmethod
    def resolve_concrete_value(p_val: Any, current_params: dict[str, Any]) -> Any:
        """Resolve a ParametricValue to a concrete value using current params."""
        if isinstance(p_val, ParametricValue):
            # If the dimension is in our current sample params, use it. 
            # Otherwise, fallback to the default value.
            return current_params.get(p_val.dim_name, p_val.value)
        return p_val

    @classmethod
    def plan(cls, model: 'Model') -> StudyPlan:
        """
        Expand the model's config into a plan.
        
        Logic:
        1. Outer Loop: Cartesian product of tasks x devices x pulses x solvers.
           Each unique combination = one ModelRun (Compilation Unit).
        2. Inner Loop: Cartesian product of sweep_space.
           Each combination = one RunResult (Execution Sample).
        """
        from musiq.workflow.model import Model
        config = model.config
        
        # 1. Define the discrete configuration space
        task_ids = list(config.tasks.keys())
        device_ids = list(config.devices.keys())
        pulse_ids = list(config.pulses.keys())
        solver_ids = list(config.solvers.keys())
        
        # 2. Define the continuous parameter space
        parameter_list = config.parameter_list
        if parameter_list is None:
            param_dims = []
            param_values = []
        else:
            param_dims = list(parameter_list.parameters.keys())
            param_values = [parameter_list.parameters[dim].values for dim in param_dims]
        
        # Generate all parameter combinations
        param_combinations = list(product(*param_values)) if param_dims else [()]
        
        all_samples: list[StudySample] = []
        run_groups: dict[str, list[StudySample]] = {}
        
        # Expand outer and inner loops
        for t_id, d_id, p_id, s_id in product(task_ids, device_ids, pulse_ids, solver_ids):
            # Use a concise run_id based on solver_id to avoid "run_default_default..."
            run_id = f"run_{s_id}"
            run_groups[run_id] = []
            
            for p_combo in param_combinations:
                # Map the combination back to dimension names
                current_params = dict(zip(param_dims, p_combo))
                
                sample = StudySample(
                    task_id=t_id,
                    device_id=d_id,
                    pulse_id=p_id,
                    solver_id=s_id,
                    params=current_params
                )
                all_samples.append(sample)
                run_groups[run_id].append(sample)
                
        return StudyPlan(run_groups=run_groups, all_samples=all_samples)

__all__ = ["StudyPlanner", "StudyPlan", "StudySample"]