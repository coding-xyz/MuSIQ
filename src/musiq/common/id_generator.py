"""Centralized ID generation for workflow artifacts."""

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from musiq.workflow.model import Model

# Analysis scope prefixes
ANALYSIS_PREFIXES = {
    "case": "case",           # Single case analysis
    "parametric": "sweep",    # Parameter sweep analysis
    "comprehensive": "summary", # Comprehensive analysis
}

class IDGenerator:
    """Generates unique sequential IDs for workflow artifacts."""
    
    @staticmethod
    def next_run_id(model: "Model", tag: str | None = None) -> str:
        """Generate next available run_id."""
        existing = set(model.runs.keys())
        prefix = tag if tag else "run"
        
        # If a custom tag is provided and doesn't conflict, use it directly
        if tag and prefix not in existing:
            return prefix
            
        idx = 0
        while f"{prefix}_{idx}" in existing:
            idx += 1
        return f"{prefix}_{idx}"
    
    @staticmethod
    def next_param_id(run_obj) -> str:
        """Generate next available param_id for a run."""
        existing = set(run_obj.results.keys())
        idx = 0
        while f"param_{idx}" in existing:
            idx += 1
        return f"param_{idx}"
    
    @staticmethod
    def next_analysis_id(
        model: "Model", 
        scope: Literal["case", "parametric", "comprehensive"] = "case",
        tag: str | None = None
    ) -> str:
        """Generate next available analysis_id based on analysis scope.
        
        Args:
            model: The model containing existing analyses
            scope: The analysis scope - "case", "parametric", or "comprehensive"
            tag: Optional custom tag to use as prefix instead of scope prefix
            
        Returns:
            A unique analysis ID with appropriate prefix
        """
        prefix = tag if tag else ANALYSIS_PREFIXES.get(scope, "case")
        existing = set(model.analyses.keys())
        
        # If a custom tag is provided and doesn't conflict, use it directly
        if tag and prefix not in existing:
            return prefix
            
        idx = 0
        while f"{prefix}_{idx}" in existing:
            idx += 1
        return f"{prefix}_{idx}"
    
    @staticmethod
    def next_study_name(run_obj, base: str = "step") -> str:
        """Generate next available study name for a run."""
        # provenance might be in result.provenance
        existing = {r.provenance.study_name for r in run_obj.results.values() 
                   if hasattr(r.provenance, 'study_name') and r.provenance.study_name}
        idx = 0
        while f"{base}_{idx}" in existing:
            idx += 1
        return f"{base}_{idx}"
    
    @staticmethod
    def next_shot_id(run_obj) -> str:
        """Generate next available shot_id for a result."""
        existing = set()
        for result in run_obj.results.values():
            existing.update(result.trajectories.keys())
        idx = 0
        while f"shot_{idx}" in existing:
            idx += 1
        return f"shot_{idx}"