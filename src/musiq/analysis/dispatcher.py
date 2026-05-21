"""Dispatcher for hierarchical analysis kinds."""

from __future__ import annotations

from typing import Any
from musiq.analysis.registry import AnalysisRegistry, AnalysisLevel, AnalysisKind
from musiq.analysis.case.single_qubit.analysis import build_single_qubit_analysis
from musiq.analysis.case.readout.analysis import build_readout_analysis
from musiq.analysis.readout_chain import build_iq_analysis

# Global registry instance
registry = AnalysisRegistry()

def initialize_analysis_registry():
    """Register all known analysis handlers."""
    # CASE level
    registry.register_kind(
        AnalysisLevel.CASE, 
        AnalysisKind.SINGLE_QUBIT, 
        build_single_qubit_analysis
    )
    registry.register_kind(
        AnalysisLevel.CASE, 
        AnalysisKind.READOUT, 
        build_readout_analysis
    )
    
    # COMPREHENSIVE level
    registry.register_kind(
        AnalysisLevel.COMPREHENSIVE, 
        AnalysisKind.IQ, 
        build_iq_analysis
    )

# Initialize on import
initialize_analysis_registry()

def dispatch_analysis(level: str, kind: str, **kwargs: Any) -> Any:
    """
    Dispatch analysis request to the appropriate handler based on level and kind.
    
    Args:
        level: String representation of AnalysisLevel (e.g., "CASE").
        kind: String representation of AnalysisKind (e.g., "READOUT").
        **kwargs: Arguments passed to the handler.
        
    Returns:
        The result of the analysis handler.
    """
    try:
        l_enum = AnalysisLevel[level.upper()]
        k_enum = AnalysisKind[kind.upper()]
        handler = registry.get_handler(l_enum, k_enum)
        return handler(**kwargs)
    except KeyError as e:
        raise KeyError(f"Unsupported analysis configuration: {level}.{kind}") from e

__all__ = ["registry", "dispatch_analysis", "initialize_analysis_registry"]
