"""Pulse construction public API.

This package exposes the high-level helpers most users need when working with
pulse generation:

- ``DefaultPulseLowering`` for converting ``CircuitIR`` into ``PulseIR``
- ``PulseCompiler`` for building pulse sequences
- ``instantiate_operation_recipe`` for resolving concrete pulse recipes
"""

from musiq.pulse.catalog import instantiate_operation_recipe
from musiq.pulse.lowering import DefaultLowering, DefaultPulseLowering, IPulseLowering
from musiq.pulse.sequence import PulseCompiler

__all__ = [
    "DefaultLowering",
    "DefaultPulseLowering",
    "IPulseLowering",
    "PulseCompiler",
    "instantiate_operation_recipe",
]
