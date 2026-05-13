"""Compatibility exports for legacy ``musiq.common.schemas`` imports.

New code should import IR/spec dataclasses from ``musiq.schemas`` or its grouped
submodules. This module remains as a stable facade for existing callers.
"""

from musiq.schemas import *
