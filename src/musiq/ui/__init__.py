"""UI helper public API.

The ``musiq.ui`` package contains lightweight helpers intended for notebooks,
simple scripts, and result inspection. The most common public helper is
``plot_default(model)``.
"""

from musiq.ui.notebook import plot_default

__all__ = ["plot_default"]
