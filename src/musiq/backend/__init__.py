"""Backend compilation public API.

The backend package is responsible for converting normalized circuits into
model specifications and executable artifacts. The most common public entry
points are ``CompilePipeline`` and ``load_backend_config``.
"""

from musiq.backend.compile_pipeline import CompilePipeline
from musiq.backend.config import load_backend_config

__all__ = ["CompilePipeline", "load_backend_config"]
