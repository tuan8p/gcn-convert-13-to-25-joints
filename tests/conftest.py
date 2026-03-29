"""pytest configuration: skip torch-dependent tests when torch is unavailable."""
from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
