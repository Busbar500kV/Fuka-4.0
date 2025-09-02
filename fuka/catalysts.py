"""
Compatibility shim.

Historically, catalysts lived here. We now implement them in `physics.py`
as a 3D-aware system. To keep old imports working (e.g., `from fuka.catalysts import CatalystsSystem`),
we re-export the new classes from `physics.py`.
"""
from __future__ import annotations
from .physics import CatalystsCfg, CatalystsSystem

__all__ = ["CatalystsCfg", "CatalystsSystem"]