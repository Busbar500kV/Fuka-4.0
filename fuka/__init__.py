# fuka/__init__.py
"""
Fuka core package.

Contains:
- engine   : main simulation loop
- physics  : update rules
- bath     : bath scaling
- external_source : stimulus injector
- guess_field     : top-down nudges
- observer : stats logger
- encoder  : cumulative edge encoder
- recorder : Parquet recorder
- runner   : CLI entrypoint
"""

# Public API
from .engine import Engine
from .runner import run

__all__ = ["Engine", "run"]