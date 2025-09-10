# analytics/__init__.py
"""
Analytics package: indices + UI.

Modules:
- build_indices : rebuild per-table indices + manifest
- streamlit_app : Streamlit-based control panel for runs
"""

from .build_indices import rebuild_indices

__all__ = ["rebuild_indices"]