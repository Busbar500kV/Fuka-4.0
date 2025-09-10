# fuka/render/__init__.py
"""
Fuka rendering helpers.

Exports:
- pack_npz : pack shards into canonical NPZ for Manim
"""

from .pack_npz import pack

__all__ = ["pack"]