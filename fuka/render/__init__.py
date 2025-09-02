# fuka/render/__init__.py

# Export a friendly alias so older imports keep working.
from .pack_npz import pack_to_npz as pack_npz, pack_to_npz

__all__ = ["pack_npz", "pack_to_npz"]