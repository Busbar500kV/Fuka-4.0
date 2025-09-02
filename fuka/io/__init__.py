# fuka/io/__init__.py
from .discovery import discover, discover_table, files_from_index, files_from_manifest, files_from_gsutil
__all__ = ["discover", "discover_table", "files_from_index", "files_from_manifest", "files_from_gsutil"]