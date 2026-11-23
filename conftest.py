"""The repo root is a ComfyUI custom-node package (`__init__.py` imports `.nodes`) whose directory
name is not a Python identifier. pytest collects the root as a package and imports its
`__init__.py` under the bare name `__init__`, where the relative import fails. Load the pack the
way ComfyUI does (as a package, here named `comfyui_genia`) and register it under that bare name
too, so pytest finds it already imported. Needs ComfyUI on `sys.path` (`PYTHONPATH=<ComfyUI>`)."""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "comfyui_genia", _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)])
_pack = importlib.util.module_from_spec(_spec)
sys.modules["comfyui_genia"] = _pack
_spec.loader.exec_module(_pack)
sys.modules["__init__"] = _pack
