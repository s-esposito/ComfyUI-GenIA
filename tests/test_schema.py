"""CPU-only schema check. Run from the ComfyUI env with ComfyUI importable:
`PYTHONPATH=/path/to/ComfyUI pytest -v` (the root `conftest.py` loads the pack as `comfyui_genia`)."""

import asyncio

import comfyui_genia
from comfyui_genia.nodes import GenIAExampleScene, GenIAReconstruct


def test_entrypoint_registers_the_nodes():
    extension = asyncio.run(comfyui_genia.comfy_entrypoint())
    assert asyncio.run(extension.get_node_list()) == [GenIAReconstruct, GenIAExampleScene]


def test_reconstruct_schema():
    schema = GenIAReconstruct.define_schema()
    assert (schema.node_id, schema.display_name, schema.category) == (
        "GenIAReconstruct", "GenIA Reconstruct", "3d/GenIA")
    by_id = {i.id: i for i in schema.inputs}
    assert list(by_id) == ["images", "objects", "quality", "seed", "pose_refinement", "low_vram",
                           "place_on_ground", "background", "extra_overrides"]
    assert [t.io_type for t in by_id["objects"].io_types] == ["MASK", "SAM3_TRACK_DATA"]
    assert by_id["quality"].options == ["draft", "fast", "full"] and by_id["quality"].default == "fast"
    # The rarely changed ones are behind "show advanced"; the seed stays put between runs
    # (a "randomize" default would re-run GenIA on every queue).
    assert [i for i, x in by_id.items() if x.advanced] == ["seed", "pose_refinement", "low_vram",
                                                           "place_on_ground", "background", "extra_overrides"]
    assert by_id["seed"].control_after_generate == "fixed"
    assert [o.get_io_type() for o in schema.outputs] == ["MESH", "LOAD3D_CAMERA", "SPLAT", "VIDEO"]


def test_example_scene_schema():
    schema = GenIAExampleScene.define_schema()
    assert schema.node_id == "GenIAExampleScene" and schema.inputs[0].id == "scene"
    assert schema.is_output_node   # runs alone ("Run selected") to preview the scene
    assert [o.get_io_type() for o in schema.outputs] == ["IMAGE", "MASK"]


def test_seed_defaults_to_genias():
    """42 is GenIA's own default, which the demo app runs with: same inputs, same result."""
    by_id = {i.id: i for i in GenIAReconstruct.define_schema().inputs}
    assert by_id["seed"].default == 42
