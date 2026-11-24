"""CPU-only schema check. Run from the ComfyUI env with ComfyUI importable:
`PYTHONPATH=/path/to/ComfyUI pytest -v` (the root `conftest.py` loads the pack as `comfyui_genia`)."""

import asyncio

import pytest

import comfyui_genia
from comfyui_genia.nodes import GenIAReconstructImage, GenIAReconstructMultiview


@pytest.fixture(scope="module")
def schema():
    return GenIAReconstructImage.define_schema()


def test_entrypoint_registers_the_node():
    extension = asyncio.run(comfyui_genia.comfy_entrypoint())
    assert asyncio.run(extension.get_node_list()) == [GenIAReconstructImage, GenIAReconstructMultiview]


def test_identity(schema):
    assert schema.node_id == "GenIAReconstructImage"
    assert schema.display_name == "GenIA Reconstruct (Image)"
    assert schema.category == "3d/GenIA"


def test_inputs(schema):
    got = {i.id: (i.get_io_type(), i.optional) for i in schema.inputs}
    assert got == {
        "image": ("IMAGE", False),
        "mask": ("MASK", True),
        "seed": ("INT", False),
        "quality": ("COMBO", False),
        "low_vram": ("COMBO", False),
        "place_on_ground": ("BOOLEAN", False),
    }
    by_id = {i.id: i for i in schema.inputs}
    assert by_id["low_vram"].options == ["auto", "on", "off"]
    assert by_id["quality"].options == ["fast", "full"] and by_id["quality"].default == "fast"
    assert by_id["place_on_ground"].default is True


def test_outputs(schema):
    assert [o.get_io_type() for o in schema.outputs] == ["MESH", "LOAD3D_CAMERA", "SPLAT"]
    assert [o.display_name for o in schema.outputs] == ["mesh", "camera_info", "splat"]


def test_multiview_schema():
    schema = GenIAReconstructMultiview.define_schema()
    assert schema.node_id == "GenIAReconstructMultiview" and schema.category == "3d/GenIA"
    assert [i.id for i in schema.inputs] == ["images", "objects", "seed", "quality", "low_vram",
                                            "place_on_ground"]
    assert schema.inputs[1].get_io_type() == "COMFY_AUTOGROW_V3"
    assert [o.get_io_type() for o in schema.outputs] == ["MESH", "LOAD3D_CAMERA", "SPLAT"]
