"""CPU checks of what the node does around GenIA: inputs, staging, reading GenIA's outputs back."""
import json
import struct

import numpy as np
import pytest
import torch
from PIL import Image

from comfyui_genia import bridge, geometry, nodes


def test_inputs_one_mask_per_object_at_the_image_size():
    image = torch.rand(1, 8, 10, 3)
    mask = torch.zeros(3, 4, 5)
    mask[0, :2] = 1          # object 1
    mask[2, 2:] = 1          # object 2; mask 1 is empty and dropped
    rgb, masks = nodes.inputs(image, mask)
    assert rgb.dtype == np.uint8 and rgb.shape == (1, 8, 10, 3)
    assert masks.shape == (1, 2, 8, 10) and masks[0, 0, :4].all() and not masks[0, 0, 4:].any()


def test_inputs_need_a_mask():
    with pytest.raises(ValueError, match="masks"):
        nodes.inputs(torch.rand(1, 4, 4, 3), None)
    with pytest.raises(ValueError, match="empty"):
        nodes.inputs(torch.rand(1, 4, 4, 3), torch.zeros(1, 4, 4))


def test_stage_image_writes_object_ids(tmp_path):
    masks = np.zeros((1, 2, 4, 4), bool)
    masks[0, 0, 0], masks[0, 1, 3] = True, True
    bridge.stage(tmp_path, "image", "s", np.zeros((1, 4, 4, 3), np.uint8), masks)
    ids = np.asarray(Image.open(tmp_path / "image/s/0000_mask.png"))
    assert ids[0].tolist() == [1] * 4 and ids[3].tolist() == [2] * 4 and ids[1:3].max() == 0
    assert json.loads((tmp_path / "image/s/0000_classes.json").read_text())["2"]["instance"] == 2


def test_stage_multiview_writes_each_view(tmp_path):
    masks = np.zeros((3, 2, 4, 4), bool)
    masks[2, 1, 1] = True                                 # object 2 in view 2, row 1
    bridge.stage(tmp_path, "multiview", "s", np.zeros((3, 4, 4, 3), np.uint8), masks)
    scene = tmp_path / "multiview/s"
    assert sorted(p.name for p in (scene / "rgbs").iterdir()) == ["00000.png", "00001.png", "00002.png"]
    assert np.asarray(Image.open(scene / "segmentations/00002.png"))[1].tolist() == [2] * 4
    assert (scene / "segmentations/classes.json").is_file()


def test_multiview_inputs_object_major_masks():
    images = torch.rand(3, 8, 8, 3)
    first, second = torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)   # at half the resolution
    first[:, :2], second[1, 2:] = 1, 1
    rgbs, masks = nodes.inputs(images, torch.cat([first, second]))
    assert rgbs.shape == (3, 8, 8, 3) and masks.shape == (3, 2, 8, 8)
    assert masks[:, 0, :4].all() and masks[1, 1, 4:].all() and not masks[0, 1].any()
    with pytest.raises(ValueError, match="every view"):
        nodes.inputs(images, torch.ones(4, 8, 8))


def test_inputs_from_a_sam3_track():
    pytest.importorskip("comfy.ldm.sam3.tracker")
    images = torch.rand(2, 8, 16, 3)
    bits = torch.zeros(2, 1, 4, 1, dtype=torch.uint8)    # (views, objects, Hm, Wm / 8)
    bits[1, 0, :2] = 0xFF                                # object 1, view 1, top half
    rgbs, masks = nodes.inputs(images, {"packed_masks": bits, "orig_size": (8, 16), "n_frames": 2})
    assert masks.shape == (2, 1, 8, 16) and not masks[0].any()
    assert masks[1, 0, :3].all() and not masks[1, 0, 5:].any()
    with pytest.raises(ValueError, match="3 views"):
        nodes.inputs(torch.rand(3, 8, 16, 3), {"packed_masks": bits})


def test_example_scene_is_object_major(monkeypatch):
    masks = np.zeros((2, 3, 4, 4), bool)                  # 2 views, 3 objects
    for v in range(2):
        for k in range(3):
            masks[v, k, k, v] = True
    monkeypatch.setattr(bridge, "load_example", lambda name: (np.zeros((2, 4, 4, 3), np.uint8), masks))
    images, objects = nodes.GenIAExampleScene.execute("multiview/x").args
    assert images.shape == (2, 4, 4, 3) and objects.shape == (6, 4, 4)
    rgbs, back = nodes.inputs(images, objects)          # what Reconstruct makes of it
    assert (back == masks).all()


def test_load_example_reads_ids(tmp_path, monkeypatch):
    scene = tmp_path / "demo/data/multiview/s"
    (scene / "rgbs").mkdir(parents=True)
    (scene / "segmentations").mkdir()
    for v in range(2):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(scene / f"rgbs/{v:05d}.png")
        ids = np.zeros((4, 4), np.uint8)
        ids[v, :] = 3 if v else 1                         # object id 3 only in view 1
        Image.fromarray(ids).save(scene / f"segmentations/{v:05d}.png")
    monkeypatch.setattr(bridge, "genia_root", lambda: tmp_path)
    assert bridge.example_scenes() == ["multiview/s"]
    images, masks = bridge.load_example("multiview/s")
    assert images.shape == (2, 4, 4, 3) and masks.shape == (2, 2, 4, 4)
    assert masks[0, 0, 0].all() and masks[1, 1, 1].all() and not masks[1, 0].any()


def test_scene_name_is_the_inputs():
    a = np.zeros((1, 4, 4, 3), np.uint8)
    m = np.zeros((1, 1, 4, 4), bool)
    assert bridge.scene_name(a, m) == bridge.scene_name(a.copy(), m.copy())
    m2 = m.copy()
    m2[0, 0, 0, 0] = True
    assert bridge.scene_name(a, m) != bridge.scene_name(a, m2)


def _glb(path, verts, faces, colours, matrix):
    """A one-triangle-list GLB, its node placed by ``matrix`` (4x4, row-major)."""
    pos, idx = verts.astype("<f4").tobytes(), faces.astype("<u4").tobytes()
    col = (colours * 255).astype("u1").tobytes()
    binary = idx + pos + col
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "matrix": np.asarray(matrix).T.ravel().tolist()}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "COLOR_0": 2}, "indices": 0}]}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(idx)},
                            {"buffer": 0, "byteOffset": len(idx), "byteLength": len(pos)},
                            {"buffer": 0, "byteOffset": len(idx) + len(pos), "byteLength": len(col)}],
            "accessors": [{"bufferView": 0, "componentType": 5125, "count": faces.size, "type": "SCALAR"},
                          {"bufferView": 1, "componentType": 5126, "count": len(verts), "type": "VEC3"},
                          {"bufferView": 2, "componentType": 5121, "count": len(verts), "type": "VEC4",
                           "normalized": True}]}
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binary), 0x004E4942) + binary
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)


def test_read_glb_mesh_applies_the_node_matrix(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    colours = np.array([[1, 0, 0, 1]] * 3, np.float32)
    matrix = np.eye(4)
    matrix[:3, 3] = (2, 3, 4)
    _glb(tmp_path / "m.glb", verts, np.array([[0, 1, 2]]), colours, matrix)
    v, f, c = geometry.read_glb_mesh(tmp_path / "m.glb")
    assert np.allclose(v, verts + [2, 3, 4]) and f.tolist() == [[0, 1, 2]] and np.allclose(c, colours)


def _ply(path, rows):
    names = list(rows)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows['x'])}",
              *[f"property float {n}" for n in names], "end_header"]
    data = np.stack([rows[n] for n in names], 1).astype("<f4").tobytes()
    path.write_bytes(("\n".join(header) + "\n").encode() + data)


def test_gaussians_from_ply_activates_and_orders_sh(tmp_path):
    n = 2
    rows = {"x": [1, 2], "y": [3, 4], "z": [5, 6], "nx": [0] * n, "ny": [0] * n, "nz": [0] * n,
            "f_dc_0": [.1, .1], "f_dc_1": [.2, .2], "f_dc_2": [.3, .3],
            **{f"f_rest_{i}": [i, i] for i in range(9)},   # 3 channels x 3 degree-1 coefficients
            "opacity": [0, 0], "scale_0": [0, 0], "scale_1": [np.log(2)] * n, "scale_2": [0, 0],
            "rot_0": [2, 2], "rot_1": [0, 0], "rot_2": [0, 0], "rot_3": [0, 0]}
    _ply(tmp_path / "g.ply", {k: np.array(v, np.float32) for k, v in rows.items()})
    pos, scales, rots, opac, sh = geometry.gaussians_from_ply(tmp_path / "g.ply")
    assert np.allclose(pos[1], [2, 4, 6]) and np.allclose(scales[0], [1, 2, 1])
    assert np.allclose(rots[0], [1, 0, 0, 0]) and np.allclose(opac, 0.5)
    assert sh.shape == (2, 4, 3) and np.allclose(sh[0, 0], [.1, .2, .3])
    assert np.allclose(sh[0, 1], [0, 3, 6])   # coefficient 1 of channels R, G, B (channel-major)


def test_camera_info_of_the_input_camera():
    K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
    cam = geometry.camera_info(np.eye(4), K, target=np.array([0, 0, -2.0]))
    assert cam["quaternion"] == pytest.approx({"x": 0, "y": 0, "z": 0, "w": 1})   # R3 -> three.js
    assert cam["position"] == pytest.approx({"x": 0, "y": 0, "z": 0})
    assert cam["aspect"] == pytest.approx(640 / 480)
    assert cam["fov"] == pytest.approx(np.degrees(2 * np.arctan(240 / 500)))
    turned = np.eye(4)
    turned[:3, :3] = np.diag([-1, 1, -1])     # a half-turn about y: the quaternion's stable case
    assert geometry.camera_info(turned, K, np.zeros(3))["quaternion"] == pytest.approx(
        {"x": 0, "y": 1, "z": 0, "w": 0}, abs=1e-9)


def test_to_comfy_places_everything_on_the_ground(tmp_path):
    verts = np.array([[0, 1, 2], [1, 1.5, 2], [0, 1.2, 3]], np.float32)   # R3: y down, so y-up -1.5..-1
    _glb(tmp_path / "w.glb", verts, np.array([[0, 1, 2]]), np.ones((3, 4), np.float32), np.eye(4))
    _ply(tmp_path / "w.ply", {**{k: np.array([0, 1.5], np.float32) for k in ("x", "y", "z")},
                             **{f"f_dc_{i}": np.zeros(2, np.float32) for i in range(3)},
                             "opacity": np.zeros(2, np.float32),
                             **{f"scale_{i}": np.zeros(2, np.float32) for i in range(3)},
                             "rot_0": np.ones(2, np.float32),
                             **{f"rot_{i}": np.zeros(2, np.float32) for i in (1, 2, 3)}})
    camera = {"c2w": np.eye(4).tolist(), "K": [[500, 0, 64], [0, 500, 64], [0, 0, 1]]}
    Holder = lambda *a, **k: (a, k)
    (mv, mk), cams, (sv, _) = geometry.to_comfy(tmp_path / "w.glb", [tmp_path / "w.ply"] * 2,
                                                [camera], True, Holder, Holder)
    cam = cams[0]
    assert mv[0].shape == (1, 3, 3) and mv[0][0, :, 1].min().item() == pytest.approx(0)
    assert mk["unlit"] and mk["vertex_colors"][0, 0, :3].tolist() == pytest.approx([1, 1, 1])
    assert sv[0][0, :, 1].tolist() == pytest.approx([-1.5, 0] * 2)     # both PLYs, lifted by 1.5 in R3 (y down)
    assert cam["position"]["y"] == pytest.approx(1.5)                   # and the camera too


def test_srgb_to_linear():
    assert geometry.srgb_to_linear(np.array([0.0, 0.5, 1.0])).tolist() == pytest.approx(
        [0, 0.214, 1], abs=1e-3)


def test_run_progress_is_one_bar_for_the_whole_run():
    p = bridge.RunProgress("image", objects=2, finetune_iterations=100)
    f0, text = p.block("START")
    assert f0 == 0 and text.startswith("Step 1/9 · Depth and cameras")
    f1, _ = p.block("SHAPE_INIT")
    f2, text = p.loop("Object 1 · Stage 1", 5, 10)            # half of object 1 of 2
    f3, _ = p.loop("Object 2 · Stage 1", 0, 10)               # object 2 starts where 1 ends
    assert f0 < f1 < f2 < f3 and "object 1/2 · Stage 1 5/10" in text
    f4, _ = p.loop("Object 1 · Stage 2", 0, 10)               # a later loop never goes back
    assert f4 == f3
    f5, text = p.block("FINETUNE")
    assert f5 > f4 and "Step 7/9 · Fine-tuning" in text
    _, text = p.loop("Object 1 · Finetune", 21, 40)            # not "Fine-tuning · Finetune"
    assert "Step 7/9 · Fine-tuning · object 1/2 · 21/40" in text
    assert p.block("FINAL")[0] < 1 and p.block("SHAPE_INIT")[0] == p.fraction   # no going back


def test_run_progress_scales_finetune_with_its_iterations():
    full = bridge.RunProgress("image", 1, 100)
    fast = bridge.RunProgress("image", 1, 40)
    keys = [k for k, _l, _s in bridge.STEPS]
    i = keys.index("FINETUNE")
    assert fast.shares[i] < full.shares[i] and abs(sum(fast.shares) - 1) < 1e-9
    assert bridge.RunProgress("image", 1, 0).shares[i] == 0


def test_finetune_iterations_from_the_overrides():
    assert bridge.finetune_iterations(["a=1"]) == 100
    assert bridge.finetune_iterations(["++finetuning.num_iterations=40"]) == 40
    assert bridge.finetune_iterations(["finetuning=none", "++finetuning.num_iterations=40"]) == 0
    assert bridge.finetune_iterations(["++finetuning.num_iterations=40", "finetuning=none"]) == 0


def test_copy_template_inputs_never_overwrites(tmp_path, monkeypatch):
    data = tmp_path / "genia/demo/data"
    for source in bridge.TEMPLATE_INPUTS.values():
        (data / source).parent.mkdir(parents=True, exist_ok=True)
        (data / source).write_bytes(b"demo")
    monkeypatch.setattr(bridge, "genia_root", lambda: tmp_path / "genia")
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "genia_pablo_0.png").write_bytes(b"mine")
    assert sorted(bridge.copy_template_inputs(inputs)) == ["genia_office_buddies.png",
                                                           "genia_pablo_1.png"]
    assert (inputs / "genia_pablo_0.png").read_bytes() == b"mine"
    assert bridge.copy_template_inputs(inputs) == []
