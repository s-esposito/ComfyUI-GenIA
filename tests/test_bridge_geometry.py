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
    assert rgb.dtype == np.uint8 and rgb.shape == (8, 10, 3)
    assert masks.shape == (2, 8, 10) and masks[0, :4].all() and not masks[0, 4:].any()


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


def test_multiview_inputs_one_batch_per_object():
    images = torch.rand(3, 8, 8, 3)
    first, second = torch.zeros(3, 8, 8), torch.zeros(3, 4, 4)
    first[:, :4], second[1, 2:] = 1, 1                    # second at a lower resolution
    rgbs, masks = nodes.multiview_inputs(images, {"mask10": second, "mask1": first, "mask2": None})
    assert rgbs.shape == (3, 8, 8, 3) and masks.shape == (3, 2, 8, 8)
    assert masks[:, 0, :4].all() and masks[1, 1, 4:].all() and not masks[0, 1].any()
    with pytest.raises(ValueError, match="one per view"):
        nodes.multiview_inputs(images, {"mask0": torch.ones(2, 8, 8)})
    with pytest.raises(ValueError, match="two or more"):
        nodes.multiview_inputs(images[:1], {"mask0": torch.ones(1, 8, 8)})


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


def test_flip_is_a_half_turn_about_x():
    rot_x = lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    q = np.array([[np.cos(.2), np.sin(.2), 0, 0]])                       # 0.4 rad about x
    pos, rots, sh = geometry.flip_gaussians(np.array([[1., 2., 3.]]), q, np.ones((1, 4, 3)))
    assert np.allclose(pos, [[1, -2, -3]])
    w, x, y, z = rots[0]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(R, geometry.FLIP @ rot_x(.4))
    assert sh[0, :, 0].tolist() == [1, -1, -1, 1]


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
    (mv, mk), cams, (sv, _) = geometry.to_comfy(tmp_path / "w.glb", tmp_path / "w.ply", [camera],
                                                True, Holder, Holder)
    cam = cams[0]
    assert mv[0].shape == (1, 3, 3) and mv[0][0, :, 1].min().item() == pytest.approx(0)
    assert sv[0][0, :, 1].tolist() == pytest.approx([1.5, 0])          # splats lifted by 1.5
    assert cam["position"]["y"] == pytest.approx(1.5)                   # and the camera too
