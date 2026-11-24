"""GenIA's outputs as ComfyUI's 3D types: numpy/torch only (ComfyUI's environment has neither
trimesh nor plyfile).

GenIA's world is R3 (x right, y down, z forward: OpenCV's, the input camera's for one image).
ComfyUI's MESH and LOAD3D_CAMERA are right-handed y-up (three.js, a camera looking down -z), a
half-turn about x away (``FLIP``, which maps R3's camera axes onto three.js' too, so a camera's
rotation conjugates by it). Its SPLAT is 3DGS' own frame, R3 again (its viewer turns it upright
by the same half-turn), so the splats keep GenIA's frame.
"""
import json
import struct

import numpy as np
import torch

#: R3 -> y-up: a half-turn about x.
FLIP = np.diag([1.0, -1.0, -1.0])


def read_glb_mesh(path):
    """``(vertices (N, 3), faces (M, 3), colours (N, 4) in [0, 1] or None)`` of a GLB: every
    triangle primitive of every node, in the scene's frame (node matrices applied)."""
    data = open(path, "rb").read()
    n_json = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + n_json])
    binary = data[20 + n_json + 8:]
    dtypes = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def accessor(i):
        a = gltf["accessors"][i]
        view = gltf["bufferViews"][a["bufferView"]]
        start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        dtype, width = np.dtype(dtypes[a["componentType"]]), widths[a["type"]]
        out = np.frombuffer(binary, dtype, a["count"] * width, start).reshape(a["count"], width)
        if a.get("normalized"):
            out = out / np.iinfo(dtype).max
        return out.astype(np.float32) if out.dtype != np.uint32 else out

    def node_matrix(node):
        if "matrix" in node:
            return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T   # column-major
        m = np.eye(4)
        if "scale" in node:
            m = np.diag([*node["scale"], 1.0]) @ m
        if "rotation" in node:
            x, y, z, w = node["rotation"]
            m = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
                          [0, 0, 0, 1]]) @ m
        if "translation" in node:
            t = np.eye(4)
            t[:3, 3] = node["translation"]
            m = t @ m
        return m

    verts, faces, colours = [], [], []
    offset = 0

    def visit(i, parent):
        nonlocal offset
        node = gltf["nodes"][i]
        world = parent @ node_matrix(node)
        for prim in gltf["meshes"][node["mesh"]]["primitives"] if "mesh" in node else []:
            if prim.get("mode", 4) != 4:
                continue
            v = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
            v = v @ world[:3, :3].T + world[:3, 3]
            f = (accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
                 if "indices" in prim else np.arange(len(v)).reshape(-1, 3))
            c = accessor(prim["attributes"]["COLOR_0"]) if "COLOR_0" in prim["attributes"] else None
            if c is not None and c.shape[1] == 3:
                c = np.concatenate([c, np.ones((len(c), 1), np.float32)], 1)
            verts.append(v)
            faces.append(f + offset)
            colours.append(c if c is not None else np.ones((len(v), 4), np.float32))
            offset += len(v)
        for child in node.get("children", []):
            visit(child, world)

    scene = gltf["scenes"][gltf.get("scene", 0)]
    for root in scene["nodes"]:
        visit(root, np.eye(4))
    return (np.concatenate(verts).astype(np.float32), np.concatenate(faces),
            np.concatenate(colours).astype(np.float32))


def read_ply_vertices(path):
    """A binary little-endian PLY's ``vertex`` element as ``{property: (N,) array}``."""
    data = open(path, "rb").read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: not a binary little-endian PLY")
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
             "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4", "short": "<i2",
             "ushort": "<u2"}
    count, props, element = 0, [], None
    for line in header:
        words = line.split()
        if words[:1] == ["element"]:
            element = words[1]
            if element == "vertex":
                count = int(words[2])
            elif not props:
                raise ValueError(f"{path}: elements before 'vertex' are not supported")
        elif words[:1] == ["property"] and element == "vertex":
            props.append((words[2], types[words[1]]))
    rows = np.frombuffer(data, np.dtype(props), count, end)
    return {name: rows[name] for name, _ in props}


def gaussians_from_ply(path):
    """A 3DGS PLY (GenIA's ``save_gaussian_ply``: log scales, logit opacities, wxyz rotations,
    SH channel-major) as activated arrays ``(positions, scales, rotations, opacities, sh)``,
    ``sh`` being (N, K, 3)."""
    v = read_ply_vertices(path)
    positions = np.stack([v["x"], v["y"], v["z"]], 1)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1))
    rotations = np.stack([v[f"rot_{i}"] for i in range(4)], 1)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    opacities = 1 / (1 + np.exp(-v["opacity"][:, None]))
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[:, None]          # (N, 1, 3)
    rest_names = sorted((k for k in v if k.startswith("f_rest_")), key=lambda k: int(k[7:]))
    if rest_names:
        rest = np.stack([v[k] for k in rest_names], 1)                       # (N, 3 * (K-1))
        rest = rest.reshape(len(rest), 3, -1).transpose(0, 2, 1)             # channel-major
        sh = np.concatenate([dc, rest], 1)
    else:
        sh = dc
    return positions, scales, rotations, opacities, sh


def _quaternion_wxyz(R):
    """A rotation matrix's unit quaternion (w, x, y, z), from its largest component (stable
    for every rotation, half-turns included)."""
    trace = np.trace(R)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q = (s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = ((R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s)
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4)
    q = np.array(q)
    return q / np.linalg.norm(q)


def camera_info(c2w, K, target):
    """ComfyUI's ``LOAD3D_CAMERA`` for an R3 camera ``c2w`` (4x4) with intrinsics ``K`` (pixel
    corners: the image is 2 cx by 2 cy), looking at ``target`` (y-up world)."""
    c2w = np.asarray(c2w, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    rotation = FLIP @ c2w[:3, :3] @ FLIP          # its axes are three.js' camera axes
    position = FLIP @ c2w[:3, 3]
    width, height = 2 * K[0, 2], 2 * K[1, 2]
    q = _quaternion_wxyz(rotation)
    xyz = lambda p: {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
    return {
        "position": xyz(position), "target": xyz(target), "zoom": 1,
        "cameraType": "perspective",
        "quaternion": {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": float(q[0])},
        "useCustomUp": True,
        "fov": float(np.degrees(2 * np.arctan(height / (2 * K[1, 1])))),
        "aspect": float(width / height),
    }


def srgb_to_linear(c):
    """sRGB values in [0, 1] as linear ones (glTF's COLOR_0, ComfyUI's MESH colours)."""
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def merged_gaussians(paths):
    """``gaussians_from_ply`` of several PLYs as one set (SH padded to the most bands)."""
    parts = [gaussians_from_ply(p) for p in paths]
    k = max(part[4].shape[1] for part in parts)
    parts = [(*part[:4], np.pad(part[4], ((0, 0), (0, k - part[4].shape[1]), (0, 0))))
             for part in parts]
    return tuple(np.concatenate(arrays) for arrays in zip(*parts))


def to_comfy(mesh_glb, splat_plys, cameras, place_on_ground, MESH, SPLAT):
    """``(MESH, [LOAD3D_CAMERA], SPLAT)`` from GenIA's world mesh, splat PLYs (merged: the
    objects, and the background if given) and input cameras (``poses.json``'s entries), in
    ComfyUI's frames (mesh and cameras y-up, splats 3DGS'); with ``place_on_ground`` all are
    raised so the mesh rests on y = 0 (up). ``MESH``/``SPLAT`` are ComfyUI's classes.

    GenIA's vertex colours are display (sRGB) values with the photo's shading in them: they
    become linear (ComfyUI's and glTF's convention) on an unlit mesh, which ComfyUI's viewer
    and Save GLB (``KHR_materials_unlit``) then show as they are, as GenIA's demo does."""
    verts, faces, colours = read_glb_mesh(mesh_glb)
    verts = verts @ FLIP.T
    colours = np.concatenate([srgb_to_linear(colours[:, :3]), colours[:, 3:]], 1)
    positions, scales, rotations, opacities, sh = merged_gaussians(splat_plys)
    lift = np.array([0.0, -verts[:, 1].min() if place_on_ground else 0.0, 0.0])
    verts, positions = verts + lift, positions + FLIP @ lift   # the lift, in each one's frame
    target = (verts.min(0) + verts.max(0)) / 2
    infos = []
    for camera in cameras:
        c2w = np.asarray(camera["c2w"], dtype=np.float64).copy()
        c2w[:3, 3] += FLIP @ lift   # the lift, in R3 (FLIP is its own inverse)
        infos.append(camera_info(c2w, camera["K"], target))
    t = lambda a, dtype=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)[None]
    mesh = MESH(t(verts), t(faces, torch.int64), vertex_colors=t(colours), unlit=True)
    splat = SPLAT(t(positions), t(scales), t(rotations), t(opacities), t(sh))
    return mesh, infos, splat
