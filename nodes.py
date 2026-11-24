"""ComfyUI-GenIA nodes (v3 API)."""
import io
import shlex
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import IO, UI, InputImpl

from . import bridge, geometry

#: ComfyUI's SAM3 video tracker output (``comfy_extras/nodes_sam3.py``).
SAM3_TRACK = IO.Custom("SAM3_TRACK_DATA")

#: Object tints of the example scene's preview.
TINTS = np.array([[230, 60, 60], [60, 200, 90], [70, 120, 240], [240, 190, 40], [200, 80, 220],
                  [40, 200, 210], [250, 130, 40], [150, 150, 150]], np.float32)

#: The node's ``quality`` -> the GenIA demo app's run options ticked (``demo/method.py``).
QUALITY = {"draft": {"No test-time refinement"}, "fast": set(), "full": {"Full quality"}}
ICP = "ICP pose refinement"


class GenIAReconstruct(IO.ComfyNode):
    """One photo, or several views of one static scene, with its objects' masks -> a mesh,
    the cameras, a Gaussian splat and a turntable video, via GenIA."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GenIAReconstruct",
            display_name="GenIA Reconstruct",
            category="3d/GenIA",
            description="Reconstruct the objects in one photo, or in several views of one static "
                        "scene (their cameras are estimated): what the photos observe is "
                        "reconstructed, what they do not is generated.",
            is_experimental=True,
            inputs=[
                IO.Image.Input("images", tooltip="One photo, or the views of one static scene "
                                                 "as a batch."),
                IO.MultiType.Input(
                    "objects", types=[IO.Mask, SAM3_TRACK],
                    tooltip="Which objects to reconstruct. A MASK batch: one mask per object "
                            "(for several views: each object's masks in view order, the "
                            "objects one after another), e.g. from SAM3 Detect with "
                            "individual_masks. Or SAM3 Video Track run over the views: one "
                            "object per tracked object."),
                IO.Combo.Input("quality", options=list(QUALITY), default="fast",
                               tooltip="full: GenIA's paper settings. fast: FINETUNE 40 "
                                       "iterations instead of 100, ~35-64% less time, slightly "
                                       "softer texture. draft: no FINETUNE."),
                IO.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff,
                             control_after_generate=IO.ControlAfterGenerate.fixed,
                             advanced=True,
                             tooltip="42: GenIA's default, the demo app's results."),
                IO.Combo.Input("pose_refinement", options=["photometric", "icp"],
                               default="photometric", advanced=True,
                               tooltip="photometric: a per-frame appearance pass, then a "
                                       "render-based pose refinement. icp: no early appearance "
                                       "pass, an ICP refinement against the shape."),
                IO.Combo.Input("low_vram", options=["auto", "on", "off"], default="auto",
                               advanced=True,
                               tooltip="Trade speed for GPU memory (fits a 24 GB GPU); auto: on "
                                       "below 44 GiB of free GPU memory."),
                IO.Boolean.Input("place_on_ground", default=True, advanced=True,
                                 tooltip="Rest the reconstruction on the ground plane (y = 0)."),
                IO.Boolean.Input("background", default=True, advanced=True,
                                 tooltip="Add the rest of the photos to the splat: the scene "
                                         "around the objects, as GenIA's demo shows it."),
                IO.String.Input("extra_overrides", default="", advanced=True,
                                tooltip="GenIA Hydra overrides added last, e.g. "
                                        "finetuning.num_iterations=60 (the demo app's "
                                        "\"Extra Hydra overrides\")."),
            ],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.Load3DCamera.Output(display_name="camera_info",
                                       tooltip="The input camera (for several views, the "
                                               "first view's): Save Splat seen from it shows "
                                               "the input view."),
                IO.Splat.Output(display_name="splat"),
                IO.Video.Output(display_name="orbit", tooltip="A turntable of the result."),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, images, objects, quality, seed=42, pose_refinement="photometric",
                low_vram="auto", place_on_ground=True, background=True,
                extra_overrides="") -> IO.NodeOutput:
        rgbs, masks = inputs(images, objects)
        options = QUALITY[quality] | ({ICP} if pose_refinement == "icp" else set())
        mesh, cameras, splat, orbit = _run(rgbs, masks, seed, options, low_vram,
                                           place_on_ground, background, extra_overrides,
                                           cls.hidden.unique_id)
        return IO.NodeOutput(mesh, cameras[0], splat, orbit)


class GenIAExampleScene(IO.ComfyNode):
    """One of the GenIA demo's bundled scenes: its photos and its objects' masks."""

    @classmethod
    def define_schema(cls):
        scenes = bridge.example_scenes()
        default = "image/sloth" if "image/sloth" in scenes else next(iter(scenes), None)
        return IO.Schema(
            node_id="GenIAExampleScene",
            display_name="GenIA Example Scene",
            category="3d/GenIA",
            description="A scene bundled with GenIA's demo (its demo/data): one photo or "
                        "several views, with its objects' masks, ready for GenIA Reconstruct.",
            is_output_node=True,   # runs on its own too ("Run selected"): a preview of the scene
            inputs=[IO.Combo.Input("scene", options=scenes,
                                   default=default,
                                   tooltip="image/...: one photo; multiview/...: several views.")],
            outputs=[IO.Image.Output(display_name="images"),
                     IO.Mask.Output(display_name="objects")],
        )

    @classmethod
    def execute(cls, scene) -> IO.NodeOutput:
        images, masks = bridge.load_example(scene)           # (V, H, W, 3), (V, N, H, W)
        v, n, h, w = masks.shape
        objects = masks.transpose(1, 0, 2, 3).reshape(n * v, h, w)   # object-major
        return IO.NodeOutput(torch.from_numpy(images).float() / 255,
                             torch.from_numpy(objects).float(),
                             ui=UI.PreviewImage(preview(images, masks), cls=cls))


def preview(images: np.ndarray, masks: np.ndarray) -> torch.Tensor:
    """The views with each object tinted (its colour at 45%), as an IMAGE batch."""
    out = images.astype(np.float32)
    for k in range(masks.shape[1]):
        m = masks[:, k, ..., None]
        out = np.where(m, 0.55 * out + 0.45 * TINTS[k % len(TINTS)], out)
    return torch.from_numpy(out.clip(0, 255) / 255)


def _run(rgbs, masks, seed, options, low_vram, place_on_ground, background, extra, node_id):
    """GenIA on ``rgbs`` (V, H, W, 3) uint8 with ``masks`` (V, N, H, W) bool, in ComfyUI: its
    models make room, its progress drives the node's bar and status text, its cancel stops
    the run. ``(MESH, [LOAD3D_CAMERA] per view, SPLAT, VIDEO or None)``."""
    import comfy.model_management as mm
    import comfy.utils
    import folder_paths
    from server import PromptServer

    mm.unload_all_models()   # GenIA's models need the room
    mm.soft_empty_cache()
    bar = comfy.utils.ProgressBar(1000)   # the whole run (bridge.RunProgress)

    def progress(fraction, text):
        bar.update_absolute(round(fraction * 1000), 1000)
        if node_id is not None:
            PromptServer.instance.send_progress_text(text, node_id)

    try:
        glb, ply, background_ply, cameras, orbit = bridge.reconstruct(
            rgbs, masks, seed=seed, low_vram=low_vram, options=options, extra=shlex.split(extra),
            work_dir=Path(folder_paths.get_temp_directory()) / "genia",
            progress=progress, interrupted=mm.processing_interrupted)
    except RuntimeError as exc:
        if str(exc) == "stopped":
            mm.throw_exception_if_processing_interrupted()
        raise
    plys = [ply, background_ply] if background and background_ply.is_file() else [ply]
    mesh, infos, splat = geometry.to_comfy(glb, plys, cameras, place_on_ground,
                                           IO.Mesh.Type, IO.Splat.Type)
    video = InputImpl.VideoFromFile(io.BytesIO(orbit)) if orbit else None
    return mesh, infos, splat, video


def _resized(mask: np.ndarray, shape) -> np.ndarray:
    """A float mask at ``shape`` (H, W), as booleans."""
    if mask.shape != tuple(shape):
        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (shape[1], shape[0]), Image.NEAREST)) / 255.0
    return mask > 0.5


def _track_masks(track: dict, views: int, shape) -> np.ndarray:
    """SAM3 Video Track's objects as ``(views, N, H, W)`` bool, at ``shape`` (H, W)."""
    from comfy.ldm.sam3.tracker import unpack_masks

    packed = track.get("packed_masks")
    if packed is None or packed.shape[1] == 0:
        raise ValueError("SAM3 tracked no object: GenIA has nothing to reconstruct.")
    if packed.shape[0] != views:
        raise ValueError(f"The SAM3 track has {packed.shape[0]} frames for {views} views: run "
                         "SAM3 Video Track over the same images.")
    m = unpack_masks(packed).float()                                    # (V, N, Hm, Wm)
    m = torch.nn.functional.interpolate(m, size=tuple(shape), mode="bilinear",
                                        align_corners=False)
    return (m > 0.5).cpu().numpy()


def inputs(images, objects):
    """ComfyUI's IMAGE batch and objects (a MASK batch, object-major over the views, or SAM3
    Video Track data) as GenIA wants them: ``(uint8 (V, H, W, 3), bool (V, N, H, W))``,
    objects empty in every view left out."""
    rgbs = (images[..., :3].clamp(0, 1) * 255).round().byte().cpu().numpy()
    views, shape = len(rgbs), rgbs.shape[1:3]
    if objects is None:
        raise ValueError("GenIA needs the objects' masks: connect a MASK (one per object) or "
                         "SAM3 Video Track's output.")
    if isinstance(objects, dict):
        masks = _track_masks(objects, views, shape)
    else:
        m = objects if objects.ndim == 3 else objects[None]
        if len(m) % views:
            raise ValueError(f"{len(m)} masks for {views} views: give each object's mask in "
                             f"every view (objects one after another, {views} masks each).")
        m = np.stack([_resized(x, shape) for x in m.float().cpu().numpy()])
        masks = m.reshape(len(m) // views, views, *shape).transpose(1, 0, 2, 3)
    masks = masks[:, masks.any(axis=(0, 2, 3))]
    if not masks.shape[1]:
        raise ValueError("Every mask is empty: GenIA has no object to reconstruct.")
    return rgbs, masks
