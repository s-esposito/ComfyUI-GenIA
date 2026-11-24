"""ComfyUI-GenIA nodes (v3 API)."""
import re
from pathlib import Path

import numpy as np
from PIL import Image

from comfy_api.latest import IO

from . import bridge, geometry

def _run_inputs():
    """The inputs both nodes share, after their images and masks."""
    return [
        IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                     control_after_generate=True),
        IO.Combo.Input("quality", options=["fast", "full"], default="fast",
                       tooltip="full: GenIA's paper settings (FINETUNE 100 iterations); "
                               "fast: 40, ~35-64% less time, slightly softer texture."),
        IO.Combo.Input("low_vram", options=["auto", "on", "off"], default="auto",
                       tooltip="Trade speed for GPU memory (fits a 24 GB GPU); auto: on "
                               "below 44 GiB of free GPU memory."),
        IO.Boolean.Input("place_on_ground", default=True,
                         tooltip="Rest the reconstruction on the ground plane (y = 0)."),
    ]


def _run(setting, images, masks, seed, quality, low_vram, place_on_ground):
    """GenIA on ``images`` (V, H, W, 3) uint8 with ``masks`` (V, N, H, W) bool, in ComfyUI:
    its models make room, its progress drives the node's bar, its cancel stops the run.
    ``(MESH, [LOAD3D_CAMERA] per view, SPLAT)``."""
    import comfy.model_management as mm
    import comfy.utils
    import folder_paths

    mm.unload_all_models()   # GenIA's models need the room
    mm.soft_empty_cache()
    bar = {}

    def progress(task, n, total):
        if total and bar.get("task") != task:
            bar.update(task=task, bar=comfy.utils.ProgressBar(total))
        if total:
            bar["bar"].update_absolute(n, total)

    try:
        glb, ply, cameras = bridge.reconstruct(
            setting, images, masks, seed=seed, low_vram=low_vram, quality=quality,
            work_dir=Path(folder_paths.get_temp_directory()) / "genia",
            progress=progress, interrupted=mm.processing_interrupted)
    except RuntimeError as exc:
        if str(exc) == "stopped":
            mm.throw_exception_if_processing_interrupted()
        raise
    return geometry.to_comfy(glb, ply, cameras, place_on_ground, IO.Mesh.Type, IO.Splat.Type)


class GenIAReconstructImage(IO.ComfyNode):
    """Single image (+ object masks) -> mesh, camera and Gaussian splat via GenIA."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GenIAReconstructImage",
            display_name="GenIA Reconstruct (Image)",
            category="3d/GenIA",
            description="Reconstruct the objects in one image with GenIA: what the image observes "
                        "is reconstructed, what it does not is generated.",
            is_experimental=True,
            inputs=[
                IO.Image.Input("image", tooltip="The input image."),
                IO.Mask.Input(
                    "mask", optional=True,
                    tooltip="Object masks: one mask per object as a batch (multi-object), or a "
                            "single white-on-black mask (object = white). Required.",
                ),
                *_run_inputs(),
            ],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Splat.Output(display_name="splat"),
            ],
        )

    @classmethod
    def execute(cls, image, seed, quality, low_vram, place_on_ground, mask=None) -> IO.NodeOutput:
        rgb, masks = inputs(image, mask)
        mesh, cameras, splat = _run("image", rgb[None], masks[None], seed, quality, low_vram,
                                    place_on_ground)
        return IO.NodeOutput(mesh, cameras[0], splat)


class GenIAReconstructMultiview(IO.ComfyNode):
    """Several views of one static scene (+ object masks per view) -> mesh, one camera per
    view and Gaussian splat via GenIA."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GenIAReconstructMultiview",
            display_name="GenIA Reconstruct (Multi-view)",
            category="3d/GenIA",
            description="Reconstruct the objects seen in several views of one static scene with "
                        "GenIA; the views' cameras are estimated.",
            is_experimental=True,
            inputs=[
                IO.Image.Input("images", tooltip="The views, as one batch (two or more)."),
                IO.Autogrow.Input("objects", template=IO.Autogrow.TemplatePrefix(
                    IO.Mask.Input("mask", tooltip="One object: its mask in every view, as a "
                                                  "batch in the views' order."),
                    prefix="mask", min=1, max=20)),
                *_run_inputs(),
            ],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.Load3DCamera.Output(display_name="camera_info",
                                       tooltip="Every view's camera, as a camera list."),
                IO.Splat.Output(display_name="splat"),
            ],
        )

    @classmethod
    def execute(cls, images, objects: IO.Autogrow.Type, seed, quality, low_vram,
                place_on_ground) -> IO.NodeOutput:
        rgbs, masks = multiview_inputs(images, objects)
        return IO.NodeOutput(*_run("multiview", rgbs, masks, seed, quality, low_vram,
                                   place_on_ground))


def _resized(mask: np.ndarray, shape) -> np.ndarray:
    """A float mask at ``shape`` (H, W), as booleans."""
    if mask.shape != tuple(shape):
        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (shape[1], shape[0]), Image.NEAREST)) / 255.0
    return mask > 0.5


def inputs(image, mask):
    """ComfyUI's IMAGE and MASK as GenIA wants them: ``(uint8 (H, W, 3), bool (N, H, W))``,
    one mask per object, at the image's size."""
    if mask is None:
        raise ValueError("GenIA needs the objects' masks: connect a MASK (one per object as a "
                         "batch, or one white-on-black mask), e.g. from a segmentation node.")
    rgb = (image[0, ..., :3].clamp(0, 1) * 255).round().byte().cpu().numpy()
    m = mask if mask.ndim == 3 else mask[None]
    masks = [one for one in (_resized(x, rgb.shape[:2]) for x in m.float().cpu().numpy())
             if one.any()]
    if not masks:
        raise ValueError("Every mask is empty: GenIA has no object to reconstruct.")
    return rgb, np.stack(masks)


def multiview_inputs(images, objects: dict):
    """ComfyUI's view batch and per-object mask batches as ``(uint8 (V, H, W, 3), bool
    (V, N, H, W))``: object ``k`` is the ``k``-th connected mask input, in order."""
    rgbs = (images[..., :3].clamp(0, 1) * 255).round().byte().cpu().numpy()
    views = len(rgbs)
    if views < 2:
        raise ValueError("Multi-view needs two or more views in `images`; use GenIA Reconstruct "
                         "(Image) for one.")
    masks = []
    # Autogrow names them mask0, mask1, ... (in the API: "objects.mask0").
    for name, batch in sorted(objects.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0)):
        if batch is None:
            continue
        batch = batch if batch.ndim == 3 else batch[None]
        if len(batch) != views:
            raise ValueError(f"{name} has {len(batch)} masks for {views} views: give one per view.")
        one = np.stack([_resized(m, rgbs.shape[1:3]) for m in batch.float().cpu().numpy()])
        if one.any():
            masks.append(one)
    if not masks:
        raise ValueError("Every mask is empty: GenIA has no object to reconstruct.")
    return rgbs, np.stack(masks, axis=1)
