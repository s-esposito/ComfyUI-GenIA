"""The way from a ComfyUI node to GenIA: stage the inputs as a scene, run GenIA, read it back.

GenIA runs one of two ways, decided once per ComfyUI process:

* **In process**, when GenIA imports in ComfyUI's own environment (ComfyUI installed into
  GenIA's environment, the recommended setup): its models load into ComfyUI's process once
  and stay, and each run happens there.
* **In a worker process** otherwise: GenIA's own environment runs a warm worker (GenIA's
  ``demo/ui/worker.py``) whose Python ``GENIA_PYTHON`` names; the models stay loaded there.

Both are GenIA's demo runners (``demo/ui/{inprocess,worker}.py``), driven with the recipe its
demo app uses (``demo/method.py``), so a ComfyUI run is exactly an app run. GenIA's checkout
is ``GENIA_ROOT``, else the one an importable ``genia`` package comes from.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_runner = None    # GenIA's runner, kept for the process (its models stay loaded)
_method = None    # GenIA's demo/method.py


def genia_root() -> Path:
    if root := os.environ.get("GENIA_ROOT"):
        return Path(root).resolve()
    try:
        import genia.core
    except ImportError:
        raise RuntimeError("GenIA not found: set GENIA_ROOT to its checkout (and GENIA_PYTHON "
                           "to its environment's python, unless ComfyUI runs in it)") from None
    return Path(genia.core.__file__).resolve().parents[1]


def _davis_palette() -> list[int]:
    """The DAVIS palette GenIA's masks use (any palette would do: only the indices matter)."""
    out = []
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        out += [r, g, b]
    return out


def _save_ids(masks: np.ndarray, path: Path) -> None:
    """Object masks ``(N, H, W)`` as one palette PNG of ids (object ``i`` = mask ``i - 1``; a
    later mask wins an overlap), as GenIA's scenes store them."""
    ids = np.zeros(masks.shape[1:], np.uint8)
    for i, m in enumerate(masks, start=1):
        ids[m] = i
    image = Image.fromarray(ids, mode="P")
    image.putpalette(_davis_palette())
    image.save(path)


def stage(root: Path, setting: str, scene: str, images: np.ndarray, masks: np.ndarray) -> None:
    """``root/setting/scene/`` as GenIA's demo reads it (laid out like its ``demo/data``):
    ``images`` (V, H, W, 3) uint8, ``masks`` (V, N, H, W) bool, one per view and object.
    ``image``: ``0000.png`` + ``0000_mask.png``; ``multiview``: ``rgbs/`` + ``segmentations/``."""
    scene_dir = root / setting / scene
    classes = {str(i): {"class": f"object {i}", "instance": i}
               for i in range(1, masks.shape[1] + 1)}
    if setting == "image":
        scene_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(images[0]).save(scene_dir / "0000.png")
        _save_ids(masks[0], scene_dir / "0000_mask.png")
        classes_path = scene_dir / "0000_classes.json"
    else:
        for sub in ("rgbs", "segmentations"):
            (scene_dir / sub).mkdir(parents=True, exist_ok=True)
        for v, (image, view_masks) in enumerate(zip(images, masks)):
            Image.fromarray(image).save(scene_dir / "rgbs" / f"{v:05d}.png")
            _save_ids(view_masks, scene_dir / "segmentations" / f"{v:05d}.png")
        classes_path = scene_dir / "segmentations" / "classes.json"
    classes_path.write_text(json.dumps(classes))


def runner(roomy: bool):
    """GenIA's runner for this process (created, and its models loaded, on first use)."""
    global _runner, _method
    if _runner is not None:
        return _runner, _method
    root = genia_root()
    for path in (str(root.parent), str(root / "demo")):   # appended: ComfyUI's modules win
        if path not in sys.path:
            sys.path.append(path)
    import method
    from ui.inprocess import InProcess
    from ui.worker import Worker

    try:
        import genia.core.run  # noqa: F401 — GenIA's own dependencies import here
        in_process = True
    except ImportError:
        in_process = False
    warm = method.worker_overrides(roomy)
    if in_process:
        os.environ.update(method.ENV)
        _runner = InProcess(warm=warm)
    else:
        python = os.environ.get("GENIA_PYTHON")
        if not python:
            raise RuntimeError("GenIA's dependencies are not in ComfyUI's environment: set "
                               "GENIA_PYTHON to the python of GenIA's environment")
        _runner = Worker(method.ENTRY, {**os.environ, **method.ENV}, warm=warm, cwd=root,
                         python=python)
    _method = method
    _runner.start()
    return _runner, _method


def reconstruct(setting, images, masks, *, seed, low_vram, quality, work_dir: Path,
                progress=None, interrupted=lambda: False):
    """Run GenIA's ``setting`` (``image`` | ``multiview``) on ``images`` (V, H, W, 3) with object
    ``masks`` (V, N, H, W); ``(mesh_glb, splat_ply, cameras)``: the paths of its world mesh and
    splats, and ``poses.json``'s cameras in view order.

    ``progress(task, n, total)`` gets GenIA's loop progress; ``interrupted()`` is polled
    while it runs (True stops it: RuntimeError("stopped")).
    """
    import torch

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    free = torch.cuda.mem_get_info()[0] / 2**30
    # ComfyUI runs nodes under torch.inference_mode(): models loaded there would be inference
    # tensors, which GenIA's test-time optimisation cannot backpropagate through. (Its run
    # itself is on a thread of its own, which does not inherit the mode.)
    with torch.inference_mode(False):
        run, method = runner(roomy=total >= 40)
    scene = time.strftime("comfy_%Y%m%d_%H%M%S")
    data_root, run_dir = work_dir / "data", work_dir / "runs" / scene
    stage(data_root, setting, scene, images, masks)
    low = {"auto": free < method.LOW_VRAM_BELOW_GIB, "on": True, "off": False}[low_vram]
    # fast = the app's default (GenIA's demo/method.py OPTIONS, "Full quality" unticked)
    args = [*(["++finetuning.num_iterations=40"] if quality == "fast" else []),
            f"processing.seed={seed % 2**32}",
            "output.save_compressed_ply=false",   # the plain 3DGS PLY ComfyUI's reader takes
            "output.save_viz_orbit=false",
            *run.warm, f"output.output_dir={run_dir}"]
    overrides = method.run_overrides(setting, data_root, scene, low, args)
    log, pending = [], ""
    output = run.run(overrides)
    for chunk in output:
        log.append(chunk)
        pending += chunk
        *lines, pending = pending.split("\n")
        for line in lines:
            if progress and (cut := line.find("@@progress ")) >= 0:
                try:
                    p = json.loads(line[cut + len("@@progress "):])
                    progress(p["task"], p["n"], p["total"])
                except (ValueError, KeyError):
                    pass
        if interrupted():
            run.stop()
            for _ in output:   # until it has ended (in process: at its next loop step)
                pass
            raise RuntimeError("stopped")
    if run.returncode:
        tail = "".join(log)[-3000:]
        raise RuntimeError(f"GenIA failed (exit code {run.returncode}):\n{tail}")
    final = run_dir / "final"
    cameras = sorted(json.loads((final / "poses.json").read_text())["cameras"],
                     key=lambda c: (c["view"], c["frame"]))
    return final / "meshes" / "world.glb", final / "gaussians" / "world.ply", cameras
