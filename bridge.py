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
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_runner = None    # GenIA's runner, kept for the process (its models stay loaded)
_method = None    # GenIA's demo/method.py

#: The scenes whose block caches are kept (the most recently run; a scene is its inputs).
KEEP_SCENES = 3

#: A tqdm bar's line (its progress reaches the node as ``@@progress`` lines instead).
_BAR = re.compile(r"\d+%\|")
_log = logging.getLogger("GenIA")
#: A block's per-object result line, e.g. "FINETUNE obj 1: PSNR 22.85 → 24.12 ...".
_RESULT = re.compile(r"^\s+[A-Z_0-9]+ obj \d+: ")

#: A block's header in GenIA's log, e.g. "──── SHAPE_INIT: Shape Init (strategy=parallel) ────".
_BLOCK = re.compile(r"^─+ ([A-Z][A-Z_0-9]*): (.+?) ─+$")


#: The run's steps, in order: (log key, label, share of a full-quality run by setting), the
#: shares measured on the demo's recorded runs (RTX 4090; FINETUNE's scales with its
#: iterations). ``START``: before the first block (models, depth and cameras); ``FINAL``: from
#: GenIA's "EVALUATION: Final state" on (the exports).
STEPS = (
    ("START", "Depth and cameras", {"image": 0.10, "multiview": 0.05}),
    ("SHAPE_INIT", "Shape", {"image": 0.055, "multiview": 0.065}),
    ("APPEARANCE_INIT", "Appearance", {"image": 0.03, "multiview": 0.05}),
    ("POSE_INIT", "Pose", {"image": 0.015, "multiview": 0.01}),
    ("GLOBAL_POSE_REFINE_1", "Pose refinement", {"image": 0.025, "multiview": 0.08}),
    ("APPEARANCE_INIT_2", "Appearance from every view", {"image": 0.055, "multiview": 0.065}),
    ("FINETUNE", "Fine-tuning", {"image": 0.61, "multiview": 0.5}),
    ("GLOBAL_POSE_REFINE_2", "Final pose refinement", {"image": 0.07, "multiview": 0.14}),
    ("FINAL", "Saving the results", {"image": 0.04, "multiview": 0.04}),
)


class RunProgress:
    """One progress for the whole run, from GenIA's log: each step (``STEPS``) takes its share
    of the bar, each object an equal part of its step, a loop its object's part; the bar never
    goes back. ``block(key)`` at a block's header, ``loop(task, n, total)`` at a loop's
    ``@@progress`` line; both return ``(fraction, text)``."""

    def __init__(self, setting: str, objects: int, finetune_iterations: int = 100) -> None:
        shares = [share[setting] * (finetune_iterations / 100 if key == "FINETUNE" else 1)
                  for key, _label, share in STEPS]
        self.shares = [s / sum(shares) for s in shares]
        self.starts = [0.0, *itertools.accumulate(self.shares)][:-1]
        self.objects, self.step, self.seen = max(objects, 1), 0, []
        self.fraction, self.t0 = 0.0, time.time()

    def block(self, key: str):
        keys = [k for k, _l, _s in STEPS]
        if key in keys and keys.index(key) > self.step:
            self.step, self.seen = keys.index(key), []
        return self._at(self.starts[self.step], "")

    def loop(self, task: str, n: int, total: int):
        part, obj, name = 0.0, "", task
        if task.startswith("Object ") and " · " in task:
            key, name = task.split(" · ", 1)
            if key not in self.seen:
                self.seen.append(key)
            k = self.seen.index(key)
            part = (k + n / total) / self.objects if total else k / self.objects
            obj = f"object {k + 1}/{self.objects}" if self.objects > 1 else ""
        elif total:
            part = n / total
        if _same(name, STEPS[self.step][1]):   # "Finetune" only restates "Fine-tuning"
            name = ""
        loop = " ".join(x for x in (name, f"{n}/{total}" if total else "") if x)
        detail = " · ".join(x for x in (obj, loop) if x)
        return self._at(self.starts[self.step] + self.shares[self.step] * min(part, 1), detail)

    def _at(self, fraction: float, detail: str):
        self.fraction = max(self.fraction, min(fraction, 1.0))
        label = STEPS[self.step][1]
        text = f"Step {self.step + 1}/{len(STEPS)} · {label}" + (f" · {detail}" if detail else "")
        elapsed = time.time() - self.t0
        if self.fraction > 0.15:
            left = elapsed * (1 - self.fraction) / self.fraction
            text += " · <1 min left" if left < 60 else f" · ~{round(left / 60)} min left"
        return self.fraction, text


def _same(a: str, b: str) -> bool:
    """Whether a loop's name only restates its step's (letters only, one a prefix of the
    other): "Finetune" and "Fine-tuning"."""
    a, b = (re.sub(r"[^a-z]", "", x.lower()) for x in (a, b))
    return bool(a and b) and (a.startswith(b[:6]) or b.startswith(a[:6]))


def finetune_iterations(overrides) -> int:
    """FINETUNE's iterations in a run's overrides: 0 with ``finetuning=none`` (a group choice,
    whatever its place), else the last ``finetuning.num_iterations``, else 100."""
    if "finetuning=none" in overrides:
        return 0
    iterations = 100
    for o in overrides:
        if o.lstrip("+").startswith("finetuning.num_iterations="):
            iterations = int(o.split("=", 1)[1])
    return iterations


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


def example_scenes() -> list[str]:
    """The GenIA demo's bundled scenes, as ``setting/scene`` (none when GenIA is not found)."""
    try:
        data = genia_root() / "demo" / "data"
    except RuntimeError:
        return []
    return sorted(f"{d.parent.name}/{d.name}" for setting in ("image", "multiview")
                  for d in (data / setting).glob("*") if d.is_dir())


#: Photos the SAM3 templates open (``example_workflows/``), copied from GenIA's demo scenes
#: into ComfyUI's input folder: ``{name in input/: path under demo/data}``.
TEMPLATE_INPUTS = {
    "genia_office_buddies.png": "image/office_buddies/0000.png",
    "genia_pablo_0.png": "multiview/pablo/rgbs/00000.png",
    "genia_pablo_1.png": "multiview/pablo/rgbs/00001.png",
}


def copy_template_inputs(input_dir: Path) -> list[str]:
    """Put the templates' photos in ComfyUI's ``input_dir`` (never over an existing file), so
    every template runs as it opens. The names copied; none when GenIA is not found."""
    try:
        data = genia_root() / "demo" / "data"
    except RuntimeError:
        return []
    copied = []
    for name, source in TEMPLATE_INPUTS.items():
        target = Path(input_dir) / name
        if not target.exists() and (data / source).is_file():
            shutil.copyfile(data / source, target)
            copied.append(name)
    return copied


def load_example(name: str) -> tuple[np.ndarray, np.ndarray]:
    """A bundled scene (``setting/scene``) as ``(images (V, H, W, 3) uint8, masks (V, N, H, W)
    bool)``: its views, and each object's mask in every view (objects in id order)."""
    scene_dir = genia_root() / "demo" / "data" / name
    if name.startswith("image/"):
        rgbs, segs = [scene_dir / "0000.png"], [scene_dir / "0000_mask.png"]
    else:
        rgbs = sorted((scene_dir / "rgbs").glob("*.png"))
        segs = [scene_dir / "segmentations" / p.name for p in rgbs]
    images = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in rgbs])
    ids = np.stack([np.asarray(Image.open(p)) for p in segs])
    objects = [i for i in np.unique(ids) if i > 0]
    return images, np.stack([ids == i for i in objects], axis=1)


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


def scene_name(images: np.ndarray, masks: np.ndarray) -> str:
    """The scene's name: a hash of its inputs, so the same inputs are the same scene, whose
    block caches a later run resumes from (``processing.block_cache_keyed``)."""
    h = hashlib.sha256()
    for a in (images, masks):
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return "comfy_" + h.hexdigest()[:16]


def _prune(runs: Path, data: Path, keep: Path) -> None:
    """Keep the ``KEEP_SCENES`` most recently run scenes (and, of ``keep``'s runs, none but
    its caches: what a run returns is read into memory)."""
    for old in keep.glob("2*"):   # timestamped run dirs
        shutil.rmtree(old, ignore_errors=True)
    scenes = sorted((d for d in runs.glob("comfy_*") if d.is_dir()),
                    key=lambda d: d.stat().st_mtime, reverse=True)
    for old in scenes[KEEP_SCENES:]:
        shutil.rmtree(old, ignore_errors=True)
        for setting_dir in data.glob("*"):
            shutil.rmtree(setting_dir / old.name, ignore_errors=True)


def reconstruct(images, masks, *, seed, low_vram, options=(), extra=(), work_dir: Path,
                progress=None, interrupted=lambda: False):
    """Run GenIA on ``images`` (V, H, W, 3) with object ``masks`` (V, N, H, W): one view is
    its ``image`` setting, more its ``multiview``. ``options``: the labels of the GenIA demo
    app's run options to tick (``demo/method.py`` OPTIONS, e.g. "Full quality"); ``extra``:
    Hydra overrides added last. GenIA's log goes to ComfyUI's console (logger "GenIA").

    Returns ``(mesh_glb, splat_ply, background_ply, cameras, orbit_mp4)``: the paths of its
    world mesh, objects' splats and background splats, ``poses.json``'s cameras in view
    order, and the turntable video's bytes. A run on inputs seen before resumes after the
    last block whose settings did not change.

    ``progress(fraction, text)`` gets the whole run's progress (``RunProgress``) and what it
    is doing; ``interrupted()`` is polled while it runs (True stops it: RuntimeError("stopped")).
    """
    import torch

    setting = "image" if len(images) == 1 else "multiview"
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    free = torch.cuda.mem_get_info()[0] / 2**30
    # ComfyUI runs nodes under torch.inference_mode(): models loaded there would be inference
    # tensors, which GenIA's test-time optimisation cannot backpropagate through. (Its run
    # itself is on a thread of its own, which does not inherit the mode.)
    with torch.inference_mode(False):
        run, method = runner(roomy=total >= 40)
    known = {o[0] for o in method.OPTIONS}
    if unknown := set(options) - known:
        raise ValueError(f"GenIA's demo has no option {sorted(unknown)}; it has {sorted(known)}")
    scene = scene_name(images, masks)
    data_root, scene_runs = work_dir / "data", work_dir / "runs" / scene
    run_dir = scene_runs / time.strftime("%Y%m%d_%H%M%S")   # its cache: scene_runs/cache
    stage(data_root, setting, scene, images, masks)
    scene_runs.mkdir(parents=True, exist_ok=True)
    os.utime(scene_runs)
    _prune(work_dir / "runs", data_root, scene_runs)
    low = {"auto": free < method.LOW_VRAM_BELOW_GIB, "on": True, "off": False}[low_vram]
    args = [*(a for label, _info, on, off in method.OPTIONS
              for a in (on if label in options else off)),
            f"processing.seed={seed % 2**32}",
            # Resume a scene seen before, after the last block whose settings are unchanged.
            "processing.resume_from_cache=true", "output.save_cache=true",
            "processing.block_cache_keyed=true",
            "output.save_compressed_ply=false",   # the plain 3DGS PLY ComfyUI's reader takes
            *run.warm, *extra, f"output.output_dir={run_dir}"]
    overrides = method.run_overrides(setting, data_root, scene, low, args)
    log, pending = [], ""
    tracker = RunProgress(setting, masks.shape[1], finetune_iterations(overrides))
    if progress:
        progress(*tracker.block("START"))
    output = run.run(overrides)
    for chunk in output:
        log.append(chunk)
        pending += chunk
        *lines, pending = pending.replace("\r", "\n").split("\n")
        for line in lines:
            m = _BLOCK.match(line.strip())
            if line.strip() and "@@progress " not in line and not _BAR.search(line):
                # The blocks and their results at INFO; the rest at DEBUG (--verbose DEBUG).
                level = logging.INFO if m or _RESULT.search(line) else logging.DEBUG
                _log.log(level, line.rstrip())
            if not progress:
                continue
            if m:
                final = m.group(1) == "EVALUATION" and m.group(2).startswith("Final state")
                if final or m.group(1) != "EVALUATION":
                    progress(*tracker.block("FINAL" if final else m.group(1)))
            elif (cut := line.find("@@progress ")) >= 0:
                try:
                    p = json.loads(line[cut + len("@@progress "):])
                    progress(*tracker.loop(p["task"], p["n"], p["total"]))
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
    orbit = final / "viz" / "orbit" / "orbit.mp4"
    gaussians = final / "gaussians"
    return (final / "meshes" / "world.glb", gaussians / "world.ply",
            gaussians / "background.ply", cameras, orbit.read_bytes() if orbit.is_file() else None)
