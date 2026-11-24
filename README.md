# ComfyUI-GenIA

GenIA generative 3D reconstruction as ComfyUI nodes:
reconstruct what the image observes, generate consistently what it does not. The node design
(one reconstruct node, image + optional mask in, a mesh and a camera out) is inspired by
[ComfyUI-Cupid3D](https://github.com/Nekodificador/ComfyUI-Cupid3D); no code is shared.

## Status

Work in progress. **M5: both nodes run GenIA** (checked on an RTX 4090 with ComfyUI v0.46.2,
both ways below).

| Node | Inputs | Outputs |
|------|--------|---------|
| GenIA Reconstruct (Image) (`3d/GenIA`) | `image` IMAGE, `mask` MASK (one mask per object as a batch, or white-on-black; required), `seed` INT, `quality` fast/full, `low_vram` auto/on/off, `place_on_ground` BOOLEAN | `mesh` MESH, `camera_info` LOAD3D_CAMERA, `splat` SPLAT |
| GenIA Reconstruct (Multi-view) (`3d/GenIA`) | `images` IMAGE (the views, a batch of two or more), `objects`: one `mask` input per object (it grows as you connect them), each a MASK batch with the object's mask in every view; then as above | `mesh` MESH, `camera_info` LOAD3D_CAMERA (every view's camera, a camera list), `splat` SPLAT |

`quality`: `full` is GenIA's paper settings; `fast` (default) runs FINETUNE 40 iterations
instead of 100: 35-64% less time, slightly softer texture. The outputs are in ComfyUI's
y-up frame: the mesh (vertex colours) and splats of every object, the input camera, and with
`place_on_ground` all three raised so the mesh rests on y = 0.

## How it runs GenIA

GenIA depends on CUDA extensions (gsplat, pytorch3d, spconv, ...) built against one torch,
so the node runs it one of two ways, decided when it first runs:

- **In ComfyUI's process (recommended)**: ComfyUI installed into GenIA's environment. GenIA
  imports there, its models load into ComfyUI's process on the first run (~25 s) and stay;
  ComfyUI's own models are unloaded before each run to make room. ComfyUI's requirements add
  packages to GenIA's environment and change none of GenIA's.
- **In a worker process**: ComfyUI in an environment of its own (e.g. an existing install).
  The node starts GenIA's warm worker with GenIA's environment's Python (`GENIA_PYTHON`);
  the models stay loaded there between runs.

Either way it is GenIA's own demo recipe (`demo/method.py`, `demo/ui/{inprocess,worker}.py`
in GenIA's checkout): the node stages the image and masks as a scene, runs it, and reads the
run's `final/` back as ComfyUI's MESH / LOAD3D_CAMERA / SPLAT. GenIA's progress drives the
node's progress bar; ComfyUI's cancel stops the run.

## Install

1. Install GenIA (`bash install.sh` in its checkout: a conda env, Linux, CUDA 12.8).
2. Clone this repo into `ComfyUI/custom_nodes/` (or symlink it there).
3. Either:
   - **recommended**: run ComfyUI from GenIA's environment, after
     `pip install -r ComfyUI/requirements.txt` in it; or
   - keep ComfyUI's environment and start it with `GENIA_ROOT=<GenIA checkout>` and
     `GENIA_PYTHON=<GenIA's env>/bin/python`.

Tests (CPU only, from the ComfyUI env):

```bash
PYTHONPATH=/path/to/ComfyUI pytest -v
```

## License

This pack is licensed under the **GNU GPL v3.0** (see `LICENSE`). GenIA itself is licensed
under **CC BY-NC 4.0** (non-commercial); it is not part of this repository (the node loads it
from your GenIA checkout), and its license governs your use of GenIA's code, weights and
outputs.
