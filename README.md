# ComfyUI-GenIA

GenIA generative 3D reconstruction as ComfyUI nodes:
reconstruct what the image observes, generate consistently what it does not. The node design
(one reconstruct node, image + optional mask in, a mesh and a camera out) is inspired by
[ComfyUI-Cupid3D](https://github.com/Nekodificador/ComfyUI-Cupid3D); no code is shared.

## Status

Work in progress. **M4: node skeleton only.** The node registers in ComfyUI with its final
inputs and outputs, but running it raises `NotImplementedError` until the bridge lands (M5).

| Node | Inputs | Outputs |
|------|--------|---------|
| GenIA Reconstruct (Image) (`3d/GenIA`) | `image` IMAGE, `mask` MASK (optional: one mask per object as a batch, or white-on-black), `seed` INT, `low_vram` auto/on/off, `place_on_ground` BOOLEAN | `mesh` MESH, `camera_info` LOAD3D_CAMERA, `splat` SPLAT |

Planned: a multi-view node (image batch + mask batch -> MESH + list of cameras + SPLAT).

## Architecture: two environments

GenIA depends on CUDA extensions (pytorch3d, gsplat, nvdiffrast, spconv, SAM 3D, ...) that do
not belong in ComfyUI's Python environment. The pack therefore keeps the two apart:

- **ComfyUI env**: this pack, which only needs what ComfyUI itself ships.
- **`genia` env**: GenIA and its dependencies. The node stages the inputs into a scene
  directory, runs GenIA there as a **subprocess** (`demo/run_demo.py --data-root <dir> --scene <name>`)
  and reads its `final/` outputs back into ComfyUI's MESH / LOAD3D_CAMERA / SPLAT types.

## Install (sketch)

1. Install GenIA in its own conda env (`genia`), following GenIA's instructions.
2. Clone this repo into `ComfyUI/custom_nodes/` (or symlink it there) and restart ComfyUI.
3. (M5) Point the pack at the GenIA checkout and its env's Python.

Tests (CPU only, from the ComfyUI env):

```bash
PYTHONPATH=/path/to/ComfyUI pytest -v
```

## License

This pack is licensed under the **GNU GPL v3.0** (see `LICENSE`). GenIA itself is licensed
under **CC BY-NC 4.0** (non-commercial); it is not part of this repository and runs as a
separate process, so its license governs your use of GenIA's code, weights and outputs.
