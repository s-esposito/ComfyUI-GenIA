"""ComfyUI-GenIA nodes (v3 API)."""

from comfy_api.latest import IO


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
                            "single white-on-black mask (object = white).",
                ),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                IO.Combo.Input("low_vram", options=["auto", "on", "off"], default="auto",
                               tooltip="Trade speed for GPU memory; auto decides from the free VRAM."),
                IO.Boolean.Input("place_on_ground", default=True,
                                 tooltip="Rest the reconstruction on the ground plane (y = 0)."),
            ],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Splat.Output(display_name="splat"),
            ],
        )

    @classmethod
    def execute(cls, image, seed, low_vram, place_on_ground, mask=None) -> IO.NodeOutput:
        raise NotImplementedError("bridge lands in M5")
