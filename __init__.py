"""ComfyUI-GenIA: GenIA generative reconstruction as ComfyUI nodes."""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO

from .nodes import GenIAReconstructImage


class GenIAExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [GenIAReconstructImage]


async def comfy_entrypoint() -> GenIAExtension:
    return GenIAExtension()
