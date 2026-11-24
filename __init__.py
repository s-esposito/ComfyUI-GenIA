"""ComfyUI-GenIA: GenIA generative reconstruction as ComfyUI nodes."""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO

from .nodes import GenIAExampleScene, GenIAReconstruct


class GenIAExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        import folder_paths

        from .bridge import copy_template_inputs
        copy_template_inputs(folder_paths.get_input_directory())

    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [GenIAReconstruct, GenIAExampleScene]


async def comfy_entrypoint() -> GenIAExtension:
    return GenIAExtension()
