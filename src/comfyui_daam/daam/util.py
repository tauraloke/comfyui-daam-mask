from __future__ import annotations
from typing import List, Generic, TypeVar

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.modules.attention import CrossAttention

import torch.nn as nn

ModuleType = TypeVar("ModuleType")
ModuleListType = TypeVar("ModuleListType", bound=List)


class ModuleLocator(Generic[ModuleType]):
    def locate(self, model: nn.Module) -> List[ModuleType]:
        raise NotImplementedError


class UNetCrossAttentionLocator(ModuleLocator[CrossAttention]):
    def locate(self, model: UNetModel, layer_idx: int) -> List[CrossAttention]:
        """
        Locate all cross-attention modules in a UNetModel.

        Args:
            model (`UNetModel`): The model to locate the cross-attention modules in.

        Returns:
            `List[(str, int, int)]`: The list of ComfyUI's cross-attention block tags. ("input" | "middle" | "output", layer_id, transformer_index)
        """
        block_tags = []

        def extract_from_blocks(name, transformer_blocks):
            for i, unet_block in enumerate(transformer_blocks):
                if not layer_idx or i == layer_idx:
                    for module in unet_block.modules():
                        if module.__class__.__name__ == "SpatialTransformer":
                            spatial_transformer = module
                            for transformer_index in range(
                                len(spatial_transformer.transformer_blocks)
                            ):
                                block_tags.append((name, i, transformer_index))

        extract_from_blocks("input", model.input_blocks)
        extract_from_blocks("middle", [model.middle_block])
        extract_from_blocks("output", model.output_blocks)

        return block_tags
