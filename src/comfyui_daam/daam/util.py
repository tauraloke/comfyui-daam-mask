from __future__ import annotations
from typing import List, Generic, TypeVar

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.modules.attention import CrossAttention

import torch.nn as nn
import functools

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


def is_output_connected(
    prompt,
    cur_node_id,
    output_index,
) -> bool:
    for node_id in prompt:
        if node_id == cur_node_id:
            continue

        for node_input in prompt[node_id]["inputs"].values():
            if isinstance(node_input, list) and node_input == [
                cur_node_id,
                output_index,
            ]:
                return True

    return False


ModuleType = TypeVar("ModuleType")
ModuleListType = TypeVar("ModuleListType", bound=List)


class ModuleLocator(Generic[ModuleType]):
    def locate(self, model: nn.Module) -> List[ModuleType]:
        raise NotImplementedError


class ObjectHooker(Generic[ModuleType]):
    def __init__(self, module: ModuleType):
        self.module: ModuleType = module
        self.hooked = False
        self.old_state = dict()

    def __enter__(self):
        self.hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def hook(self):
        if self.hooked:
            raise RuntimeError("Already hooked module")

        self.old_state = dict()
        self.hooked = True
        self._hook_impl()

        return self

    def unhook(self):
        if not self.hooked:
            raise RuntimeError("Module is not hooked")

        for k, v in self.old_state.items():
            if k.startswith("old_fn_"):
                setattr(self.module, k[7:], v)

        self.hooked = False
        self._unhook_impl()

        return self

    def monkey_patch(self, fn_name, fn):
        self.old_state[f"old_fn_{fn_name}"] = getattr(self.module, fn_name)
        setattr(self.module, fn_name, functools.partial(fn, self.module))

    def monkey_super(self, fn_name, *args, **kwargs):
        return self.old_state[f"old_fn_{fn_name}"](*args, **kwargs)

    def _hook_impl(self):
        raise NotImplementedError

    def _unhook_impl(self):
        pass


class AggregateHooker(ObjectHooker[ModuleListType]):
    def _hook_impl(self):
        for h in self.module:
            h.hook()

    def _unhook_impl(self):
        for h in self.module:
            h.unhook()

    def register_hook(self, hook: ObjectHooker):
        self.module.append(hook)
