from __future__ import annotations
from collections import defaultdict
from typing import Type
from einops import rearrange

from comfy.ldm.flux.layers import DoubleStreamBlock, apply_mod

import torch

from .base_patcher import BaseAttentionPatcher
from ..attention import attention_sub_quad_patched

from ..util import ObjectHooker, AggregateHooker


__all__ = ["FluxAttentionPatcher"]


class FluxAttentionPatcher(BaseAttentionPatcher):
    def __init__(
        self,
        model_patcher,
        img_height: int,
        img_width: int,
        context_size: tuple[int, int] = (77, 77),  # [0] for positive, [1] for negative
        enable_heat_maps: tuple[bool, bool] = (
            True,
            True,
        ),  # [0] for positive, [1] for negative
        weighted: bool = False,
        head_idx: int = 0,
    ):
        super().__init__(
            model_patcher,
            img_height,
            img_width,
            context_size=context_size,
            enable_heat_maps=enable_heat_maps,
            weighted=weighted,
            head_idx=head_idx,
        )
        self.tracers = [
            fluxtrace(
                model_patcher,
                img_height,
                img_width,
                heat_maps=self.heat_maps,
                context_size=context_size[0],
            )
        ]

    def patch(self):
        for tracer in self.tracers:
            tracer.hook()

    def unpatch(self):
        for tracer in self.tracers:
            tracer.unhook()


class ModelFunctionWrapper:
    def __init__(self):
        self.cond_or_uncond = None

    def model_wrapper(self, unet_apply_function, unet_params) -> torch.Tensor:
        c = unet_params["c"]

        self.cond_or_uncond = unet_params["cond_or_uncond"]

        output = unet_apply_function(unet_params["input"], unet_params["timestep"], **c)
        return output


class FluxDiffusionHeatMapHooker(AggregateHooker):
    def __init__(
        self,
        model,
        heigth: int,
        width: int,
        heat_maps,
        context_size: tuple[int, int] = (77, 77),
        weighted: bool = False,
        layer_idx: int = None,
        head_idx: int = None,
    ):
        # batch index, factor, attention
        self.heat_maps = heat_maps
        self.model_wrapper = ModelFunctionWrapper()

        selected_blocks = model.model.diffusion_model.double_blocks
        self.block_len = len(selected_blocks)

        modules = [
            DoubleStreamBlockHooker(
                x,
                heigth,
                width,
                heat_maps,
                context_size=context_size,
                weighted=weighted,
                head_idx=head_idx,
                cond_or_uncond_tracker=self.model_wrapper,
                block_len=self.block_len,
            )
            for x in selected_blocks
        ]

        self.height = heigth
        self.width = width
        self.model = model

        super().__init__(modules)

    @property
    def all_heat_maps(self):
        return self.forward_hook.all_heat_maps

    def reset(self):
        map(lambda module: module.reset(), self.module)
        return self.forward_hook.all_heat_maps.clear()

    def _hook_impl(self):
        self.model.set_model_unet_function_wrapper(self.model_wrapper.model_wrapper)

        for h in self.module:
            h.hook()

    def _unhook_impl(self):
        self.model.set_model_unet_function_wrapper(None)

        for h in self.module:
            h.unhook()


class DoubleStreamBlockHooker(ObjectHooker[DoubleStreamBlock]):
    def __init__(
        self,
        module: DoubleStreamBlock,
        img_height: int,
        img_width: int,
        heat_maps: defaultdict(defaultdict),
        context_size: tuple[int, int] = (77, 77),
        weighted: bool = False,
        head_idx: int = 0,
        cond_or_uncond_tracker=None,
        block_len=19,
    ):
        super().__init__(module)
        self.heat_maps = heat_maps
        self.context_size = context_size
        self.weighted = weighted
        self.head_idx = head_idx
        self.img_height = img_height
        self.img_width = img_width
        self.cond_or_uncond_tracker = cond_or_uncond_tracker
        self.block_len = block_len

    def _forward(
        hk_self,
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        pe: torch.Tensor,
        attn_mask=None,
        modulation_dims_img=None,
        modulation_dims_txt=None,
    ):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = apply_mod(
            img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img
        )
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(
            img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = apply_mod(
            txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt
        )
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(
            txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        if self.flipped_img_txt:
            # run actual attention
            attn, attn_probs = hk_self._attn_patched(
                self,
                torch.cat((img_q, txt_q), dim=2),
                torch.cat((img_k, txt_k), dim=2),
                torch.cat((img_v, txt_v), dim=2),
                pe=pe,
                mask=attn_mask,
            )

            # TODO: Extract the attention probabilities in case of flipped

            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        else:
            # run actual attention
            attn, attn_probs = hk_self._attn_patched(
                self,
                torch.cat((txt_q, img_q), dim=2),
                torch.cat((txt_k, img_k), dim=2),
                torch.cat((txt_v, img_v), dim=2),
                pe=pe,
                mask=attn_mask,
            )
            # https://github.com/wooyeolbaek/attention-map-diffusers/blob/main/attention_map_diffusers/modules.py#L2182
            # ([24, 4352, 4352]) -> ([24, 4096, 256])
            attn_probs = attn_probs[:, txt.shape[1] :, : txt.shape[1]].cpu()

            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # Process the attention probabilities
        # TODO: Variable height
        height = 64
        attn_probs = rearrange(
            attn_probs,
            "(batch heads) (height width) txt_tokens -> batch txt_tokens heads height width",
            heads=txt_q.shape[1],
            height=height,
        )
        attn_probs = attn_probs.sum(dim=2, keepdim=True).cpu()

        hk_self._add_heat_map_if_matches(
            self, attn_probs, hk_self.cond_or_uncond_tracker.cond_or_uncond
        )

        # calculate the img bloks
        img = img + apply_mod(
            self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img
        )
        img = img + apply_mod(
            self.img_mlp(
                apply_mod(
                    self.img_norm2(img),
                    (1 + img_mod2.scale),
                    img_mod2.shift,
                    modulation_dims_img,
                )
            ),
            img_mod2.gate,
            None,
            modulation_dims_img,
        )

        # calculate the txt bloks
        txt += apply_mod(
            self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt
        )
        txt += apply_mod(
            self.txt_mlp(
                apply_mod(
                    self.txt_norm2(txt),
                    (1 + txt_mod2.scale),
                    txt_mod2.shift,
                    modulation_dims_txt,
                )
            ),
            txt_mod2.gate,
            None,
            modulation_dims_txt,
        )

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt

    def _attn_patched(
        hk_self,
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pe: torch.Tensor,
        mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_shape = q.shape
        k_shape = k.shape

        if pe is not None:
            q = q.to(dtype=pe.dtype).reshape(*q.shape[:-1], -1, 1, 2)
            k = k.to(dtype=pe.dtype).reshape(*k.shape[:-1], -1, 1, 2)
            q = (
                (pe[..., 0] * q[..., 0] + pe[..., 1] * q[..., 1])
                .reshape(*q_shape)
                .type_as(v)
            )
            k = (
                (pe[..., 0] * k[..., 0] + pe[..., 1] * k[..., 1])
                .reshape(*k_shape)
                .type_as(v)
            )

        heads = q.shape[1]
        x, attn_probs = attention_sub_quad_patched(
            q, k, v, heads, skip_reshape=True, mask=mask
        )
        return x, attn_probs

    def _hook_impl(self):
        self.monkey_patch("forward", self._forward)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))

    def _add_heat_map_if_matches(
        hk_self,
        self,
        attn_probs: torch.Tensor,  # [batch, txt_tokens, 1, height, width]
        cond_or_uncond: list,
    ):
        batch_size = attn_probs.shape[0]
        factor = 2

        for batch_index in range(batch_size):
            sim_slice = attn_probs[batch_index]

            heat_map_added = False

            if len(cond_or_uncond) == 2:
                # Combined conditional/unconditional in the same batch
                half = batch_size // 2
                if batch_index >= half and hk_self.heat_maps["pos"] is not None:
                    # Conditional batch
                    hk_self.heat_maps["pos"].add_heat_map(
                        batch_index - half, factor, sim_slice
                    )
                elif hk_self.heat_maps["neg"] is not None:
                    # Unconditional batch
                    hk_self.heat_maps["neg"].add_heat_map(
                        batch_index, factor, sim_slice
                    )

                heat_map_added = True
            elif cond_or_uncond == [0] and hk_self.heat_maps["pos"] is not None:
                # Separate conditional batch
                hk_self.heat_maps["pos"].add_heat_map(batch_index, factor, sim_slice)

                heat_map_added = True

            elif cond_or_uncond == [1] and hk_self.heat_maps["neg"] is not None:
                # Separate unconditional batch
                hk_self.heat_maps["neg"].add_heat_map(batch_index, factor, sim_slice)

                heat_map_added = True

            if heat_map_added:
                for pos_or_neg_heat_maps in hk_self.heat_maps.values():
                    if (
                        pos_or_neg_heat_maps is not None
                        and pos_or_neg_heat_maps.count() >= hk_self.block_len
                    ):
                        # Flush the heatmaps for each timestep
                        pos_or_neg_heat_maps.save_heat_map_timestep()


fluxtrace: Type[FluxDiffusionHeatMapHooker] = FluxDiffusionHeatMapHooker
