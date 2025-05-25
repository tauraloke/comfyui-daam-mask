from __future__ import annotations
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn.functional as F

import math

from ..util import UNetCrossAttentionLocator
from ..attention import attention_sub_quad_patched


class RawHeatMaps:
    def __init__(self):
        self.heat_maps = defaultdict(lambda: defaultdict(list))
        self.heat_maps_count = 0

        self.all_heat_maps = []

    def add_heat_map(self, batch_index, factor, heat_map):
        self.heat_maps[batch_index][factor].append(heat_map)
        self.heat_maps_count += 1

    def save_heat_map_timestep(self):
        if len(self.heat_maps) > 0:
            self.all_heat_maps.append(deepcopy(self.heat_maps))

        self.heat_maps.clear()
        self.heat_maps_count = 0

    def count(self):
        return self.heat_maps_count


class BaseAttentionPatcher:
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
        self.model_patcher = model_patcher
        self.img_height = img_height
        self.img_width = img_width
        self.heat_maps = {
            "pos": RawHeatMaps() if enable_heat_maps[0] else None,
            "neg": RawHeatMaps() if enable_heat_maps[1] else None,
        }
        self.context_size = context_size
        self.weighted = weighted
        self.head_idx = head_idx

    def patch(self, layer_idx: int = None):
        raise NotImplementedError("Error: patch() not implemented")

    def unpatch(self):
        raise NotImplementedError("Error: unpatch() not implemented")

    @property
    def all_heat_maps(self):
        pos_heat_maps = (
            self.heat_maps["pos"].all_heat_maps if self.heat_maps["pos"] else None
        )
        neg_heat_maps = (
            self.heat_maps["neg"].all_heat_maps if self.heat_maps["neg"] else None
        )
        return pos_heat_maps, neg_heat_maps

    @torch.no_grad()
    def _up_sample_attn(self, x, value, factor, method="bicubic"):
        # type: (torch.Tensor, torch.Tensor, int, Literal['bicubic', 'conv']) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Up samples the attention map in x using interpolation to the maximum size of (64, 64), as assumed in the Stable
        Diffusion model.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.
            method (`str`): the method to use; one of `'bicubic'` or `'conv'`.

        Returns:
            `torch.Tensor`: the up-sampled attention map of shape (tokens, 1, height, width).
        """
        weight = torch.full((factor, factor), 1 / factor**2, device=x.device)
        weight = weight.view(1, 1, factor, factor)

        ratio = (x.size(1) / (self.img_height * self.img_width)) ** 0.5
        h = int(round(self.img_height * ratio))
        w = int(round(self.img_width * ratio))

        h_fix = w_fix = 64
        if h >= w:
            w_fix = int((w * h_fix) / h)
        else:
            h_fix = int((h * w_fix) / w)

        maps = []
        x = x.permute(2, 0, 1)
        value = value.permute(1, 0, 2)
        weights = 1

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.unsqueeze(1).view(map_.size(0), 1, h, w)

                if method == "bicubic":
                    map_ = F.interpolate(map_, size=(h_fix, w_fix), mode="bicubic")
                    maps.append(map_.squeeze(1))
                else:
                    maps.append(
                        F.conv_transpose2d(map_, weight, stride=factor).squeeze(1)
                    )

        if self.weighted:
            weights = value.norm(p=1, dim=-1, keepdim=True).unsqueeze(-1)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)

        if self.head_idx:
            maps = maps[:, self.head_idx : self.head_idx + 1, :, :]

        return (weights * maps).sum(1, keepdim=True).cpu()


class CrossAttentionPatcher(BaseAttentionPatcher):
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
        self.saved_model_options = None

    def patch(self, layer_idx: int = None):
        self.block_tags = UNetCrossAttentionLocator().locate(
            self.model_patcher.model.diffusion_model, layer_idx=layer_idx
        )

        self.saved_model_options = self.model_patcher.model_options.copy()

        for block_name, layer_idx, transformer_index in self.block_tags:
            self.model_patcher.set_model_patch_replace(
                self._attn2_patched, "attn2", block_name, layer_idx, transformer_index
            )

    def unpatch(self):
        if hasattr(self, "block_tags"):
            for block_name, layer_idx, transformer_index in self.block_tags:
                self.model_patcher.set_model_patch_replace(
                    None, "attn2", block_name, layer_idx, transformer_index
                )

        if self.saved_model_options is not None:
            self.model_patcher.model_options = self.saved_model_options.copy()

    def _attn2_patched(self, q, context=None, value=None, extra_options={}, mask=None):
        k = context
        v = value

        n_heads = extra_options["n_heads"]
        _dim_head = extra_options["dim_head"]
        attn_precision = extra_options["attn_precision"]
        cond_or_uncond = extra_options["cond_or_uncond"]

        out, attn_map = attention_sub_quad_patched(
            q, k, v, n_heads, mask=mask, attn_precision=attn_precision
        )

        self._add_heat_map_if_matches(
            attn_map,
            v,
            cond_or_uncond,
            batch_size=q.shape[0],
        )

        return out

    def _add_heat_map_if_matches(
        self,
        attn_probs: torch.Tensor,
        value: torch.Tensor,
        cond_or_uncond: list,
        batch_size: int,
    ):
        def calc_factor_base(w, h):
            z = max(w / 64, h / 64)
            factor_b = min(w, h) * z
            return factor_b

        batch_size_attention = attn_probs.shape[0]
        slice_size = batch_size_attention // batch_size
        factor_base = calc_factor_base(self.img_width, self.img_height)

        for batch_index in range(batch_size):
            start_idx = batch_index * slice_size
            end_idx = (batch_index + 1) * slice_size

            sim_slice = attn_probs[start_idx:end_idx]
            factor = int(math.sqrt(factor_base // sim_slice.shape[1]))

            if factor < 1:
                return False

            factor //= 1

            heat_map_added = False
            # ComfyUI determines whether run cond/uncond in 2 batches or 1 batch for each, based on free memory.
            # This will affect how we collect the heatmaps, so we need to keep track the cond_or_uncond variable.
            # len(cond_or_uncond) = 1: either cond [0] or uncond [1]
            # len(cond_or_uncond) = 2: uncond and cond in the same run [1, 0]
            # https://github.com/comfyanonymous/ComfyUI/blob/92cdc692f47188e6e4c48c5666ac802281240a37/comfy/samplers.py#L260
            if len(cond_or_uncond) == 2 and sim_slice.shape[-1] == max(
                self.context_size
            ):
                # Combined conditional/unconditional in the same batch
                half = batch_size // 2
                if batch_index >= half and self.heat_maps["pos"] is not None:
                    # Conditional batch
                    maps = self._up_sample_attn(sim_slice, value, factor)
                    self.heat_maps["pos"].add_heat_map(batch_index - half, factor, maps)
                elif self.heat_maps["neg"] is not None:
                    # Unconditional batch
                    maps = self._up_sample_attn(sim_slice, value, factor)
                    self.heat_maps["neg"].add_heat_map(batch_index, factor, maps)

                heat_map_added = True
            elif (
                sim_slice.shape[-1] == self.context_size[0]
                and cond_or_uncond == [0]
                and self.heat_maps["pos"] is not None
            ):
                # Separate conditional batch
                maps = self._up_sample_attn(sim_slice, value, factor)
                self.heat_maps["pos"].add_heat_map(batch_index, factor, maps)

                heat_map_added = True

            elif (
                sim_slice.shape[-1] == self.context_size[1]
                and cond_or_uncond == [1]
                and self.heat_maps["neg"] is not None
            ):
                # Separate unconditional batch
                maps = self._up_sample_attn(sim_slice, value, factor)
                self.heat_maps["neg"].add_heat_map(batch_index, factor, maps)

                heat_map_added = True

            if heat_map_added:
                for pos_or_neg_heat_maps in self.heat_maps.values():
                    if (
                        pos_or_neg_heat_maps is not None
                        and pos_or_neg_heat_maps.count() >= len(self.block_tags)
                    ):
                        # Flush the heatmaps for each timestep
                        pos_or_neg_heat_maps.save_heat_map_timestep()
