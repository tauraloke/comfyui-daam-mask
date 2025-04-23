from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from typing import Type, Literal
import math
from einops import rearrange, repeat

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.modules.attention import CrossAttention, default
from comfy import model_management

from comfy.cli_args import args

import torch
import torch.nn.functional as F
from torch import einsum

from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator


__all__ = ['trace', 'DiffusionHeatMapHooker']


class ModelFunctionWrapper:
    """
    ComfyUI determines whether run cond/uncond in 2 batches or 1 batch for each, based on free memory.
    This will affect how we collect the heatmaps, so we need to keep track the cond_or_uncond variable.
    len(cond_or_uncond) = 1: either cond [0] or uncond [1]
    len(cond_or_uncond) = 2: uncond and cond in the same run [1, 0]
    https://github.com/comfyanonymous/ComfyUI/blob/92cdc692f47188e6e4c48c5666ac802281240a37/comfy/samplers.py#L260
    """

    def __init__(self):
        self.cond_or_uncond = None

    def cond_uncond_as_two_batches(self):
        """
        Returns True if the model is running cond/uncond in 2 batches.
        This is the case when there is enough memory to run both in the same time.
        """
        return self.cond_or_uncond and self.cond_or_uncond == 2

    def model_wrapper(self, unet_apply_function, unet_params) -> torch.Tensor:
        c = unet_params["c"]

        self.cond_or_uncond = len(unet_params["cond_or_uncond"])

        output = unet_apply_function(
            unet_params["input"], unet_params["timestep"], **c)
        return output


class UNetForwardHooker(ObjectHooker[UNetModel]):
    def __init__(self, module: UNetModel, heat_maps: defaultdict(defaultdict)):
        super().__init__(module)
        self.all_heat_maps = []
        self.heat_maps = heat_maps

    def _hook_impl(self):
        self.monkey_patch('forward', self._forward)

    def _unhook_impl(self):
        pass

    def _forward(hk_self, self, *args, **kwargs):
        super_return = hk_self.monkey_super('forward', *args, **kwargs)

        if len(hk_self.heat_maps) > 0:
            hk_self.all_heat_maps.append(deepcopy(hk_self.heat_maps))

        hk_self.heat_maps.clear()

        return super_return


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(self, model, heigth: int, width: int, context_size: int = 77, weighted: bool = False, layer_idx: int = None, head_idx: int = None):
        # batch index, factor, attention
        heat_maps = defaultdict(lambda: defaultdict(list))

        self.model_wrapper = ModelFunctionWrapper()

        modules = [UNetCrossAttentionHooker(x, heigth, width, heat_maps, context_size=context_size, weighted=weighted, head_idx=head_idx, cond_or_uncond_tracker=self.model_wrapper)
                   for x in UNetCrossAttentionLocator().locate(model.model.diffusion_model, layer_idx)]
        self.forward_hook = UNetForwardHooker(
            model.model.diffusion_model, heat_maps)
        modules.append(self.forward_hook)

        self.height = heigth
        self.width = width
        self.model = model
        self.last_prompt = ''

        super().__init__(modules)

    @property
    def all_heat_maps(self):
        return self.forward_hook.all_heat_maps

    def reset(self):
        map(lambda module: module.reset(), self.module)
        return self.forward_hook.all_heat_maps.clear()

    def _hook_impl(self):
        self.model.set_model_unet_function_wrapper(
            self.model_wrapper.model_wrapper)

        for h in self.module:
            h.hook()

    def _unhook_impl(self):
        self.model.set_model_unet_function_wrapper(None)

        for h in self.module:
            h.unhook()

# Adapted from A1111's DAAM Extension and Official ComfyUI
# https://github.com/kousw/stable-diffusion-webui-daam/blob/b23fb574bf691f0bdf503e5617a0b3578160c7a1/scripts/daam/trace.py#L204
# https://github.com/comfyanonymous/ComfyUI/blob/6fc5dbd52ab70952020e6bc486c4d851a7ba6625/comfy/ldm/modules/attention.py#L613


class UNetCrossAttentionHooker(ObjectHooker[CrossAttention]):
    def __init__(self, module: CrossAttention, img_height: int, img_width: int, heat_maps: defaultdict(defaultdict), context_size: int = 77, weighted: bool = False, head_idx: int = 0, cond_or_uncond_tracker=None):
        super().__init__(module)
        self.heat_maps = heat_maps
        self.context_size = context_size
        self.weighted = weighted
        self.head_idx = head_idx
        self.img_height = img_height
        self.img_width = img_width
        self.calledCount = 0
        self.cond_or_uncond_tracker = cond_or_uncond_tracker

    def reset(self):
        self.heat_maps.clear()
        self.calledCount = 0

    @torch.no_grad()
    def _up_sample_attn(self, x, value, factor, method='bicubic'):
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
        weight = torch.full((factor, factor), 1 / factor ** 2, device=x.device)
        weight = weight.view(1, 1, factor, factor)

        h = int(math.sqrt((self.img_height * x.size(1)) / self.img_width))
        w = int(self.img_width * h / self.img_height)

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

                if method == 'bicubic':
                    map_ = F.interpolate(map_, size=(
                        h_fix, w_fix), mode='bicubic')
                    maps.append(map_.squeeze(1))
                else:
                    maps.append(F.conv_transpose2d(
                        map_, weight, stride=factor).squeeze(1))

        if self.weighted:
            weights = value.norm(p=1, dim=-1, keepdim=True).unsqueeze(-1)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)

        if self.head_idx:
            maps = maps[:, self.head_idx:self.head_idx+1, :, :]

        return (weights * maps).sum(1, keepdim=True).cpu()

    def _forward(hk_self, self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        heads = self.heads
        b, sequence_length, dim_head = q.shape
        dim_head //= heads

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = hk_self._hooked_attention(
            self, q, k, v, b, sequence_length, dim_head, mask)

        hk_self.calledCount += 1

        return self.to_out(out)

    # Capture attentions and aggregate them.
    def _hooked_attention(hk_self, self, q, k, v, batch_size, sequence_length, dim_head, mask, use_context: bool = True):
        def calc_factor_base(w, h):
            z = max(w/64, h/64)
            factor_b = min(w, h) * z
            return factor_b

        factor_base = calc_factor_base(hk_self.img_width, hk_self.img_height)
        batch_size_attention = q.shape[0]
        slice_size = batch_size_attention // batch_size

        out = torch.zeros(
            (batch_size_attention, sequence_length,
             dim_head), device=q.device, dtype=q.dtype
        )

        def get_attn_precision(attn_precision, current_dtype):
            FORCE_UPCAST_ATTENTION_DTYPE = model_management.force_upcast_attention_dtype()

            if args.dont_upcast_attention:
                return None

            if FORCE_UPCAST_ATTENTION_DTYPE is not None and current_dtype in FORCE_UPCAST_ATTENTION_DTYPE:
                return FORCE_UPCAST_ATTENTION_DTYPE[current_dtype]
            return attn_precision

        attn_precision = get_attn_precision(self.attn_precision, q.dtype)

        scale = dim_head ** -0.5

        for batch_index in range(batch_size):
            start_idx = batch_index * slice_size
            end_idx = (batch_index + 1) * slice_size

            # force cast to fp32 to avoid overflowing
            if attn_precision == torch.float32:
                sim_slice = einsum('b i d, b j d -> b i j',
                                   q[start_idx:end_idx].float(), k[start_idx:end_idx].float()) * scale
            else:
                sim_slice = einsum(
                    'b i d, b j d -> b i j', q[start_idx:end_idx], k[start_idx:end_idx]) * scale

            if mask is not None:
                if mask.dtype == torch.bool:
                    # TODO: check if this bool part matches pytorch attention
                    mask = rearrange(mask, 'b ... -> b (...)')
                    max_neg_value = -torch.finfo(sim_slice.dtype).max
                    mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
                    sim_slice.masked_fill_(~mask, max_neg_value)
                else:
                    if len(mask.shape) == 2:
                        bs = 1
                    else:
                        bs = mask.shape[0]
                    mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(
                        batch_size, self.heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
                    sim_slice.add_(mask)

            factor = int(math.sqrt(factor_base // sim_slice.shape[1]))

            # Attention
            sim_slice = sim_slice.softmax(-1)

            out_slice = einsum('b i j, b j d -> b i d',
                               sim_slice.to(v.dtype), v[start_idx:end_idx])

            if use_context and sim_slice.shape[-1] == hk_self.context_size:
                if hk_self.cond_or_uncond_tracker and hk_self.cond_or_uncond_tracker.cond_uncond_as_two_batches():
                    # Cond and uncond in the same run
                    # The first half of batch is unconditional, the second half is conditional.
                    if batch_index >= batch_size // 2:
                        if factor >= 1:
                            factor //= 1
                            maps = hk_self._up_sample_attn(
                                sim_slice, v, factor)
                            hk_self.heat_maps[batch_index -
                                              batch_size // 2][factor].append(maps)

                elif hk_self.calledCount % 2 == 1:
                    # Cond and uncond in different runs. Keep the second calls for the conditional.
                    if factor >= 1:
                        factor //= 1
                        maps = hk_self._up_sample_attn(sim_slice, v, factor)
                        hk_self.heat_maps[batch_index][factor].append(maps)

            out[start_idx:end_idx] = out_slice

        del q, k

        out = (
            out.unsqueeze(0)
            .reshape(batch_size, self.heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, -1, self.heads * dim_head)
        )

        return out

    def _hook_impl(self):
        self.monkey_patch('forward', self._forward)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
