# Patched Attention Functions for Capturing Attention Probabilities
# https://github.com/comfyanonymous/ComfyUI/blob/16417b40d9411c6e3a63949aa0f3582be25b28db/comfy/ldm/modules/sub_quadratic_attention.py

from comfy import model_management
from comfy.ldm.modules.attention import get_attn_precision
from comfy.ldm.modules.sub_quadratic_attention import (
    SummarizeChunk,
    ComputeQueryChunkAttn,
    dynamic_slice,
    _summarize_chunk,
    _query_chunk_attention,
)
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from functools import partial
import math
import logging


def attention_sub_quad_patched(
    query,
    key,
    value,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
):
    attn_precision = get_attn_precision(attn_precision, query.dtype)

    if skip_reshape:
        b, _, _, dim_head = query.shape
    else:
        b, _, dim_head = query.shape
        dim_head //= heads

    if skip_reshape:
        query = query.reshape(b * heads, -1, dim_head)
        value = value.reshape(b * heads, -1, dim_head)
        key = key.reshape(b * heads, -1, dim_head).movedim(1, 2)
    else:
        query = (
            query.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
        )
        value = (
            value.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
        )
        key = (
            key.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 3, 1)
            .reshape(b * heads, dim_head, -1)
        )

    dtype = query.dtype
    upcast_attention = attn_precision == torch.float32 and query.dtype != torch.float32
    if upcast_attention:
        bytes_per_token = torch.finfo(torch.float32).bits // 8
    else:
        bytes_per_token = torch.finfo(query.dtype).bits // 8
    batch_x_heads, q_tokens, _ = query.shape
    _, _, k_tokens = key.shape

    mem_free_total, _ = model_management.get_free_memory(query.device, True)

    kv_chunk_size_min = None
    kv_chunk_size = None
    query_chunk_size = None

    for x in [4096, 2048, 1024, 512, 256]:
        count = mem_free_total / (batch_x_heads * bytes_per_token * x * 4.0)
        if count >= k_tokens:
            kv_chunk_size = k_tokens
            query_chunk_size = x
            break

    if query_chunk_size is None:
        query_chunk_size = 512

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = (
            mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1])
            .expand(b, heads, -1, -1)
            .reshape(-1, mask.shape[-2], mask.shape[-1])
        )

    hidden_states, attn_probs = _efficient_dot_product_attention_patched(
        query,
        key,
        value,
        query_chunk_size=query_chunk_size,
        kv_chunk_size=kv_chunk_size,
        kv_chunk_size_min=kv_chunk_size_min,
        use_checkpoint=False,
        upcast_attention=upcast_attention,
        mask=mask,
    )

    hidden_states = hidden_states.to(dtype)
    if skip_output_reshape:
        hidden_states = hidden_states.unflatten(0, (-1, heads))
    else:
        hidden_states = (
            hidden_states.unflatten(0, (-1, heads)).transpose(1, 2).flatten(start_dim=2)
        )
    return hidden_states, attn_probs


def _get_attention_scores_no_kv_chunking_patched(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    scale: float,
    upcast_attention: bool,
    mask,
) -> Tensor:
    if upcast_attention:
        with torch.autocast(enabled=False, device_type="cuda"):
            query = query.float()
            key_t = key_t.float()
            attn_scores = torch.baddbmm(
                torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
                query,
                key_t,
                alpha=scale,
                beta=0,
            )
    else:
        attn_scores = torch.baddbmm(
            torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
            query,
            key_t,
            alpha=scale,
            beta=0,
        )

    if mask is not None:
        attn_scores += mask
    try:
        attn_probs = attn_scores.softmax(dim=-1)
        del attn_scores
    except model_management.OOM_EXCEPTION:
        logging.warning(
            "ran out of memory while running softmax in  _get_attention_scores_no_kv_chunking, trying slower in place softmax instead"
        )
        attn_scores -= attn_scores.max(dim=-1, keepdim=True).values  # noqa: F821 attn_scores is not defined
        torch.exp(attn_scores, out=attn_scores)
        summed = torch.sum(attn_scores, dim=-1, keepdim=True)
        attn_scores /= summed
        attn_probs = attn_scores

    hidden_states_slice = torch.bmm(attn_probs.to(value.dtype), value)

    # Also return the attention prob
    return hidden_states_slice, attn_probs


def _efficient_dot_product_attention_patched(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    query_chunk_size=1024,
    kv_chunk_size=None,
    kv_chunk_size_min=None,
    use_checkpoint=True,
    upcast_attention=False,
    mask=None,
):
    """Computes efficient dot-product attention given query, transposed key, and value.
    This is efficient version of attention presented in
    https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
    Args:
      query: queries for calculating attention with shape of
        `[batch * num_heads, tokens, channels_per_head]`.
      key_t: keys for calculating attention with shape of
        `[batch * num_heads, channels_per_head, tokens]`.
      value: values to be used in attention with shape of
        `[batch * num_heads, tokens, channels_per_head]`.
      query_chunk_size: int: query chunks size
      kv_chunk_size: Optional[int]: key/value chunks size. if None: defaults to sqrt(key_tokens)
      kv_chunk_size_min: Optional[int]: key/value minimum chunk size. only considered when kv_chunk_size is None. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
      use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
    Returns:
      Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
    """
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, _, k_tokens = key_t.shape
    scale = q_channels_per_head**-0.5

    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)

    if mask is not None and len(mask.shape) == 2:
        mask = mask.unsqueeze(0)

    def get_query_chunk(chunk_idx: int) -> Tensor:
        return dynamic_slice(
            query,
            (0, chunk_idx, 0),
            (batch_x_heads, min(query_chunk_size, q_tokens), q_channels_per_head),
        )

    def get_mask_chunk(chunk_idx: int) -> Tensor:
        if mask is None:
            return None
        if mask.shape[1] == 1:
            return mask
        chunk = min(query_chunk_size, q_tokens)
        return mask[:, chunk_idx : chunk_idx + chunk]

    summarize_chunk: SummarizeChunk = partial(
        _summarize_chunk, scale=scale, upcast_attention=upcast_attention
    )
    summarize_chunk: SummarizeChunk = (
        partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk
    )
    compute_query_chunk_attn: ComputeQueryChunkAttn = (
        partial(
            _get_attention_scores_no_kv_chunking_patched,
            scale=scale,
            upcast_attention=upcast_attention,
        )
        if k_tokens <= kv_chunk_size
        # fast-path for when there's just 1 key-value chunk per query chunk (this is just sliced attention btw)
        else (
            partial(
                _query_chunk_attention,
                kv_chunk_size=kv_chunk_size,
                summarize_chunk=summarize_chunk,
            )
        )
    )

    if q_tokens <= query_chunk_size:
        # fast-path for when there's just 1 query chunk
        return compute_query_chunk_attn(
            query=query,
            key_t=key_t,
            value=value,
            mask=mask,
        )

    # TODO: maybe we should use torch.empty_like(query) to allocate storage in-advance,
    # and pass slices to be mutated, instead of torch.cat()ing the returned slices

    # Collects the output and attention map chunks
    output_chunks = []
    attn_probs_chunks = []

    for i in range(math.ceil(q_tokens / query_chunk_size)):
        output_chunk, attn_map_chunk = compute_query_chunk_attn(
            query=get_query_chunk(i * query_chunk_size),
            key_t=key_t,
            value=value,
            mask=get_mask_chunk(i * query_chunk_size),
        )
        output_chunks.append(output_chunk)
        attn_probs_chunks.append(attn_map_chunk)

    res = torch.cat(output_chunks, dim=1)
    attn_probs = torch.cat(attn_probs_chunks, dim=1)

    return res, attn_probs
