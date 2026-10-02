# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""memory_efficient_attention on top of PyTorch SDPA, for when mslk is missing.

Runs on any device PyTorch's ``scaled_dot_product_attention`` supports (CPU,
MPS, ...). Backward comes from autograd. Conventions:

- Layouts: BMK ``[B, M, K]``, BMHK ``[B, M, H, K]`` and BMGHK
  ``[B, M, G, H, K]``. The output has the query's layout with last dim ``Kv``
  (value's last dim); ``K != Kv`` is allowed. The output is contiguous.
- ``scale`` defaults to ``1 / sqrt(K)``. ``p`` is the dropout probability
  (SDPA's ``dropout_p``); since SDPA on MPS has no dropout, ``p > 0`` uses an
  explicit softmax(QK^T)V implementation on every device.
- ``attn_bias``:

  * ``None``: no mask.
  * exactly ``LowerTriangularMask`` (type check, subclasses differ): SDPA's
    ``is_causal=True`` (both are top-left aligned).
  * ``torch.Tensor``: additive bias in xformers convention, broadcastable to
    ``[B, *GH, Mq, Mk]`` (``[B, Mq, Mk]`` for BMK, ``[B, H, Mq, Mk]`` for BMHK).
  * ``LowerTriangularMaskWithTensorBias``: its tensor as above plus a causal
    mask.
  * exactly ``BlockDiagonalMask``, ``BlockDiagonalCausalMask`` or
    ``BlockDiagonalCausalFromBottomRightMask``: no mask is built. q is split
    along M by ``q_seqinfo`` and k/v by ``k_seqinfo``, and each block runs
    its own attention: unmasked, ``is_causal=True`` (top-left, like
    ``LowerTriangularMask``), or bottom-right causal (``is_causal`` when
    ``Mq_i == Mk_i``, else a small ``[Mq_i, Mk_i]`` mask). Blocks with the
    same ``(Mq_i, Mk_i)`` are stacked on the batch dim into one call (a view
    when they are consecutive). Memory is O(sum_i Mq_i * Mk_i).
  * any other ``AttentionBias``: ``bias.materialize(...)`` as an additive
    mask. This costs O(Mq * Mk) memory.
  * Varlen biases (``BlockDiagonal*``) expect the packed layout, so ``B``
    must be 1.

- MQA/GQA: when key and value are both expanded with stride 0 along the
  head dim (BMHK dim 2, BMGHK dim 3), the compact K/V (one head, or one per
  group) is passed to SDPA with ``enable_gqa=True`` instead of copying K/V
  for every query head; the explicit (dropout, LSE) paths group the query
  heads the same way. Other stride patterns are copied as before.
- Rows where every key is masked produce an output of 0 (and zero gradients),
  like the xformers kernels; their LSE is ``-inf``. This relies on SDPA's
  safe softmax (PyTorch >= 2.5).
- LSE (from ``memory_efficient_attention_forward_requires_grad``) is float32
  with shape ``query.shape[:1] + query.shape[2:-1] + (Mq,)``, i.e. ``[B, Mq]``
  for BMK, ``[B, H, Mq]`` for BMHK and ``[B, G, H, Mq]`` for BMGHK. It is
  computed from the pre-dropout scores.
- ``op`` must be ``None``: there are no kernels to choose from.
- ``output_dtype``: the output is computed in the query dtype and then cast.
- ``memory_efficient_attention_backward`` recomputes the forward with
  autograd; it ignores ``output``/``lse`` and does not support ``p > 0``
  (the dropout mask cannot be replayed).
- ``memory_efficient_attention_partial`` returns ``(out, lse)`` (``out``
  float32 by default, like mslk) and ``merge_attentions`` combines such
  partial results computed over disjoint keys/values; see their docstrings.
"""

import math
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import torch

from .. import attn_bias as _public_attn_bias
from . import attn_bias as _vendored_attn_bias

# Accept the bias classes of both mslk (when installed) and the vendored copy,
# so this module also works when called directly next to mslk.
_CAUSAL_TYPES = frozenset(
    {_public_attn_bias.LowerTriangularMask, _vendored_attn_bias.LowerTriangularMask}
)
_CAUSAL_WITH_TENSOR_TYPES = (
    _public_attn_bias.LowerTriangularMaskWithTensorBias,
    _vendored_attn_bias.LowerTriangularMaskWithTensorBias,
)
_BIAS_TYPES = (_public_attn_bias.AttentionBias, _vendored_attn_bias.AttentionBias)


def _both(name: str) -> FrozenSet[type]:
    return frozenset(
        {getattr(_public_attn_bias, name), getattr(_vendored_attn_bias, name)}
    )


# Exact types (not subclasses) that run block by block instead of materializing
# the whole [Mq, Mk] mask.
_BLOCK_DIAGONAL_CAUSAL_TYPES = _both("BlockDiagonalCausalMask")
_BLOCK_DIAGONAL_BOTTOM_RIGHT_TYPES = _both("BlockDiagonalCausalFromBottomRightMask")
_BLOCK_DIAGONAL_TYPES = (
    _both("BlockDiagonalMask")
    | _BLOCK_DIAGONAL_CAUSAL_TYPES
    | _BLOCK_DIAGONAL_BOTTOM_RIGHT_TYPES
)
# Private switch for tests and benchmarks: False forces the dense
# (materialized-mask) path for the block-diagonal biases too.
_USE_BLOCK_DIAGONAL_PATH = True

_NO_OP_MSG = (
    "xformers.ops.fmha: the `op` argument is not supported because mslk is not "
    "installed; only the PyTorch SDPA fallback is available. Pass op=None."
)


def _check_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if not (query.ndim == key.ndim == value.ndim) or query.ndim not in (3, 4, 5):
        raise ValueError(
            "query, key and value must all be BMK (3-D), BMHK (4-D) or BMGHK (5-D); "
            f"got shapes {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )
    if (
        query.shape[0] != key.shape[0]
        or key.shape[:-1] != value.shape[:-1]
        or query.shape[2:] != key.shape[2:]
        or query.shape[2:-1] != value.shape[2:-1]
    ):
        raise ValueError(
            "Incompatible shapes for query/key/value: "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )


def _to_sdpa(x: torch.Tensor) -> torch.Tensor:
    """[B, M, *GH, K] -> [B, GH, M, K]"""
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x.transpose(1, 2)
    B, M, G, H, K = x.shape
    return x.permute(0, 2, 3, 1, 4).reshape(B, G * H, M, K)


def _from_sdpa(x: torch.Tensor, query_shape: Tuple[int, ...]) -> torch.Tensor:
    """[B, GH, M, K] -> [B, M, *GH, K]"""
    if len(query_shape) == 3:
        return x.squeeze(1)
    if len(query_shape) == 4:
        return x.transpose(1, 2)
    B, _, G, H, _ = query_shape
    return x.reshape(B, G, H, x.shape[-2], x.shape[-1]).permute(0, 3, 1, 2, 4)


def _compact_kv(
    key: torch.Tensor, value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Narrow K/V to one head when both are stride-0 expanded along H.

    xformers expresses MQA/GQA by expanding K/V along the head dim (BMHK dim
    2, BMGHK dim 3). ``narrow`` keeps autograd reaching the expanded tensor
    (the expand's backward sums the gradient over the heads). In SDPA layout
    the compact K/V have ``G`` heads (``1`` for BMHK) and query head
    ``g * H + h`` uses K/V head ``g``, which is SDPA's ``enable_gqa`` grouping.
    """
    if key.ndim not in (4, 5):
        return key, value
    dim = key.ndim - 2
    if key.shape[dim] > 1 and key.stride(dim) == 0 and value.stride(dim) == 0:
        return key.narrow(dim, 0, 1), value.narrow(dim, 0, 1)
    return key, value


def _grouped_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` where ``a`` has ``Hq`` heads and ``b`` has ``Hkv`` heads.

    a: [B, Hq, M, X], b: [B, Hkv, X, N] with Hq a multiple of Hkv; query head
    i uses b's head ``i // (Hq // Hkv)``. The ``Hq // Hkv`` query heads of a
    group are folded into M, so b is never copied per query head.
    """
    B, Hq, M, X = a.shape
    Hkv = b.shape[1]
    if Hkv == Hq:
        return a @ b
    out = a.reshape(B, Hkv, (Hq // Hkv) * M, X) @ b
    return out.view(B, Hq, M, b.shape[-1])


def _tensor_bias_to_mask(
    bias: torch.Tensor, query: torch.Tensor, Mk: int
) -> torch.Tensor:
    """xformers-convention [B, *GH, Mq, Mk] tensor -> [B, GH, Mq, Mk] SDPA mask"""
    B, Mq = query.shape[:2]
    gh = query.shape[2:-1]
    target = (B, *gh, Mq, Mk)
    if bias.ndim > len(target):
        raise ValueError(
            f"attn_bias tensor of shape {tuple(bias.shape)} has more dimensions than "
            f"expected {target}"
        )
    # mslk's kernels reject these instead of converting, so do the same: code
    # that works on the fallback must also work with mslk.
    if bias.dtype != query.dtype:
        raise ValueError(
            "attn_bias tensor should have the same dtype as the query\n"
            f"  query.dtype    : {query.dtype}\n"
            f"  attn_bias.dtype: {bias.dtype}"
        )
    if bias.device != query.device:
        raise ValueError(
            "Attention bias and Query/Key/Value should be on the same device\n"
            f"  query.device: {query.device}\n"
            f"  attn_bias   : {bias.device}"
        )
    bias = bias.expand(target)
    return bias.reshape(B, math.prod(gh), Mq, Mk)


def _bias_to_mask(
    attn_bias: Any, query: torch.Tensor, Mk: int
) -> Tuple[Optional[torch.Tensor], bool]:
    """Returns (additive SDPA mask or None, is_causal)."""
    if attn_bias is None:
        return None, False
    if type(attn_bias) in _CAUSAL_TYPES:
        return None, True
    if isinstance(attn_bias, torch.Tensor):
        return _tensor_bias_to_mask(attn_bias, query, Mk), False

    B, Mq = query.shape[:2]
    if isinstance(attn_bias, _CAUSAL_WITH_TENSOR_TYPES):
        causal = _vendored_attn_bias.LowerTriangularMask().materialize(
            (Mq, Mk), dtype=query.dtype, device=query.device
        )
        return _tensor_bias_to_mask(attn_bias._bias, query, Mk) + causal, False
    if isinstance(attn_bias, _BIAS_TYPES):
        if hasattr(attn_bias, "q_seqinfo") and B != 1:
            raise ValueError(
                f"{type(attn_bias).__name__} expects the packed layout with batch "
                f"size 1 (sequences concatenated along M), got batch size {B}"
            )
        GH = math.prod(query.shape[2:-1])
        mask_Mk = Mk
        block_tables = getattr(attn_bias, "block_tables", None)
        if block_tables is not None:
            # Paged biases take the logical (unpaged) key length, and return a
            # mask that only reaches the last page block_tables uses.
            mask_Mk = block_tables.numel() * getattr(attn_bias, "page_size")
        mask = attn_bias.materialize(
            (B, GH, Mq, mask_Mk), dtype=query.dtype, device=query.device
        )
        if mask.shape[-1] < Mk:
            # Keys in pages that no sequence uses are never attended to.
            mask = torch.nn.functional.pad(
                mask, (0, Mk - mask.shape[-1]), value=-math.inf
            )
        return mask, False
    raise TypeError(f"Unsupported attn_bias type: {type(attn_bias)}")


def _attention_with_dropout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    is_causal: bool,
    p: float,
    scale: float,
) -> torch.Tensor:
    """Explicit attention with dropout, in SDPA layout.

    SDPA on MPS does not support dropout, so dropout always takes this path,
    on every device, for consistent behavior.
    """
    scores = _grouped_matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        mask = _vendored_attn_bias.LowerTriangularMask().materialize(
            scores.shape[-2:], dtype=q.dtype, device=q.device
        )
    if mask is not None:
        scores = scores + mask
    # Fully masked rows: output 0 and no NaN in the gradients.
    fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    scores = scores.masked_fill(fully_masked, 0.0)
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    attn = attn.masked_fill(fully_masked, 0.0)
    attn = torch.nn.functional.dropout(attn, p=p, training=True)
    return _grouped_matmul(attn, v)


def _core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    is_causal: bool,
    p: float,
    scale: float,
    compute_lse: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Attention in SDPA layout: returns (out [B, GH, Mq, Kv], lse [B, GH, Mq]).

    k and v may have fewer heads than q (see ``_compact_kv``).
    """
    if p == 0.0:
        # Only pass enable_gqa when needed, so other calls stay unchanged.
        gqa: Dict[str, bool] = {"enable_gqa": True} if k.shape[1] != q.shape[1] else {}
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal, scale=scale, **gqa
        )
    else:
        out = _attention_with_dropout(q, k, v, mask, is_causal, p, scale)

    lse = None
    if compute_lse:
        with torch.no_grad():
            scores = _grouped_matmul(q.float(), k.float().transpose(-2, -1)) * scale
            if is_causal:
                mask = _vendored_attn_bias.LowerTriangularMask().materialize(
                    scores.shape[-2:], dtype=torch.float32, device=scores.device
                )
            if mask is not None:
                scores = scores + mask.float()
            lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def _attention_block_diagonal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Any,
    p: float,
    scale: float,
    compute_lse: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-block attention for the exact ``_BLOCK_DIAGONAL_TYPES``.

    Blocks sharing ``(Mq_i, Mk_i)`` run as one SDPA call, stacked on the batch
    dim. Returns (out in the query layout, lse [GH, Mq]).
    """
    if query.shape[0] != 1:
        raise ValueError(
            f"{type(attn_bias).__name__} expects the packed layout with batch "
            f"size 1 (sequences concatenated along M), got batch size "
            f"{query.shape[0]}"
        )
    q_starts = attn_bias.q_seqinfo.seqstart_py
    k_starts = attn_bias.k_seqinfo.seqstart_py
    if q_starts[-1] != query.shape[1] or k_starts[-1] != key.shape[1]:
        raise ValueError(
            f"{type(attn_bias).__name__} covers {q_starts[-1]} queries and "
            f"{k_starts[-1]} keys, but got Mq={query.shape[1]}, Mk={key.shape[1]}"
        )
    bottom_right = type(attn_bias) in _BLOCK_DIAGONAL_BOTTOM_RIGHT_TYPES
    causal = bottom_right or type(attn_bias) in _BLOCK_DIAGONAL_CAUSAL_TYPES

    # (Mq_i, Mk_i) -> block indices, in order of first appearance
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i, (q_start, k_start) in enumerate(zip(q_starts, k_starts[:-1])):
        if q_starts[i + 1] > q_start:  # blocks without queries contribute nothing
            shape = (q_starts[i + 1] - q_start, k_starts[i + 1] - k_start)
            groups.setdefault(shape, []).append(i)

    def consecutive(idx: List[int]) -> bool:
        return idx[-1] - idx[0] == len(idx) - 1

    def gather(x: torch.Tensor, starts: List[int], idx: List[int], m: int):
        """Blocks ``idx`` (all of length m) of x[0] -> [n, m, *GH, K]"""
        if consecutive(idx):  # a view, no copy
            begin = starts[idx[0]]
            return x[0, begin : begin + m * len(idx)].unflatten(0, (len(idx), m))
        return torch.stack([x[0, starts[i] : starts[i] + m] for i in idx])

    # first block index -> (out [M_chunk, *GH, Kv], lse [GH, M_chunk])
    chunks: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    for (mq, mk), idx in groups.items():
        q = _to_sdpa(gather(query, q_starts, idx, mq))
        k = _to_sdpa(gather(key, k_starts, idx, mk))
        v = _to_sdpa(gather(value, k_starts, idx, mk))
        mask: Optional[torch.Tensor] = None
        is_causal = False
        if causal and mk > 0:
            if bottom_right and mq != mk:
                mask = _vendored_attn_bias.LowerTriangularFromBottomRightMask().materialize(
                    (mq, mk), dtype=q.dtype, device=q.device
                )
            else:
                is_causal = True
        out, lse = _core(q, k, v, mask, is_causal, p, scale, compute_lse)
        out = _from_sdpa(out, (len(idx), mq, *query.shape[2:]))  # [n, mq, *GH, Kv]
        if consecutive(idx):
            lse = None if lse is None else lse.transpose(0, 1).flatten(1)
            chunks[idx[0]] = (out.flatten(0, 1), lse)
        else:
            lse_blocks = [None] * len(idx) if lse is None else lse.unbind(0)
            for i, o, block_lse in zip(idx, out.unbind(0), lse_blocks):
                chunks[i] = (o, block_lse)

    ordered = [chunks[i] for i in sorted(chunks)]
    out = torch.cat([o for o, _ in ordered]).unsqueeze(0)
    lse = None
    if compute_lse:
        lse = torch.cat([x for _, x in ordered], dim=-1)  # type: ignore[misc]
    return out, lse


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Any,
    p: float,
    scale: Optional[float],
    output_dtype: Optional[torch.dtype],
    compute_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _check_inputs(query, key, value)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    if (
        _USE_BLOCK_DIAGONAL_PATH
        and type(attn_bias) in _BLOCK_DIAGONAL_TYPES
        and query.shape[1] > 0
    ):
        key, value = _compact_kv(key, value)
        out, lse = _attention_block_diagonal(
            query, key, value, attn_bias, p, scale, compute_lse
        )
    else:
        mask, is_causal = _bias_to_mask(attn_bias, query, key.shape[1])
        key, value = _compact_kv(key, value)
        q, k, v = _to_sdpa(query), _to_sdpa(key), _to_sdpa(value)
        out, lse = _core(q, k, v, mask, is_causal, p, scale, compute_lse)
        out = _from_sdpa(out, query.shape)
    out = out.contiguous()
    if output_dtype is not None:
        out = out.to(output_dtype)
    if lse is not None:
        lse = lse.reshape(query.shape[:1] + query.shape[2:-1] + query.shape[1:2])
    return out, lse


def _check_op(op: Any) -> None:
    if op is not None:
        raise NotImplementedError(_NO_OP_MSG)


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, Any]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Any = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Attention with autograd support. See the module docstring."""
    _check_op(op)
    return _attention(query, key, value, attn_bias, p, scale, output_dtype)[0]


def memory_efficient_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, Any]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Any = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Forward only, without building an autograd graph."""
    _check_op(op)
    with torch.no_grad():
        return _attention(query, key, value, attn_bias, p, scale, output_dtype)[0]


def memory_efficient_attention_forward_requires_grad(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, Any]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Any = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward returning ``(out, lse)``, without building an autograd graph.

    ``lse`` is float32 with shape ``[B, Mq]`` (BMK), ``[B, H, Mq]`` (BMHK) or
    ``[B, G, H, Mq]`` (BMGHK).
    """
    _check_op(op)
    with torch.no_grad():
        out, lse = _attention(
            query, key, value, attn_bias, p, scale, output_dtype, compute_lse=True
        )
    assert lse is not None
    return out, lse


def memory_efficient_attention_backward(
    grad: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, Any]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients ``(grad_q, grad_k, grad_v)``, by recomputing the forward.

    ``output`` and ``lse`` are not used. Dropout (``p > 0``) is not supported,
    since the forward's dropout mask cannot be reproduced.
    """
    _check_op(op)
    if p != 0.0:
        raise NotImplementedError(
            "memory_efficient_attention_backward with dropout (p > 0) is not "
            "supported by the PyTorch SDPA fallback; use memory_efficient_attention "
            "with autograd instead."
        )
    with torch.enable_grad():
        q, k, v = (t.detach().requires_grad_() for t in (query, key, value))
        out, _ = _attention(q, k, v, attn_bias, 0.0, scale, None)
        gq, gk, gv = torch.autograd.grad(out, (q, k, v), grad.to(out.dtype))
    return gq, gk, gv


def memory_efficient_attention_partial(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, Any]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Any = None,
    output_dtype: Optional[torch.dtype] = None,
    _allow_backward: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention against part of the keys/values: returns ``(out, lse)``.

    Outputs of calls with the same query and disjoint keys/values can be
    combined with :func:`merge_attentions` into the attention over all of them.
    As in mslk, ``out`` is float32 by default (float64 for a float64 query);
    here it is computed in the query dtype and then cast. ``lse`` is float32
    with the same shape as in ``memory_efficient_attention_forward_requires_grad``
    (for varlen ``BlockDiagonal*`` biases the packed ``[1, *GH, M]`` layout).
    Rows without any visible key get ``out = 0`` and ``lse = -inf``.

    There is no backward pass, unless ``_allow_backward=True``: then ``out``
    is differentiable but ``lse`` is not, so only the gradient of ``out`` is
    used. This makes it easy to get wrong gradients (as in mslk). Dropout is
    not supported.
    """
    if p != 0.0:
        raise NotImplementedError("dropout is not supported.")
    _check_op(op)
    if output_dtype is None:
        output_dtype = torch.float64 if query.dtype is torch.float64 else torch.float32
    is_grad = (
        _allow_backward
        and torch.is_grad_enabled()
        and any(x.requires_grad for x in (query, key, value))
    )
    if not is_grad:
        with torch.no_grad():
            out, lse = _attention(
                query, key, value, attn_bias, 0.0, scale, output_dtype, True
            )
    else:
        if query.ndim == 5:
            raise ValueError("gradients not supported for 5D tensors")
        out, lse = _attention(
            query, key, value, attn_bias, 0.0, scale, output_dtype, True
        )
    assert lse is not None
    return out, lse


def _merge(
    attn_split: List[torch.Tensor],
    lse_split: List[torch.Tensor],
    write_lse: bool,
    output_dtype: Optional[torch.dtype],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """attn_split: [B, M, G, H, Kq] each, lse_split: [B, G, H, M] each."""
    acc_dtype = (
        torch.float64
        if any(x.dtype is torch.float64 for x in [*attn_split, *lse_split])
        else torch.float32
    )
    # [B, G, H, M] -> [B, M, G, H]
    lses = [x.to(acc_dtype).permute(0, 3, 1, 2) for x in lse_split]
    lse_max = lses[0]
    for lse in lses[1:]:
        lse_max = torch.maximum(lse_max, lse)
    # All chunks -inf: use 0 so the exps are exp(-inf) = 0, not NaN.
    lse_max = lse_max.masked_fill(torch.isneginf(lse_max), 0.0)
    numerator = torch.zeros(
        attn_split[0].shape, dtype=acc_dtype, device=attn_split[0].device
    )
    sumexp = torch.zeros_like(lse_max)
    for attn, lse in zip(attn_split, lses):
        weight = torch.exp(lse - lse_max)
        sumexp = sumexp + weight
        numerator = numerator + attn.to(acc_dtype) * weight.unsqueeze(-1)
    # Rows where every chunk is fully masked (sumexp == 0) output 0.
    out = numerator / sumexp.masked_fill(sumexp == 0, 1.0).unsqueeze(-1)
    out = out.to(output_dtype or attn_split[0].dtype)
    lse_out = None
    if write_lse:
        # log(0) = -inf for fully masked rows
        lse_out = (lse_max + torch.log(sumexp)).permute(0, 2, 3, 1)
        lse_out = lse_out.to(lse_split[0].dtype)
    return out, lse_out


class _MergeAttentions(torch.autograd.Function):
    """Lets merge_attentions run on inputs that require grad (like mslk);
    the backward raises, as mslk's does."""

    @staticmethod
    # type: ignore
    def forward(ctx, output_dtype, num_chunks, *tensors):
        out, lse = _merge(
            list(tensors[:num_chunks]), list(tensors[num_chunks:]), True, output_dtype
        )
        return out, lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out, grad_lse):  # type: ignore[override]
        raise NotImplementedError(
            "Backward pass is not implemented for merge_attentions. "
            "If it was, it would be easy to get wrong attention gradients, "
            "because the gradients of the LSEs "
            "don't get propagated by attention backward."
        )


def merge_attentions(
    attn_split: Union[torch.Tensor, Sequence[torch.Tensor]],
    lse_split: Union[torch.Tensor, Sequence[torch.Tensor]],
    write_lse: bool = True,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Combine attention computed on disjoint parts of K/V for the same query.

    ``Out = sum_i Out_i * exp(LSE_i) / sum_i exp(LSE_i)`` and
    ``LSE = log(sum_i exp(LSE_i))``, accumulated in float32 (float64 if an
    input is float64). Chunks with ``LSE = -inf`` contribute nothing; rows
    where every chunk has ``LSE = -inf`` get ``Out = 0`` and ``LSE = -inf``.

    Args:
        attn_split: a list of ``[B, M, G, H, Kq]`` or ``[B, M, H, Kq]`` tensors,
            or one stacked ``[num_chunks, B, M, (G,) H, Kq]`` tensor.
        lse_split: a list of ``[B, G, H, M]`` or ``[B, H, M]`` tensors, or one
            stacked ``[num_chunks, B, (G,) H, M]`` tensor.
        write_lse: whether to return the merged LSE.
        output_dtype: dtype of the merged output (default: the chunks' dtype).
    Returns:
        ``(out [B, M, (G,) H, Kq], lse [B, (G,) H, M] or None)``; ``lse`` has
        the dtype of ``lse_split``.

    As in mslk, inputs may require grad (then ``write_lse`` must be True), but
    calling backward raises ``NotImplementedError``.
    """
    attn_is_concat = isinstance(attn_split, torch.Tensor)
    lse_is_concat = isinstance(lse_split, torch.Tensor)
    attn_list: List[torch.Tensor] = list(
        attn_split.unbind(0)  # type: ignore[union-attr]
        if attn_is_concat
        else attn_split
    )
    lse_list: List[torch.Tensor] = list(
        lse_split.unbind(0) if lse_is_concat else lse_split  # type: ignore[union-attr]
    )
    requires_grad = torch.is_grad_enabled() and any(
        x.requires_grad for x in [*attn_list, *lse_list]
    )
    if requires_grad and not write_lse:
        raise ValueError("write_lse should be true if inputs require gradients.")

    num_chunks = len(attn_list)
    if len(lse_list) != num_chunks:
        raise ValueError(
            "Incompatible number of LSE and attention chunks: "
            f"{len(attn_list)=}, {len(lse_list)=}"
        )
    if num_chunks == 0:
        raise ValueError("merge_attentions needs at least one chunk")

    is_bmhk = attn_list[0].ndim == 4
    for i in range(num_chunks):
        if attn_list[i].ndim != lse_list[i].ndim + 1 or attn_list[i].ndim not in (
            4,
            5,
        ):
            raise ValueError(
                f"Incompatible input shapes for chunk {i}: "
                f"{attn_list[i].shape=}, {lse_list[i].shape=}"
            )
        if (attn_list[i].ndim == 4) != is_bmhk:
            raise ValueError("All chunks must be either BMHK or BMGHK")
        if is_bmhk:
            attn_list[i] = attn_list[i].unsqueeze(2)
            lse_list[i] = lse_list[i].unsqueeze(1)

    B, M, G, H, Kq = attn_list[0].shape
    for i in range(num_chunks):
        if attn_list[i].shape != (B, M, G, H, Kq):
            raise ValueError(
                f"Incompatible input shapes for attention chunk {i}: "
                f"{attn_list[i].shape=}, {(B, M, G, H, Kq)=}"
            )
        if lse_list[i].shape != (B, G, H, M):
            raise ValueError(
                f"Incompatible input shapes for LSE chunk {i}: "
                f"{lse_list[i].shape=}, {(B, G, H, M)=}"
            )

    lse_out: Optional[torch.Tensor]
    if requires_grad:
        attn_out, lse_out = _MergeAttentions.apply(
            output_dtype, num_chunks, *attn_list, *lse_list
        )
    else:
        attn_out, lse_out = _merge(attn_list, lse_list, write_lse, output_dtype)

    if is_bmhk:
        attn_out = attn_out[:, :, 0]
        if lse_out is not None:
            lse_out = lse_out[:, 0]
    return attn_out, lse_out
