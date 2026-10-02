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
"""

import math
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

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
    bias = bias.to(device=query.device, dtype=query.dtype).expand(target)
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
        mask = attn_bias.materialize(
            (B, GH, Mq, Mk), dtype=query.dtype, device=query.device
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
    scores = (q @ k.transpose(-2, -1)) * scale
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
    return attn @ v


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
    """Attention in SDPA layout: returns (out [B, GH, Mq, Kv], lse [B, GH, Mq])."""
    if p == 0.0:
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal, scale=scale
        )
    else:
        out = _attention_with_dropout(q, k, v, mask, is_causal, p, scale)

    lse = None
    if compute_lse:
        with torch.no_grad():
            scores = (q.float() @ k.float().transpose(-2, -1)) * scale
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
        out, lse = _attention_block_diagonal(
            query, key, value, attn_bias, p, scale, compute_lse
        )
    else:
        mask, is_causal = _bias_to_mask(attn_bias, query, key.shape[1])
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
