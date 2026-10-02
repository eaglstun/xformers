# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the PyTorch SDPA fallback of tree attention and attn_bias_utils.

They call xformers.ops.fmha._fallback.* directly, so they also run where mslk
is installed (only the tests of the public modules' fallback wiring are
skipped there).
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from xformers.ops.fmha import attn_bias as fmha_attn_bias
from xformers.ops.fmha._backend import HAS_MSLK
from xformers.ops.fmha._fallback import (
    attn_bias_utils as fb_utils,
    sdpa,
    tree_attention as fb_tree,
)


def _mps_usable() -> bool:
    """is_available() is not enough: on GitHub's macOS runners (VMs) it is True,
    but every MPS allocation fails with "MPS backend out of memory"."""
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.ones(1, device="mps").add_(1).cpu()
    except RuntimeError:
        return False
    return True


DEVICES = [
    "cpu",
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(not _mps_usable(), reason="MPS is not usable"),
    ),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOLERANCES: Dict[torch.dtype, Dict[str, Any]] = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=3e-3, rtol=3e-3),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}

# A small EAGLE tree, from https://github.com/SafeAILab/EAGLE/blob/e98fc7c/model/choices.py
# (truncated), and an unsorted Medusa-style one.
# fmt: off
EAGLE_SMALL = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (2, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 0, 0, 0)]
MEDUSA_UNSORTED = [(0,), (0, 0), (1,), (0, 1), (2,), (0, 0, 0), (1, 0), (0, 2), (3,), (0, 3)]
# fmt: on

TREES = {
    "eagle_small": EAGLE_SMALL,
    "medusa_unsorted": MEDUSA_UNSORTED,
    "full_1x1": fb_tree.construct_full_tree_choices(1, 1),
    "full_3x2": fb_tree.construct_full_tree_choices(3, 2),
    "full_2x3": fb_tree.construct_full_tree_choices(2, 3),
    "arbitrary_2_3_1": fb_tree.construct_tree_choices([2, 3, 1]),
}


# --- TreeAttnMetadata --------------------------------------------------------


def _brute_force_tree(
    tree_choices: List[Tuple[int, ...]],
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """Returns (mask, parent node, depth) for nodes in TreeAttnMetadata order
    (node 0 is the root, node i + 1 is sorted(tree_choices)[i])."""
    nodes = [()] + sorted(tree_choices, key=lambda x: (len(x), x))
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    mask = torch.full((n, n), -math.inf)
    for i, node in enumerate(nodes):
        # A node sees itself and every ancestor, up to and including the root.
        for length in range(len(node) + 1):
            mask[i, index[node[:length]]] = 0
    parents = [0] + [index[node[:-1]] for node in nodes[1:]]
    depths = [len(node) for node in nodes]
    return mask, parents, depths


@pytest.mark.parametrize("tree_name", list(TREES))
def test_tree_attn_metadata(tree_name: str) -> None:
    tree_choices = TREES[tree_name]
    meta = fb_tree.TreeAttnMetadata.from_tree_choices(tree_choices, torch.float32)
    mask, parents, depths = _brute_force_tree(tree_choices)
    n = len(tree_choices) + 1

    torch.testing.assert_close(meta.attention_bias, mask)
    assert meta.tree_seq_position_ids.tolist() == depths
    assert meta.parent_node_indices.tolist() == parents[1:]
    num_children = [parents[1:].count(i) for i in range(n)]
    assert meta.num_children_per_node.tolist() == num_children

    # Every retrieval path is the root-to-leaf chain of a distinct leaf.
    leaves = {i for i in range(n) if num_children[i] == 0}
    seen_leaves = set()
    for path, length in zip(meta.retrieval_indices.tolist(), meta.path_lengths):
        assert all(x == -1 for x in path[length:])
        path = path[:length]
        assert path[0] == 0
        for parent, child in zip(path, path[1:]):
            assert parents[child] == parent
        seen_leaves.add(path[-1])
    assert seen_leaves == leaves
    assert len(meta.path_lengths) == len(leaves)

    assert meta.subtree_sizes[-1] == n
    assert sum(meta.num_nodes_per_level.tolist()) == n


def test_tree_size_helpers() -> None:
    for depth, branching in [(1, 1), (2, 3), (3, 2), (4, 1)]:
        choices = fb_tree.construct_full_tree_choices(depth, branching)
        assert len(choices) + 1 == fb_tree.get_full_tree_size(depth + 1, branching)
    assert fb_tree.use_triton_splitk_for_prefix(1, 1, 10)
    assert not fb_tree.use_triton_splitk_for_prefix(256, 1, 128)


# --- tree_attention vs a dense reference ---------------------------------------


def _make_paged(
    cache: torch.Tensor, block_tables: torch.Tensor, page_size: int
) -> torch.Tensor:
    """[B, Mk, ...] cache -> [1, B * Mk, ...] where logical page j of row b is
    stored at physical page block_tables[b, j] (like tests/test_tree_attention.py)."""
    B, Mk = cache.shape[:2]
    pages = cache.reshape(B * (Mk // page_size), page_size, *cache.shape[2:])
    paged = torch.empty_like(pages)
    paged[block_tables.flatten().long()] = pages
    return paged.reshape(1, B * Mk, *cache.shape[2:])


def _dense_reference(
    q: torch.Tensor,
    spec_k: torch.Tensor,
    spec_v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    spec_attn_bias: torch.Tensor,
    prefix_attn_bias: fmha_attn_bias.AttentionBias,
) -> torch.Tensor:
    """softmax(q @ [cache; spec]^T * scale + [prefix | spec] bias) @ [cache; spec]
    in float32, per batch element. All inputs BMGHK, caches unpaged."""
    B, tree_size_q, G, H, D = q.shape
    Mk = cache_k.shape[1]
    # [B * tq, B * Mk] -> diagonal blocks [B, tq, Mk]
    prefix_mask = prefix_attn_bias.materialize(
        (B * tree_size_q, B * Mk), dtype=torch.float32, device=q.device
    )
    outs = []
    for b in range(B):
        mask = torch.cat(
            [
                prefix_mask[
                    b * tree_size_q : (b + 1) * tree_size_q, b * Mk : (b + 1) * Mk
                ],
                spec_attn_bias.float(),
            ],
            dim=-1,
        )
        k = torch.cat([cache_k[b], spec_k[b]]).float()  # [Mk + tkv, G, H, D]
        v = torch.cat([cache_v[b], spec_v[b]]).float()
        scores = torch.einsum("qghd,kghd->ghqk", q[b].float(), k) / math.sqrt(D)
        attn = torch.softmax(scores + mask, dim=-1)
        outs.append(torch.einsum("ghqk,kghd->qghd", attn, v))
    return torch.stack(outs)


def _tree_inputs(
    tree_choices: List[Tuple[int, ...]],
    B: int,
    G: int,
    square: bool,
    device: str,
    dtype: torch.dtype,
    Mk: int = 64,
    H: int = 4,
    D: int = 16,
    seed: int = 0,
):
    torch.manual_seed(seed)
    tree_size_kv = len(tree_choices) + 1
    tree_size_q = tree_size_kv if square else max(tree_size_kv // 2, 1)
    q = torch.randn([B, tree_size_q, G, H, D], device=device, dtype=dtype)
    # Shared K/V heads, expanded with stride 0 (multi-query), like the upstream test.
    spec_k = torch.randn([B, tree_size_kv, G, 1, D], device=device, dtype=dtype)
    spec_v = torch.randn_like(spec_k)
    spec_k = spec_k.expand(-1, -1, -1, H, -1)
    spec_v = spec_v.expand(-1, -1, -1, H, -1)
    cache_k = torch.randn([B, Mk, G, 1, D], device=device, dtype=dtype)
    cache_v = torch.randn_like(cache_k)
    cache_k = cache_k.expand(-1, -1, -1, H, -1)
    cache_v = cache_v.expand(-1, -1, -1, H, -1)
    # Uneven lengths, including a full row.
    kv_lens = [Mk] + [random.Random(seed + b).randint(1, Mk) for b in range(1, B)]
    meta = fb_tree.TreeAttnMetadata.from_tree_choices(
        tree_choices, dtype, torch.device(device)
    )
    spec_attn_bias = meta.attention_bias[-tree_size_q:]
    prefix_attn_bias = fmha_attn_bias.BlockDiagonalPaddedKeysMask.from_seqlens(
        q_seqlen=[tree_size_q] * B, kv_seqlen=kv_lens, kv_padding=Mk
    )
    return q, spec_k, spec_v, cache_k, cache_v, spec_attn_bias, prefix_attn_bias


@pytest.mark.parametrize("paged", [False, True], ids=["unpaged", "paged"])
@pytest.mark.parametrize("fmt", ["BMHK", "BMGHK"])
@pytest.mark.parametrize("square", [True, False], ids=["square", "rect"])
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("tree_name", ["eagle_small", "full_2x3"])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
def test_tree_attention(
    device: str,
    dtype: torch.dtype,
    tree_name: str,
    B: int,
    square: bool,
    fmt: str,
    paged: bool,
) -> None:
    G = 2 if fmt == "BMGHK" else 1
    q, spec_k, spec_v, cache_k, cache_v, spec_bias, prefix_bias = _tree_inputs(
        TREES[tree_name], B, G, square, device, dtype
    )
    ref = _dense_reference(q, spec_k, spec_v, cache_k, cache_v, spec_bias, prefix_bias)

    run_cache_k, run_cache_v = cache_k, cache_v
    run_prefix_bias: fmha_attn_bias.AttentionBias = prefix_bias
    if paged:
        page_size = 16
        Mk = cache_k.shape[1]
        block_tables = torch.randperm(
            B * (Mk // page_size), generator=torch.Generator().manual_seed(1)
        ).view(B, Mk // page_size)
        block_tables = block_tables.to(device=device, dtype=torch.int32)
        run_cache_k = _make_paged(cache_k, block_tables, page_size)
        run_cache_v = _make_paged(cache_v, block_tables, page_size)
        run_prefix_bias = prefix_bias.make_paged(
            block_tables,
            page_size=page_size,
            paged_type=fmha_attn_bias.PagedBlockDiagonalPaddedKeysMask,
        )

    args = [q, spec_k, spec_v, run_cache_k, run_cache_v]
    if fmt == "BMHK":
        args = [x.squeeze(2) for x in args]
        ref = ref.squeeze(2)
    out = fb_tree.tree_attention(
        args[0], args[1], args[2], args[3], args[4], spec_bias, run_prefix_bias
    )

    assert out.shape == args[0].shape
    assert out.dtype == dtype
    assert out.device.type == device
    torch.testing.assert_close(out.float(), ref, **TOLERANCES[dtype])


@pytest.mark.parametrize("device", DEVICES)
def test_tree_attention_gappy_prefix(device: str) -> None:
    """A non-padded prefix bias (gappy keys) goes through the generic path."""
    q, spec_k, spec_v, cache_k, cache_v, spec_bias, prefix_bias = _tree_inputs(
        EAGLE_SMALL, 2, 1, True, device, torch.float32
    )
    Mk = cache_k.shape[1]
    kv_lens = prefix_bias.k_seqinfo.seqlen_py
    gappy = fmha_attn_bias.BlockDiagonalGappyKeysMask.from_seqlens(
        q_seqlen=[q.shape[1]] * len(kv_lens),
        kv_seqstarts=[b * Mk for b in range(len(kv_lens))] + [len(kv_lens) * Mk],
        kv_seqlen=kv_lens,
    )
    out = fb_tree.tree_attention(q, spec_k, spec_v, cache_k, cache_v, spec_bias, gappy)
    ref = _dense_reference(q, spec_k, spec_v, cache_k, cache_v, spec_bias, prefix_bias)
    torch.testing.assert_close(out, ref, **TOLERANCES[torch.float32])


def _small_inputs(cache_dtype: Optional[torch.dtype] = None):
    q, spec_k, spec_v, cache_k, cache_v, spec_bias, prefix_bias = _tree_inputs(
        EAGLE_SMALL, 2, 1, True, "cpu", torch.float32
    )
    if cache_dtype is not None:
        cache_k = torch.zeros(cache_k.shape, dtype=cache_dtype)
        cache_v = torch.zeros(cache_v.shape, dtype=cache_dtype)
    return q, spec_k, spec_v, cache_k, cache_v, spec_bias, prefix_bias


class _SomeOp:
    pass


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(prefix_op=_SomeOp), "prefix_op"),
        (dict(suffix_op=_SomeOp), "suffix_op"),
        (dict(autotune=True), "autotune"),
        (dict(quantized_kv_scales=(torch.ones(1), torch.ones(1))), "quantized"),
        (dict(q_fp8=torch.ones(1)), "quantized"),
    ],
    ids=["prefix_op", "suffix_op", "autotune", "kv_scales", "q_fp8"],
)
def test_tree_attention_unsupported_args(kwargs, match: str) -> None:
    with pytest.raises(NotImplementedError, match=match):
        fb_tree.tree_attention(*_small_inputs(), **kwargs)


@pytest.mark.parametrize(
    "cache_dtype", [torch.uint8, torch.float8_e4m3fn], ids=["uint8", "fp8"]
)
def test_tree_attention_quantized_cache(cache_dtype: torch.dtype) -> None:
    with pytest.raises(NotImplementedError, match="quantized"):
        fb_tree.tree_attention(*_small_inputs(cache_dtype))


# --- public modules in fallback mode -----------------------------------------

fallback_only = pytest.mark.skipif(HAS_MSLK, reason="mslk is installed")


@fallback_only
def test_public_tree_attention_module() -> None:
    import xformers.ops.tree_attention as public

    assert public.tree_attention is fb_tree.tree_attention
    assert public.TreeAttnMetadata is fb_tree.TreeAttnMetadata
    assert "SplitKAutotune" not in public.__all__
    for name in public.__all__:
        assert getattr(public, name) is getattr(fb_tree, name)
    with pytest.raises(NotImplementedError, match="mslk"):
        public.SplitKAutotune
    with pytest.raises(NotImplementedError, match="mslk"):
        from xformers.ops.tree_attention import SplitKAutotune  # noqa: F401
    with pytest.raises(AttributeError):
        public.does_not_exist


@fallback_only
def test_public_attn_bias_utils_module() -> None:
    import xformers.attn_bias_utils as public

    for name in ["create_attn_bias", "pack_kv_cache", "ref_attention"]:
        assert getattr(public, name) is getattr(fb_utils, name)
    assert public.ref_attention_bmhk is fb_utils.ref_attention_bmhk


# --- attn_bias_utils ---------------------------------------------------------

AB = fmha_attn_bias
PACKED_BIAS_TYPES = [
    AB.BlockDiagonalMask,
    AB.BlockDiagonalCausalMask,
    AB.BlockDiagonalCausalFromBottomRightMask,
    AB.BlockDiagonalCausalLocalAttentionMask,
    AB.BlockDiagonalPaddedKeysMask,
    AB.BlockDiagonalCausalWithOffsetPaddedKeysMask,
    AB.BlockDiagonalCausalLocalAttentionPaddedKeysMask,
    AB.PagedBlockDiagonalPaddedKeysMask,
    AB.BlockDiagonalGappyKeysMask,
    AB.BlockDiagonalCausalWithOffsetGappyKeysMask,
]
DENSE_BIAS_TYPES = [
    type(None),
    torch.Tensor,
    AB.LowerTriangularMask,
    AB.LowerTriangularFromBottomRightMask,
    AB.LowerTriangularMaskWithTensorBias,
    AB.LocalAttentionFromBottomRightMask,
]


@pytest.mark.parametrize(
    "bias_type", DENSE_BIAS_TYPES + PACKED_BIAS_TYPES, ids=lambda t: t.__name__
)
@pytest.mark.parametrize("fmt", ["BMHK", "BMGHK"])
def test_create_attn_bias_and_ref_attention(bias_type, fmt: str) -> None:
    B, H, G, q_len, kv_len, D, page_size = 3, 2, 2, 8, 32, 16, 8
    packed = bias_type in PACKED_BIAS_TYPES
    bias = fb_utils.create_attn_bias(
        bias_type,
        batch_size=B,
        num_heads=H,
        num_heads_groups=G,
        q_len=q_len,
        kv_len=kv_len,
        device="cpu",
        dtype=torch.float32,
        requires_grad=False,
        fmt=fmt,
        op=_SomeOp,
        page_size=page_size,
    )
    if bias_type is type(None):
        assert bias is None
    elif bias_type is torch.Tensor:
        assert isinstance(bias, torch.Tensor)
    else:
        assert type(bias) is bias_type

    gh = (G, H) if fmt == "BMGHK" else (H,)
    batch, mq, mk = (1, B * q_len, B * kv_len) if packed else (B, q_len, kv_len)
    torch.manual_seed(0)
    q = torch.randn(batch, mq, *gh, D)
    k = torch.randn(batch, mk, *gh, D)
    v = torch.randn(batch, mk, *gh, D)

    ref = fb_utils.ref_attention(q, k, v, bias)
    assert ref.shape == q.shape
    # Rows without any visible key: NaN in the reference, 0 in the fallback.
    ref = torch.nan_to_num(ref, nan=0.0)
    out = sdpa.memory_efficient_attention(q, k, v, bias)
    torch.testing.assert_close(out, ref.to(out.dtype), atol=1e-4, rtol=1e-4)


def test_ref_attention_bmk() -> None:
    B, H, M, D = 2, 3, 5, 8
    bias = fb_utils.create_attn_bias(
        torch.Tensor, B, H, 1, M, M, "cpu", torch.float32, False, "BMK"
    )
    assert bias.shape == (B * H, M, M)
    q, k, v = (torch.randn(B * H, M, D) for _ in range(3))
    ref = fb_utils.ref_attention(q, k, v, bias)
    expected = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D) + bias, -1) @ v
    torch.testing.assert_close(ref, expected)


def test_pack_kv_cache_cpu() -> None:
    B, MAX_T, H, D, BLOCK_N = 3, 20, 2, 4, 8
    kv_seqlens = [5, 20, 9]
    cache_k = torch.randn(B, MAX_T, H, D)
    cache_v = torch.randn(B, MAX_T, H, D)
    block_tables, packed_k, packed_v = fb_utils.pack_kv_cache(
        cache_k, cache_v, kv_seqlens, BLOCK_N
    )
    assert block_tables.device.type == "cpu"
    for b, n in enumerate(kv_seqlens):
        for t in range(n):
            page = int(block_tables[b, t // BLOCK_N])
            pos = page * BLOCK_N + t % BLOCK_N
            torch.testing.assert_close(packed_k[0, pos], cache_k[b, t])
            torch.testing.assert_close(packed_v[0, pos], cache_v[b, t])
