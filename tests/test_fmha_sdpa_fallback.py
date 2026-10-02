# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Tests for the PyTorch-SDPA fallback used when `mslk` is not installed.
# They call the fallback module directly, so they also run where mslk exists.

import functools
import math
from typing import Callable, List, NamedTuple, Optional, Tuple

import pytest
import torch
from xformers.ops.fmha import _backend
from xformers.ops.fmha._fallback import attn_bias as fb, sdpa

# ---------------------------------------------------------------------------
# Devices / dtypes
# ---------------------------------------------------------------------------

_DEVICES = ["cpu", "mps"]
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@functools.lru_cache(maxsize=None)
def _sdpa_supported(device: str, dtype: torch.dtype) -> bool:
    """Detect (don't guess) whether SDPA runs for this device/dtype."""
    if device == "mps" and not torch.backends.mps.is_available():
        return False
    try:
        x = torch.randn([1, 1, 4, 8], device=device).to(dtype)
        torch.nn.functional.scaled_dot_product_attention(x, x, x)
        mask = torch.zeros([1, 1, 4, 4], device=device, dtype=dtype)
        torch.nn.functional.scaled_dot_product_attention(x, x, x, attn_mask=mask)
    except Exception:
        return False
    return True


def _skip_if_unsupported(device: str, dtype: torch.dtype) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    if not _sdpa_supported(device, dtype):
        pytest.skip(f"SDPA does not support {dtype} on {device}")


# Forward tolerances, in line with the cutlass/flash ops in
# tests/test_mem_eff_attention.py (ERROR_ATOL / ERROR_RTOL).
_FW_ATOL = {torch.float32: 3e-4, torch.float16: 4e-3, torch.bfloat16: 2e-2}
_FW_RTOL = {torch.float32: 2e-5, torch.float16: 4e-4, torch.bfloat16: 2e-2}
# Backward tolerances (grads accumulate more error).
_BW_ATOL = {torch.float32: 5e-4, torch.float16: 2e-2, torch.bfloat16: 1e-1}
_BW_RTOL = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}


def _assert_close(out: torch.Tensor, ref: torch.Tensor, msg: str, atol, rtol):
    assert out.shape == ref.shape, f"{msg}: shape {out.shape} vs {ref.shape}"
    out = out.detach().float().cpu()
    ref = ref.detach().float().cpu()
    diff = (out - ref).abs()
    assert torch.allclose(out, ref, atol=atol, rtol=rtol), (
        f"{msg}: max abs diff {diff.max().item():.3g} "
        f"(atol={atol}, rtol={rtol}); "
        f"failing elements: {int((diff > atol + rtol * ref.abs()).sum())}"
        f"/{diff.numel()}"
    )


# ---------------------------------------------------------------------------
# Reference implementation (float32, CPU, independent of the fallback)
# ---------------------------------------------------------------------------


def _to_bhmk(x: torch.Tensor) -> torch.Tensor:
    """xformers layout -> [B, H, M, K] float32 on CPU."""
    x = x.detach().float().cpu()
    if x.ndim == 3:  # BMK
        x = x.unsqueeze(2)
    elif x.ndim == 5:  # BMGHK
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
    return x.transpose(1, 2)


def _from_bhmk(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """[B, H, M, K] -> layout of `like` (with last dim of x)."""
    x = x.transpose(1, 2)
    if like.ndim == 3:
        return x.squeeze(2)
    if like.ndim == 5:
        return x.reshape(*like.shape[:-1], x.shape[-1])
    return x


def _ref_scores(q, k, bias: Optional[torch.Tensor], scale: Optional[float]):
    qh, kh = _to_bhmk(q), _to_bhmk(k)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    s = qh @ kh.transpose(-1, -2) * scale
    if bias is not None:
        s = s + bias
    return s


def _ref_attention(q, k, v, bias: Optional[torch.Tensor], scale=None):
    """Returns output in the layout of q (float32, cpu)."""
    s = _ref_scores(q, k, bias, scale)
    out = s.softmax(-1) @ _to_bhmk(v)
    return _from_bhmk(out, q)


def _ref_attention_autograd(q, k, v, bias, scale, grad_out):
    """Reference grads: re-run in float32 on CPU with autograd."""
    qr, kr, vr = (x.detach().float().cpu().requires_grad_(True) for x in (q, k, v))
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    def bhmk(x):
        if x.ndim == 3:
            x = x.unsqueeze(2)
        elif x.ndim == 5:
            x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
        return x.transpose(1, 2)

    s = bhmk(qr) @ bhmk(kr).transpose(-1, -2) * scale
    if bias is not None:
        s = s + bias
    out = _from_bhmk(s.softmax(-1) @ bhmk(vr), qr)
    out.backward(grad_out.detach().float().cpu())
    return out.detach(), qr.grad, kr.grad, vr.grad


def _causal_mask_ref(mq: int, mk: int, from_bottomright: bool = False):
    """Explicit triu(-inf) causal mask, independent of attn_bias."""
    shift = (mk - mq) if from_bottomright else 0
    return torch.triu(
        torch.full([mq, mk], -math.inf, dtype=torch.float32), diagonal=shift + 1
    )


# ---------------------------------------------------------------------------
# Bias cases
# ---------------------------------------------------------------------------

# A bias factory gets (B, H, Mq, Mk, device, dtype) and returns
#   (attn_bias for the op, reference additive bias [B|1, H|1, Mq, Mk] fp32 cpu)


class BiasCase(NamedTuple):
    name: str
    B: int
    Mq: int
    Mk: int
    make: Callable
    tensor_based: bool = False  # bias shape depends on BMHK heads


def _no_bias(B, H, Mq, Mk, device, dtype):
    return None, None


def _causal(B, H, Mq, Mk, device, dtype):
    bias = fb.LowerTriangularMask()
    ref = _causal_mask_ref(Mq, Mk)
    # Cross-check our ground truth against the class's own definition
    torch.testing.assert_close(bias.materialize((Mq, Mk)), ref)
    return bias, ref


def _materialized(bias):
    def ref(B, H, Mq, Mk):
        return bias.materialize((B, H, Mq, Mk), dtype=torch.float32, device="cpu")

    return ref


def _bottom_right(B, H, Mq, Mk, device, dtype):
    bias = fb.LowerTriangularFromBottomRightMask()
    ref = _causal_mask_ref(Mq, Mk, from_bottomright=True)
    torch.testing.assert_close(bias.materialize((Mq, Mk)), ref)
    return bias, ref


def _bottom_right_local(B, H, Mq, Mk, device, dtype):
    bias = fb.LowerTriangularFromBottomRightMask().make_local_attention(3)
    return bias, _materialized(bias)(B, H, Mq, Mk)


def _local_bottom_right(B, H, Mq, Mk, device, dtype):
    bias = fb.LocalAttentionFromBottomRightMask(window_left=2, window_right=1)
    return bias, _materialized(bias)(B, H, Mq, Mk)


def _tensor_bias(B, H, Mq, Mk, device, dtype):
    t = torch.randn([B, H, Mq, Mk]).to(dtype)
    return t.to(device), t.float()


def _causal_with_tensor_bias(B, H, Mq, Mk, device, dtype):
    t = torch.randn([B, H, Mq, Mk]).to(dtype)
    bias = fb.LowerTriangularMaskWithTensorBias(t.to(device))
    ref = _causal_mask_ref(Mq, Mk) + t.float()
    # cross-check against the class's own materialize (on a CPU copy)
    torch.testing.assert_close(
        fb.LowerTriangularMaskWithTensorBias(t.float()).materialize((B, H, Mq, Mk)),
        ref,
    )
    return bias, ref


def _block_diag_factory(cls_name: str, q_seqlens, kv_seqlens, transform=None):
    def make(B, H, Mq, Mk, device, dtype):
        cls = getattr(fb, cls_name)
        bias = cls.from_seqlens(q_seqlens, kv_seqlens, device=torch.device(device))
        if transform is not None:
            bias = transform(bias)
        ref = bias.materialize((1, H, Mq, Mk), dtype=torch.float32, device="cpu")
        return bias, ref

    return make


_QS = [7, 33, 24]
_KS = [9, 20, 30]
_KS_GE = [9, 33, 30]  # each >= _QS (for bottom-right causal)

BIAS_CASES: List[BiasCase] = [
    BiasCase("none", 2, 5, 7, _no_bias),
    BiasCase("causal_sq", 2, 8, 8, _causal),
    BiasCase("causal_mq<mk", 2, 5, 9, _causal),
    BiasCase("causal_mq>mk", 2, 9, 5, _causal),
    BiasCase("causal_bottomright", 2, 5, 9, _bottom_right),
    BiasCase("causal_bottomright_local", 2, 6, 10, _bottom_right_local),
    BiasCase("local_bottomright", 2, 10, 10, _local_bottom_right),
    BiasCase("tensor", 2, 6, 9, _tensor_bias, tensor_based=True),
    BiasCase("causal_tensor", 2, 7, 7, _causal_with_tensor_bias, tensor_based=True),
    BiasCase(
        "causal_tensor_mq<mk", 2, 5, 8, _causal_with_tensor_bias, tensor_based=True
    ),
    BiasCase(
        "blockdiag",
        1,
        sum(_QS),
        sum(_QS),
        _block_diag_factory("BlockDiagonalMask", _QS, None),
    ),
    BiasCase(
        "blockdiag_qkv",
        1,
        sum(_QS),
        sum(_KS),
        _block_diag_factory("BlockDiagonalMask", _QS, _KS),
    ),
    BiasCase(
        "blockdiag_causal",
        1,
        sum(_QS),
        sum(_QS),
        _block_diag_factory("BlockDiagonalCausalMask", _QS, None),
    ),
    BiasCase(
        "blockdiag_causal_qkv",
        1,
        sum(_QS),
        sum(_KS),
        _block_diag_factory("BlockDiagonalCausalMask", _QS, _KS),
    ),
    BiasCase(
        "blockdiag_causal_bottomright",
        1,
        sum(_QS),
        sum(_KS_GE),
        _block_diag_factory("BlockDiagonalCausalFromBottomRightMask", _QS, _KS_GE),
    ),
    BiasCase(
        "blockdiag_causal_local",
        1,
        sum(_QS),
        sum(_QS),
        _block_diag_factory(
            "BlockDiagonalMask", _QS, None, lambda b: b.make_local_attention(4)
        ),
    ),
]
_BIAS_BY_NAME = {c.name: c for c in BIAS_CASES}


def _make_inputs(
    layout: str,
    B: int,
    Mq: int,
    Mk: int,
    device: str,
    dtype: torch.dtype,
    K: int = 16,
    Kv: Optional[int] = None,
    H: int = 3,
    G: int = 2,
    expand_kv: bool = False,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Kv = K if Kv is None else Kv
    if layout == "BMK":
        qs, ks, vs = [B, Mq, K], [B, Mk, K], [B, Mk, Kv]
    elif layout == "BMHK":
        qs, ks, vs = [B, Mq, H, K], [B, Mk, H, K], [B, Mk, H, Kv]
    elif layout == "BMGHK":
        qs, ks, vs = [B, Mq, G, H, K], [B, Mk, G, H, K], [B, Mk, G, H, Kv]
        if expand_kv:
            ks, vs = [B, Mk, 1, H, K], [B, Mk, 1, H, Kv]
    else:
        raise ValueError(layout)
    q = torch.randn(qs).to(dtype).to(device)
    k = torch.randn(ks).to(dtype).to(device)
    v = torch.randn(vs).to(dtype).to(device)
    if requires_grad:
        q, k, v = (x.requires_grad_(True) for x in (q, k, v))
    if layout == "BMGHK" and expand_kv:
        k = k.expand(B, Mk, G, H, K)
        v = v.expand(B, Mk, G, H, Kv)
    return q, k, v


def _num_heads(q: torch.Tensor) -> int:
    if q.ndim == 3:
        return 1
    if q.ndim == 5:
        return q.shape[2] * q.shape[3]
    return q.shape[2]


def _make_bias(case: BiasCase, q, Mq, Mk, device, dtype):
    H = _num_heads(q)
    return case.make(case.B, H, Mq, Mk, device, dtype)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", BIAS_CASES, ids=[c.name for c in BIAS_CASES])
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("device", _DEVICES)
def test_forward_biases(device, dtype, case: BiasCase):
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(0)
    q, k, v = _make_inputs("BMHK", case.B, case.Mq, case.Mk, device, dtype)
    bias, ref_bias = _make_bias(case, q, case.Mq, case.Mk, device, dtype)
    out = sdpa.memory_efficient_attention(q, k, v, attn_bias=bias)
    assert out.dtype == dtype
    assert out.device.type == device
    ref = _ref_attention(q, k, v, ref_bias)
    _assert_close(out, ref, f"out[{case.name}]", _FW_ATOL[dtype], _FW_RTOL[dtype])

    out_fw = sdpa.memory_efficient_attention_forward(q, k, v, attn_bias=bias)
    _assert_close(out_fw, ref, "forward", _FW_ATOL[dtype], _FW_RTOL[dtype])


_LAYOUT_BIASES = ["none", "causal_mq<mk", "causal_bottomright", "blockdiag_causal_qkv"]


@pytest.mark.parametrize("bias_name", _LAYOUT_BIASES)
@pytest.mark.parametrize("layout", ["BMK", "BMHK", "BMGHK", "BMGHK_expanded"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("device", _DEVICES)
def test_forward_layouts(device, dtype, layout, bias_name):
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(1)
    case = _BIAS_BY_NAME[bias_name]
    expand = layout == "BMGHK_expanded"
    q, k, v = _make_inputs(
        layout.split("_")[0],
        case.B,
        case.Mq,
        case.Mk,
        device,
        dtype,
        K=16,
        Kv=24,
        expand_kv=expand,
    )
    bias, ref_bias = _make_bias(case, q, case.Mq, case.Mk, device, dtype)
    out = sdpa.memory_efficient_attention(q, k, v, attn_bias=bias)
    assert out.shape == (*q.shape[:-1], v.shape[-1])
    ref = _ref_attention(q, k, v, ref_bias)
    _assert_close(out, ref, f"out[{layout}]", _FW_ATOL[dtype], _FW_RTOL[dtype])


@pytest.mark.parametrize("bias_name", ["none", "causal_sq", "tensor", "blockdiag"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("device", _DEVICES)
def test_forward_custom_scale_and_kv_dim(device, dtype, bias_name):
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(2)
    case = _BIAS_BY_NAME[bias_name]
    q, k, v = _make_inputs("BMHK", case.B, case.Mq, case.Mk, device, dtype, K=8, Kv=32)
    bias, ref_bias = _make_bias(case, q, case.Mq, case.Mk, device, dtype)
    scale = 0.37
    out = sdpa.memory_efficient_attention(q, k, v, attn_bias=bias, scale=scale)
    assert out.shape[-1] == 32
    ref = _ref_attention(q, k, v, ref_bias, scale=scale)
    _assert_close(out, ref, "out[scale]", _FW_ATOL[dtype], _FW_RTOL[dtype])
    # The custom scale must actually be used.
    ref_default = _ref_attention(q, k, v, ref_bias)
    assert not torch.allclose(out.float().cpu(), ref_default, atol=1e-2)


@pytest.mark.parametrize("device", _DEVICES)
def test_output_dtype(device):
    _skip_if_unsupported(device, torch.float16)
    torch.manual_seed(3)
    q, k, v = _make_inputs("BMHK", 2, 5, 7, device, torch.float16)
    out = sdpa.memory_efficient_attention(q, k, v, output_dtype=torch.float32)
    assert out.dtype == torch.float32
    ref = _ref_attention(q, k, v, None)
    _assert_close(out, ref, "out", _FW_ATOL[torch.float16], _FW_RTOL[torch.float16])


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

_BW_BIASES = [
    "none",
    "causal_mq<mk",
    "causal_bottomright",
    "tensor",
    "blockdiag_causal_qkv",
    "blockdiag_causal_bottomright",
]

_BW_CONFIGS = [
    pytest.param("cpu", torch.float32, id="cpu-float32"),
    pytest.param("mps", torch.float32, id="mps-float32"),
    pytest.param("mps", torch.float16, id="mps-float16"),
]


@pytest.mark.parametrize("bias_name", _BW_BIASES)
@pytest.mark.parametrize("device,dtype", _BW_CONFIGS)
def test_backward_autograd(device, dtype, bias_name):
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(4)
    case = _BIAS_BY_NAME[bias_name]
    q, k, v = _make_inputs(
        "BMHK", case.B, case.Mq, case.Mk, device, dtype, Kv=24, requires_grad=True
    )
    bias, ref_bias = _make_bias(case, q, case.Mq, case.Mk, device, dtype)
    out = sdpa.memory_efficient_attention(q, k, v, attn_bias=bias)
    grad_out = torch.randn(out.shape).to(dtype).to(device)
    out.backward(grad_out)

    ref_out, rdq, rdk, rdv = _ref_attention_autograd(q, k, v, ref_bias, None, grad_out)
    _assert_close(out, ref_out, "out", _FW_ATOL[dtype], _FW_RTOL[dtype])
    for name, g, rg in (("dq", q.grad, rdq), ("dk", k.grad, rdk), ("dv", v.grad, rdv)):
        assert g is not None, name
        assert g.dtype == dtype
        _assert_close(g, rg, name, _BW_ATOL[dtype], _BW_RTOL[dtype])


@pytest.mark.parametrize("layout", ["BMK", "BMGHK"])
@pytest.mark.parametrize("device,dtype", _BW_CONFIGS)
def test_backward_layouts(device, dtype, layout):
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(5)
    q, k, v = _make_inputs(layout, 2, 6, 9, device, dtype, requires_grad=True)
    bias = fb.LowerTriangularMask()
    out = sdpa.memory_efficient_attention(q, k, v, attn_bias=bias)
    grad_out = torch.randn(out.shape).to(dtype).to(device)
    out.backward(grad_out)
    _, rdq, rdk, rdv = _ref_attention_autograd(
        q, k, v, _causal_mask_ref(6, 9), None, grad_out
    )
    for name, g, rg in (("dq", q.grad, rdq), ("dk", k.grad, rdk), ("dv", v.grad, rdv)):
        _assert_close(g, rg, name, _BW_ATOL[dtype], _BW_RTOL[dtype])


@pytest.mark.parametrize("bias_name", ["none", "causal_sq", "blockdiag_causal"])
@pytest.mark.parametrize("device,dtype", _BW_CONFIGS)
def test_forward_requires_grad_and_backward(device, dtype, bias_name):
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(6)
    case = _BIAS_BY_NAME[bias_name]
    q, k, v = _make_inputs("BMHK", case.B, case.Mq, case.Mk, device, dtype)
    bias, ref_bias = _make_bias(case, q, case.Mq, case.Mk, device, dtype)
    scale = 0.29

    out, lse = sdpa.memory_efficient_attention_forward_requires_grad(
        q, k, v, attn_bias=bias, scale=scale
    )
    ref_out = _ref_attention(q, k, v, ref_bias, scale=scale)
    _assert_close(out, ref_out, "out", _FW_ATOL[dtype], _FW_RTOL[dtype])

    B, Mq, H = q.shape[0], q.shape[1], q.shape[2]
    assert lse.dtype == torch.float32
    assert tuple(lse.shape) == (B, H, Mq), lse.shape
    ref_lse = _ref_scores(q, k, ref_bias, scale).logsumexp(-1)
    lse_atol = 2e-4 if dtype == torch.float32 else 2e-2
    _assert_close(lse, ref_lse, "lse", lse_atol, 2e-4)

    grad_out = torch.randn(out.shape).to(dtype).to(device)
    dq, dk, dv = sdpa.memory_efficient_attention_backward(
        grad_out, out, lse, q, k, v, attn_bias=bias, scale=scale
    )
    _, rdq, rdk, rdv = _ref_attention_autograd(q, k, v, ref_bias, scale, grad_out)
    for name, g, rg in (("dq", dq, rdq), ("dk", dk, rdk), ("dv", dv, rdv)):
        assert g.shape == rg.shape and g.dtype == dtype, name
        _assert_close(g, rg, name, _BW_ATOL[dtype], _BW_RTOL[dtype])


# ---------------------------------------------------------------------------
# Unsupported paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        sdpa.memory_efficient_attention,
        sdpa.memory_efficient_attention_forward,
        sdpa.memory_efficient_attention_forward_requires_grad,
    ],
    ids=lambda f: f.__name__,
)
def test_op_not_implemented(fn):
    q, k, v = _make_inputs("BMHK", 1, 4, 4, "cpu", torch.float32)
    with pytest.raises(NotImplementedError):
        fn(q, k, v, op=object())


def test_backward_op_not_implemented():
    q, k, v = _make_inputs("BMHK", 1, 4, 4, "cpu", torch.float32)
    out, lse = sdpa.memory_efficient_attention_forward_requires_grad(q, k, v)
    with pytest.raises(NotImplementedError):
        sdpa.memory_efficient_attention_backward(
            torch.ones_like(out), out, lse, q, k, v, op=object()
        )


# ---------------------------------------------------------------------------
# Public wiring (only meaningful when mslk is absent)
# ---------------------------------------------------------------------------

_mslk_missing = pytest.mark.skipif(
    _backend.FMHA_BACKEND == "mslk",
    reason="mslk is in use; public API routes to mslk, not the fallback",
)


@_mslk_missing
def test_public_api_routes_to_fallback():
    import xformers.ops as xops
    from xformers.ops.fmha import attn_bias as public_attn_bias

    assert hasattr(xops, "memory_efficient_attention")
    assert xops.memory_efficient_attention_partial is (
        sdpa.memory_efficient_attention_partial
    )
    assert xops.merge_attentions is sdpa.merge_attentions
    assert public_attn_bias.BlockDiagonalCausalMask is fb.BlockDiagonalCausalMask
    assert public_attn_bias.LowerTriangularMask is fb.LowerTriangularMask
    assert xops.LowerTriangularMask is fb.LowerTriangularMask

    torch.manual_seed(7)
    case = _BIAS_BY_NAME["blockdiag_causal_qkv"]
    q, k, v = _make_inputs("BMHK", case.B, case.Mq, case.Mk, "cpu", torch.float32)
    bias, ref_bias = _make_bias(case, q, case.Mq, case.Mk, "cpu", torch.float32)
    out = xops.memory_efficient_attention(q, k, v, attn_bias=bias)
    expected = sdpa.memory_efficient_attention(q, k, v, attn_bias=bias)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    _assert_close(
        out,
        _ref_attention(q, k, v, ref_bias),
        "out",
        _FW_ATOL[torch.float32],
        _FW_RTOL[torch.float32],
    )

    with pytest.raises(NotImplementedError):
        xops.memory_efficient_attention(q, k, v, op=object())


@_mslk_missing
def test_public_fmha_namespace():
    from xformers.ops import fmha

    for name in (
        "memory_efficient_attention",
        "memory_efficient_attention_forward",
        "memory_efficient_attention_forward_requires_grad",
        "memory_efficient_attention_backward",
        "memory_efficient_attention_partial",
        "merge_attentions",
        "AttentionBias",
        "LowerTriangularMask",
        "BlockDiagonalMask",
    ):
        assert hasattr(fmha, name), name
    assert fmha.AttentionBias is fb.AttentionBias


# ---------------------------------------------------------------------------
# Block-diagonal fast path (per-block SDPA instead of a materialized mask)
# ---------------------------------------------------------------------------

_BD_CLASSES = [
    "BlockDiagonalMask",
    "BlockDiagonalCausalMask",
    "BlockDiagonalCausalFromBottomRightMask",
]


class BDConfig(NamedTuple):
    name: str
    q_seqlens: List[int]
    kv_seqlens: Optional[List[int]]
    bottom_right_ok: bool = True  # every kv seqlen >= q seqlen


_BD_CONFIGS = [
    BDConfig("uneven", [7, 33, 24], None),
    BDConfig("q!=kv", [7, 33, 24], [9, 20, 30], bottom_right_ok=False),
    BDConfig("q<kv", [7, 33, 24], [9, 33, 30]),
    BDConfig("many_equal", [16] * 12, [24] * 12),
    # Same shapes in non-consecutive blocks: exercises the gather/stack path
    BDConfig("interleaved", [5, 9, 5, 9, 5, 9], [7, 9, 7, 9, 7, 12]),
    # A query block with no keys: its rows are fully masked
    BDConfig("empty_kv", [4, 6, 3], [5, 0, 4], bottom_right_ok=False),
    BDConfig("empty_q", [4, 0, 3], [5, 6, 4]),
]
_BD_CASES = [
    pytest.param(cls_name, cfg, id=f"{cls_name}-{cfg.name}")
    for cls_name in _BD_CLASSES
    for cfg in _BD_CONFIGS
    if cfg.bottom_right_ok or "BottomRight" not in cls_name
]


def _bd_bias(cls_name: str, cfg: BDConfig, device: str):
    cls = getattr(fb, cls_name)
    return cls.from_seqlens(cfg.q_seqlens, cfg.kv_seqlens, device=torch.device(device))


def _ref_attention_safe(q, k, v, bias, grad_out):
    """float32 reference with autograd where fully masked rows output 0."""
    qr, kr, vr = (x.detach().float().cpu().requires_grad_(True) for x in (q, k, v))

    def bhmk(x):
        if x.ndim == 3:
            x = x.unsqueeze(2)
        elif x.ndim == 5:
            x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
        return x.transpose(1, 2)

    s = bhmk(qr) @ bhmk(kr).transpose(-1, -2) * (q.shape[-1] ** -0.5) + bias
    lse = s.detach().logsumexp(-1)
    masked = torch.isneginf(s).all(-1, keepdim=True)
    attn = s.masked_fill(masked, 0).softmax(-1).masked_fill(masked, 0)
    out = _from_bhmk(attn @ bhmk(vr), qr)
    out.backward(grad_out.detach().float().cpu())
    return out.detach(), lse, (qr.grad, kr.grad, vr.grad)


def _run_fallback(q, k, v, bias, grad_out, dense: bool, monkeypatch, p=0.0):
    """Returns (out, lse, (dq, dk, dv)) from the fast or the dense path."""
    monkeypatch.setattr(sdpa, "_USE_BLOCK_DIAGONAL_PATH", not dense)
    q, k, v = (x.detach().requires_grad_(True) for x in (q, k, v))
    out = sdpa.memory_efficient_attention(q, k, v, attn_bias=bias, p=p)
    out.backward(grad_out)
    _, lse = sdpa.memory_efficient_attention_forward_requires_grad(
        q, k, v, attn_bias=bias
    )
    return out.detach(), lse, (q.grad, k.grad, v.grad)


def _check_bd_against(fast, other, dtype, msg, fully_masked=None):
    out, lse, grads = fast
    o_out, o_lse, o_grads = other
    _assert_close(out, o_out, f"out vs {msg}", _FW_ATOL[dtype], _FW_RTOL[dtype])
    assert torch.equal(torch.isneginf(lse.cpu()), torch.isneginf(o_lse.cpu()))
    finite = torch.isfinite(o_lse.cpu())
    lse_atol = 2e-4 if dtype == torch.float32 else 2e-2
    _assert_close(
        lse.cpu()[finite], o_lse.cpu()[finite], f"lse vs {msg}", lse_atol, 2e-4
    )
    for name, g, og in zip(("dq", "dk", "dv"), grads, o_grads):
        assert g is not None and g.dtype == dtype, name
        assert torch.isfinite(g).all(), f"{name} has NaN/inf"
        _assert_close(g, og, f"{name} vs {msg}", _BW_ATOL[dtype], _BW_RTOL[dtype])


@pytest.mark.parametrize("cls_name,cfg", _BD_CASES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("device", _DEVICES)
def test_block_diagonal_fast_path(device, dtype, cls_name, cfg: BDConfig, monkeypatch):
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(8)
    bias = _bd_bias(cls_name, cfg, device)
    Mq = sum(cfg.q_seqlens)
    Mk = sum(cfg.kv_seqlens or cfg.q_seqlens)
    q, k, v = _make_inputs("BMHK", 1, Mq, Mk, device, dtype, Kv=24)
    grad_out = torch.randn([1, Mq, q.shape[2], 24]).to(dtype).to(device)
    ref_bias = bias.materialize((1, q.shape[2], Mq, Mk), dtype=torch.float32)

    fast = _run_fallback(q, k, v, bias, grad_out, False, monkeypatch)
    dense = _run_fallback(q, k, v, bias, grad_out, True, monkeypatch)
    ref = _ref_attention_safe(q, k, v, ref_bias, grad_out)
    _check_bd_against(fast, ref, dtype, "reference")
    _check_bd_against(fast, dense, dtype, "dense path")

    # Fully masked rows: output exactly 0, lse -inf
    masked_rows = torch.isneginf(ref_bias[0, 0]).all(-1)
    if masked_rows.any():
        out, lse, _ = fast
        assert (out[0, masked_rows.to(out.device)] == 0).all()
        assert torch.isneginf(lse[0, :, masked_rows.to(lse.device)]).all()


@pytest.mark.parametrize("cls_name,cfg", _BD_CASES)
@pytest.mark.parametrize("device", _DEVICES)
def test_block_diagonal_fast_path_dropout_branch(device, cls_name, cfg, monkeypatch):
    """p > 0 runs the explicit-softmax branch; with dropout replaced by the
    identity it must match the reference exactly like p == 0."""
    dtype = torch.float32
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(9)
    monkeypatch.setattr(
        torch.nn.functional, "dropout", lambda x, p=0.5, training=True: x
    )
    bias = _bd_bias(cls_name, cfg, device)
    Mq = sum(cfg.q_seqlens)
    Mk = sum(cfg.kv_seqlens or cfg.q_seqlens)
    q, k, v = _make_inputs("BMHK", 1, Mq, Mk, device, dtype)
    grad_out = torch.randn([1, Mq, q.shape[2], q.shape[3]]).to(device)
    ref_bias = bias.materialize((1, q.shape[2], Mq, Mk), dtype=torch.float32)

    fast = _run_fallback(q, k, v, bias, grad_out, False, monkeypatch, p=0.3)
    ref = _ref_attention_safe(q, k, v, ref_bias, grad_out)
    _check_bd_against(fast, ref, dtype, "reference")


@pytest.mark.parametrize("device", _DEVICES)
def test_block_diagonal_fast_path_real_dropout(device):
    _skip_if_unsupported(device, torch.float32)
    torch.manual_seed(10)
    bias = fb.BlockDiagonalCausalMask.from_seqlens([4, 6, 3], [5, 0, 4])
    q, k, v = _make_inputs("BMHK", 1, 13, 9, device, torch.float32, requires_grad=True)
    out = sdpa.memory_efficient_attention(q, k, v, attn_bias=bias, p=0.5)
    out.sum().backward()
    assert (out[0, 4:10] == 0).all()  # the block without keys
    assert not torch.allclose(out, sdpa.memory_efficient_attention(q, k, v, bias))
    for g in (q.grad, k.grad, v.grad):
        assert torch.isfinite(g).all()


@pytest.mark.parametrize("cls_name", _BD_CLASSES)
@pytest.mark.parametrize("layout", ["BMK", "BMGHK", "BMGHK_expanded"])
@pytest.mark.parametrize("device", _DEVICES)
def test_block_diagonal_fast_path_layouts(device, layout, cls_name, monkeypatch):
    dtype = torch.float32
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(11)
    cfg = _BD_CONFIGS[2]  # q<kv: valid for all three classes
    bias = _bd_bias(cls_name, cfg, device)
    Mq, Mk = sum(cfg.q_seqlens), sum(cfg.kv_seqlens)
    q, k, v = _make_inputs(
        layout.split("_")[0],
        1,
        Mq,
        Mk,
        device,
        dtype,
        Kv=24,
        expand_kv=layout.endswith("expanded"),
    )
    grad_out = torch.randn([*q.shape[:-1], 24]).to(device)
    H = _num_heads(q)
    ref_bias = bias.materialize((1, H, Mq, Mk), dtype=torch.float32)

    fast = _run_fallback(q, k, v, bias, grad_out, False, monkeypatch)
    assert fast[1].shape == q.shape[:1] + q.shape[2:-1] + q.shape[1:2]
    out_ref, lse_ref, grads_ref = _ref_attention_safe(q, k, v, ref_bias, grad_out)
    lse_ref = lse_ref.reshape(fast[1].shape)
    _check_bd_against(fast, (out_ref, lse_ref, grads_ref), dtype, "reference")


def test_block_diagonal_fast_path_is_used(monkeypatch):
    calls = []
    real = sdpa._attention_block_diagonal

    def spy(*args, **kwargs):
        calls.append(type(args[3]).__name__)
        return real(*args, **kwargs)

    monkeypatch.setattr(sdpa, "_attention_block_diagonal", spy)
    q, k, v = _make_inputs("BMHK", 1, 10, 10, "cpu", torch.float32)
    for cls_name in _BD_CLASSES:
        bias = getattr(fb, cls_name).from_seqlens([4, 6])
        sdpa.memory_efficient_attention(q, k, v, attn_bias=bias)
    # Subclasses with other semantics keep the dense path
    local = fb.BlockDiagonalMask.from_seqlens([4, 6]).make_local_attention(2)
    sdpa.memory_efficient_attention(q, k, v, attn_bias=local)
    assert calls == _BD_CLASSES


def test_block_diagonal_fast_path_errors():
    bias = fb.BlockDiagonalMask.from_seqlens([4, 6])
    q, k, v = _make_inputs("BMHK", 2, 10, 10, "cpu", torch.float32)
    with pytest.raises(ValueError, match="batch size 1"):
        sdpa.memory_efficient_attention(q, k, v, attn_bias=bias)
    q, k, v = _make_inputs("BMHK", 1, 11, 10, "cpu", torch.float32)
    with pytest.raises(ValueError, match="covers"):
        sdpa.memory_efficient_attention(q, k, v, attn_bias=bias)


# ---------------------------------------------------------------------------
# memory_efficient_attention_partial / merge_attentions
# ---------------------------------------------------------------------------

_LSE_ATOL = {torch.float32: 2e-4, torch.float16: 2e-2, torch.bfloat16: 2e-2}


def _lse_shape(q: torch.Tensor) -> Tuple[int, ...]:
    return tuple(q.shape[:1] + q.shape[2:-1] + q.shape[1:2])


def _check_lse(lse, ref_lse, dtype, msg="lse"):
    lse, ref_lse = lse.float().cpu(), ref_lse.reshape(lse.shape).float().cpu()
    assert torch.equal(torch.isneginf(lse), torch.isneginf(ref_lse)), msg
    finite = torch.isfinite(ref_lse)
    _assert_close(lse[finite], ref_lse[finite], msg, _LSE_ATOL[dtype], 2e-4)


@pytest.mark.parametrize(
    "bias_name", ["none", "causal_mq<mk", "tensor", "blockdiag_causal_qkv"]
)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("device", _DEVICES)
def test_partial_matches_reference(device, dtype, bias_name):
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(20)
    case = _BIAS_BY_NAME[bias_name]
    q, k, v = _make_inputs("BMHK", case.B, case.Mq, case.Mk, device, dtype, Kv=24)
    bias, ref_bias = _make_bias(case, q, case.Mq, case.Mk, device, dtype)
    out, lse = sdpa.memory_efficient_attention_partial(q, k, v, attn_bias=bias)
    # Like mslk: float32 output by default, float32 LSE in the packed layout
    assert out.dtype == torch.float32 and out.shape == (*q.shape[:-1], 24)
    assert lse.dtype == torch.float32 and tuple(lse.shape) == _lse_shape(q)
    assert not out.requires_grad
    _assert_close(
        out, _ref_attention(q, k, v, ref_bias), "out", _FW_ATOL[dtype], _FW_RTOL[dtype]
    )
    _check_lse(lse, _ref_scores(q, k, ref_bias, None).logsumexp(-1), dtype)


@pytest.mark.parametrize("device", _DEVICES)
def test_partial_output_dtype(device):
    _skip_if_unsupported(device, torch.float16)
    torch.manual_seed(21)
    q, k, v = _make_inputs("BMHK", 2, 5, 7, device, torch.float16)
    out, lse = sdpa.memory_efficient_attention_partial(
        q, k, v, output_dtype=torch.float16
    )
    assert out.dtype == torch.float16 and lse.dtype == torch.float32
    _assert_close(
        out, _ref_attention(q, k, v, None), "out", _FW_ATOL[torch.float16], 4e-4
    )
    if device == "cpu":
        q, k, v = (x.double() for x in (q, k, v))
        out, _ = sdpa.memory_efficient_attention_partial(q, k, v)
        assert out.dtype == torch.float64


def _split_bounds(Mk: int, n: int) -> List[Tuple[int, int]]:
    edges = [round(i * Mk / n) for i in range(n + 1)]
    return list(zip(edges[:-1], edges[1:]))


@pytest.mark.parametrize("stacked", [False, True], ids=["list", "stacked"])
@pytest.mark.parametrize("num_chunks", [2, 3])
@pytest.mark.parametrize("layout", ["BMHK", "BMGHK"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("device", _DEVICES)
def test_split_kv_merge(device, dtype, layout, num_chunks, stacked):
    """Causal attention over 12 keys, split into chunks with a bias each: the
    later chunks have fully masked rows (the early queries see none of their
    keys). Merging must give the full attention and its LSE."""
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(22)
    B, M = 2, 12
    q, k, v = _make_inputs(layout, B, M, M, device, dtype, Kv=24)
    ref_bias = _causal_mask_ref(M, M)
    outs, lses = [], []
    for i, (a, b) in enumerate(_split_bounds(M, num_chunks)):
        if i == 0:
            # top-left causal on keys [0, b) is the full causal mask restricted
            bias: object = fb.LowerTriangularMask()
        else:
            bias = ref_bias[:, a:b].to(dtype).to(device)
        out, lse = sdpa.memory_efficient_attention_partial(
            q, k[:, a:b], v[:, a:b], attn_bias=bias
        )
        assert torch.isneginf(lse[..., :a]).all()  # fully masked rows
        assert (out[:, :a] == 0).all()
        outs.append(out)
        lses.append(lse)
    if stacked:
        merged, lse = sdpa.merge_attentions(torch.stack(outs), torch.stack(lses))
    else:
        merged, lse = sdpa.merge_attentions(outs, lses)
    assert merged.dtype == torch.float32 and merged.shape == outs[0].shape
    assert lse is not None and lse.dtype == torch.float32 and lse.shape == lses[0].shape
    assert torch.isfinite(merged).all()
    _assert_close(
        merged,
        _ref_attention(q, k, v, ref_bias),
        "merged",
        _FW_ATOL[dtype],
        _FW_RTOL[dtype],
    )
    _check_lse(lse, _ref_scores(q, k, ref_bias, None).logsumexp(-1), dtype)

    merged2, no_lse = sdpa.merge_attentions(
        outs, lses, write_lse=False, output_dtype=dtype
    )
    assert no_lse is None and merged2.dtype == dtype
    _assert_close(merged2, merged, "write_lse=False", _FW_ATOL[dtype], 0)


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("device", _DEVICES)
def test_split_kv_merge_block_diagonal(device, dtype):
    """Varlen: each sequence's keys split in two; chunk 1 has an empty part for
    one sequence (fully masked rows). Uses the packed [1, H, M] LSE layout."""
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(23)
    q_lens, kv1, kv2 = [5, 7, 3], [4, 6, 2], [3, 0, 5]
    kv_full = [a + b for a, b in zip(kv1, kv2)]
    q, k, v = _make_inputs("BMHK", 1, sum(q_lens), sum(kv_full), device, dtype)
    starts = [0]
    for n in kv_full:
        starts.append(starts[-1] + n)

    def pick(x, part):
        pieces = []
        for s, a, b in zip(starts, kv1, kv2):
            pieces.append(x[:, s : s + a] if part == 0 else x[:, s + a : s + a + b])
        return torch.cat(pieces, dim=1)

    outs, lses = [], []
    for part, lens in enumerate((kv1, kv2)):
        bias = fb.BlockDiagonalMask.from_seqlens(
            q_lens, lens, device=torch.device(device)
        )
        out, lse = sdpa.memory_efficient_attention_partial(
            q, pick(k, part), pick(v, part), attn_bias=bias
        )
        assert tuple(lse.shape) == (1, q.shape[2], sum(q_lens))
        outs.append(out)
        lses.append(lse)
    assert torch.isneginf(lses[1][:, :, 5:12]).all()

    merged, lse = sdpa.merge_attentions(outs, lses)
    full_bias = fb.BlockDiagonalMask.from_seqlens(q_lens, kv_full)
    ref_bias = full_bias.materialize(
        (1, q.shape[2], sum(q_lens), sum(kv_full)), dtype=torch.float32
    )
    _assert_close(
        merged,
        _ref_attention(q, k, v, ref_bias),
        "merged",
        _FW_ATOL[dtype],
        _FW_RTOL[dtype],
    )
    _check_lse(lse, _ref_scores(q, k, ref_bias, None).logsumexp(-1), dtype)


@pytest.mark.parametrize("layout", ["BMHK", "BMGHK"])
@pytest.mark.parametrize("device", _DEVICES)
def test_merge_fully_masked(device, layout):
    """A chunk whose keys are all masked, and rows masked in every chunk."""
    _skip_if_unsupported(device, torch.float32)
    torch.manual_seed(24)
    q, k, v = _make_inputs(layout, 1, 6, 8, device, torch.float32)
    lse_shape = _lse_shape(q)
    ninf = torch.full(lse_shape, -math.inf, device=device)
    out_a, lse_a = sdpa.memory_efficient_attention_partial(q, k, v)
    # chunk b: everything masked
    out_b = torch.zeros_like(out_a)
    merged, lse = sdpa.merge_attentions([out_a, out_b], [lse_a, ninf])
    torch.testing.assert_close(merged, out_a)
    torch.testing.assert_close(lse, lse_a)
    # every chunk masked: 0 output, -inf LSE, no NaN
    merged, lse = sdpa.merge_attentions([out_b, out_b], [ninf, ninf.clone()])
    assert (merged == 0).all() and torch.isneginf(lse).all()


def test_merge_errors_and_autograd():
    q, k, v = _make_inputs("BMHK", 1, 4, 6, "cpu", torch.float32)
    out, lse = sdpa.memory_efficient_attention_partial(q, k, v)
    with pytest.raises(ValueError, match="number of LSE"):
        sdpa.merge_attentions([out, out], [lse])
    with pytest.raises(ValueError):
        sdpa.merge_attentions([out], [lse[0]])
    with pytest.raises(ValueError):
        sdpa.merge_attentions([out, out[:, :2]], [lse, lse[..., :2]])

    # Like mslk: inputs may require grad (with write_lse), backward raises
    out = out.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="write_lse"):
        sdpa.merge_attentions([out, out], [lse, lse], write_lse=False)
    merged, merged_lse = sdpa.merge_attentions([out, out], [lse, lse])
    assert merged.requires_grad
    torch.testing.assert_close(merged.detach(), out.detach())
    torch.testing.assert_close(merged_lse, lse + math.log(2))
    with pytest.raises(NotImplementedError, match="merge_attentions"):
        merged.sum().backward()


def test_partial_errors():
    q, k, v = _make_inputs("BMHK", 1, 4, 4, "cpu", torch.float32)
    with pytest.raises(NotImplementedError, match="dropout"):
        sdpa.memory_efficient_attention_partial(q, k, v, p=0.1)
    with pytest.raises(NotImplementedError):
        sdpa.memory_efficient_attention_partial(q, k, v, op=object())
    q, k, v = _make_inputs("BMGHK", 1, 4, 4, "cpu", torch.float32, requires_grad=True)
    with pytest.raises(ValueError, match="5D"):
        sdpa.memory_efficient_attention_partial(q, k, v, _allow_backward=True)
    # without _allow_backward: no graph, even for inputs requiring grad
    out, lse = sdpa.memory_efficient_attention_partial(q, k, v)
    assert not out.requires_grad and not lse.requires_grad


@pytest.mark.parametrize("bias_name", ["none", "causal_mq<mk", "blockdiag_causal_qkv"])
@pytest.mark.parametrize("layout", ["BMK", "BMHK"])
@pytest.mark.parametrize("device,dtype", _BW_CONFIGS)
def test_partial_allow_backward(device, dtype, layout, bias_name):
    """With _allow_backward, `out` is differentiable (the LSE is not): over all
    the keys, its grads are those of full attention."""
    _skip_if_unsupported(device, dtype)
    torch.manual_seed(25)
    case = _BIAS_BY_NAME[bias_name]
    q, k, v = _make_inputs(
        layout, case.B, case.Mq, case.Mk, device, dtype, requires_grad=True
    )
    bias, ref_bias = _make_bias(case, q, case.Mq, case.Mk, device, dtype)
    out, lse = sdpa.memory_efficient_attention_partial(
        q, k, v, attn_bias=bias, _allow_backward=True
    )
    assert out.requires_grad and not lse.requires_grad
    assert out.dtype == torch.float32
    grad_out = torch.randn(out.shape).to(device)
    out.backward(grad_out)
    ref_out, rdq, rdk, rdv = _ref_attention_autograd(q, k, v, ref_bias, None, grad_out)
    _assert_close(out, ref_out, "out", _FW_ATOL[dtype], _FW_RTOL[dtype])
    for name, g, rg in (("dq", q.grad, rdq), ("dk", k.grad, rdk), ("dv", v.grad, rdv)):
        assert g is not None and g.dtype == dtype, name
        _assert_close(g, rg, name, _BW_ATOL[dtype], _BW_RTOL[dtype])


@pytest.mark.parametrize("n_pages", [4, 6], ids=["exact_fit", "spare_pages"])
@pytest.mark.parametrize("device", _DEVICES)
def test_paged_bias_matches_page_gather(device: str, n_pages: int) -> None:
    """Paged biases take the logical key length in materialize() and the
    physical K/V cache may hold pages that block_tables never uses."""
    _skip_if_unsupported(device, torch.float32)
    torch.manual_seed(0)
    page, B, Mq, H, D = 16, 2, 3, 4, 32
    kv_lens = [20, 9]
    block_tables = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32, device=device)
    q = torch.randn(1, B * Mq, H, D, device=device)
    k = torch.randn(1, n_pages * page, H, D, device=device)
    v = torch.randn_like(k)
    bias = fb.BlockDiagonalPaddedKeysMask.from_seqlens(
        [Mq] * B, kv_padding=2 * page, kv_seqlen=kv_lens
    ).make_paged(block_tables, page, paged_type=fb.PagedBlockDiagonalPaddedKeysMask)
    out = sdpa.memory_efficient_attention(q, k, v, attn_bias=bias)

    refs = []
    for b in range(B):
        pages = [int(p) for p in block_tables[b]]
        kk = torch.cat([k[0, p * page : (p + 1) * page] for p in pages])
        vv = torch.cat([v[0, p * page : (p + 1) * page] for p in pages])
        kk, vv = kk[: kv_lens[b]], vv[: kv_lens[b]]
        scores = torch.einsum("mhd,nhd->hmn", q[0, b * Mq : (b + 1) * Mq], kk)
        refs.append(torch.einsum("hmn,nhd->mhd", (scores / D**0.5).softmax(-1), vv))
    _assert_close(out[0], torch.cat(refs), "out", 1e-5, 1e-5)


@pytest.mark.parametrize(
    "q_dtype, bias_dtype",
    [(torch.float16, torch.float32), (torch.float32, torch.float16)],
)
def test_tensor_bias_dtype_mismatch_raises(
    q_dtype: torch.dtype, bias_dtype: torch.dtype
) -> None:
    """Like mslk's kernels (see TestAttnBias.test_f16_biasf32 in
    test_mem_eff_attention.py), reject instead of silently casting, so code
    that runs on the fallback also runs with mslk."""
    q = torch.randn(1, 8, 2, 16, dtype=q_dtype)
    bias = torch.randn(1, 2, 8, 8, dtype=bias_dtype)
    for bias_arg in (bias, fb.LowerTriangularMaskWithTensorBias(bias)):
        with pytest.raises(ValueError, match="same dtype"):
            sdpa.memory_efficient_attention(q, q, q, attn_bias=bias_arg)


def test_tensor_bias_device_mismatch_raises() -> None:
    _skip_if_unsupported("mps", torch.float32)
    q = torch.randn(1, 8, 2, 16, device="mps")
    bias = torch.randn(1, 2, 8, 8)
    with pytest.raises(ValueError, match="same device"):
        sdpa.memory_efficient_attention(q, q, q, attn_bias=bias)
