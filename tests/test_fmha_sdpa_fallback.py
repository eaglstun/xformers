# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Tests for the PyTorch-SDPA fallback used when `mslk` is not installed.
# They call the fallback module directly, so they also run where mslk exists.

import functools
import importlib.util
import math
from typing import Callable, List, NamedTuple, Optional, Tuple

import pytest
import torch
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
    importlib.util.find_spec("mslk") is not None,
    reason="mslk is installed; public API routes to mslk, not the fallback",
)


@_mslk_missing
def test_public_api_routes_to_fallback():
    import xformers.ops as xops
    from xformers.ops.fmha import attn_bias as public_attn_bias

    assert hasattr(xops, "memory_efficient_attention")
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
        "AttentionBias",
        "LowerTriangularMask",
        "BlockDiagonalMask",
    ):
        assert hasattr(fmha, name), name
    assert fmha.AttentionBias is fb.AttentionBias
