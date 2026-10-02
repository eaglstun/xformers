# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Tests for the plain-PyTorch path of xformers.ops.scaled_index_add and
# xformers.ops.index_select_cat, used when the Triton kernels are not available
# (always on macOS). The Triton kernels are patched out, so these tests exercise
# the fallback even on a machine with Triton.

import random
from typing import Any, Dict, Optional

import pytest
import torch

import xformers.ops as xops
from xformers.ops import indexing


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
# grad_scaling sums over B_src * M rows: compare against an fp32 reference with a
# tolerance relative to the size of the reduction.
SCALING_TOLERANCES: Dict[torch.dtype, Dict[str, Any]] = {
    torch.float32: dict(atol=1e-3, rtol=1e-4),
    torch.float16: dict(atol=5e-2, rtol=5e-3),
    torch.bfloat16: dict(atol=5e-1, rtol=2e-2),
}


@pytest.fixture(autouse=True)
def _no_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "scaled_index_add_fwd",
        "scaled_index_add_bwd",
        "index_select_cat_fwd",
        "index_select_cat_bwd",
    ):
        monkeypatch.setattr(indexing, name, None)


def test_fallback_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the Triton kernels set to None, the PyTorch fallback runs (and the
    Triton entry points are never reached)."""
    calls = []

    def spy(fn):
        def wrapper(*args, **kwargs):
            calls.append(fn.__name__)
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        indexing,
        "_scaled_index_add_fwd_torch",
        spy(indexing._scaled_index_add_fwd_torch),
    )
    monkeypatch.setattr(
        indexing,
        "_scaled_index_add_bwd_torch",
        spy(indexing._scaled_index_add_bwd_torch),
    )
    assert indexing.scaled_index_add_fwd is None
    inp = torch.randn([4, 3, 8], requires_grad=True)
    src = torch.randn([2, 3, 8], requires_grad=True)
    out = xops.scaled_index_add(inp.clone(), torch.tensor([3, 1]), src)
    out.sum().backward()
    assert calls == ["_scaled_index_add_fwd_torch", "_scaled_index_add_bwd_torch"]

    s = torch.randn([5, 6], requires_grad=True)
    out = xops.index_select_cat([s], [torch.tensor([4, 0])])
    out.sum().backward()
    assert s.grad is not None


def _ref_scaled_index_add(
    inp: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    scaling: Optional[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    src_scaled = src.float() if scaling is None else scaling.float() * src.float()
    return torch.index_add(
        inp.float(), dim=0, source=src_scaled, index=index, alpha=alpha
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_scaling", [False, True])
@pytest.mark.parametrize("alpha", [1.0, 0.73])
@pytest.mark.parametrize(
    "shape",
    [
        (8, 257, 384, 4),  # DINOv2-like: half the batch is kept
        (6, 1, 33, 3),
        (5, 7, 64, 5),  # all rows
        (12, 50, 128, 1),
    ],
)
def test_scaled_index_add(device, dtype, with_scaling, alpha, shape) -> None:
    torch.manual_seed(0)
    B_out, M, D, B_src = shape

    inp = torch.randn([B_out, M, D], device=device, dtype=dtype, requires_grad=True)
    src = torch.randn([B_src, M, D], device=device, dtype=dtype, requires_grad=True)
    tensors = {"inp": inp, "src": src}
    perm = list(range(B_out))
    random.Random(B_out).shuffle(perm)
    index = torch.tensor(perm[:B_src], dtype=torch.int64, device=device)
    scaling = None
    if with_scaling:
        scaling = torch.randn([D], device=device, dtype=dtype, requires_grad=True)
        tensors["scaling"] = scaling

    ref_out = _ref_scaled_index_add(inp, index, src, scaling, alpha)
    grad_output = torch.randn_like(ref_out).to(dtype)
    ref_out.backward(grad_output.float())
    ref_grads = {}
    for k, v in tensors.items():
        assert v.grad is not None, k
        ref_grads[k] = v.grad.float()
    for v in tensors.values():
        v.grad = None

    x = inp.clone()
    out = xops.scaled_index_add(x, index, src, scaling, alpha)
    # In-place: the returned tensor is the (modified) input.
    assert out.data_ptr() == x.data_ptr()
    assert out.dtype == dtype and out.shape == inp.shape
    torch.testing.assert_close(out.float(), ref_out, **TOLERANCES[dtype])

    out.backward(grad_output)
    for k, v in tensors.items():
        assert v.grad is not None, k
        assert v.grad.shape == v.shape and v.grad.dtype == dtype, k
        tol = SCALING_TOLERANCES[dtype] if k == "scaling" else TOLERANCES[dtype]
        torch.testing.assert_close(v.grad.float(), ref_grads[k], **tol, msg=k)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "batches", [[(8, 257 * 4)], [(48, 25), (192, 50)], [(3, 1), (5, 7), (4, 2)]]
)
def test_index_select_cat(device, dtype, batches) -> None:
    torch.manual_seed(0)
    D = 16
    num_rows = sum(B * seqlen for B, seqlen in batches)
    src = torch.randn([num_rows, D], device=device, dtype=dtype, requires_grad=True)
    indices = []
    sources = []
    rows_begin = 0
    for B, seqlen in batches:
        index = list(range(B))
        random.Random(B).shuffle(index)
        indices.append(
            torch.tensor(index[: max(1, B // 2)], dtype=torch.int64, device=device)
        )
        sources.append(
            src[rows_begin : rows_begin + B * seqlen].reshape([B, seqlen * D])
        )
        rows_begin += B * seqlen

    ref_out = torch.cat([s[i].flatten() for s, i in zip(sources, indices)], dim=0)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    assert src.grad is not None
    ref_grad = src.grad.clone()
    src.grad = None

    out = xops.index_select_cat(sources, indices)
    assert out.dtype == dtype
    # Pure copies: exact.
    torch.testing.assert_close(out, ref_out, atol=0, rtol=0)
    out.backward(grad_out)
    assert src.grad is not None
    torch.testing.assert_close(src.grad, ref_grad, atol=0, rtol=0)
