# Plan: PyTorch SDPA fallback for fMHA when `mslk` is missing

Status: approved 2026-10-02 · Scope: fork-only (`eaglstun/xformers`, branch `macos-support`)

## Why

Since the fmha code moved to `mslk` (ca6d2aa0), xformers is a pure-Python
wheel that installs anywhere, but `xformers.ops.memory_efficient_attention`
only exists when `mslk` is importable. `mslk` only ships manylinux CUDA/ROCm
wheels on `download.pytorch.org`; on PyPI (what macOS resolves) it is an empty
`0.0.0` placeholder with no Python package. Result on Apple Silicon: `import
xformers` works, the attention API is simply absent.

Experiment (mslk 1.0.0 Python layer on macOS, native lib skipped): the
dispatcher runs and rejects every op with `device=mps (supported: {'cuda'})`,
and `attn_bias` classes `materialize()` fine on MPS. The bias module is pure
Python (only `torch` + stdlib). So the gap is "an op that runs on non-CUDA
devices" plus "a way to get the bias classes without mslk".

## Decisions

- **Option A**: an xformers-native fallback, no `mslk` required. (Option B —
  getting mslk's Python layer onto macOS and registering an SDPA op in its
  dispatcher — was rejected: needs a repackaged mslk or upstream changes.)
- Fallback is active **whenever `mslk` is not importable** (CPU, MPS, anything),
  not only for MPS tensors. Gives CPU support and Linux CI coverage for free.
- `requirements.txt`: `mslk; sys_platform == "linux"` so macOS stops installing
  the empty placeholder.
- Fork-only; no upstreaming effort for now.

## Design

### 1. Vendored bias classes

- Copy `mslk/attention/fmha/attn_bias.py` from **mslk 1.3.0** verbatim to
  `xformers/ops/fmha/_fallback/attn_bias.py`, with a header noting source,
  version, and BSD-3 license (mslk is BSD-3, Meta).
- `xformers/ops/fmha/attn_bias.py`: import from `mslk` if available, else from
  the vendored copy. Keep the same exported names.

### 2. SDPA backend — `xformers/ops/fmha/_fallback/sdpa.py`

Implements attention with `torch.nn.functional.scaled_dot_product_attention`.
Backward comes from autograd (no custom BwOp).

- Layouts: xformers `BMHK [B, M, H, K]` ↔ SDPA `[B, H, M, K]` via
  `transpose(1, 2)`. `BMK` (3-D): add a head dim. `BMGHK` (5-D, GQA): fold
  `G·H` into heads (keys/values may be stride-0 expanded; that is fine).
- `scale=` → SDPA `scale`; `p` → `dropout_p`; `K != Kv` allowed.
- Bias mapping:

  | Bias                      | Handling                                                                                                            |
  | ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
  | `None`                    | plain SDPA                                                                                                          |
  | `LowerTriangularMask`     | `is_causal=True` (both top-left aligned)                                                                            |
  | `torch.Tensor`            | additive `attn_mask` (broadcast to `[B, H, Mq, Mk]`)                                                                |
  | any other `AttentionBias` | `bias.materialize((B, H, Mq, Mk) or as the class requires, dtype=query.dtype, device=query.device)` → additive mask |

- Generic materialization is O(Mq·Mk) memory. Acceptable for v1; a v2 could
  loop per block for `BlockDiagonal*`.

### 3. Public surface in fallback mode

Provided: `memory_efficient_attention`, `memory_efficient_attention_forward`,
`memory_efficient_attention_forward_requires_grad` (returns `(out, lse)`; lse
computed in the reference way), `memory_efficient_attention_backward` (via
autograd), `AttentionBias`, `LowerTriangularMask`, `BlockDiagonalMask`, and the
full `attn_bias` module.

Not provided: an `op=` naming a real kernel (cutlass/flash/ck/…) raises a clear
`NotImplementedError`; `memory_efficient_attention_partial`,
`merge_attentions`, fp8 `ScaledTensor`, triton split-K, the `_OPS_LOOKUP`
serialization helpers.

### 4. Wiring

- `xformers/ops/fmha/__init__.py` and `xformers/ops/__init__.py`: "mslk, else
  fallback" instead of "mslk, else nothing". Warning text becomes a one-time
  notice that the PyTorch SDPA fallback is in use.
- `python -m xformers.info`: add `fmha.backend: mslk | sdpa-fallback`.
- Leave `xformers/attn_bias_utils.py` and `xformers/ops/tree_attention.py`
  alone (they still hard-require mslk); note in CHANGELOG.

### 5. Tests — `tests/test_fmha_sdpa_fallback.py`

- Naive float32 reference: `softmax(Q·Kᵀ·scale + materialized_bias) · V`.
- Every bias type above; forward output and q/k/v grads.
- Devices: cpu, mps (skip if unavailable). Dtypes: fp32, fp16, bf16 with
  per-dtype tolerances.
- Layouts: BMK, BMHK, BMGHK; K ≠ Kv; custom scale; `op=` error path.
- Tests call the fallback module directly so they also run where mslk exists.

## Out of scope

Metal kernels, per-block loop optimization, upstreaming, tree attention.
