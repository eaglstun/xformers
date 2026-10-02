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

---

## Step 2 (2026-10-02): per-block BlockDiagonal path + backend detection

v1 landed in c1747238. Two follow-ups.

### 2a. Per-block path for block-diagonal biases

Benchmark (MPS, fp16, H=16, K=64, causal, equal-length packed sequences),
v1 dense materialize vs. splitting into per-sequence SDPA calls:

| seqs × len | M | dense | split |
|---|---|---|---|
| 8 × 256 | 2048 | 7.7 ms | 1.1 ms |
| 16 × 512 | 8192 | 49.9 ms | 2.3 ms |
| 32 × 512 | 16384 | 108.0 ms | 4.4 ms |
| 16 × 1024 | 16384 | 109.6 ms | 5.6 ms |

The dense mask is also `[1, H, M, M]`: ~8 GiB at M=16k, H=16, fp16.

Design: in `sdpa.py`, for `BlockDiagonalMask`, `BlockDiagonalCausalMask` and
`BlockDiagonalCausalFromBottomRightMask` (exact types, mslk or vendored), split
q by `q_seqinfo` and k/v by `k_seqinfo` and run SDPA per block (`is_causal`
for top-left causal; bottom-right causal with `Mq_i == Mk_i` is the same as
`is_causal`, otherwise a small per-block materialized mask). Blocks with the
same `(Mq_i, Mk_i)` may be batched into one SDPA call. Works for the forward,
LSE and dropout paths. Every other bias keeps the dense path. Tests: results
must match the dense path and the reference, including uneven q/kv seqlens and
zero-length blocks if the bias classes allow them.

### 2b. Robust mslk detection

`importlib.util.find_spec("mslk")` is true for any directory named `mslk` on
`sys.path` (namespace package) — found by running a script next to an
unpacked mslk wheel, which made `import xformers.ops` crash. Replace it with
one shared check that actually imports `mslk.attention.fmha` and treats any
exception as "no mslk" (the fallback warning includes the reason). Used by
`xformers/ops/__init__.py`, `xformers/ops/fmha/__init__.py`,
`xformers/ops/fmha/attn_bias.py` and `xformers/info.py`.

---

## Step 3 (2026-10-02): the remaining mslk-only modules

Step 2 landed in 24701c04. Two modules still hard-import mslk:
`xformers/attn_bias_utils.py` and `xformers/ops/tree_attention.py`.

### 3a. `memory_efficient_attention_partial` and `merge_attentions`

Add both to the SDPA fallback (`sdpa.py`), matching mslk 1.3.0 semantics:
`partial` returns `(out, lse)` with no dropout and (by default) no backward;
`merge_attentions` combines chunks via
`Out = Σ Out_i·exp(LSE_i) / Σ exp(LSE_i)`, accepting lists or stacked
tensors, BMHK and BMGHK, `write_lse`, `output_dtype`, and chunks where every
key is masked (LSE `-inf`). Export both from `xformers.ops.fmha` and
`xformers.ops` in fallback mode. These were "not provided" in step 1 and are
generally useful (split-KV, ring/context-parallel attention), not just for
tree attention.

### 3b. Tree attention and attn_bias_utils

- `attn_bias_utils`: vendor mslk 1.3.0's file into `_fallback/`, adapting only
  the imports and the `triton_splitk` op checks.
- `tree_attention`: vendor `TreeAttnMetadata`, the `_prepare_*` helpers,
  `construct_*tree_choices`, `get_full_tree_size` and
  `use_triton_splitk_for_prefix` unchanged. Reimplement `tree_attention()` on
  the fallback's `partial` (prefix vs. KV cache) + `forward_requires_grad`
  (suffix with the tree mask) + `merge_attentions`. Not supported: `prefix_op`
  / `suffix_op` other than None, `autotune`, fp8/uint8 KV caches,
  `SplitKAutotune` (it subclasses a Triton kernel).
- `xformers/attn_bias_utils.py` and `xformers/ops/tree_attention.py` choose
  mslk or the fallback via `xformers.ops.fmha._backend.HAS_MSLK`.
- Tests compare `tree_attention` against a dense reference (full q over
  `cat(cache, spec)` with the combined mask) on cpu and mps.

---

## Step 4 (2026-10-02): CI

- `.github/workflows/fallback_test.yml`: runs the three fallback test files on
  `macos-14` (Python 3.12, mslk absent) and `ubuntu-24.04` (Python 3.10, CPU
  torch, `XFORMERS_FMHA_BACKEND=sdpa`). Installs with
  `--no-build-isolation` so the build doesn't fetch a second torch.
- `gpu_test_gh.yml`: guarded with `github.repository ==
  'facebookresearch/xformers'`, since its GPU runners don't exist on forks.
- The `linters` workflow runs `ufmt` and repo-wide `mypy`/`flake8` outside
  pre-commit, so the verbatim-vendored `attn_bias.py`/`attn_bias_utils.py` are
  excluded in `pyproject.toml` (`[tool.ufmt]`, `[[tool.mypy.overrides]]`) and
  `.flake8`. `_fallback/tree_attention.py` is mostly rewritten, so it gets the
  standard xformers header instead.

---

## Step 5 (2026-10-02): Node 24 actions, MPS runner, troubleshooting

- Actions bumped to Node 24 versions: checkout v7, setup-python v7, cache v6,
  peaceiris/actions-gh-pages v4.
- First CI run: GitHub's `macos-14` VMs report `mps.is_available() == True`
  but every MPS allocation fails ("MPS backend out of memory", 0 bytes
  allocated). Tests now probe with a real allocation and skip; hosted CI is
  CPU-only.
- MPS coverage comes from `mps_test.yml` on a self-hosted runner
  (`~/actions-runner`, labels `self-hosted, macOS, ARM64, mps`), run on
  demand as the owner's user. Safety: `workflow_dispatch` only (never
  `pull_request`), fork-PR approval set to `all_external_contributors`, and
  the fork's default branch is `macos-support` (manual triggers require the
  workflow on the default branch). The job fails if MPS isn't usable or any
  MPS test skips.
- Correction to step 1: mslk *does* ship Windows wheels (CUDA 13.0,
  `win_amd64`), so the requirement marker is now `sys_platform != "darwin"`.
- README: the compile-era troubleshooting (NVCC, TORCH_CUDA_ARCH_LIST, ninja)
  is replaced by mslk-centric troubleshooting.
