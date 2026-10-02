<img src="./docs/assets/logo.png" width=800>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/xformers/blob/main/docs/source/xformers_mingpt.ipynb)
<br/><!--
![PyPI](https://img.shields.io/pypi/v/xformers)
![PyPI - License](https://img.shields.io/pypi/l/xformers)
[![Documentation Status](https://github.com/facebookresearch/xformers/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/facebookresearch/xformers/actions/workflows/gh-pages.yml/badge.svg)
-->
[![CircleCI](https://circleci.com/gh/facebookresearch/xformers.svg?style=shield)](https://app.circleci.com/pipelines/github/facebookresearch/xformers/)
[![Codecov](https://codecov.io/gh/facebookresearch/xformers/branch/main/graph/badge.svg?token=PKGKDR4JQM)](https://codecov.io/gh/facebookresearch/xformers)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<br/>
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<!--
[![Downloads](https://pepy.tech/badge/xformers)](https://pepy.tech/project/xformers)
-->

---

> **About this fork** ([eaglstun/xformers](https://github.com/eaglstun/xformers), branch `macos-support`):
> upstream xFormers gets its attention kernels from the `mslk` package, which only ships for Linux with CUDA or ROCm.
> Without it, `xformers.ops.memory_efficient_attention` used to be missing entirely.
> This fork adds a fallback built on PyTorch's `scaled_dot_product_attention`, so the attention API works on macOS (Apple Silicon, MPS) and on CPU.
> See [Running without mslk](#running-without-mslk-macos-cpu) below.

## xFormers - Toolbox to Accelerate Research on Transformers

xFormers is:

- **Customizable building blocks**: Independent/customizable building blocks that can be used without boilerplate code. The components are domain-agnostic and xFormers is used by researchers in vision, NLP and more.
- **Research first**: xFormers contains bleeding-edge components, that are not yet available in mainstream libraries like PyTorch.
- **Built with efficiency in mind**: Because speed of iteration matters, components are as fast and memory-efficient as possible. xFormers contains its own CUDA kernels, but dispatches to other libraries when relevant.

## Installing xFormers

- **(RECOMMENDED, linux & win) Install latest stable with pip**: Requires [PyTorch 2.10.0](https://pytorch.org/get-started/locally/)

```bash
# [linux & win] cuda 12.6 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
# [linux & win] cuda 12.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128
# [linux & win] cuda 13.0 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu130
# [linux only] (EXPERIMENTAL) rocm 7.1 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/rocm7.1
```

- **Development binaries**:

```bash
# Same requirements as for the stable version above
pip install --pre -U xformers
```

- **Install from source**: If you want to use with another version of PyTorch for instance (including nightly-releases)

```bash
# (Optional) Makes the build much faster
pip install ninja
# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
# NOTE: pytorch must already be installed!
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# (this can take dozens of minutes)
```

- **macOS (Apple Silicon) or CPU-only, from this fork**: xFormers is pure Python, so this installs in seconds. `mslk` is not installed (it isn't available for macOS), and the [SDPA fallback](#running-without-mslk-macos-cpu) is used instead.

```bash
pip install torch
pip install --no-build-isolation git+https://github.com/eaglstun/xformers.git@macos-support
```

## Benchmarks

**Memory-efficient MHA**
![Benchmarks for ViTS](./docs/plots/mha/mha_vit.png)
_Setup: A100 on f16, measured total time for a forward+backward pass_

Note that this is exact attention, not an approximation, just by calling [`xformers.ops.memory_efficient_attention`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)

**More benchmarks**

xFormers provides many components, and more benchmarks are available in [BENCHMARKS.md](BENCHMARKS.md).

### (Optional) Testing the installation

This command will provide information on an xFormers installation, and what kernels are built/available:

```python
python -m xformers.info
```

## Using xFormers

### Key Features

1. Optimized building blocks, beyond PyTorch primitives
   1. Memory-efficient exact attention - up to 10x faster
   2. sparse attention
   3. block-sparse attention
   4. fused softmax
   5. fused linear layer
   6. fused layer norm
   7. fused dropout(activation(x+bias))
   8. fused SwiGLU

### Running without mslk (macOS, CPU)

When `mslk` can't be imported (not installed, or installed but broken), `xformers.ops.fmha` uses a fallback built on [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). It runs on any device SDPA supports, including MPS and CPU. A one-line warning at import says which backend is in use, and `python -m xformers.info` reports it:

```text
fmha.backend:                                      sdpa-fallback
fmha.fallback_reason:                              the 'mslk' package is not installed
```

```python
import torch
import xformers.ops as xops
from xformers.ops import fmha

q = torch.randn(1, 128, 8, 64, device="mps", dtype=torch.float16)  # [B, M, H, K]
out = xops.memory_efficient_attention(q, q, q, attn_bias=fmha.attn_bias.LowerTriangularMask())
```

**Supported:**

- `memory_efficient_attention`, plus its `_forward`, `_forward_requires_grad` and `_backward` variants. The backward pass comes from autograd.
- `memory_efficient_attention_partial` and `merge_attentions`.
- Every `attn_bias` type, including the block-diagonal, padded, paged and local-attention variants, and plain tensors.
  - `LowerTriangularMask` maps to SDPA's `is_causal`.
  - `BlockDiagonalMask`, `BlockDiagonalCausalMask` and `BlockDiagonalCausalFromBottomRightMask` run one sequence at a time, batching sequences of the same length, so memory scales with the sequence lengths rather than the packed total.
  - The other biases are materialized as a dense `[Mq, Mk]` mask.
- The BMK, BMHK and BMGHK layouts, custom `scale`, dropout (`p`), and value head dimensions that differ from the key's.
- `xformers.ops.tree_attention` and `xformers.attn_bias_utils`.

**Not supported** (these raise `NotImplementedError` and need mslk):

- choosing a kernel with `op=`
- fp8 inputs and quantized KV caches
- Triton split-K, including `tree_attention`'s `autotune` and `SplitKAutotune`
- `memory_efficient_attention_backward` with dropout

**Speed:** the fallback is as fast as PyTorch's SDPA on your device. It isn't a fused memory-efficient kernel, and biases that get materialized cost O(Mq·Mk) memory.

Set `XFORMERS_FMHA_BACKEND=sdpa` to force the fallback even when mslk is installed, for example to test it on Linux. The fallback's tests are in `tests/test_fmha_sdpa_fallback.py`, `tests/test_fmha_backend_detection.py` and `tests/test_tree_attention_fallback.py`. CI runs them on macOS and Linux (`.github/workflows/fallback_test.yml`).

### Install troubleshooting

- NVCC and the current CUDA runtime match. Depending on your setup, you may be able to change the CUDA runtime with `module unload cuda; module load cuda/xx.x`, possibly also `nvcc`
- the version of GCC that you're using matches the current NVCC capabilities
- the `TORCH_CUDA_ARCH_LIST` env variable is set to the architectures that you want to support. A suggested setup (slow to build but comprehensive) is `export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"`
- If the build from source OOMs, it's possible to reduce the parallelism of ninja with `MAX_JOBS` (eg `MAX_JOBS=2`)
- If getting error message `Filename longer than 260 characters` on Windows, make sure long paths are enabled at OS level, and also execute the command `git config --global core.longpaths true`

### License

xFormers has a BSD-style license, as found in the [LICENSE](LICENSE) file.
It includes code from the [triton-lang/kernels](https://github.com/triton-lang/kernels) repo.
The SDPA fallback in `xformers/ops/fmha/_fallback/` includes files vendored from `mslk` 1.3.0 (BSD-3-Clause, Meta Platforms, Inc.), as noted at the top of each file.

## Citing xFormers

If you use xFormers in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{xFormers2022,
  author =       {Benjamin Lefaudeux and Francisco Massa and Diana Liskovich and Wenhan Xiong and Vittorio Caggiano and Sean Naren and Min Xu and Jieru Hu and Marta Tintore and Susan Zhang and Patrick Labatut and Daniel Haziza and Luca Wehrstedt and Jeremy Reizenstein and Grigory Sizov},
  title =        {xFormers: A modular and hackable Transformer modelling library},
  howpublished = {\url{https://github.com/facebookresearch/xformers}},
  year =         {2022}
}
```

## Credits

The following repositories are used in xFormers, either in close to original form or as an inspiration:

- [Sputnik](https://github.com/google-research/sputnik)
- [GE-SpMM](https://github.com/hgyhungry/ge-spmm)
- [Triton](https://github.com/openai/triton)
- [LucidRain Reformer](https://github.com/lucidrains/reformer-pytorch)
- [RevTorch](https://github.com/RobinBruegger/RevTorch)
- [Nystromformer](https://github.com/mlpen/Nystromformer)
- [FairScale](https://github.com/facebookresearch/fairscale/)
- [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models)
- [CUTLASS](https://github.com/nvidia/cutlass)
- [Flash-Attention](https://github.com/HazyResearch/flash-attention)
