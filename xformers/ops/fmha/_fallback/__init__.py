# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-PyTorch fallback for xformers.ops.fmha, used when `mslk` is missing.

- ``attn_bias``: the bias classes, vendored verbatim from mslk 1.3.0.
- ``sdpa``: memory_efficient_attention on top of
  ``torch.nn.functional.scaled_dot_product_attention`` (any device).

Nothing here imports mslk, so it is importable everywhere (and testable even
where mslk is installed).
"""
