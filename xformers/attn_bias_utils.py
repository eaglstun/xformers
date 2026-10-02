# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa

from xformers.ops.fmha._backend import HAS_MSLK as _HAS_MSLK

# Without mslk, use the copy vendored from mslk 1.3.0 (it works on any device).
if _HAS_MSLK:
    from mslk.attention.fmha.attn_bias_utils import (
        create_attn_bias,
        pack_kv_cache,
        ref_attention,
        ref_attention_bmhk,
    )
else:
    from xformers.ops.fmha._fallback.attn_bias_utils import (
        create_attn_bias,
        pack_kv_cache,
        ref_attention,
        ref_attention_bmhk,
    )
