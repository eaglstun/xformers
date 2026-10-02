# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from .fmha._backend import HAS_MSLK as _HAS_MSLK

__all__ = [
    "construct_full_tree_choices",
    "construct_tree_choices",
    "get_full_tree_size",
    "tree_attention",
    "TreeAttnMetadata",
    "use_triton_splitk_for_prefix",
]

if _HAS_MSLK:
    from mslk.attention.fmha.tree_attention import (  # noqa: E402, F401
        construct_full_tree_choices,
        construct_tree_choices,
        get_full_tree_size,
        SplitKAutotune,
        tree_attention,
        TreeAttnMetadata,
        use_triton_splitk_for_prefix,
    )

    __all__.append("SplitKAutotune")
else:
    # Without mslk, tree_attention runs on the PyTorch SDPA fallback; see
    # xformers/ops/fmha/_fallback/tree_attention.py for what it supports.
    from .fmha._fallback.tree_attention import (  # noqa: F401
        construct_full_tree_choices,
        construct_tree_choices,
        get_full_tree_size,
        tree_attention,
        TreeAttnMetadata,
        use_triton_splitk_for_prefix,
    )

    def __getattr__(name: str) -> Any:
        if name == "SplitKAutotune":
            # Not an AttributeError: `from ... import SplitKAutotune` would turn
            # that into a generic "cannot import name" and hide this message.
            raise NotImplementedError(
                "xformers.ops.tree_attention.SplitKAutotune is not available: it "
                "subclasses mslk's Triton split-K kernel, and mslk is not installed "
                "(the PyTorch SDPA fallback is in use)."
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
