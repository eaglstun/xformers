# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Decides, once, whether xformers.ops.fmha is backed by mslk or by the
PyTorch SDPA fallback.

``importlib.util.find_spec("mslk")`` is not enough: it is true for any
directory named ``mslk`` on ``sys.path`` (a namespace package), and a real but
broken mslk (e.g. a native library that fails to load) only fails when
imported. So we actually import ``mslk.attention.fmha`` and treat any exception
as "no mslk", remembering why.

Setting ``XFORMERS_FMHA_BACKEND=sdpa`` forces the fallback even when mslk is
installed.
"""

import os
from typing import Optional, Tuple

_FORCE_ENV_VAR = "XFORMERS_FMHA_BACKEND"


def _detect() -> Tuple[bool, Optional[str]]:
    if os.environ.get(_FORCE_ENV_VAR, "").strip().lower() in ("sdpa", "sdpa-fallback"):
        return False, f"{_FORCE_ENV_VAR}=sdpa is set"
    try:
        import mslk.attention.fmha  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name == "mslk":
            return False, "the 'mslk' package is not installed"
        return False, _broken_reason(e)
    except Exception as e:
        return False, _broken_reason(e)
    return True, None


def _broken_reason(e: Exception) -> str:
    msg = " ".join(str(e).split())  # Keep the warning on one line.
    return f"the 'mslk' package failed to import ({type(e).__name__}: {msg})"


HAS_MSLK, MSLK_UNAVAILABLE_REASON = _detect()
FMHA_BACKEND = "mslk" if HAS_MSLK else "sdpa-fallback"
