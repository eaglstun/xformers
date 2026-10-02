# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Tests for how xformers.ops.fmha picks between mslk and the SDPA fallback.
# Each case runs in a fresh interpreter, since the choice is made at import.

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = """
import json, logging
records = []
class _H(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())
logging.getLogger("xformers").addHandler(_H())
import xformers.ops
from xformers.ops import fmha
from xformers.ops.fmha import _backend
print(json.dumps({
    "backend": fmha.FMHA_BACKEND,
    "reason": _backend.MSLK_UNAVAILABLE_REASON,
    "has_mea": hasattr(xformers.ops, "memory_efficient_attention"),
    "warnings": [m for m in records if "mslk" in m or "SDPA" in m],
}))
"""


def _mslk_importable() -> bool:
    out = subprocess.run(
        [sys.executable, "-c", "import mslk.attention.fmha"],
        cwd=_REPO_ROOT,
        capture_output=True,
    )
    return out.returncode == 0


_HAS_REAL_MSLK = _mslk_importable()


def _run(
    cwd: Path, extra_path: Optional[Path] = None, env: Optional[Dict[str, str]] = None
) -> dict:
    full_env = dict(os.environ)
    full_env.pop("XFORMERS_FMHA_BACKEND", None)
    paths = [str(_REPO_ROOT)]
    if extra_path is not None:
        paths.insert(0, str(extra_path))
    full_env["PYTHONPATH"] = os.pathsep.join(paths)
    full_env.update(env or {})
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(_HAS_REAL_MSLK, reason="mslk is installed")
def test_no_mslk_uses_fallback(tmp_path: Path) -> None:
    res = _run(tmp_path)
    assert res["backend"] == "sdpa-fallback"
    assert res["has_mea"]
    assert res["reason"] == "the 'mslk' package is not installed"
    assert len(res["warnings"]) == 1
    assert "'mslk' package is not installed" in res["warnings"][0]


@pytest.mark.skipif(_HAS_REAL_MSLK, reason="mslk is installed")
def test_empty_mslk_directory_is_not_mslk(tmp_path: Path) -> None:
    # A bare directory named mslk (e.g. an unpacked wheel next to a script)
    # is a namespace package: find_spec() finds it but it is not mslk.
    (tmp_path / "mslk").mkdir()
    for cwd, extra in ((tmp_path, None), (_REPO_ROOT, tmp_path)):
        res = _run(cwd, extra_path=extra)
        assert res["backend"] == "sdpa-fallback"
        assert res["has_mea"]
        assert "failed to import" in res["reason"]
        assert "ModuleNotFoundError" in res["reason"]


@pytest.mark.skipif(_HAS_REAL_MSLK, reason="mslk is installed")
def test_broken_mslk_falls_back_with_reason(tmp_path: Path) -> None:
    pkg = tmp_path / "mslk"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "raise OSError('libmslk.so: cannot open shared object file')\n"
    )
    res = _run(tmp_path, extra_path=tmp_path)
    assert res["backend"] == "sdpa-fallback"
    assert res["has_mea"]
    assert len(res["warnings"]) == 1
    warning = res["warnings"][0]
    assert "\n" not in warning
    assert "failed to import" in warning
    assert "OSError: libmslk.so: cannot open shared object file" in warning


def test_env_var_forces_fallback(tmp_path: Path) -> None:
    res = _run(tmp_path, env={"XFORMERS_FMHA_BACKEND": "sdpa"})
    assert res["backend"] == "sdpa-fallback"
    assert res["has_mea"]
    assert res["reason"] == "XFORMERS_FMHA_BACKEND=sdpa is set"
    assert len(res["warnings"]) == 1
    assert "XFORMERS_FMHA_BACKEND=sdpa" in res["warnings"][0]
