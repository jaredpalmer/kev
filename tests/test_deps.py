"""kev.model names pyproject's transformers range when that import fails on an older release (issue #280).

A release inside the range, and an install with no distribution metadata, keep the original ImportError.
The fake transformers trees run in a subprocess so they do not replace the transformers this process imported.
Run: uv run python -m pytest tests/test_deps.py -q
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kev.model import _TRANSFORMERS_REQUIREMENT, _transformers_outside_range

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("found,outside", [
    ("4.57.6", True), ("5.2.0", True), ("5.2.0+cu124", True), ("5.16.9", True),
    ("5.17", False), ("5.17.0", False), ("5.17.0+cu124", False), ("5.17.0rc1", False), ("5.99.0", False),
    ("6", True), ("6.0.0", True),
])
def test_range_edges(found, outside):
    assert _transformers_outside_range(found) is outside


def test_pyproject_declares_the_requirement_the_error_names():
    """The error names the pyproject dependency, and the numeric check is that same range.
    `_transformers_outside_range` writes >=5.17,<6 as tuples; this fails if those tuples drift from the string."""
    declared = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"{_TRANSFORMERS_REQUIREMENT}"' in declared
    low, high = _TRANSFORMERS_REQUIREMENT.removeprefix("transformers>=").split(",<")
    assert _transformers_outside_range(low) is False
    assert _transformers_outside_range(high) is True
    major, minor = (int(part) for part in low.split("."))
    assert _transformers_outside_range(f"{major}.{minor - 1}.9") is True


def _fake(tmp_path, version, symbol, metadata):
    package = tmp_path / "transformers"
    package.mkdir()
    init = "class AutoModel: pass\nclass AutoModelForCausalLM: pass\nclass AutoTokenizer: pass\nclass DynamicCache: pass\n"
    (package / "__init__.py").write_text(init, encoding="utf-8")
    mixin = "class LinearAttentionCacheLayerMixin: pass\n" if symbol else ""
    (package / "cache_utils.py").write_text("class DynamicCache: pass\n" + mixin, encoding="utf-8")
    if not metadata:
        return
    dist = tmp_path / f"transformers-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: transformers\nVersion: {version}\n", encoding="utf-8")


def _import(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)])}
    return subprocess.run([sys.executable, "-c", "import kev.model"], cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("version,symbol,metadata,expect", [
    ("5.2.0", False, True, "requirement"),   # the bundled env in issue #280
    ("6.0.0", False, True, "requirement"),   # the other side of the pin, when the name is gone
    ("5.17.0", False, True, "raw"),          # inside the range: do not mask a real import error
    ("5.2.0", False, False, "raw"),          # no distribution metadata: the missing name is not rewritten
    ("5.17.0", True, True, "ok"),
    ("6.0.0", True, True, "ok"),             # a release that still imports is not refused here
])
def test_import(tmp_path, version, symbol, metadata, expect):
    _fake(tmp_path, version, symbol, metadata)
    proc = _import(tmp_path)
    if expect == "ok":
        assert proc.returncode == 0, proc.stderr
        return
    assert proc.returncode != 0
    assert "cannot import name 'LinearAttentionCacheLayerMixin'" in proc.stderr
    named = f"Kev requires {_TRANSFORMERS_REQUIREMENT} (found {version}). Use Python 3.12 or 3.13 and run `uv sync`."
    if expect == "requirement":
        assert named in proc.stderr
    else:
        assert "Kev requires" not in proc.stderr
