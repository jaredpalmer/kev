"""The transformers version guard (issue #280): an ImportError naming pyproject's range, and other import errors left alone.

Runs without a usable transformers: kev.deps reads the distribution metadata and never imports transformers. The two
subprocess cases build a fake transformers (and its metadata) on PYTHONPATH.
Run: uv run python -m pytest tests/test_deps.py -q
"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from kev import deps

ROOT = Path(__file__).resolve().parents[1]
MESSAGE = (
    "Kev requires transformers>=5.17,<6; found {found}. "
    "Install with `pip install 'transformers>=5.17,<6'`."
)


@pytest.mark.parametrize("found", ["5.17", "5.17.0", "5.17.0+cpu", "5.18.2", "5.99.0"])
def test_declared_range_is_accepted(found):
    assert deps.transformers_problem(found) is None


@pytest.mark.parametrize("found", ["4.57.6", "5.2.0", "5.16.9", "6.0.0", "6.1.0"])
def test_outside_the_range_names_the_requirement(found):
    assert deps.transformers_problem(found) == MESSAGE.format(found=found)


def test_require_transformers_raises_import_error():
    with pytest.raises(ImportError) as err:
        deps.require_transformers("5.2.0")
    assert str(err.value) == MESSAGE.format(found="5.2.0")
    assert deps.require_transformers("5.17.0") is None


def test_reads_installed_metadata_and_skips_when_transformers_is_absent(monkeypatch):
    monkeypatch.setattr(deps, "version", lambda name: "5.2.0")
    assert "found 5.2.0" in deps.transformers_problem()

    def missing(name):
        raise deps.PackageNotFoundError(name)

    monkeypatch.setattr(deps, "version", missing)
    assert (
        deps.transformers_problem() is None
    )  # the import that needs transformers raises its own ModuleNotFoundError


def test_pyproject_declares_the_range_the_guard_enforces():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert deps.TRANSFORMERS_REQUIREMENT in pyproject["project"]["dependencies"]


def fake_transformers(tmp_path, version, symbol=True):
    """A PYTHONPATH tree with transformers' metadata at `version` and a stub package. Without `symbol`, cache_utils has no
    LinearAttentionCacheLayerMixin, which is what kev.model sees in 5.2.0."""
    (tmp_path / "transformers").mkdir()
    stub = "class AutoModel: pass\nclass AutoModelForCausalLM: pass\nclass AutoTokenizer: pass\nclass DynamicCache: pass\n"
    mixin = "class LinearAttentionCacheLayerMixin: pass\n" if symbol else ""
    files = {
        "__init__.py": stub,
        "cache_utils.py": "class DynamicCache: pass\n" + mixin,
    }
    for name, text in files.items():
        (tmp_path / "transformers" / name).write_text(text, encoding="utf-8")
    dist = tmp_path / f"transformers-{version}.dist-info"
    dist.mkdir()
    meta = f"Metadata-Version: 2.1\nName: transformers\nVersion: {version}\n"
    (dist / "METADATA").write_text(meta, encoding="utf-8")
    return tmp_path


def run_import(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), str(ROOT), os.environ.get("PYTHONPATH", "")]
        ),
    }
    return subprocess.run(
        [sys.executable, "-c", "import kev.model"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_import_names_the_requirement_before_the_symbol_import(tmp_path):
    proc = run_import(fake_transformers(tmp_path, "5.2.0", symbol=False))
    assert proc.returncode != 0
    assert MESSAGE.format(found="5.2.0") in proc.stderr


def test_import_does_not_mask_other_errors_when_the_version_is_allowed(tmp_path):
    proc = run_import(fake_transformers(tmp_path, "5.17.0", symbol=False))
    assert proc.returncode != 0
    assert "cannot import name 'LinearAttentionCacheLayerMixin'" in proc.stderr
    assert "Kev requires transformers" not in proc.stderr
