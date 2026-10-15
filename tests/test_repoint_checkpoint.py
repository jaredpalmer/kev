"""scripts/repoint_checkpoint.py: repointing head.pt's `base`, touching nothing else.

Runs the script as a subprocess (its CLI is the contract). No weights needed: the script
edits only head.pt and adapter_config.json metadata.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kev.checkpoint import Meta, read_meta, write_meta

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "repoint_checkpoint.py"


def run_script(*argv):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, env=env)


@pytest.fixture
def base_dir(tmp_path):
    b = tmp_path / "Qwen3.5-4B-Base"
    b.mkdir()
    (b / "config.json").write_text("{}", encoding="utf-8")
    return b


@pytest.fixture
def run_dir(tmp_path):
    """A LoRA checkpoint pointing at the hub, with extra fields that must survive."""
    r = tmp_path / "kev-4b"
    r.mkdir()
    # A placeholder revision: the real Qwen3.5 base revision has one home (kev.suite.ADMISSION_TOKENIZER), and
    # the test only needs a non-empty value to prove repoint clears it.
    write_meta(r, Meta(base="Qwen/Qwen3.5-4B-Base", base_revision="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
                       temperature=2.41, head_dim=256, extra={"epoch": 2}))
    cfg = json.dumps({"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3.5-4B-Base",
                      "revision": "a1b2c3d4"})
    (r / "adapter_config.json").write_text(cfg, encoding="utf-8")
    return r


def test_default_base_is_models_base(run_dir):
    p = run_script("--run", str(run_dir))
    assert p.returncode == 0, p.stderr
    assert "/models/base" in p.stdout
    meta = read_meta(run_dir)
    assert meta.base == "/models/base" and meta.base_revision is None
    assert meta.extra["base_repointed_from"] == "Qwen/Qwen3.5-4B-Base"
    assert meta.extra["base_repointed_to"] == "/models/base"
    assert "base_repointed_at" in meta.extra


def test_explicit_base(run_dir, base_dir):
    p = run_script("--run", str(run_dir), "--base", str(base_dir))
    assert p.returncode == 0, p.stderr
    meta = read_meta(run_dir)
    assert meta.base == str(base_dir) and meta.base_revision is None
    assert meta.temperature == 2.41 and meta.head_dim == 256 and meta.extra["epoch"] == 2
    cfg = json.loads((run_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["base_model_name_or_path"] == str(base_dir) and cfg["revision"] is None


def test_missing_target_path_is_accepted(run_dir):
    """The target normally exists only inside the container, so it need not exist on this host."""
    target = str(run_dir.parent / "does-not-exist" / "base")
    p = run_script("--run", str(run_dir), "--base", target)
    assert p.returncode == 0, p.stderr
    assert read_meta(run_dir).base == target


def test_idempotent(run_dir):
    assert run_script("--run", str(run_dir)).returncode == 0
    p = run_script("--run", str(run_dir))
    assert p.returncode == 0
    assert "already" in p.stdout
    meta = read_meta(run_dir)
    assert meta.extra["base_repointed_at"] == read_meta(run_dir).extra["base_repointed_at"]


def test_check_prints_without_writing(run_dir):
    p = run_script("--check", "--run", str(run_dir))
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "Qwen/Qwen3.5-4B-Base"
    assert read_meta(run_dir).base == "Qwen/Qwen3.5-4B-Base"


def test_missing_head_pt(run_dir):
    p = run_script("--run", str(run_dir / "nope"))
    assert p.returncode == 1
