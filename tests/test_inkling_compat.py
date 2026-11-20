"""Offline source-pin and report-serialization tests for the research launcher."""

import importlib.util
import json
import subprocess
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("inkling_compat", Path(__file__).resolve().parents[1] / "scripts" / "inkling_compat.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def checkout(path):
    root = path / "upstream"
    (root / "src/transformers").mkdir(parents=True)
    (root / "src/transformers/__init__.py").write_text("__version__ = 'fixture'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "src"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    return root, revision


def invoke(root, tmp_path, capsys):
    status = launcher.main(["--upstream", str(root), "--report", str(tmp_path / "report.json")])
    return status, capsys.readouterr().err


def test_cli_rejects_bad_revision(tmp_path, capsys):
    root, _ = checkout(tmp_path)
    status, error = invoke(root, tmp_path, capsys)
    assert status == 2
    assert "upstream revision mismatch" in error


def test_cli_rejects_untracked_source(tmp_path, monkeypatch, capsys):
    root, revision = checkout(tmp_path)
    monkeypatch.setattr(launcher, "PINNED_REVISION", revision)
    (root / "untracked.py").write_text("untracked\n", encoding="utf-8")
    status, error = invoke(root, tmp_path, capsys)
    assert status == 2
    assert "dirty or has untracked files" in error


def test_cli_rejects_dirty_source(tmp_path, monkeypatch, capsys):
    root, revision = checkout(tmp_path)
    monkeypatch.setattr(launcher, "PINNED_REVISION", revision)
    (root / "src/transformers/__init__.py").write_text("modified\n", encoding="utf-8")
    status, error = invoke(root, tmp_path, capsys)
    assert status == 2
    assert "dirty or has untracked files" in error


def test_exit_code_uses_only_aggregate_required_gate_status():
    complete = [{"name": name, "status": "pass"} for name in launcher.REQUIRED_GATES]
    complete[4]["checks"] = {"independent_rows": True, "padding_batching": True}
    assert launcher.exit_code({"gates": complete}) == 0
    for status in ("fail", "untested"):
        changed = [dict(gate) for gate in complete]
        changed[1]["status"] = status
        assert launcher.exit_code({"gates": changed}) == 1
    assert launcher.exit_code({"gates": complete[:-1]}) == 1
    with_optional = [*complete, {"name": "original_release_loader", "required": False, "status": "untested"}]
    assert launcher.exit_code({"gates": with_optional}) == 0


def test_loading_info_uses_canonical_json_without_stringifying_sets(tmp_path):
    from kev.suite import write_json

    value = launcher.loading_info_json({"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []})
    path = tmp_path / "loading-info.json"
    write_json(path, value)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []
    }
