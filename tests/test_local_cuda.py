"""Local serving tools must isolate package/cache changes and stop on a failed kernel check."""
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "local_cuda.sh"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_setup_refuses_existing_environment(tmp_path):
    env = tmp_path / "existing"
    env.mkdir()
    (env / "pyvenv.cfg").write_text("existing=true\n", encoding="utf-8")
    result = subprocess.run(["bash", str(SCRIPT), "setup", "--env", str(env), "--cache", str(tmp_path / "cache")], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Refusing to modify" in result.stderr
    assert (env / "pyvenv.cfg").read_text(encoding="utf-8") == "existing=true\n"


def test_serve_keeps_explicit_paths_and_arguments(tmp_path):
    env = tmp_path / "env with spaces"
    (env / "bin").mkdir(parents=True)
    python = env / "bin" / "python"
    stub = '#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps({"args":sys.argv[1:], "cache":os.environ["TRITON_CACHE_DIR"], "fused":os.environ.get("KEV_FUSED")}))\n'
    python.write_text(stub, encoding="utf-8")
    python.chmod(0o755)
    cache = tmp_path / "cache with spaces"
    result = subprocess.run(["bash", str(SCRIPT), "serve", "--env", str(env), "--cache", str(cache), "--", "--run", "org/model@revision", "--port", "8010"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in result.stdout.splitlines()]
    assert calls[-1]["args"] == ["-m", "kev.serve", "--run", "org/model@revision", "--port", "8010"]
    assert all(call["cache"] == str(cache / "triton") for call in calls)
    assert calls[-1]["fused"] == "1"


def test_failed_preflight_prevents_server_start(tmp_path):
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    python = env / "bin" / "python"
    stub = '#!/bin/sh\necho kernel-failed >&2\nexit 1\n'
    python.write_text(stub, encoding="utf-8")
    python.chmod(0o755)
    result = subprocess.run(["bash", str(SCRIPT), "serve", "--env", str(env), "--cache", str(tmp_path / "cache")], capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stderr.count("kernel-failed") == 1


def test_http_benchmark_reports_queueing_separately():
    from scripts.http_serving_bench import summary
    rows = [{"wall_ms": 30 + i, "model_ms": 10} for i in range(20)]
    result = summary(rows, elapsed=2)
    assert result == {"n": 20, "wall_p50_ms": 39.5, "wall_p95_ms": 48, "model_p50_ms": 10, "requests_per_s": 10}


def test_http_benchmark_creates_output_directory(tmp_path, monkeypatch):
    from scripts import http_serving_bench
    out = tmp_path / "new-directory" / "report.json"
    monkeypatch.setattr(sys, "argv", ["bench", "--url", "http://localhost:8010", "--out", str(out)])
    monkeypatch.setattr(http_serving_bench, "run", lambda *args: {"latency": {"measured": True}})
    http_serving_bench.main()
    assert json.loads(out.read_text(encoding="utf-8")) == {"latency": {"measured": True}}
