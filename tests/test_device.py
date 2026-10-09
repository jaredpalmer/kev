"""Startup selection, native-crash containment, and the actual serve argument path; no GPU or weights needed."""
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kev import device, serve
from kev.checkpoint import LoadOptions


def reported(name):
    return device.PROBE_PREFIX + name + "\n"


def test_explicit_cpu_never_discovers_accelerators(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU selection must not inspect an accelerator or launch a probe")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.backends.mps, "is_available", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    assert device.select("cpu") == "cpu"


@pytest.mark.parametrize("requested,resolved", [("auto", "cuda"), ("auto", "mps"), ("auto", "cpu"),
                                               ("cuda", "cuda"), ("mps", "mps")])
def test_selection_uses_fresh_interpreter_and_requested_precision(monkeypatch, requested, resolved):
    def run(command, **kwargs):
        assert command == [sys.executable, str(Path(device.__file__).resolve()), requested, "bfloat16"]
        assert kwargs == dict(capture_output=True, text=True, errors="replace", timeout=30, check=True)
        return SimpleNamespace(stdout=reported(resolved))
    monkeypatch.setattr(subprocess, "run", run)
    assert device.select(requested, dtype=torch.bfloat16) == resolved


@pytest.mark.parametrize("failure", [subprocess.CalledProcessError(1, "probe", stderr="HIP error: invalid device function"),
                                     subprocess.CalledProcessError(1, "probe", stderr=" \n"),
                                     subprocess.CalledProcessError(-11, "probe"), subprocess.TimeoutExpired("probe", 30),
                                     OSError("cannot launch interpreter")])
@pytest.mark.parametrize("requested", ["auto", "cuda", "mps"])
def test_failed_probe_falls_back_only_for_auto(monkeypatch, failure, requested):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(subprocess, "run", run)
    if requested == "auto":
        with pytest.warns(RuntimeWarning, match="Falling back to CPU.*KEV_DTYPE=bf16"):
            assert device.default_device() == "cpu"
    else:
        with pytest.raises(RuntimeError, match="--device cpu"):
            device.select(requested)


def test_invalid_probe_output_is_not_a_device(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="not a device\n"))
    with pytest.warns(RuntimeWarning, match="unexpected probe result"):
        assert device.select() == "cpu"
    with pytest.raises(ValueError, match="unsupported device"):
        device.select("npu")


def test_probe_ignores_stdout_banners_and_keeps_the_marked_line(monkeypatch):
    """A driver banner, including a line that is itself a device name, must not override the child's report."""
    banner = "ROCm 6.2 init\ncuda\n\n" + reported("mps") + "shutdown\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=banner))
    assert device.select() == "mps"
    assert device.select("mps") == "mps"
    with pytest.raises(RuntimeError, match="unexpected probe result 'mps'"):
        device.select("cuda")


@pytest.mark.parametrize("cuda,mps,resolved", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_probe_allocates_multiplies_and_synchronizes(monkeypatch, cuda, mps, resolved):
    calls = []
    class Matrix:
        def __matmul__(self, other):
            calls.append("matmul")
            return self
    def ones(*shape, **kwargs):
        calls.append((shape, kwargs))
        return Matrix()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch, "ones", ones)
    monkeypatch.setattr(device, "sync", lambda dev: calls.append(("sync", dev)))
    assert device._probe("auto", torch.bfloat16) == resolved
    assert calls == ([] if resolved == "cpu" else [((8, 8), {"device": resolved, "dtype": torch.bfloat16}),
                                                   "matmul", ("sync", resolved)])


@pytest.mark.skipif(os.name != "posix", reason="POSIX native signal exit status")
@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_real_sigsegv_in_probe_cannot_kill_parent(tmp_path, monkeypatch, requested):
    # Only the child imports this torch stand-in. It dies where a native allocation could die.
    fake_torch = '''
import contextlib, os, resource, signal
from types import SimpleNamespace
float32 = "float32"
cuda = SimpleNamespace(is_available=lambda: True)
no_grad = contextlib.nullcontext
def ones(*args, **kwargs):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.kill(os.getpid(), signal.SIGSEGV)
'''
    (tmp_path / "torch.py").write_text(fake_torch, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    if requested == "auto":
        with pytest.warns(RuntimeWarning, match="SIGSEGV"):
            assert device.select() == "cpu"
    else:
        with pytest.raises(RuntimeError, match="SIGSEGV"):
            device.select(requested)


@pytest.fixture
def startup(monkeypatch):
    """Replace weight loading and the socket server, leaving serve.main, selection and option defaults intact."""
    loaded = []
    model = SimpleNamespace(hybrid=True, backend="torch", dtype="float32")
    class Checkpoint:
        def __init__(self, run):
            self.requested = self.path = run
        def load(self, dev, opts):
            loaded.append((dev, opts))
            return None, model
    monkeypatch.setattr(serve, "Checkpoint", Checkpoint)
    monkeypatch.setattr(serve, "Server", lambda ck, tok, model, dev: SimpleNamespace(truncate_states=False))
    monkeypatch.setattr(serve.app.state, "server", None, raising=False)
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: loaded.append(kwargs))
    for name in os.environ:
        if name.startswith("KEV_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(serve, "fused_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["kev.serve", "--run", "fixture/model", "--port", "8019"])
    return loaded


@pytest.mark.parametrize("requested,resolved,hip", [(None, "cuda", None), ("auto", "cpu", None),
                                                  ("cpu", "cpu", None), ("cuda", "cuda", "6.4"),
                                                  ("cuda", "cuda", None), ("mps", "mps", None)])
def test_serve_main_selects_device_and_defaults(startup, monkeypatch, requested, resolved, hip):
    if requested is not None:
        sys.argv += ["--device", requested]
    monkeypatch.setattr(torch.version, "hip", hip)
    def run(command, **kwargs):
        assert command[-2:] == [requested or "auto", "bfloat16"]
        assert requested != "cpu", "explicit CPU should never launch a probe"
        return SimpleNamespace(stdout=reported(resolved))
    monkeypatch.setattr(subprocess, "run", run)
    serve.main()
    dev, opts = startup[0]
    assert dev == resolved
    assert opts.dtype == (None if dev == "cpu" else torch.bfloat16)
    assert opts.attn == ("sdpa" if dev == "mps" else None)
    assert opts.backend == "auto"
    assert opts.cuda_graphs == (True if dev == "cuda" and hip is None else None)
    assert opts.fused == (True if dev == "cuda" and hip is None else None)
    assert startup[1] == {"host": "127.0.0.1", "port": 8019}


@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_serve_probe_failure_precedes_weight_loading(startup, monkeypatch, requested):
    sys.argv += ["--device", requested]
    def run(*args, **kwargs):
        raise subprocess.CalledProcessError(-11, "probe")
    monkeypatch.setattr(subprocess, "run", run)
    if requested == "auto":
        with pytest.warns(RuntimeWarning, match="Falling back to CPU"):
            serve.main()
        assert startup[0][0] == "cpu"
        assert startup[0][1].cuda_graphs is None and startup[0][1].fused is None
    else:
        with pytest.raises(RuntimeError, match="--device cpu"):
            serve.main()
        assert not startup


def test_serve_preserves_explicit_options_and_probes_that_dtype(startup, monkeypatch):
    sys.argv += ["--device", "cuda"]
    opts = LoadOptions(dtype=torch.float32, cuda_graphs=False, fused=False, backend="torch", attn="eager")
    monkeypatch.setattr(LoadOptions, "from_env", classmethod(lambda cls: opts))
    def run(command, **kwargs):
        assert command[-1] == "float32"
        return SimpleNamespace(stdout=reported("cuda"))
    monkeypatch.setattr(subprocess, "run", run)
    serve.main()
    assert startup[0] == ("cuda", opts)


def test_serve_rejects_invalid_device_before_loading(startup):
    sys.argv += ["--device", "bogus"]
    with pytest.raises(SystemExit) as error:
        serve.main()
    assert error.value.code == 2 and not startup


def test_serve_cpu_honors_bf16_environment_without_probe(startup, monkeypatch):
    sys.argv += ["--device", "cpu"]
    monkeypatch.setenv("KEV_DTYPE", "bf16")
    def forbidden(*args, **kwargs):
        pytest.fail("explicit CPU must skip probing")
    monkeypatch.setattr(subprocess, "run", forbidden)
    serve.main()
    assert startup[0][0] == "cpu"
    assert startup[0][1].dtype == torch.bfloat16


def test_child_marker_ignores_import_banners_and_undecodable_bytes(tmp_path, monkeypatch):
    """The real child: a torch stand-in prints a device-named banner and a non-UTF-8 byte during import.
    The parent must keep the marked line, which is cpu because this stand-in reports no accelerator."""
    fake_torch = '''
import sys
from types import SimpleNamespace
print("ROCm 6.2 init")
print("cuda")
sys.stdout.buffer.write(b"\\xff\\xfe junk\\n")
sys.stdout.flush()
float32 = "float32"
cuda = SimpleNamespace(is_available=lambda: False)
backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
'''
    (tmp_path / "torch.py").write_text(fake_torch, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert device.select() == "cpu"


@pytest.mark.skipif(os.name != "posix", reason="POSIX core rlimit")
def test_probe_child_disables_core_dumps(tmp_path, monkeypatch):
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    if hard == 0:
        pytest.skip("core dumps already forbidden by the hard limit")
    report = tmp_path / "core.txt"
    fake_torch = '''
import contextlib, os, resource
float32 = "float32"
no_grad = contextlib.nullcontext
def ones(*args, **kwargs):
    with open(os.environ["CORE_REPORT"], "w", encoding="utf-8") as out:
        out.write(str(resource.getrlimit(resource.RLIMIT_CORE)[0]))
    raise SystemExit(0)
'''
    (tmp_path / "torch.py").write_text(fake_torch, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("CORE_REPORT", str(report))
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (min(hard, 1024), hard))
        subprocess.run([sys.executable, str(Path(device.__file__).resolve()), "cuda", "float32"],
                       check=True, capture_output=True, text=True, timeout=30)
    finally:
        resource.setrlimit(resource.RLIMIT_CORE, (soft, hard))
    assert report.read_text(encoding="utf-8") == "0"


def _benchmark_report(*args, **kwargs):
    return {"objective": 0, "clean": {}, "coverage": {}}, []


def test_benchmark_help_and_usage_do_not_probe(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("probe"))
    from kev import benchmark
    monkeypatch.setattr(sys, "argv", ["kev.benchmark", "--help"])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 0
    monkeypatch.setattr(sys, "argv", ["kev.benchmark", "--out", "unused"])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_benchmark_skips_the_probe_for_cpu_and_remote(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("probe"))
    data = tmp_path / "data.jsonl"
    data.write_text("", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    from kev import benchmark
    monkeypatch.setattr(benchmark, "load_records", lambda path: [])
    monkeypatch.setattr(benchmark, "evaluate_records", _benchmark_report)
    monkeypatch.setattr(benchmark, "LocalPredictor", lambda *a, **k: SimpleNamespace(temperature=1.0, environment={}))
    monkeypatch.setattr(sys, "argv", ["kev.benchmark", "--run", "fixture", "--data", str(data), "--device", "cpu", "--out", str(out)])
    benchmark.main()
    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.setattr(sys, "argv", ["kev.benchmark", "--remote", "http://127.0.0.1:9", "--data", str(data),
                                      "--device", "cuda", "--out", str(remote)])
    benchmark.main()


def test_benchmark_explicit_cuda_failure_precedes_loading(monkeypatch):
    def crash(*args, **kwargs):
        raise subprocess.CalledProcessError(-11, "probe")
    monkeypatch.setattr(subprocess, "run", crash)
    loaded = []
    from kev import benchmark
    monkeypatch.setattr(benchmark, "LocalPredictor", lambda *a, **k: loaded.append(a) or SimpleNamespace(temperature=1.0, environment={}))
    monkeypatch.setattr(sys, "argv", ["kev.benchmark", "--run", "fixture", "--data", "missing.jsonl", "--device", "cuda", "--out", "unused"])
    with pytest.raises(RuntimeError, match="--device cpu"):
        benchmark.main()
    assert not loaded


def test_benchmark_probes_the_dtype_it_will_load(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text("", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    seen = []
    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(stdout=reported("cuda"))
    monkeypatch.setenv("KEV_DTYPE", "bf16")
    monkeypatch.setattr(subprocess, "run", run)
    from kev import benchmark
    monkeypatch.setattr(benchmark, "load_records", lambda path: [])
    monkeypatch.setattr(benchmark, "evaluate_records", _benchmark_report)
    monkeypatch.setattr(benchmark, "LocalPredictor", lambda *a, **k: SimpleNamespace(temperature=1.0, environment={}))
    monkeypatch.setattr(sys, "argv", ["kev.benchmark", "--run", "fixture", "--data", str(data), "--device", "cuda", "--out", str(out)])
    benchmark.main()
    assert seen[0][-2:] == ["cuda", "bfloat16"]


def test_experiment_help_and_aggregate_do_not_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("probe"))
    from kev import experiment
    monkeypatch.setattr(sys, "argv", ["kev.experiment", "--help"])
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 0
    monkeypatch.setattr(sys, "argv", ["kev.experiment", "--aggregate", "--out", str(tmp_path)])
    experiment.main()


def test_base_mmlu_probe_help_does_not_probe(monkeypatch):
    import importlib.util
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("probe"))
    path = Path(__file__).resolve().parents[1] / "scripts" / "base_mmlu_probe.py"
    spec = importlib.util.spec_from_file_location("base_mmlu_probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["base_mmlu_probe", "--help"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 0
