"""The accelerator this process uses: cuda, then mps, then cpu."""
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

# This file is the probe child (`python kev/device.py`). A native crash inside torch's import must not dump a core
# into the caller's directory (issue #170). The limit is set before the import; importing this module as kev.device is not the child.
if __name__ == "__main__" and os.name == "posix":
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import torch

DEVICES = ("cpu", "cuda", "mps")
# The child prints `kev-device <resolved>` as its result. ROCm and other wheels write startup text to stdout
# around it; a bare "cuda" line in that text is not a result. Before the allocation it prints `kev-try <device>`,
# and `kev-try cuda mps` when a later child should try MPS, so a crash still says what was attempted.
PROBE_PREFIX = "kev-device "
TRY_PREFIX = "kev-try "
DEVICE_HELP = ("auto probes in a child process and falls back to cpu (CUDA/ROCm, then MPS; "
               "a second probe runs only when the CUDA child died and reported MPS available; "
               "each probe waits KEV_DEVICE_PROBE_TIMEOUT seconds, default 30); "
               "an explicit accelerator fails closed; cpu skips discovery")


def default_device():
    return select()


def probe_timeout():
    """Seconds one probe may run. KEV_DEVICE_PROBE_TIMEOUT overrides the default.

    Auto runs a second probe only when the first died on CUDA and reported that MPS is available.
    """
    raw = os.environ.get("KEV_DEVICE_PROBE_TIMEOUT", "30")
    try:
        timeout = float(raw)
    except ValueError:
        raise ValueError(f"KEV_DEVICE_PROBE_TIMEOUT must be a positive number of seconds; got {raw!r}") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"KEV_DEVICE_PROBE_TIMEOUT must be a positive number of seconds; got {raw!r}")
    return timeout


def select(device="auto", *, dtype=torch.float32, timeout=None):
    """Resolve a device before loading weights. A fresh interpreter contains native driver crashes.

    Explicit CPU never inspects accelerators. Auto tries CUDA/ROCm and then MPS. The MPS probe is that same child
    when CUDA is absent, and a second child only when the CUDA child died after reporting MPS available, so MPS
    runs at most once (issue #255). An explicit accelerator fails closed. This tests one allocation/matmul/sync,
    not every model kernel or whether the checkpoint fits.
    """
    if device != "auto" and device not in DEVICES:
        raise ValueError(f"unsupported device {device!r}; choose auto or one of {DEVICES}")
    if device == "cpu":
        return device
    if timeout is None:
        timeout = probe_timeout()
    failures = []
    attempt = device
    while attempt:
        try:
            return _run_probe(attempt, dtype, timeout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            tried, retry_mps = _child_report(error)
            failures.append((tried or attempt, error))
            # A child that dies before printing kev-try (during torch import) would die the same way again.
            attempt = "mps" if device == "auto" and attempt == "auto" and retry_mps else None
    message = "Device probe failed (" + "; ".join(f"{name}: {_probe_detail(error)}" for name, error in failures) + ")."
    if device != "auto":
        raise RuntimeError(f"{message} Fix the accelerator runtime or pass --device cpu.") from failures[-1][1]
    warnings.warn(f"{message} Falling back to CPU; pass --device cpu to skip probing. "
                  "CPU defaults to fp32; KEV_DTYPE=bf16 reduces serving weight memory.", RuntimeWarning, stacklevel=2)
    return "cpu"


def _probe_detail(error):
    detail = str(error)
    if isinstance(error, subprocess.CalledProcessError) and (stderr := (error.stderr or "").strip()):
        detail += f": {stderr.splitlines()[-1]}"
    return detail


def _child_report(error):
    """The device the child printed before it might die, and whether that line offered an MPS fallback."""
    text = getattr(error, "stdout", None) or ""
    lines = [line[len(TRY_PREFIX):].strip() for line in text.splitlines() if line.startswith(TRY_PREFIX)]
    parts = lines[-1].split() if lines else []
    tried = parts[0] if parts and parts[0] in DEVICES else ""
    return tried, tried == "cuda" and "mps" in parts[1:]


def _run_probe(device, dtype, timeout):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), device, str(dtype).removeprefix("torch.")],
                            capture_output=True, text=True, errors="replace", timeout=timeout, check=True)
    reported = [line[len(PROBE_PREFIX):].strip() for line in result.stdout.splitlines() if line.startswith(PROBE_PREFIX)]
    resolved = reported[-1] if reported else ""
    if resolved not in DEVICES or (device != "auto" and resolved != device):
        tail = next((line.strip() for line in reversed(result.stdout.splitlines()) if line.strip()), "")
        raise ValueError(f"unexpected probe result {(resolved or tail)!r}")
    return resolved


def _mps_available():
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend is not None and backend.is_available())


def _probe(device, dtype):
    """Child-process entry point: no accelerator operations from here may run in the selecting process."""
    fallback = ""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            if _mps_available():
                fallback = " mps"
        elif _mps_available():
            device = "mps"
        else:
            return "cpu"
    if device != "cpu":
        # Flushed first: a timeout, a native crash, or a dtype the backend rejects still names the device.
        print(TRY_PREFIX + device + fallback, flush=True)
        with torch.no_grad():
            matrix = torch.ones(8, 8, device=device, dtype=dtype)
            matrix @ matrix
            sync(device)
    return device


def sync(device):
    """Wait for queued kernels, so wall-clock timings around a forward pass are real."""
    if device == "mps": torch.mps.synchronize()
    elif device == "cuda": torch.cuda.synchronize()


def empty_cache(device):
    if device == "mps": torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()


def out_of_memory(e):
    """Whether a torch allocator ran out: CUDA raises torch.OutOfMemoryError, MPS a plain RuntimeError with this message.
    The MLX backend's Metal errors are neither."""
    return isinstance(e, torch.OutOfMemoryError) or isinstance(e, RuntimeError) and str(e).startswith("MPS backend out of memory")


def allocated_bytes(device):
    """Bytes currently allocated on the device (MPS) or the peak since the process started (CUDA); 0 on CPU."""
    if device == "mps": return torch.mps.current_allocated_memory()
    if device == "cuda": return torch.cuda.max_memory_allocated()
    return 0


if __name__ == "__main__":
    print(PROBE_PREFIX + _probe(sys.argv[1], getattr(torch, sys.argv[2])), flush=True)
