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
# The child prints one line with this prefix. ROCm and other wheels write startup text to stdout around it;
# a bare "cuda" line in that text is not a result.
PROBE_PREFIX = "kev-device "
# Printed before the availability check (`auto`) and again before the allocation (`cuda`, `mps`, or `cpu`).
# A driver banner is not one of these lines, and neither line starts with PROBE_PREFIX, so it cannot be a result.
PROBE_START_PREFIX = "kev-probing "
DEVICE_HELP = ("auto probes in a child process and falls back to cpu (CUDA/ROCm, then MPS; "
               "MPS runs once, and a second probe runs only after the child has reached CUDA; "
               "each probe waits KEV_DEVICE_PROBE_TIMEOUT seconds, default 30); "
               "an explicit accelerator fails closed; cpu skips discovery")


def default_device():
    return select()


def probe_timeout():
    """Seconds one probe may run. KEV_DEVICE_PROBE_TIMEOUT overrides the default.

    Auto runs a second probe only when the first child reached the CUDA check or a CUDA allocation.
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

    Explicit CPU never inspects accelerators. Auto tries CUDA/ROCm, then MPS, then CPU. The child prints
    `kev-probing auto` before the availability check and `kev-probing <device>` before the allocation.
    The parent skips the MPS child when this child printed `mps` or `cpu`, or never printed `auto` or `cuda`.
    The MPS child still runs when the first child dies during the CUDA check or the CUDA allocation.
    An explicit accelerator fails closed. This tests one allocation/matmul/sync, not every model kernel or whether the checkpoint fits.
    """
    if device != "auto" and device not in DEVICES:
        raise ValueError(f"unsupported device {device!r}; choose auto or one of {DEVICES}")
    if device == "cpu":
        return device
    if timeout is None:
        timeout = probe_timeout()
    failures = []
    for attempt in (("auto", "mps") if device == "auto" else (device,)):
        try:
            return _run_probe(attempt, dtype, timeout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            announced = _probe_announcements(error)
            # Name the device the child reached. A child that printed only "auto", or nothing, keeps the attempt name.
            failures.append((announced[-1] if announced and announced[-1] in DEVICES else attempt, error))
            if attempt == "auto" and (
                "mps" in announced or "cpu" in announced or not ("auto" in announced or "cuda" in announced)
            ):
                break
    message = "Device probe failed (" + "; ".join(f"{attempt}: {_probe_detail(error)}" for attempt, error in failures) + ")."
    if device != "auto":
        raise RuntimeError(f"{message} Fix the accelerator runtime or pass --device cpu.") from failures[-1][1]
    warnings.warn(f"{message} Falling back to CPU; pass --device cpu to skip probing. "
                  "CPU defaults to fp32; KEV_DTYPE=bf16 reduces serving weight memory.", RuntimeWarning, stacklevel=2)
    return "cpu"


def _probe_announcements(error):
    """`kev-probing` payloads the child printed, in order. Empty when it died before the first print.

    TimeoutExpired.output stays bytes even with text=True. communicate() decodes the stream only when it returns.
    """
    output = getattr(error, "output", None) or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return [line[len(PROBE_START_PREFIX):] for line in output.splitlines() if line.startswith(PROBE_START_PREFIX)]


def _probe_detail(error):
    detail = str(error)
    if isinstance(error, subprocess.CalledProcessError) and (stderr := (error.stderr or "").strip()):
        detail += f": {stderr.splitlines()[-1]}"
    return detail


def _run_probe(device, dtype, timeout):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), device, str(dtype).removeprefix("torch.")],
                            capture_output=True, text=True, errors="replace", timeout=timeout, check=True)
    reported = [line[len(PROBE_PREFIX):].strip() for line in result.stdout.splitlines() if line.startswith(PROBE_PREFIX)]
    resolved = reported[-1] if reported else ""
    if resolved not in DEVICES or (device != "auto" and resolved != device):
        tail = next((line.strip() for line in reversed(result.stdout.splitlines()) if line.strip()), "")
        raise ValueError(f"unexpected probe result {(resolved or tail)!r}")
    return resolved


def _probe(device, dtype):
    """Child-process entry point: no accelerator operations from here may run in the selecting process."""
    if device == "auto":
        # Import has finished. A hang in the availability check still leaves this line, so the parent can try MPS.
        print(PROBE_START_PREFIX + "auto", flush=True)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(PROBE_START_PREFIX + device, flush=True)
    if device != "cpu":
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
