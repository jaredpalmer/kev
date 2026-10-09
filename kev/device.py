"""The accelerator this process uses: cuda, then mps, then cpu."""
import os
import subprocess
import sys
import warnings
from pathlib import Path

import torch

DEVICES = ("cpu", "cuda", "mps")
# The child prints one line with this prefix. ROCm and other wheels write startup text to stdout around it;
# a bare "cuda" line in that text is not a result.
PROBE_PREFIX = "kev-device "


def default_device():
    return select()


def select(device="auto", *, dtype=torch.float32):
    """Resolve a device before loading weights. A fresh interpreter contains native driver crashes.

    Explicit CPU never inspects accelerators. Auto falls back on probe failure; an explicit accelerator fails closed.
    This tests one allocation/matmul/sync, not every model kernel or whether the checkpoint fits in memory.
    """
    if device != "auto" and device not in DEVICES:
        raise ValueError(f"unsupported device {device!r}; choose auto or one of {DEVICES}")
    if device == "cpu":
        return device
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), device, str(dtype).removeprefix("torch.")],
                                capture_output=True, text=True, errors="replace", timeout=30, check=True)
        reported = [line[len(PROBE_PREFIX):].strip() for line in result.stdout.splitlines() if line.startswith(PROBE_PREFIX)]
        resolved = reported[-1] if reported else ""
        if resolved not in DEVICES or device != "auto" and resolved != device:
            tail = next((line.strip() for line in reversed(result.stdout.splitlines()) if line.strip()), "")
            raise ValueError(f"unexpected probe result {(resolved or tail)!r}")
        return resolved
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        detail = str(error)
        if isinstance(error, subprocess.CalledProcessError) and (stderr := (error.stderr or "").strip()):
            detail += f": {stderr.splitlines()[-1]}"
        message = f"Device probe failed ({detail})."
        if device != "auto":
            raise RuntimeError(f"{message} Fix the accelerator runtime or pass --device cpu.") from error
        warnings.warn(f"{message} Falling back to CPU; pass --device cpu to skip probing. "
                      "CPU defaults to fp32; KEV_DTYPE=bf16 reduces serving weight memory.", RuntimeWarning, stacklevel=2)
        return "cpu"


def _probe(device, dtype):
    """Child-process entry point: no accelerator operations from here may run in the selecting process."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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
    # A native crash must not dump a core into the caller's directory (issue #170's failure did).
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    print(PROBE_PREFIX + _probe(sys.argv[1], getattr(torch, sys.argv[2])), flush=True)
