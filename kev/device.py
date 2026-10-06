"""The accelerator this process uses: cuda, then mps, then cpu."""
import torch


def _cuda_usable():
    """A tiny kernel on `cuda`, since `is_available()` can be True with no kernel that actually runs: a ROCm or CUDA
    wheel can report the device and its name correctly while carrying no compiled code object for the local GPU
    architecture (common on RDNA3.5 APUs, e.g. gfx1151; #170). This reduces but cannot close the risk: the same
    mismatch can also crash the process outright (a SIGSEGV, never raised as a Python exception) rather than raising
    cleanly, which no amount of try/except on this side can catch."""
    try:
        (torch.zeros(2, 2, device="cuda") @ torch.zeros(2, 2, device="cuda")).cpu()
        torch.cuda.synchronize()
        return True
    except Exception as e:
        print(f"cuda reported available but a smoke kernel failed ({e}); falling back to mps/cpu")
        return False


def default_device():
    if torch.cuda.is_available() and _cuda_usable(): return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


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
