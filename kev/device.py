"""The accelerator this process uses: cuda, then mps, then cpu."""
import torch


def _cuda_usable():
    """Whether CUDA is not just reported but actually runnable.

    torch.cuda.is_available() is True on ROCm wheels that ship no code objects
    for the local GPU (e.g. gfx1151), where the first kernel then fails (issue
    #170). Probe with one tiny matmul instead of trusting the flag.
    """
    if not torch.cuda.is_available():
        return False
    try:
        with torch.no_grad():
            probe = torch.zeros(8, 8, device="cuda", dtype=torch.float16)
            (probe @ probe).cpu()
        return True
    except Exception:
        return False


def default_device():
    return "cuda" if _cuda_usable() else "mps" if torch.backends.mps.is_available() else "cpu"


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
