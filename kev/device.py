"""The accelerator this process uses: cuda, then mps, then cpu."""
import os

import torch


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def limit_memory(device, fraction=None):
    """Cap this process's share of a shared card. Returns the fraction applied, or None.

    A sidecar beside a much larger server needs a ceiling it cannot cross. The caching allocator only ever grows -- one
    long request expands it and the memory is never handed back -- so the cap has to be in place before the first
    allocation, not tidied up afterwards. KEV_GPU_MEMORY_FRACTION sets it; unset or 0 means no cap, which is what a
    process with the card to itself wants."""
    if fraction is None: fraction = float(os.environ.get("KEV_GPU_MEMORY_FRACTION", "0"))
    if not fraction or device != "cuda": return None
    torch.cuda.set_per_process_memory_fraction(fraction)
    return fraction


def sync(device):
    """Wait for queued kernels, so wall-clock timings around a forward pass are real."""
    if device == "mps": torch.mps.synchronize()
    elif device == "cuda": torch.cuda.synchronize()


def empty_cache(device):
    if device == "mps": torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()


def allocated_bytes(device):
    """Bytes currently allocated on the device (MPS) or the peak since the process started (CUDA); 0 on CPU."""
    if device == "mps": return torch.mps.current_allocated_memory()
    if device == "cuda": return torch.cuda.max_memory_allocated()
    return 0
