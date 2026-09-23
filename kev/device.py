"""The accelerator this process uses: cuda, then xpu, then mps, then cpu."""
import torch


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    """Wait for queued kernels, so wall-clock timings around a forward pass are real."""
    if device == "mps": torch.mps.synchronize()
    elif device == "cuda": torch.cuda.synchronize()
    elif device == "xpu": torch.xpu.synchronize()


def empty_cache(device):
    if device == "mps": torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()
    elif device == "xpu": torch.xpu.empty_cache()


def allocated_bytes(device):
    """Bytes currently allocated on the device (MPS) or the peak since the process started (CUDA/XPU); 0 on CPU."""
    if device == "mps": return torch.mps.current_allocated_memory()
    if device == "cuda": return torch.cuda.max_memory_allocated()
    if device == "xpu": return torch.xpu.memory_allocated()
    return 0
