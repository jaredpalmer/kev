"""The accelerator this process uses: cuda, then mps, then cpu."""
import torch

DEVICES = ("cpu", "mps", "cuda", "npu")


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def select(name):
    """Resolve a `--device` value: one of DEVICES, with an optional index ("npu:7", "cuda:1"). An Ascend device also
    needs torch_npu imported before anything touches it, so every entry point that takes a device goes through here."""
    kind, _, index = str(name).partition(":")
    if kind not in DEVICES or (index and not index.isdigit()):
        raise ValueError(f"unknown device {name!r}: expected {', '.join(DEVICES)}, optionally with an index like npu:0")
    if kind == "npu":
        import torch_npu  # noqa: F401  # registers torch.npu and the Ascend backend; must precede any .to("npu")
    return str(name)


def sync(device):
    """Wait for queued kernels, so wall-clock timings around a forward pass are real."""
    if device == "mps": torch.mps.synchronize()
    elif device == "cuda": torch.cuda.synchronize()
    elif str(device).startswith("npu"): torch.npu.synchronize()


def empty_cache(device):
    if device == "mps": torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()
    elif str(device).startswith("npu"): torch.npu.empty_cache()


def out_of_memory(e):
    """Whether a torch allocator ran out: CUDA raises torch.OutOfMemoryError, MPS a plain RuntimeError with this message.
    The MLX backend's Metal errors are neither."""
    return isinstance(e, torch.OutOfMemoryError) or isinstance(e, RuntimeError) and str(e).startswith("MPS backend out of memory")


def allocated_bytes(device):
    """Bytes currently allocated on the device (MPS) or the peak since the process started (CUDA/NPU); 0 on CPU."""
    if device == "mps": return torch.mps.current_allocated_memory()
    if device == "cuda": return torch.cuda.max_memory_allocated()
    if str(device).startswith("npu"): return torch.npu.max_memory_allocated()
    return 0
