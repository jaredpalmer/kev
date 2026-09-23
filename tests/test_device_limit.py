"""A sidecar sharing a card with a much larger server must not be able to grow into its memory.

The caching allocator only ever grows: once a long request has expanded it, the memory is not handed
back, so the ceiling has to be set before the first allocation rather than tidied up afterwards.
"""
from kev import device


def test_no_cap_when_the_environment_says_nothing(monkeypatch):
    monkeypatch.delenv("KEV_GPU_MEMORY_FRACTION", raising=False)
    assert device.limit_memory("cuda") is None


def test_cpu_and_mps_are_never_capped(monkeypatch):
    monkeypatch.setenv("KEV_GPU_MEMORY_FRACTION", "0.2")
    assert device.limit_memory("cpu") is None
    assert device.limit_memory("mps") is None


def test_cuda_takes_the_fraction_from_the_environment(monkeypatch):
    applied = []
    monkeypatch.setenv("KEV_GPU_MEMORY_FRACTION", "0.2")
    monkeypatch.setattr(device.torch.cuda, "set_per_process_memory_fraction", applied.append)
    assert device.limit_memory("cuda") == 0.2
    assert applied == [0.2]


def test_an_explicit_fraction_beats_the_environment(monkeypatch):
    applied = []
    monkeypatch.setenv("KEV_GPU_MEMORY_FRACTION", "0.2")
    monkeypatch.setattr(device.torch.cuda, "set_per_process_memory_fraction", applied.append)
    assert device.limit_memory("cuda", 0.5) == 0.5
    assert applied == [0.5]
