"""kev.device.default_device: usability probe and --device override (issue #170).

No GPU needed: every CUDA-present case is simulated by monkeypatching torch.
Run: python -m pytest tests/test_device.py -q
"""
import torch

from kev.device import _cuda_usable, default_device


def test_cpu_machine_returns_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device() == "cpu"


def test_mps_when_no_usable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device() == "mps"


def test_broken_rocm_wheel_falls_back_to_cpu(monkeypatch):
    """is_available() True but no kernel image for the local arch (#170)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(*args, **kwargs):
        raise RuntimeError("HIP error: invalid device function")

    monkeypatch.setattr(torch, "zeros", boom)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _cuda_usable() is False
    assert default_device() == "cpu"


def test_serve_device_flag_parsing():
    """--device auto/cuda/mps/cpu is accepted by kev.serve (issue #170)."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cuda", "mps", "cpu"))
    assert ap.parse_args([]).device == "auto"
    assert ap.parse_args(["--device", "cpu"]).device == "cpu"
