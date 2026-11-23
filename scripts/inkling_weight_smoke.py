"""One-shot, public Inkling-2-Mini plumbing smoke helpers and remote runner."""

from __future__ import annotations

import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
MODEL_REVISION = "7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6"
TRANSFORMERS_REVISION = "6173a2f679ff4ed393afc499aefc569da7e68bc1"
EXPECTED_TEXT_CONFIG = {
    "model_type": "inkling2_loop_text",
    "hidden_size": 2304,
    "num_hidden_layers": 24,
}
VISUAL_PREFIX = "model.visual."
REMOTE_DEVICE_PATH = Path("/tmp/kev_device.py")
REMOTE_MODEL_PATH = Path("/tmp/kev_model.py")

H200_PRICE_PER_HOUR = 4.54
CPU_PRICE_PER_CORE_HOUR = 0.0473
CPU_PRICE_PER_CORE_HOUR_SANDBOX = 0.1419
RAM_PRICE_PER_GIB_HOUR = 0.008
RAM_PRICE_PER_GIB_HOUR_SANDBOX = 0.024
EGRESS_PRICE_PER_GIB = 0.04
SANDBOX_MULTIPLIER = 3
SANDBOX_SECONDS = 3000
END_TO_END_WATCHDOG_SECONDS = 3300
SANDBOX_CPU_CORES = 8
SANDBOX_RAM_GIB = 128
IMAGE_BUILD_ALLOWANCE_SECONDS = 300
IMAGE_BUILD_CPU_CORES = 2
IMAGE_BUILD_RAM_GIB = 8
MAX_EGRESS_GIB = 1 / 1024
MAX_COST_USD = 20.0
MAX_RESULT_BYTES = 65536

BENIGN_RECORD = {
    "state": "The library closes at 6 PM. A visitor arrives at 5 PM.",
    "questions": [
        {
            "instr": "Did the visitor arrive before closing?",
            "options": ["Yes", "No"],
            "label": 0,
        }
    ],
}


class SmokeFailure(RuntimeError):
    def __init__(self, stage: str, details: dict):
        super().__init__(stage)
        self.stage = stage
        self.details = details


def cost_breakdown(sandbox_seconds: float = SANDBOX_SECONDS) -> dict[str, float]:
    sandbox = sandbox_seconds / 3600 * (
        SANDBOX_MULTIPLIER * H200_PRICE_PER_HOUR
        + SANDBOX_CPU_CORES * CPU_PRICE_PER_CORE_HOUR_SANDBOX
        + SANDBOX_RAM_GIB * RAM_PRICE_PER_GIB_HOUR_SANDBOX
    )
    build_allowance = IMAGE_BUILD_ALLOWANCE_SECONDS / 3600 * SANDBOX_MULTIPLIER * (
        IMAGE_BUILD_CPU_CORES * CPU_PRICE_PER_CORE_HOUR
        + IMAGE_BUILD_RAM_GIB * RAM_PRICE_PER_GIB_HOUR
    )
    egress = SANDBOX_MULTIPLIER * MAX_EGRESS_GIB * EGRESS_PRICE_PER_GIB
    return {
        "sandbox": sandbox,
        "image_build_allowance": build_allowance,
        "egress": egress,
        "total": sandbox + build_allowance + egress,
    }


def cost_bound() -> float:
    return cost_breakdown()["total"]


def claim_marker(path: Path | str, metadata: dict) -> Path:
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as marker_file:
        json.dump(metadata, marker_file, sort_keys=True)
        marker_file.write("\n")
        marker_file.flush()
        os.fsync(marker_file.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def update_marker(path: Path | str, updates: dict) -> None:
    path = Path(path)
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    _write_private_json(path, current)


def _write_private_json(path: Path, value: dict) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=path.parent)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
        json.dump(value, temp_file, sort_keys=True)
        temp_file.write("\n")
        temp_file.flush()
        os.fsync(temp_file.fileno())
    os.replace(temp_name, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_report(path: Path | str, report: dict) -> None:
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    _write_private_json(path, report)


def _string_list(value, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"loading_info.{field} must contain strings")
    return sorted(value)


def summarize_loading_info(info: dict) -> dict:
    missing = _string_list(info.get("missing_keys"), "missing_keys")
    unexpected = _string_list(info.get("unexpected_keys"), "unexpected_keys")
    mismatched = _string_list(info.get("mismatched_keys"), "mismatched_keys")
    errors = _string_list(info.get("error_msgs"), "error_msgs")
    visual = [key for key in unexpected if key.startswith(VISUAL_PREFIX)]
    non_visual = [key for key in unexpected if not key.startswith(VISUAL_PREFIX)]
    return {
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "unexpected_visual_count": len(visual),
        "unexpected_visual_examples": visual[:10],
        "unexpected_non_visual_count": len(non_visual),
        "unexpected_non_visual_examples": non_visual[:10],
        "mismatched_count": len(mismatched),
        "mismatched_examples": mismatched[:10],
        "error_count": len(errors),
        "error_examples": errors[:5],
    }


def loading_info_passes(summary: dict) -> bool:
    return not any(
        summary[key]
        for key in (
            "missing_count",
            "unexpected_non_visual_count",
            "mismatched_count",
            "error_count",
        )
    )


def _install_runtime() -> None:
    source = f"https://github.com/huggingface/transformers/archive/{TRANSFORMERS_REVISION}.zip"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "torch==2.8.0",
        "accelerate==1.15.0",
        source,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise SmokeFailure(
            "runtime_install",
            {"exit_code": completed.returncode, "stderr_tail": completed.stderr[-2000:]},
        )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load uploaded kev/model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_remote_smoke(device: str = "cuda") -> dict:
    stage = "runtime_install"
    try:
        _install_runtime()
        stage = "imports"
        import torch
        from transformers import AutoTokenizer
        from transformers.models.inkling2_loop.modeling_inkling2_loop import (
            Inkling2LoopForCausalLM,
            Inkling2LoopNormedEmbedding,
        )
        from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

        torch.manual_seed(20261120)
        torch.set_grad_enabled(False)
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        device_code = _load_module("kev_device", REMOTE_DEVICE_PATH)
        if device_code.select(device, dtype=torch.bfloat16, timeout=30) != device:
            raise RuntimeError(f"requested device {device} is unavailable")
        model_code = _load_module("kev_model", REMOTE_MODEL_PATH)

        stage = "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
        )
        added = tokenizer.add_special_tokens({"additional_special_tokens": model_code.SPECIAL})
        boundary_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in model_code.SPECIAL}
        if len(set(boundary_ids.values())) != len(model_code.SPECIAL):
            raise RuntimeError("Kev boundary tokens do not have unique tokenizer ids")
        if tokenizer.unk_token_id in boundary_ids.values():
            raise RuntimeError("a Kev boundary token resolved to the unknown-token id")

        stage = "model_load"
        text_config = Inkling2LoopTextConfig.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
        )
        causal_lm, loading_info = Inkling2LoopForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            config=text_config,
            dtype=torch.bfloat16,
            quantization_config=None,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        backbone = causal_lm.model
        lm_head_shape = list(causal_lm.lm_head.weight.shape)
        del causal_lm
        config_summary = {
            "model_type": backbone.config.model_type,
            "hidden_size": backbone.config.hidden_size,
            "num_hidden_layers": backbone.config.num_hidden_layers,
        }
        if config_summary != EXPECTED_TEXT_CONFIG:
            raise SmokeFailure("text_config", config_summary)
        loading_summary = summarize_loading_info(loading_info)
        if not loading_info_passes(loading_summary):
            raise SmokeFailure("strict_text_loading_info", loading_summary)
        if len(tokenizer) != backbone.embed_tokens.num_embeddings:
            backbone.resize_token_embeddings(len(tokenizer))
        if not isinstance(backbone.embed_tokens, Inkling2LoopNormedEmbedding):
            raise RuntimeError("token resize did not preserve Inkling normed embedding")
        if backbone.embed_tokens.num_embeddings != len(tokenizer):
            raise RuntimeError("tokenizer and model embedding sizes differ")
        backbone.config.use_cache = False
        device_name = f"{device}:0" if device == "cuda" else device
        backbone_dtype = str(next(backbone.parameters()).dtype)
        backbone.to(device_name)
        backbone.eval()

        stage = "score"
        from torch import nn

        class PlumbingDecisionModel(model_code.DecisionModel):
            def __init__(self, lm, head, pad_id):
                nn.Module.__init__(self)
                self.lm, self.head = lm, head
                self.device = torch.device(device_name)
                self.pad_id = pad_id
                self.hybrid = model_code.is_hybrid(lm.config)
                self.option_isolation = False

        head = model_code.PointerHead(backbone.config.hidden_size).to(device_name)
        pointer_head_dtype = str(next(head.parameters()).dtype)
        head.eval()
        view = PlumbingDecisionModel(backbone, head, model_code.pad_id(tokenizer))
        view.eval()
        encoded = model_code.encode(tokenizer, BENIGN_RECORD)
        _, _, branches = model_code.rows_of(encoded)
        if len(branches) != 1:
            raise RuntimeError("the plumbing example must encode to exactly one independent causal row")
        with torch.inference_mode():
            logits = view.forward_rows_batch([encoded], shared_prefix=False)[0][0]
        if logits.numel() != 2 or not torch.isfinite(logits).all():
            raise RuntimeError("the one-example pointer readout was non-finite or had wrong option count")
        device_code.sync(device)
        peak_gpu = device_code.allocated_bytes(device) / (1024**3)
        peak_host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        return {
            "status": "passed",
            "interpretation": "PLUMBING SMOKE ONLY; random untrained pointer head; no quality verdict",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "text_config": config_summary,
            "loading_info": loading_summary,
            "loaded_class": "Inkling2LoopForCausalLM",
            "extracted_backbone": "model",
            "lm_head_shape": lm_head_shape,
            "added_boundary_tokens": added,
            "boundary_token_ids": boundary_ids,
            "embedding_class": type(backbone.embed_tokens).__name__,
            "embedding_rows": backbone.embed_tokens.num_embeddings,
            "use_cache": backbone.config.use_cache,
            "backbone_parameter_dtype": backbone_dtype,
            "pointer_head_parameter_dtype": pointer_head_dtype,
            "independent_causal_rows": len(branches),
            "option_count": int(logits.numel()),
            "random_pointer_logits": [float(value) for value in logits.float().cpu()],
            "label_used_for_scoring": False,
            "peak_cuda_allocated_gib": round(peak_gpu, 3),
            "peak_process_rss_gib": round(peak_host, 3),
            "training_steps": 0,
            "dataset_reads": 0,
        }
    except SmokeFailure as exc:
        return {"status": "failed", "stage": exc.stage, "details": exc.details}
    except Exception as exc:
        return {
            "status": "failed",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }


def main() -> int:
    stdout, stderr = sys.stdout, sys.stderr
    with open(os.devnull, "w", encoding="utf-8") as sink:
        sys.stdout = sink
        sys.stderr = sink
        try:
            result = run_remote_smoke()
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    payload = json.dumps(result, sort_keys=True)
    if len(payload.encode("utf-8")) > MAX_RESULT_BYTES:
        result = {"status": "failed", "stage": "report_size"}
        payload = json.dumps(result, sort_keys=True)
    print(payload, file=stdout)
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
