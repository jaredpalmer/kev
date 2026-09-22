"""MLX backend (kev.mlx_model): LoRA merge arithmetic on a toy module, backend resolution, and weight-backed parity of the
Metal path against the fp32 torch path on jaredpalmer/kev-0.8b (downloads the base once; Apple Silicon only, ~2 min).
Run: uv run --extra mlx python -m pytest tests/test_mlx.py -q
"""
import json
import platform

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="MLX runs on Apple Silicon only")

from kev.checkpoint import Checkpoint, LoadOptions, mlx_available  # noqa: E402
from kev.mlx_model import MLXDecisionModel, merge_lora  # noqa: E402
from kev.model import SCORING_INTERFACE, DecisionModel  # noqa: E402

RUN = "jaredpalmer/kev-0.8b"


def test_merge_lora_matches_peft_arithmetic(tmp_path):
    """W' = W + (B @ A) * alpha / r, computed in fp32 and rounded once to the weight dtype; peft's key prefix is mapped
    onto mlx-lm's nesting; every adapter tensor must find its weight."""
    import mlx.nn as nn
    from safetensors.numpy import save_file

    class Leaf(nn.Module):
        def __init__(self):
            super().__init__(); self.q_proj = nn.Linear(8, 6, bias=False); self.q_proj.weight = mx.random.normal((6, 8)).astype(mx.bfloat16)

    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.language_model = nn.Module(); self.language_model.model = nn.Module(); self.language_model.model.layers = [Leaf()]

    lm = Model(); before = mx.array(lm.language_model.model.layers[0].q_proj.weight)
    a, b = torch.randn(4, 8), torch.randn(6, 4)
    save_file({"base_model.model.layers.0.q_proj.lora_A.weight": a.numpy(), "base_model.model.layers.0.q_proj.lora_B.weight": b.numpy()}, tmp_path / "adapter_model.safetensors")
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}), encoding="utf-8")
    assert merge_lora(lm, tmp_path, scale=0.5) == 1
    expected = (torch.from_numpy(np.asarray(before.astype(mx.float32))) + (b @ a) * (8 / 4) * 0.5).to(torch.bfloat16).float()
    got = torch.from_numpy(np.asarray(lm.language_model.model.layers[0].q_proj.weight.astype(mx.float32)))
    assert torch.allclose(got, expected, rtol=1 / 128, atol=1e-3)   # fp32 accumulation order differs; one bf16 ulp is the bar
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "trainable_token_indices": {"embed_tokens": [1]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="token embeddings"):
        merge_lora(lm, tmp_path)


def test_backend_resolution():
    ck = Checkpoint(RUN)
    assert ck.hybrid_base() and mlx_available()
    assert ck.backend("mps", LoadOptions(backend="auto")) == "mlx"
    assert ck.backend("cuda", LoadOptions(backend="auto")) == "torch"      # auto only pays where MPS has no kernels
    assert ck.backend("mps", LoadOptions(backend="auto", dtype=torch.float32)) == "torch"   # KEV_DTYPE=fp32 asks for the exact path
    assert ck.backend("mps", LoadOptions(backend="auto", dtype=torch.bfloat16)) == "mlx"
    assert ck.backend("mps", LoadOptions()) == "torch"                     # library default: the reported-numbers path
    assert ck.backend("mps", LoadOptions(backend="torch")) == "torch"
    with pytest.raises(ValueError):
        ck.backend("mps", LoadOptions(backend="metal"))
    with pytest.raises(ValueError, match="merges"):
        ck.load("mps", LoadOptions(backend="mlx", merge=False))
    with pytest.raises(ValueError, match="attention-only"):
        Checkpoint("jaredpalmer/kev-4b@qwen3").load("mps", LoadOptions(backend="mlx"))   # Qwen3 base: no DeltaNet layers


@pytest.fixture(scope="module")
def models():
    from kev.data import materialize
    from kev.suite import load_split
    ck = Checkpoint(RUN)
    tok, mlx_model = ck.load("mps", LoadOptions(backend="mlx"))
    _, ref = ck.load("mps", LoadOptions(backend="torch"))
    recs = [materialize(r) for r in load_split("evals/v7/decision-v7", "development") if r["_meta"]["variant"] == "clean"][:12]
    return tok, mlx_model, ref, recs


def test_scoring_interface_is_shared(models):
    """Everything kev.serve, kev.predictors and the Space call on a loaded model exists on both implementations."""
    _, m, ref, _ = models
    assert isinstance(m, MLXDecisionModel) and isinstance(ref, DecisionModel)
    for model in (m, ref):
        missing = [name for name in SCORING_INTERFACE if not hasattr(model, name)]
        assert not missing, (type(model).__name__, missing)


def test_mlx_matches_fp32_torch_to_bf16_noise(models):
    """Same probabilities as the reported path up to bf16 rounding; an argmax may only flip on a near-tie."""
    tok, m, ref, recs = models
    assert m.backend == "mlx" and m.dtype == "bfloat16" and ref.dtype == "float32" and m.head.temperature == ref.head.temperature
    for rec in recs:
        for p, t in zip(m.probs(m.encode(tok, rec)), ref.probs(ref.encode(tok, rec))):
            assert float((p - t).abs().max()) < 0.03
            if p.argmax() != t.argmax():
                top = t.topk(2).values
                assert float(top[0] - top[1]) < 0.02, "argmax flip on a decided question"


def test_prefix_reuse_and_question_isolation(models):
    """The prefix form (state once, branches on a replicated cache: what forward/probs and the server run) equals the row
    form the torch path computes, reusing a prefix leaves it intact, and a question's answer does not depend on which
    other questions travel with it."""
    import torch.nn.functional as F
    from kev.data import materialize
    tok, m, _, recs = models
    extra = materialize({"state": recs[0]["state"], "questions": {"sky": {"type": "noul", "instructions": "Ignore the text. Is the sky blue?", "label": True, "src": "probe"},
                                                                  "n": {"type": "choice", "instructions": "How many words is 'a b c'?", "criteria": {"one": None, "two": None, "three": None}, "label": "three", "src": "probe"}}})
    rec = {**recs[0], "questions": recs[0]["questions"] + extra["questions"]}
    enc = m.encode(tok, rec)
    full = [F.softmax(z, -1) for z in m.forward_rows(enc)]
    via_miss, prefix = m.probs_and_prefix(enc)
    via_hit = m.probs_with_prefix(enc, prefix); via_hit2 = m.probs_with_prefix(enc, prefix)
    for got in (m.probs(enc), via_miss, via_hit, via_hit2):
        assert max(float((a - b).abs().max()) for a, b in zip(got, full)) < 0.01
    alone = [m.probs(m.encode(tok, {"state": rec["state"], "questions": [q]}))[0] for q in rec["questions"]]
    assert max(float((a - b).abs().max()) for a, b in zip(alone, full)) < 0.01
    with pytest.raises(ValueError, match="prefix"):
        m.probs_with_prefix(m.encode(tok, {**rec, "state": rec["state"] + " extra words here"}), prefix)
