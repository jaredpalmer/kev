"""MLX backend (kev.mlx_model): LoRA merge arithmetic on a toy module, backend resolution, and weight-backed parity of the
Metal path against the fp32 torch path on a pinned jaredpalmer/kev-0.8b (downloads the base once; Apple Silicon only,
~30 s once cached).
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

RUN = "jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e"   # pinned (round 15, 2026-09-24): bf16 noise is per checkpoint, so a republish must not move these bars


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


# --- full-weight checkpoints (kev.train --full_ft; kev.mlx_model.load_full): a 4-layer Qwen3.5 text model with random
# weights (norms, conv kernels and A_log included, so a missed layout rule shows), saved as a full-weight run saves it

WORDS = "it is charged twice which team billing shipping refund angry the customer".split()


def save_full(root, layer_types=("linear_attention", "full_attention", "linear_attention", "full_attention"), dtype=torch.bfloat16, seed=0):
    """-> a full-weight checkpoint directory: save_pretrained of the text backbone (unprefixed names, config.json of
    Qwen3_5TextModel), head.pt with weights="full", and a word-level tokenizer as its base."""
    from tokenizers import Tokenizer, models as tkm, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from kev.checkpoint import Meta, write_meta
    from kev.model import SPECIAL, PointerHead
    vocab = {t: i for i, t in enumerate(["<unk>", "<pad>", *SPECIAL, *WORDS])}
    base, ckpt = root / "base", root / "full"
    tk = Tokenizer(tkm.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=SPECIAL).save_pretrained(base)
    config = Qwen3_5TextConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=len(layer_types), num_attention_heads=2,
                               num_key_value_heads=1, head_dim=32, linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=32,
                               linear_value_head_dim=32, layer_types=list(layer_types), full_attention_interval=2, pad_token_id=1)
    torch.manual_seed(seed)
    lm = Qwen3_5ForCausalLM(config).model
    with torch.no_grad():
        for name, p in lm.named_parameters():   # zero-initialised norms would hide a missing 1 + w; A_log stays negative-definite
            p.copy_(torch.randn_like(p) * (0.3 if p.ndim == 1 else 0.15) - (1.0 if name.endswith("A_log") else 0.0))
    lm.to(dtype).save_pretrained(ckpt)
    head = PointerHead(64, dp=16)
    with torch.no_grad():
        for p in head.parameters(): p.mul_(4.0)   # confident enough that a wrong hidden state moves the probabilities
    write_meta(ckpt, Meta(base=str(base), head=head.state_dict(), head_dim=16, weights="full", weights_dtype={torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]))
    return ckpt


TINY_REC = {"state": "the customer is charged twice it it it", "questions": {
    "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": "billing", "src": "probe"},
    "angry": {"type": "noul", "instructions": "is the customer angry", "label": True, "src": "probe"}}}


def test_full_weight_checkpoint_runs_on_mlx_as_saved(tmp_path):
    """A full-weight checkpoint loads on MLX from its own config and shards, nothing merged: auto picks MLX for it on
    Apple Silicon, the backbone runs in the saved bf16, every weight is the saved tensor (the DeltaNet conv kernel in MLX's
    axis order, zero-centred norms as 1 + w: mlx-lm's rules for a transformers export) and there is no LM head; hidden
    states and probabilities match the torch loader of the same checkpoint (fp32 on CPU) to bf16 noise, and the prefix
    form equals the row form."""
    from safetensors.torch import load_file
    from kev.data import materialize
    from mlx.utils import tree_flatten
    ck = Checkpoint(save_full(tmp_path))
    assert ck.full and ck.hybrid_base() and ck.backend("mps", LoadOptions(backend="auto")) == "mlx"
    tok, m = ck.load("mps", LoadOptions(backend="mlx"))
    _, ref = ck.load("cpu", LoadOptions(dtype=torch.float32))
    assert isinstance(m, MLXDecisionModel) and m.dtype == "bfloat16" and "lm_head" not in m.lm.language_model
    saved = load_file(ck.file("model.safetensors"))
    params = dict(tree_flatten(m.lm.parameters()))
    assert len(params) == len(saved)
    as_torch = lambda a: torch.from_numpy(np.asarray(a.astype(mx.float32)))
    for name, value in saved.items():
        got = as_torch(params["language_model.model." + name])
        if name.endswith("conv1d.weight"): value = value.transpose(1, 2)            # [C, 1, K] -> [C, K, 1]
        if name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or name == "norm.weight": value = value + 1   # bf16 + 1 in bf16, as mlx-lm does
        assert torch.equal(got, value.float()), name
    import torch.nn.functional as F
    rec = materialize(TINY_REC)
    enc = m.encode(tok, rec)
    # bf16 noise on these random weights: torch's own bf16 pass is 0.027 from fp32 (relative norm), MLX's 0.032
    assert state_error(m, ref, enc) < 0.05
    got, target = m.probs(enc), ref.probs(ref.encode(tok, rec))
    assert max(float((p - t).abs().max()) for p, t in zip(got, target)) < 0.02
    assert max(float(p.max()) for p in target) > 0.9, "a flat head would pass any comparison"
    assert max(float((p - F.softmax(z, -1)).abs().max()) for p, z in zip(got, m.forward_rows(enc))) < 0.02


def state_error(m, ref, enc):
    """Relative error (norm) of the MLX model's hidden states over the state tokens against the torch model's."""
    ids = enc["ids"][: enc["seg"].count(0)]
    h = torch.from_numpy(np.asarray(m._hidden([ids])[0].astype(mx.float32)))
    want = ref.lm(input_ids=torch.tensor([ids])).last_hidden_state[0].detach().float()
    return float((h - want).norm() / want.norm())


def test_full_weight_checkpoint_in_fp32_matches_torch_closely(tmp_path):
    """head.pt's weights_dtype (agreeing with config.json) decides the dtype: an fp32 export runs in fp32, not rounded.
    With no bf16 rounding anywhere, MLX and torch agree to the Metal fp32 matmul's precision (0.0036 relative on these
    weights), so any layout mistake (norms without 1 + w, conv kernel axes, A_log, rotary, gating) would stand out."""
    from kev.data import materialize
    ck = Checkpoint(save_full(tmp_path, dtype=torch.float32))
    tok, m = ck.load("mps", LoadOptions(backend="mlx"))
    _, ref = ck.load("cpu", LoadOptions(dtype=torch.float32))
    assert m.dtype == "float32"
    rec = materialize(TINY_REC)
    enc = m.encode(tok, rec)
    assert state_error(m, ref, enc) < 0.01
    assert max(float((p - t).abs().max()) for p, t in zip(m.probs(enc), ref.probs(ref.encode(tok, rec)))) < 0.01


def test_full_weight_refusals_on_mlx(tmp_path):
    """Strict: a checkpoint whose tensors do not match mlx-lm's model name for name and shape, carry another dtype than
    head.pt names, or disagree with config.json is refused, never loaded partially or cast; so are lora_scale (no adapter),
    option_isolation and attention-only backbones, as for LoRA checkpoints."""
    import json, shutil
    from safetensors.torch import load_file, save_file
    from kev.checkpoint import read_meta, write_meta
    good = save_full(tmp_path / "good")
    mlx = LoadOptions(backend="mlx")

    def variant(name, tensors=None, config=None, meta=None):
        d = tmp_path / name; shutil.copytree(good, d)
        if tensors:
            save_file(tensors(load_file(d / "model.safetensors")), d / "model.safetensors", metadata={"format": "pt"})
        if config:
            c = json.loads((d / "config.json").read_text(encoding="utf-8")); config(c); (d / "config.json").write_text(json.dumps(c), encoding="utf-8")
        if meta:
            m = read_meta(d); meta(m); write_meta(d, m)
        return Checkpoint(d)

    def drop(t): t.pop("norm.weight"); return t
    def extra(t): t["lm_head.weight"] = t["embed_tokens.weight"].clone(); return t
    def upcast(t): t["layers.0.linear_attn.A_log"] = t["layers.0.linear_attn.A_log"].float(); return t
    def mlx_layout(t):
        for k in [k for k in t if k.endswith("conv1d.weight")]: t[k] = t[k].transpose(1, 2).contiguous()
        return t
    for name, kw, match in (("missing", {"tensors": drop}, r"1 tensors missing \(e.g. \['norm.weight'\]\)"),
                            ("unexpected", {"tensors": extra}, r"1 unexpected \(e.g. \['lm_head.weight'\]\)"),
                            ("dtype", {"tensors": upcast}, r"1 not bfloat16 \(e.g. \['layers.0.linear_attn.A_log'\]\)"),
                            ("shape", {"config": lambda c: c.update(intermediate_size=96)}, "with another shape"),
                            ("layout", {"tensors": mlx_layout}, "conv kernels"),
                            ("model_type", {"config": lambda c: c.update(model_type="qwen3_5_moe_text")}, "Qwen3_5TextModel"),
                            ("layer_types", {"config": lambda c: c.update(layer_types=["full_attention", "linear_attention"] * 2)}, "layer_types"),   # mlx-lm follows the interval
                            ("mislabelled", {"meta": lambda m: setattr(m, "weights_dtype", "fp32")}, "config.json records the weights as bfloat16"),
                            ("isolation", {"meta": lambda m: setattr(m, "option_isolation", True)}, "option_isolation")):
        with pytest.raises(ValueError, match=match):
            variant(name, **kw).load("mps", mlx)
    with pytest.raises(ValueError, match="lora_scale"):
        Checkpoint(good).load("mps", LoadOptions(backend="mlx", lora_scale=0.5))
    attn = Checkpoint(save_full(tmp_path / "attn", layer_types=("full_attention",) * 4))
    assert not attn.hybrid_base() and attn.backend("mps", LoadOptions(backend="auto")) == "torch"
    with pytest.raises(ValueError, match="attention-only"):
        attn.load("mps", mlx)


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
        near(m.probs(m.encode(tok, rec)), ref.probs(ref.encode(tok, rec)), bar=0.03, tie=0.02)


def test_full_weight_kev_matches_its_lora_source_bit_for_bit(models, tmp_path):
    """The pinned Kev-0.8B written as a bf16 full-weight checkpoint (scripts/merge_lora_checkpoint.py --weights_dtype bf16:
    fp32 merge, one rounding) and loaded on MLX from its own shards gives exactly the probabilities of the LoRA path (base +
    merge_lora, the same arithmetic) once the base's fp32 tensors (A_log and the gated-norm weight of each DeltaNet layer,
    which a full-weight export holds in bf16) are rounded the same way; loading it takes no more MLX memory than its weights."""
    from scripts.merge_lora_checkpoint import merge
    from scripts.mlx_full_parity import round_fp32_tensors
    tok, _, _, recs = models
    merge(RUN, tmp_path / "merged", log=lambda m: None, weights_dtype="bf16")
    ck = Checkpoint(tmp_path / "merged/checkpoint")
    assert ck.full and ck.meta.weights_dtype == "bf16" and ck.backend("mps", LoadOptions(backend="auto")) == "mlx"
    mx.clear_cache(); mx.reset_peak_memory()
    before = mx.get_active_memory()
    _, full = ck.load("mps", LoadOptions(backend="mlx"))
    weights = sum(p.stat().st_size for p in ck.shards())
    assert mx.get_peak_memory() - before < 1.01 * weights   # no merge copy, no LM head, no transients left behind
    _, lora = Checkpoint(RUN).load("mps", LoadOptions(backend="mlx"))
    assert round_fp32_tensors(lora) > 0
    for rec in recs:
        got, want = full.probs(full.encode(tok, rec)), lora.probs(lora.encode(tok, rec))
        assert all(torch.equal(a, b) for a, b in zip(got, want))


def near(got, ref, bar, tie):
    """Every question within `bar` of the reference; an argmax may only flip where the reference's top-2 margin is under `tie`."""
    for p, r in zip(got, ref):
        assert float((p - r).abs().max()) < bar
        if p.argmax() != r.argmax():
            top = r.topk(2).values
            assert float(top[0] - top[1]) < tie, "argmax flip on a decided question"


def test_prefix_reuse_and_question_isolation(models):
    """The prefix form (state once, branches on a replicated cache: what forward/probs and the server run) equals the row
    form the torch path computes, reusing a prefix leaves it intact, and a question's answer does not depend on which
    other questions travel with it. Same function in fp32 (2.6e-6 on the CPU stream), different bf16 computations: max
    |dp| 0.028 on this checkpoint, hence the 0.04 bar. Injected cache faults (state's last token dropped, DeltaNet conv or
    recurrent state zeroed) give 0.67 / 0.39 / 0.47, yet as little as 0.003 on one record, so every record is checked.
    Paths that run the same kernels on the same shapes must agree bit for bit."""
    import torch.nn.functional as F
    from kev.data import materialize
    tok, m, _, recs = models
    close = lambda got, ref: near(got, ref, bar=0.04, tie=0.04)
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b))
    extra = materialize({"state": recs[0]["state"], "questions": {"sky": {"type": "noul", "instructions": "Ignore the text. Is the sky blue?", "label": True, "src": "probe"},
                                                                  "n": {"type": "choice", "instructions": "How many words is 'a b c'?", "criteria": {"one": None, "two": None, "three": None}, "label": "three", "src": "probe"}}})
    rec = {**recs[0], "questions": recs[0]["questions"] + extra["questions"]}
    for other in recs[1:]:   # one question each: the prefix form against the row form on every record (a cache fault can be small on one)
        enc = m.encode(tok, other)
        close(m.probs(enc), [F.softmax(z, -1) for z in m.forward_rows(enc)])
    enc = m.encode(tok, rec)
    full = [F.softmax(z, -1) for z in m.forward_rows(enc)]
    via_miss, prefix = m.probs_and_prefix(enc)
    via_hit = m.probs_with_prefix(enc, prefix); via_hit2 = m.probs_with_prefix(enc, prefix)
    close(via_miss, full)
    for got in (m.probs(enc), via_hit, via_hit2):   # reusing the prefix leaves it intact
        assert same(got, via_miss)
    alone = [m.probs(m.encode(tok, {"state": rec["state"], "questions": [q]}))[0] for q in rec["questions"]]
    close(alone, full)
    import kev.mlx_model as MM
    saved, MM.row_groups = MM.row_groups, lambda lengths, prefix_len=0, budget=0: [[i] for i in range(len(lengths))]   # one row (and one cache copy) per pass: same answers
    try: chunked = m.probs_with_prefix(enc, prefix)
    finally: MM.row_groups = saved
    assert same(chunked, alone)   # one branch row on the cached prefix, whether asked alone or split out of a batch
    with pytest.raises(ValueError, match="prefix"):
        m.probs_with_prefix(m.encode(tok, {**rec, "state": rec["state"] + " extra words here"}), prefix)


def test_chunked_state_prefill_is_exact(tmp_path, monkeypatch):
    """prefix() runs the state PREFILL_CHUNK tokens per pass into the prompt cache: each pass continues the cached
    attention keys/values at the cache's rotary offset, the DeltaNet conv window (the last 3 inputs) and its fp32 recurrent
    state, so chunking is exact up to float reassociation. On the fp32 random Qwen3.5 on the CPU (Metal's fp32 matmul is a
    reduced-precision fast path), every chunk size, down to passes shorter than the conv window, gives the one-pass cache and
    answers to 3e-5 (measured: cache 4e-6, probabilities 2e-6); a conv window lost at one boundary puts the cache 1.3 off. The Metal kernels on real bf16
    checkpoints: test_chunked_state_prefill_serves_within_bf16_noise, runs/mlx-long-states/prefill-ab-*."""
    import random
    import kev.mlx_model as MM
    from kev.data import materialize
    ck = Checkpoint(save_full(tmp_path, dtype=torch.float32))
    tok, m = ck.load("mps", LoadOptions(backend="mlx"))
    rng = random.Random(0)
    rec = materialize({**TINY_REC, "state": " ".join(rng.choice(WORDS) for _ in range(44))})
    enc = m.encode(tok, rec)
    Ls = enc["seg"].count(0)
    flat = lambda cache: [np.asarray(x) for c in cache for x in c.state]   # keys and values, conv and recurrent states
    rel = lambda a, b: max(float(np.abs(x - y).max() / np.abs(y).max()) for x, y in zip(a, b))
    mx.set_default_device(mx.cpu)
    try:
        one = m.prefix(enc)
        want, want_p = flat(one[1]), m.probs_with_prefix(enc, one)
        assert m.dtype == "float32" and Ls > 44 and max(float(p.max()) for p in want_p) > 0.9, "a flat head would pass any comparison"
        for chunk in (1, 2, 3, 4, 7, 16, Ls - 1):
            monkeypatch.setattr(MM, "PREFILL_CHUNK", chunk)
            n, cache = m.prefix(enc)
            assert n == Ls and {c.offset for c in cache if hasattr(c, "offset")} == {Ls}
            assert rel(flat(cache), want) < 3e-5, chunk
            assert max(float((p - q).abs().max()) for p, q in zip(m.probs_with_prefix(enc, (n, cache)), want_p)) < 3e-5, chunk
        cache = MM.make_prompt_cache(m.lm)   # the fault the comparison must catch: the conv window dropped at a boundary
        m.text(mx.array([enc["ids"][:20]], dtype=mx.int32), cache=cache)
        for c in cache:
            if not hasattr(c, "offset"): c[0] = mx.zeros_like(c[0])
        m.text(mx.array([enc["ids"][20:Ls]], dtype=mx.int32), cache=cache)
        assert rel(flat(cache), want) > 0.1
    finally:
        mx.set_default_device(mx.gpu)


def test_chunked_state_prefill_serves_within_bf16_noise(models, monkeypatch):
    """The served chunk on real states longer than it (the three longest hard-v1 long_policy development records, 4.6k-4.9k
    tokens: five 1,024-token passes, the last one partial) keeps the answers within the MLX-vs-torch serving bar of the
    one-pass prefix and of the fp32 torch path. Chunked and one-pass bf16 differ only by rounding in kernels picked by
    shape (fp32 on the CPU: 1e-6, test_chunked_state_prefill_is_exact): measured max |dp| 0.003-0.005 against one pass
    (any chunk from 256 to 4,096 moves these states as much: runs/mlx-long-states/prefill-ab-0.8b.json) and 0.003-0.011
    against fp32 torch (one pass: 0.004-0.012), no flip, one torch top-2 margin 0.019."""
    import kev.mlx_model as MM
    from kev.data import materialize
    from kev.model import admit
    from kev.suite import load_split
    tok, m, ref, _ = models
    long = sorted((r for r in load_split("evals/hard-v1", "development") if r["_meta"]["family"] == "long_policy"), key=lambda r: -r["_meta"]["state_tokens"])[:3]
    for rec in map(materialize, long):
        enc = admit(m, tok, rec)   # the serving context: encode's default cuts the state at the training length
        Ls = enc["seg"].count(0)
        assert Ls > 4 * MM.PREFILL_CHUNK and Ls % MM.PREFILL_CHUNK
        chunked = m.probs(enc)
        with monkeypatch.context() as one_pass:
            one_pass.setattr(MM, "PREFILL_CHUNK", 10 ** 9); one = m.probs(enc)
        near(chunked, one, bar=0.03, tie=0.02)
        near(chunked, ref.probs(admit(ref, tok, rec)), bar=0.03, tie=0.02)


def test_server_refuses_or_marks_over_length_states_on_mlx(models, monkeypatch):
    """kev.serve on the MLX backend admits like the torch one (kev.model.admit): with the state limit set just under a
    real record's state, the default server answers 422 before the model runs, and a truncating server reads exactly the
    first `limit` tokens (the answers of that cut encoding) and says so."""
    from types import SimpleNamespace
    from fastapi import HTTPException
    import kev.model as M
    from kev import serve
    tok, m, _, recs = models
    rec = recs[0]
    n = m.encode(tok, rec, max_state=10 ** 6, max_branch=10 ** 6)["state_tokens"]
    monkeypatch.setattr(M, "SERVE_MAX_STATE", n - 1)
    card = SimpleNamespace(release_date=lambda: "2026-01-01")
    refuse, truncate = serve.Server(card, tok, m, "mps"), serve.Server(card, tok, m, "mps", truncate_states=True)
    try:
        with pytest.raises(HTTPException) as refused:
            refuse.probs(rec)
        assert refused.value.status_code == 422 and refused.value.detail.startswith(f"state is {n:,} tokens, over the {n - 1:,}-token limit")
        assert refuse.batches == 0
        ps, stats = truncate.probs(rec)
        assert (stats["state_tokens"], stats["state_tokens_used"]) == (n, n - 1)
        cut = m.probs(m.encode(tok, rec, max_state=n - 1, max_branch=M.SERVE_MAX_BRANCH))
        assert all(np.allclose(p, c.tolist(), atol=1e-6) for p, c in zip(ps, cut))
        monkeypatch.setattr(M, "SERVE_MAX_STATE", n)
        assert refuse.probs(rec)[1]["state_tokens_used"] == n   # exactly at the limit: admitted whole
    finally:
        refuse.close(); truncate.close()
