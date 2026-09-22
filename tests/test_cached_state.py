"""kev.cached_state: branch inputs against the packed mask, the K/V cache store, and the two mechanisms' equivalence with
the packed forward. Most tests need no weights (a character-level fake tokenizer; a tiny random Qwen3 saved to tmp_path);
the last two use Qwen/Qwen3-0.6B-Base from the HF cache when present (one is a ~1 min kev.train smoke). CPU only.
Run: uv run --extra serve python -m pytest tests/test_unit.py tests/test_cached_state.py -q
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from kev.model import SPECIAL, DecisionModel, branch_mask_batch, encode
from kev.suite import read_json
from kev.cached_state import (INT8, StateCache, branch_hidden, branch_inputs, branch_logits, cache_from_kv, chunked_loss_backward,
                              dequantize_kv, frozen_loss_backward, kv_bytes_per_token, kv_dtype, quantize_kv, size_bytes, state_kv, state_len)

BASE = "Qwen/Qwen3-0.6B-Base"
REPO = Path(__file__).resolve().parents[1]
REC = {"state": "the pump tripped twice overnight and the tank level fell", "questions": [
    {"instr": "resolved?", "options": ["yes", "no"], "label": 1},
    {"instr": "which part", "options": ["pump", "tank", "valve", "sensor"], "label": 0},
    {"instr": "severity", "options": ["low", "mid", "high"], "label": 2}]}


class FakeTokenizer:
    """One id per character; the five delimiters get their own ids (like tests/test_research.py's stand-in)."""
    pad_token_id = 0

    def __call__(self, text, **kw):
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def convert_tokens_to_ids(self, token):
        return 1000 + SPECIAL.index(token)


def spans(enc):
    S = state_len(enc)
    return [(S if q == 0 else enc["decide_idx"][q - 1] + 1, d + 1) for q, d in enumerate(enc["decide_idx"])]


@pytest.mark.parametrize("isolate", [False, True])
@pytest.mark.parametrize("qs", [None, [1], [2, 0]])
def test_branch_inputs_match_packed_mask_and_indices(isolate, qs):
    enc = encode(FakeTokenizer(), REC, option_isolation=isolate)
    S = state_len(enc); sp = spans(enc)
    ids, pos, mask, decide, opts = branch_inputs(enc, qs, "cpu")
    order = list(range(3)) if qs is None else qs
    sel = [i for q in order for i in range(*sp[q])]
    assert ids[0].tolist() == [enc["ids"][i] for i in sel] and pos[0].tolist() == [enc["pos"][i] for i in sel]
    assert all(pos[0, local].item() == S for local in (sel.index(sp[q][0]) for q in order))   # branch positions restart after the state
    assert [sel[d] for d in decide] == [enc["decide_idx"][q] for q in order]
    assert [[sel[i] for i in o] for o in opts] == [enc["opt_idx"][q] for q in order]
    packed = branch_mask_batch([enc["seg"]], "cpu", opts=[enc["opt"]] if isolate else None)[0, 0]
    cols = list(range(S)) + sel
    assert mask.shape == (1, 1, len(sel), S + len(sel))
    assert torch.equal(mask[0, 0] == 0, packed[sel][:, cols] == 0)


def test_branch_mask_rules():
    enc = encode(FakeTokenizer(), REC, option_isolation=True)
    S = state_len(enc)
    _, _, mask, decide, opts = branch_inputs(enc, [1], "cpu")
    allowed = mask[0, 0] == 0
    assert allowed[:, :S].all()                                   # every branch token sees the whole state
    o0, o1 = opts[0][0], opts[0][1]                               # </opt> of option 0 and option 1
    assert allowed[o1, S + o1] and not allowed[o1, S + o0] and not allowed[o0, S + o1]   # option spans are isolated from each other
    assert allowed[decide[0], S:].all()                           # <decide> sees its whole question
    assert allowed[o0, S] and not allowed[0, S + o1]              # instruction visible to options, options invisible to the instruction (causal)
    plain = branch_inputs(encode(FakeTokenizer(), REC), [0, 1], "cpu")[2][0, 0] == 0
    q0 = encode(FakeTokenizer(), REC)["decide_idx"][0] - S + 1
    assert plain[q0:, S:S + q0].sum() == 0 and plain[:q0, S + q0:].sum() == 0   # questions never see each other
    assert torch.diagonal(plain[:, S:]).all()


def test_state_cache_store_cast_and_eviction():
    kv = [(torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)) for _ in range(3)]
    n = size_bytes(kv)
    assert n == 3 * 2 * 40 * 4
    c = StateCache(torch.bfloat16, "cpu")
    assert c.get("a") is None and c.misses == 1
    got = c.put("a", kv)
    assert got[0][0].dtype == torch.bfloat16 and c.bytes == n // 2 and c.get("a") is got and c.hits == 1
    back = kv_dtype(got, torch.float32)
    assert back[0][0].dtype == torch.float32 and (back[0][0] - kv[0][0]).abs().max() < 0.05
    same = kv_dtype(kv, torch.float32, "cpu")
    assert same[1][1] is kv[1][1]                                 # nothing to cast: no copy
    c = StateCache(torch.float32, "cpu", max_bytes=2 * n + 1)
    c.put("a", kv); c.put("b", kv); c.put("c", kv)
    assert len(c) == 2 and c.get("a") is None and c.get("c") is not None and c.bytes == 2 * n   # oldest evicted
    c.put("d", [(k.repeat(1, 1, 2, 1), v.repeat(1, 1, 2, 1)) for k, v in kv] * 2)
    assert "d" not in c.entries and c.stats()["entries"] == 2      # larger than the cap: returned, not kept


def test_state_cache_disk_tier(tmp_path):
    kv = [(torch.randn(1, 2, 64, 16), torch.randn(1, 2, 64, 16)) for _ in range(3)]   # big enough that the payload, not torch.save's per-tensor overhead, sets the file size
    d = tmp_path / "sc"
    c = StateCache(torch.bfloat16, "disk", cache_dir=d)
    assert d.is_dir() and c.get("a") is None and c.misses == 1
    got = c.put("a", kv)
    files = list(d.iterdir())
    assert [f.name for f in files] == [hashlib.sha1(b"a").hexdigest() + ".pt"] and c.bytes == files[0].stat().st_size >= size_bytes(got)   # on-disk bytes
    back = c.get("a")
    assert c.hits == 1 and isinstance(back, list) and isinstance(back[0], tuple) and back[0][0].dtype == torch.bfloat16 and back[0][0].device.type == "cpu"
    assert all(torch.equal(x, y) for p, q in zip(back, got) for x, y in zip(p, q))
    assert c.stats()["tier"] == "disk" and c.stats()["dir"] == str(d) and c.stats()["entries"] == 1
    again = StateCache(torch.bfloat16, "disk", cache_dir=d)          # a later run over the same directory adopts the file
    assert again.get("a") is not None and (again.hits, again.misses, again.bytes, len(again)) == (1, 0, c.bytes, 1)
    n = c.bytes
    c = StateCache(torch.bfloat16, "disk", max_bytes=2 * n + 1, cache_dir=tmp_path / "capped")
    c.put("a", kv); c.put("b", kv); c.put("c", kv)
    assert len(c) == 2 and c.get("a") is None and c.get("c") is not None and c.bytes == 2 * n
    assert sorted(f.name for f in (tmp_path / "capped").iterdir()) == sorted(hashlib.sha1(k.encode()).hexdigest() + ".pt" for k in "bc")   # oldest file evicted
    c.put("d", [(k.repeat(1, 1, 2, 1), v.repeat(1, 1, 2, 1)) for k, v in kv] * 2)
    assert "d" not in c.entries and len(list((tmp_path / "capped").iterdir())) == 2 and c.bytes == 2 * n   # larger than the cap: returned, no file left
    with pytest.raises(ValueError, match="cache_dir"):
        StateCache(torch.float32, "disk")


def test_cache_from_kv_references_tensors_and_rejects_bad_configs():
    kv = [(torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)) for _ in range(2)]
    fake = SimpleNamespace(device="cpu", training=False, lm=SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention"] * 2, use_sliding_window=False),
        parameters=lambda: iter([torch.zeros(1)]), is_gradient_checkpointing=False))
    cache = cache_from_kv(fake, kv)
    assert cache.get_seq_length() == 5 and all(l.keys is kv[i][0] and l.values is kv[i][1] for i, l in enumerate(cache.layers))
    before = kv[0][0].clone()
    cache.update(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4), 0)
    assert cache.get_seq_length() == 8 and torch.equal(kv[0][0], before) and kv[0][0].shape[2] == 5   # appends never touch kv
    with pytest.raises(ValueError, match="layers"):
        cache_from_kv(fake, kv[:1])
    fake.lm.config.layer_types = ["sliding_attention", "full_attention"]
    with pytest.raises(NotImplementedError, match="sliding"):
        cache_from_kv(fake, kv)
    fake.lm.config.layer_types = ["full_attention"] * 2; fake.training = True; fake.lm.is_gradient_checkpointing = True
    with pytest.raises(RuntimeError, match="checkpointing"):
        cache_from_kv(fake, kv)


def perturb_lora(model):
    """PEFT starts lora_B at zero (adapter = identity, lora_A grads exactly zero); make the adapter matter."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n: p.normal_(0, 0.02)


def grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def grad_gap(ref, other, min_scale=1e-6):
    """(worst per-tensor max|diff| / max|ref| over tensors with a non-trivial reference gradient, global relative L2)."""
    worst, num, den = 0.0, 0.0, 0.0
    for n, g in ref.items():
        o = other[n]; num += (g - o).pow(2).sum().item(); den += g.pow(2).sum().item()
        if g.abs().max().item() >= min_scale: worst = max(worst, ((g - o).abs().max() / g.abs().max()).item())
    return worst, (num ** 0.5) / (den ** 0.5 + 1e-30)


def readout_positions(enc):
    return enc["decide_idx"] + [i for oi in enc["opt_idx"] for i in oi]


def check_equivalence(model, enc, seed=0):
    """(a) frozen == packed with the adapter disabled (hidden at readout positions, logits); (b) chunked gradients == packed
    gradients with a perturbed adapter; (c) frozen mode trains every LoRA tensor. Returns the measured gaps."""
    torch.manual_seed(seed); perturb_lora(model); model.eval()
    out = {}
    sel = readout_positions(enc)
    with torch.no_grad(), model.lm.disable_adapter():
        model.state_mode = "adapted"; h_ref = model.hidden(enc); z_ref = model(enc)
        model.state_mode = "frozen"; h_frozen = model.hidden(enc); z_frozen = model(enc)
        kv = state_kv(model, enc["ids"][:state_len(enc)], adapter=False)
        h_one = torch.cat([branch_hidden(model, enc, kv, [q])[0] for q in range(len(z_ref))])
        zeros = [(torch.zeros_like(k), torch.zeros_like(v)) for k, v in kv]
        z_nostate = branch_logits(model, enc, zeros)
    scale = h_ref[sel].abs().max().item()
    out["hidden_rel"] = (h_frozen[sel] - h_ref[sel]).abs().max().item() / scale
    out["hidden_one_at_a_time_rel"] = (h_one[[i - state_len(enc) for i in sel]] - h_ref[sel]).abs().max().item() / scale
    out["logits_rel"] = max((a - b).abs().max().item() for a, b in zip(z_frozen, z_ref)) / max(z.abs().max().item() for z in z_ref)
    out["control_no_state_rel"] = max((a - b).abs().max().item() for a, b in zip(z_nostate, z_ref)) / max(z.abs().max().item() for z in z_ref)
    model.state_mode = "adapted"
    Q = len(enc["labels"])
    def q_loss(z, q): return F.cross_entropy(z.float()[None], torch.tensor([enc["labels"][q]])) / Q
    model.zero_grad(set_to_none=True)
    loss_ref = sum(q_loss(z, q) for q, z in enumerate(model(enc))); loss_ref.backward(); g_ref = grads(model)
    for chunk in (1, 2):
        model.zero_grad(set_to_none=True)
        loss = chunked_loss_backward(model, enc, q_loss, chunk)
        out[f"chunk{chunk}_loss_gap"] = abs(loss - loss_ref.item())
        out[f"chunk{chunk}_grad_rel_inf"], out[f"chunk{chunk}_grad_rel_l2"] = grad_gap(g_ref, grads(model))
    model.zero_grad(set_to_none=True)
    model.state_mode = "frozen"
    sum(q_loss(z, q) for q, z in enumerate(model(enc))).backward()
    lora = {n: g for n, g in grads(model).items() if "lora" in n}
    out["frozen_lora_tensors"] = len(lora); out["frozen_lora_nonzero"] = sum(int(g.abs().max() > 0) for g in lora.values())
    out["frozen_lora_expected"] = sum(1 for n, p in model.named_parameters() if p.requires_grad and "lora" in n)
    out["cache_requires_grad"] = any(k.requires_grad or v.requires_grad for k, v in kv)
    model.state_mode = "adapted"
    return out


def assert_equivalent(out):
    assert out["hidden_rel"] < 1e-2 and out["hidden_one_at_a_time_rel"] < 1e-2 and out["logits_rel"] < 1e-2
    assert out["control_no_state_rel"] > 1e-2                   # the cached state is what makes the branch pass equal
    for chunk in (1, 2):
        assert out[f"chunk{chunk}_loss_gap"] < 1e-4
        assert out[f"chunk{chunk}_grad_rel_inf"] < 1e-3 and out[f"chunk{chunk}_grad_rel_l2"] < 1e-4
    assert out["frozen_lora_nonzero"] == out["frozen_lora_tensors"] == out["frozen_lora_expected"] and not out["cache_requires_grad"]


@pytest.fixture(scope="module")
def tiny_path(tmp_path_factory):
    """A random 2-layer Qwen3 (no download) saved to a temp dir."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=2048, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=False)
    path = tmp_path_factory.mktemp("tiny-qwen3"); Qwen3ForCausalLM(cfg).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny_model(tiny_path):
    """The tiny backbone with LoRA r=4 in every layer, on CPU."""
    return DecisionModel(tiny_path, FakeTokenizer(), "cpu", lora=4, head_dim=32)


def test_tiny_model_frozen_and_chunked_match_packed(tiny_model):
    enc = encode(FakeTokenizer(), REC)
    assert_equivalent(check_equivalence(tiny_model, enc))


def test_tiny_model_option_isolation_frozen_matches_packed(tiny_model):
    assert_equivalent(check_equivalence(tiny_model, encode(FakeTokenizer(), REC, option_isolation=True), seed=1))


@pytest.mark.parametrize("isolate", [False, True])
def test_tiny_model_frozen_chunked_matches_one_pass(tiny_model, isolate):
    """Frozen state: frozen_loss_backward (chunks of 1 and 2 questions, scaled) returns the loss and leaves the gradients of one
    branch pass over all questions against the same base K/V; the only difference is the attention over fewer branch rows."""
    model = tiny_model; torch.manual_seed(4); perturb_lora(model); model.eval()
    enc = encode(FakeTokenizer(), REC, option_isolation=isolate); Q = len(enc["labels"])
    kv = state_kv(model, enc["ids"][:state_len(enc)], adapter=False, grad=False)
    def q_loss(z, q): return F.cross_entropy(z.float()[None], torch.tensor([enc["labels"][q]])) / Q
    model.zero_grad(set_to_none=True)
    loss_ref = sum(q_loss(z, q) for q, z in enumerate(branch_logits(model, enc, kv))); (loss_ref * 0.5).backward(); g_ref = grads(model)
    for chunk in (1, 2):
        model.zero_grad(set_to_none=True); times = {"branch": 0.0}
        loss = frozen_loss_backward(model, enc, kv, q_loss, chunk, scale=0.5, times=times)
        g = grads(model); worst, rel = grad_gap(g_ref, g)
        assert abs(loss - loss_ref.item()) < 1e-6 and set(g) == set(g_ref) and times["branch"] > 0
        assert worst < 2e-5 and rel < 1e-5, (chunk, worst, rel)
    assert not any(k.requires_grad or v.requires_grad for k, v in kv)


def test_tiny_model_forward_batch_uses_supplied_caches_and_times(tiny_model, tmp_path):
    model = tiny_model; model.eval(); model.state_mode = "frozen"
    enc = encode(FakeTokenizer(), REC)
    enc2 = encode(FakeTokenizer(), {**REC, "questions": REC["questions"][:1]})
    try:
        with torch.no_grad():
            ref = model.forward_batch([enc, enc2])
            kv = state_kv(model, enc["ids"][:state_len(enc)], adapter=False)
            store = StateCache(torch.bfloat16, "cpu"); store.put("r", kv)
            got = model.forward_batch([enc, enc2], [store.get("r"), None])   # bf16 entry is cast for the model; None = compute
        for a, b in zip(sum(ref, []), sum(got, [])):
            assert (a - b).abs().max().item() < 5e-2 * max(a.abs().max().item(), 1)
        disk = StateCache(torch.bfloat16, "disk", cache_dir=tmp_path / "disk"); disk.put("r", kv)
        with torch.no_grad():
            from_disk = model.forward_batch([enc], [disk.get("r")])[0]   # loaded CPU tensors, cast for the model like any entry
        assert all(torch.equal(a, b) for a, b in zip(from_disk, got[0]))
        model.state_kv_dtype = torch.bfloat16   # what evaluate.load sets from head.pt: recomputed K/V rounded like the training cache
        with torch.no_grad():
            rounded = model.forward_batch([enc])[0]
        assert all(torch.equal(a, b) for a, b in zip(rounded, got[0]))
        assert model.probs(enc, store.get("r"))[0].shape == rounded[0].shape
    finally:
        model.state_mode, model.state_kv_dtype = "adapted", None
    times = {"state": 0.0, "branch": 0.0}; model.zero_grad(set_to_none=True)
    loss = chunked_loss_backward(model, enc, lambda z, q: F.cross_entropy(z[None], torch.tensor([enc["labels"][q]])), 2, scale=0.5, times=times)
    assert loss > 0 and times["state"] > 0 and times["branch"] > 0
    with pytest.raises(ValueError, match="unknown state_mode"):   # checked before any weights are loaded
        DecisionModel("unused", FakeTokenizer(), "cpu", state_mode="bogus")


def test_int8_round_trip_and_forward_through_int8_cache(tiny_model, tmp_path):
    """int8 cache entries: codes plus one fp16 scale per (layer, head, token) row, byte accounting, a small round-trip error
    (at most half a step of the row's scale), the same rounded K/V from put()'s return value, get() and kv_dtype(..., "int8"),
    and a branch pass over an int8 entry that equals the recomputed-and-rounded path evaluate.load sets up."""
    model = tiny_model; model.eval()
    enc = encode(FakeTokenizer(), REC); S = state_len(enc)
    with torch.no_grad():
        kv = state_kv(model, enc["ids"][:S], adapter=False)
    q = quantize_kv(kv)
    cfg = model.lm.config
    assert all(qk.dtype == torch.int8 and sk.dtype == torch.float16 and sk.shape == (1, cfg.num_key_value_heads, S, 1) for qk, sk, _, _ in q)
    assert size_bytes(q) == S * kv_bytes_per_token(model, INT8) == S * 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.head_dim + 2)
    assert kv_bytes_per_token(model, INT8) == kv_bytes_per_token(model, torch.float32) * (cfg.head_dim + 2) / (4 * cfg.head_dim)   # a quarter of fp32 plus the scales
    back = dequantize_kv(q)
    for (k, v), (bk, bv) in zip(kv, back):
        for t, b in ((k, bk), (v, bv)):
            step = t.abs().amax(-1, keepdim=True) / 127
            assert b.dtype == torch.float32 and ((t - b).abs() <= 0.5 * step * (1 + 2e-3) + 1e-6).all()   # fp16 scale rounding adds ~1e-3 relative
            assert ((t - b).norm() / t.norm()).item() < 5e-3
    assert all(torch.equal(a, b) for p1, p2 in zip(back, kv_dtype(kv, INT8)) for a, b in zip(p1, p2))
    for store in (StateCache(INT8, "cpu"), StateCache(INT8, "disk", cache_dir=tmp_path / "int8")):
        seen = store.put("r", kv)
        got = store.get("r")
        assert store.bytes >= size_bytes(q) and store.stats()["dtype"] == "int8" and store.hits == 1
        assert all(torch.equal(a, b) for p1, p2 in zip(seen, back) for a, b in zip(p1, p2))   # the miss that stores sees the rounded K/V
        assert all(torch.equal(a, b) for p1, p2 in zip(got, back) for a, b in zip(p1, p2))
        if store.dir is None: assert store.entries["r"][0][0].dtype == torch.int8 and store.bytes == size_bytes(q)
    model.state_mode = "frozen"
    try:
        with torch.no_grad():
            exact = model.forward_batch([enc])[0]
            via_cache = model.forward_batch([enc], [store.get("r")])[0]
            model.state_kv_dtype = INT8
            rounded = model.forward_batch([enc])[0]
        assert all(torch.equal(a, b) for a, b in zip(via_cache, rounded))
        assert all((a - b).abs().max().item() < 5e-2 * max(a.abs().max().item(), 1) for a, b in zip(exact, rounded))
        assert any(not torch.equal(a, b) for a, b in zip(exact, rounded))   # rounding is visible, small
        model.train(); model.zero_grad(set_to_none=True)
        loss = frozen_loss_backward(model, enc, store.get("r"), lambda z, qi: F.cross_entropy(z[None], torch.tensor([enc["labels"][qi]])), 2)
        assert loss > 0 and any("lora" in n and p.grad is not None and p.grad.abs().max() > 0 for n, p in model.named_parameters())
    finally:
        model.state_mode, model.state_kv_dtype = "adapted", None; model.eval()


def lora_layer_ids(model):
    """Indices of the transformer layers holding LoRA parameters (names: base_model.model.layers.N....lora_A...)."""
    return {int(n.split(".")[3]) for n, _ in model.lm.named_parameters() if "lora_A" in n}


def test_lora_layers_top_only_and_reloads(tiny_path, tmp_path):
    """lora_layers=M puts LoRA in the top M layers only, and a saved adapter reloads the same module set and function through
    PeftModel.from_pretrained on a LoRA-less DecisionModel (what evaluate.load does)."""
    torch.manual_seed(2)
    model = DecisionModel(tiny_path, FakeTokenizer(), "cpu", lora=4, head_dim=32, lora_layers=1)
    assert lora_layer_ids(model) == {1} and model.lm.peft_config["default"].layers_to_transform == [1]
    assert all("layers.1." in n for n, p in model.lm.named_parameters() if p.requires_grad)
    perturb_lora(model); model.eval(); enc = encode(FakeTokenizer(), REC)
    with torch.no_grad():
        ref = model(enc)
        with model.lm.disable_adapter(): base = model(enc)
    assert any(not torch.equal(a, b) for a, b in zip(ref, base))   # the adapter is live
    run = tmp_path / "run"; model.lm.save_pretrained(run)
    from peft import PeftModel
    loaded = DecisionModel(tiny_path, FakeTokenizer(), "cpu", lora=None, head_dim=32)
    loaded.lm = PeftModel.from_pretrained(loaded.lm, run); loaded.head.load_state_dict(model.head.state_dict()); loaded.eval()
    assert lora_layer_ids(loaded) == {1}
    with torch.no_grad():
        got = loaded(enc)
    assert all(torch.equal(a, b) for a, b in zip(ref, got))
    with pytest.raises(ValueError, match="lora_layers"):
        DecisionModel(tiny_path, FakeTokenizer(), "cpu", lora=4, lora_layers=3)


def test_state_grad_off_matches_detached_adapted_state(tiny_path):
    """state_grad=False: the packed forward is unchanged, its LoRA gradients differ from plain packed training (the hooks act)
    and equal those of the branch pass over the state K/V computed with the adapter and detached."""
    torch.manual_seed(3)
    plain = DecisionModel(tiny_path, FakeTokenizer(), "cpu", lora=4, head_dim=32)
    hooked = DecisionModel(tiny_path, FakeTokenizer(), "cpu", lora=4, head_dim=32, state_grad=False)
    perturb_lora(plain); hooked.load_state_dict(plain.state_dict()); plain.eval(); hooked.eval()   # eval: no LoRA dropout, so passes agree
    enc = encode(FakeTokenizer(), REC); Q = len(enc["labels"])
    def loss_of(zs): return sum(F.cross_entropy(z.float()[None], torch.tensor([enc["labels"][q]])) for q, z in enumerate(zs)) / Q
    plain.zero_grad(set_to_none=True); loss_of(plain(enc)).backward(); g_plain = grads(plain)
    hooked.zero_grad(set_to_none=True); zs = hooked(enc); loss_of(zs).backward(); g_hook = grads(hooked)
    assert hooked.current_state_len is None
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(zs, plain(enc)))   # same forward function
    hooked.zero_grad(set_to_none=True)
    kv = state_kv(hooked, enc["ids"][:state_len(enc)], adapter=True, grad=False)
    loss_of(branch_logits(hooked, enc, kv)).backward(); g_cached = grads(hooked)
    lora = lambda g: {n: t for n, t in g.items() if "lora" in n}
    assert set(g_plain) == set(g_hook) == set(g_cached) and grad_gap(lora(g_plain), lora(g_hook))[1] > 1e-3   # the state's share is gone
    worst, rel = grad_gap(g_cached, g_hook)
    assert worst < 1e-3 and rel < 1e-4
    with pytest.raises(ValueError, match="state_grad"):
        hooked.hidden_batch([enc, enc])


def test_cache_projection_counts_each_record_once():
    from kev.train import cache_projection
    from kev.data import materialize
    q = {"ok": {"type": "noul", "instructions": "Is it fine?", "label": 1, "src": "t"}}
    reqs = [{"_meta": {"id": "a"}, "state": "hello world", "questions": q}, {"_meta": {"id": "b"}, "state": "hi", "questions": q},
            {"_meta": {"id": "a"}, "state": "hello world", "questions": q}]   # synthetic_repeat duplicates share one cache entry
    tok = FakeTokenizer(); per_token = 8
    n_rec, n_tok, n_bytes = cache_projection(tok, reqs, per_token)
    expected = sum(len(tok(materialize(r)["state"], add_special_tokens=False).input_ids) + 1 for r in reqs[:2])
    assert (n_rec, n_tok, n_bytes) == (2, expected, expected * per_token)


def cached(repo):
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


@pytest.mark.skipif(not cached(BASE), reason=f"{BASE} not in the HF cache")
def test_qwen3_base_frozen_and_chunked_match_packed():
    from kev.model import load_tokenizer
    tok = load_tokenizer(BASE)
    model = DecisionModel(BASE, tok, "cpu", lora=16)
    enc = encode(tok, REC, strict=True)
    out = check_equivalence(model, enc)
    assert_equivalent(out)
    assert out["hidden_rel"] < 1e-4 and out["chunk1_grad_rel_l2"] < 1e-4


@pytest.mark.skipif(not cached(BASE), reason=f"{BASE} not in the HF cache")
def test_train_frozen_branch_chunk_smoke(tmp_path):
    """kev.train --state_mode frozen --branch_chunk 1 on the smoke suite (CPU, ~1 min) finishes and writes training_metrics.json
    with both the state pass (cache miss) and the branch chunks timed."""
    out = tmp_path / "smoke-frozen-chunk"
    cmd = [sys.executable, "-m", "kev.train", "--suite", "evals/v3/smoke-v3", "--epochs", "1", "--accum", "4", "--device", "cpu",
           "--state_mode", "frozen", "--branch_chunk", "1", "--out", str(out)]
    p = subprocess.run(cmd, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    m = read_json(out / "training_metrics.json")
    assert m["state_mode"] == "frozen" and m["branch_chunk"] == 1 and m["records_seen"] == m["requested_records"] > 0
    assert m["state_seconds"] > 0 and m["branch_seconds"] > 0 and m["state_tokens_computed"] > 0 and (out / "head.pt").exists()
