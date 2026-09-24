"""Numerical parity of the model's serving paths, on real weights: merged vs unmerged LoRA, prefix cache vs full pass,
shape-bucket padding, row form vs packed mask, hybrid isolation, and the --init_from warm start end to end.
Needs the smoke checkpoint (runs/smoke-hl/00-trial-0/checkpoint) and downloads Qwen/Qwen2.5-0.5B (the hybrid test also
Qwen/Qwen3.5-0.8B-Base); not run in CI.
Run: uv run --extra serve python -m pytest tests/test_model.py -q
"""
import os

import pytest

SMOKE = "runs/smoke-hl/00-trial-0/checkpoint"


@pytest.fixture
def smoke_run():
    if not os.path.exists(f"{SMOKE}/head.pt"): pytest.skip("smoke checkpoint not present")
    return SMOKE


def test_merged_load_matches_unmerged_exactly_in_fp32(smoke_run):
    import torch
    from kev.checkpoint import LoadOptions, load
    from kev.data import materialize
    from kev.suite import load_split
    recs = [materialize(r) for r in load_split("evals/smoke-v1", "development")[:3]]
    tok, a = load(smoke_run, "cpu", LoadOptions(merge=False)); _, b = load(smoke_run, "cpu", LoadOptions(merge=True))
    with torch.no_grad():
        for r in recs:
            pa, pb = torch.cat(a.probs(a.encode(tok, r))), torch.cat(b.probs(b.encode(tok, r)))
            assert (pa - pb).abs().max() < 1e-5

def test_bf16_merge_equals_fp32_merge_then_cast(smoke_run):
    """A bf16 load merges the fp32 adapter straight into the bf16 weights (one rounding in fp32 math), which must give the
    same bits as the old path: load in fp32, merge, cast. That path held an fp32 copy of the backbone (36 GB for Kev-9B)."""
    import torch
    from peft import PeftModel
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import DecisionModel
    ck = Checkpoint(smoke_run)
    tok, new = ck.load("cpu", LoadOptions(dtype=torch.bfloat16))
    old = DecisionModel(ck.meta.base, tok, "cpu", revision=ck.meta.base_revision, head_dim=ck.meta.head_dim)
    old = PeftModel.from_pretrained(old.lm, ck.path, torch_device="cpu").merge_and_unload().to(torch.bfloat16)
    a, b = old.state_dict(), new.lm.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

def test_prefix_cache_matches_full_pass(smoke_run):
    import torch
    from kev.checkpoint import load
    from kev.data import materialize
    from kev.suite import load_split
    tok, m = load(smoke_run, "cpu")
    recs = [materialize(r) for r in load_split("evals/smoke-v1", "development")[:3]]
    for r in recs:
        enc = m.encode(tok, r); full = torch.cat(m.probs(enc))
        prefix = m.prefix(enc)
        a = torch.cat(m.probs_with_prefix(enc, prefix)); b = torch.cat(m.probs_with_prefix(enc, prefix))   # reuse twice: crop() must restore the cache
        assert (full - a).abs().max() < 1e-4 and (a - b).abs().max() < 1e-6
        p2, prefix2 = m.probs_and_prefix(enc)                                                               # single-pass miss path
        assert (torch.cat(p2) - full).abs().max() < 1e-5 and prefix2[0] == prefix[0] and (prefix2[2] - prefix[2]).abs().max() < 1e-3 * prefix[2].abs().max()
        assert (torch.cat(m.probs_with_prefix(enc, prefix2)) - full).abs().max() < 1e-4
        # a different question set on the same state also reuses the prefix
        r2 = {**r, "questions": r["questions"][:1]}; enc2 = m.encode(tok, r2)
        assert (torch.cat(m.probs(enc2)) - torch.cat(m.probs_with_prefix(enc2, prefix))).abs().max() < 1e-4

def test_shape_bucket_padding_is_exact_in_fp32(smoke_run):
    import torch
    from kev.checkpoint import load
    from kev.data import materialize
    from kev.suite import load_split
    tok, m = load(smoke_run, "cpu")
    recs = [materialize(r) for r in load_split("evals/smoke-v1", "development")[:3]]
    from kev.model import branch_mask_batch
    for r in recs:
        enc = m.encode(tok, r); L = len(enc["ids"]); padded = -(-L // 64) * 64
        with torch.no_grad():
            h = m.hidden_batch([enc])[0, :L]
            ids = torch.full((1, padded), m.pad_id); ids[0, :L] = torch.tensor(enc["ids"]); pos = torch.zeros((1, padded), dtype=torch.long); pos[0, :L] = torch.tensor(enc["pos"])
            hp = m.lm(input_ids=ids, position_ids=pos, attention_mask=branch_mask_batch([enc["seg"]], "cpu", length=padded)).last_hidden_state[0, :L]
        assert (h - hp).abs().max() < 1e-4 * h.abs().max()

def test_train_path_drops_records_that_exceed_the_context():
    """Issue #5: training without --suite built records straight from the datasets and the strict encoder aborted on the
    first long passage. The on-the-fly path now applies the same context filter that suite freezing applies."""
    from kev.data import materialize
    from kev.model import fits, load_tokenizer
    tok = load_tokenizer("Qwen/Qwen2.5-0.5B")
    short = {"state": "s " * 10, "questions": {"q": {"type": "noul", "instructions": "i", "label": True, "src": "t"}}, "_meta": {"id": "a", "source": "t"}}
    long = {**short, "state": "word " * 600}
    assert fits(materialize(short), tok) and not fits(materialize(long), tok)

def test_rows_match_packed():
    """The row form (state + one branch per causal row) must reproduce the packed block-causal form on an attention-only
    backbone: each row holds exactly the tokens its question may attend to, at the same positions."""
    import torch
    from kev.model import DecisionModel, load_tokenizer, rows_of
    tok = load_tokenizer("Qwen/Qwen2.5-0.5B"); m = DecisionModel("Qwen/Qwen2.5-0.5B", tok, "cpu").eval()
    rec = {"state": "Order 4411 arrived two weeks late and the box was crushed. Two charges appear on the card.",
           "questions": [{"instr": "Is there a billing problem?", "options": ["yes", "no"], "label": 0},
                         {"instr": "Which team should handle this?", "options": ["returns", "shipping", "billing", "other"], "label": 2},
                         {"instr": "How upset is the customer?", "options": ["calm", "annoyed", "furious"], "label": 1}]}
    enc = m.encode(tok, rec)
    S, Sp, rows = rows_of(enc)
    assert len(rows) == 3 and all(r["ids"][-1] == enc["ids"][d] for r, d in zip(rows, enc["decide_idx"]))
    with torch.no_grad():
        packed = [torch.softmax(z, -1) for z in m._readout(m.hidden(enc), enc)]
        rowed = [torch.softmax(z, -1) for z in m.forward_rows_batch([enc])[0]]
    for a, b in zip(packed, rowed):
        assert (a - b).abs().max() < 1e-4, (a, b)

def test_row_batching_and_packed_fallback_do_not_change_answers(smoke_run, monkeypatch):
    """Serving bounds memory by running rows a token budget at a time (rows_per_pass) and by switching an attention-only
    backbone from the packed mask to rows once the packed sequence exceeds one serving row. Neither may move a probability:
    one row per pass against the default, and the forced row form (full pass, prefix miss and prefix hit) against the packed
    pass, on many questions."""
    import torch
    from kev import model as M
    from kev.checkpoint import load
    from kev.data import materialize
    from kev.suite import load_split
    tok, m = load(smoke_run, "cpu")
    base = materialize(load_split("evals/smoke-v1", "development")[0])
    rec = {**base, "questions": base["questions"] * 5}                  # 5x the questions: several ROW_BATCH chunks
    enc = m.encode(tok, rec)
    with torch.no_grad():
        packed = torch.cat(m.probs(enc)); prefix_packed = m.prefix(enc)
        monkeypatch.setattr(M, "SERVE_MAX_PACKED", len(enc["ids"]) - 1)   # now "too long to pack": every path takes the row form
        assert m.rows_form([enc])
        rows_full = torch.cat(m.probs(enc)); rows_miss, prefix_rows = m.probs_and_prefix(enc)
        rows_hit_from_packed_prefix = torch.cat(m.probs_with_prefix(enc, prefix_packed))   # a prefix made by the packed pass, reused by rows
        rows_hit = torch.cat(m.probs_with_prefix(enc, prefix_rows))
        monkeypatch.setattr(M, "rows_per_pass", lambda rows, prefix_len=0, budget=0: 1)   # one row per pass
        one_at_a_time = torch.cat(m.probs(enc)); one_at_a_time_hit = torch.cat(m.probs_with_prefix(enc, prefix_rows))
    for got in (rows_full, torch.cat(rows_miss), rows_hit_from_packed_prefix, rows_hit, one_at_a_time, one_at_a_time_hit):
        assert (got - packed).abs().max() < 1e-4


def test_hybrid_rows_isolation_and_prefix():
    """Qwen3.5 (Gated DeltaNet + attention): the row form isolates questions exactly, and the serving prefix path
    (state once, cache replicated per question) reproduces it. Uses the 0.8B base; slow reference kernels on CPU."""
    import torch
    from kev.model import DecisionModel, load_tokenizer
    tok = load_tokenizer("Qwen/Qwen3.5-0.8B-Base"); m = DecisionModel("Qwen/Qwen3.5-0.8B-Base", tok, "cpu").eval()
    assert m.hybrid
    rec = {"state": "Order 4411 arrived late and the box was crushed. Two charges appear on the card.",
           "questions": [{"instr": "Is there a billing problem?", "options": ["yes", "no"], "label": 0},
                         {"instr": "Which team should handle this?", "options": ["returns", "shipping", "billing", "other"], "label": 2}]}
    enc = m.encode(tok, rec)
    with torch.no_grad():
        together = m.probs(enc)
        alone = [m.probs(m.encode(tok, {"state": rec["state"], "questions": [q]}))[0] for q in rec["questions"]]
        cached, prefix = m.probs_and_prefix(enc)
        again = m.probs_with_prefix(enc, prefix); again2 = m.probs_with_prefix(enc, prefix)
        import kev.model as M
        saved, M.rows_per_pass = M.rows_per_pass, lambda rows, prefix_len=0, budget=0: 1   # one row per pass: same answers, bounded memory
        try: chunked = m.probs_with_prefix(enc, prefix)
        finally: M.rows_per_pass = saved
    for a, b, c, d, e, f in zip(together, alone, cached, again, again2, chunked):
        assert (a - b).abs().max() < 1e-4 and (a - c).abs().max() < 1e-4 and (a - d).abs().max() < 1e-4 and (a - e).abs().max() < 1e-4 and (a - f).abs().max() < 1e-4

def test_probs_routes_hybrid_through_prefix_not_forward_rows_batch(monkeypatch):
    """probs() on a hybrid backbone must reach the state-prefix path (state encoded once), not forward_rows_batch
    (state re-encoded per question): issue #77. The test above already shows the two paths agree numerically;
    this one shows probs() actually takes the cheap one rather than merely matching it by coincidence."""
    import torch
    from kev import model as M
    tok = M.load_tokenizer("Qwen/Qwen3.5-0.8B-Base"); m = M.DecisionModel("Qwen/Qwen3.5-0.8B-Base", tok, "cpu").eval()
    assert m.hybrid
    rec = {"state": "Order 4411 arrived late and the box was crushed.",
           "questions": [{"instr": "Is there a billing problem?", "options": ["yes", "no"], "label": 0}]}
    enc = m.encode(tok, rec)
    calls = {"rows_batch": 0, "prefix": 0}
    orig_rows, orig_prefix = M.DecisionModel.forward_rows_batch, M.DecisionModel.probs_and_prefix
    def spy_rows(self, encs):
        calls["rows_batch"] += 1; return orig_rows(self, encs)
    def spy_prefix(self, enc):
        calls["prefix"] += 1; return orig_prefix(self, enc)
    monkeypatch.setattr(M.DecisionModel, "forward_rows_batch", spy_rows)
    monkeypatch.setattr(M.DecisionModel, "probs_and_prefix", spy_prefix)
    with torch.no_grad():
        m.probs(enc)
    assert calls == {"rows_batch": 0, "prefix": 1}

def test_cuda_graphs_match_eager():
    """kev.cuda_graphs + kev.fused_qwen35 (CUDA only): the served path (probs_batch) gives the eager bf16 answers up to
    bf16 noise, first with its new buckets run eagerly and then replayed, for new states (two sharing one) and cached
    ones (made either way), with more rows than one pass holds, a state past GRAPH_STATE (its own eager state pass, then
    batched rows) and a row past GRAPH_ROW (the plain eager path). scripts/serving_bench.py measures the same on 200
    records against fp32."""
    import torch
    if not torch.cuda.is_available(): pytest.skip("needs CUDA")
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev import cuda_graphs
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    tok, m = Checkpoint("jaredpalmer/kev-0.8b").load("cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True, fused=True))
    q = {"instr": "Which team should handle this?", "options": ["returns", "shipping", "billing", "other"], "label": 0}
    recs = [{"state": "Order 4411 arrived late and the box was crushed. Two charges appear on the card." * k, "questions": [q] * n}
            for k, n in ((1, 1), (1, 3), (4, cuda_graphs.GRAPH_ROWS + 3), (20, 2), (300, 2))]
    recs.append({"state": "short", "questions": [{**q, "instr": "word " * (cuda_graphs.GRAPH_ROW + 10)}]})
    graphs = m.graphs
    with torch.no_grad():
        encs = [m.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH) for rec in recs + recs[:2]]
        m.graphs = None; refs = [m.probs(e) for e in encs]; eager_prefix = m.prefix(encs[2])
        m.graphs = graphs
        new_prefixes = None
        for _ in range(2):   # first run: new buckets run eagerly; second: replayed
            got, new_prefixes = m.probs_batch(encs, [None] * len(encs), [True] * len(encs))
            graphs.capture_pending()
        cached = list(new_prefixes); cached[2] = eager_prefix                    # cached states, one of them made eagerly
        hits, same = m.probs_batch(encs, cached, [True] * len(encs))
        for ref, ps, hs, p, c in zip(refs, got, hits, new_prefixes, same):
            assert p is not None and c is p or c is eager_prefix
            assert all((a - b).abs().max() < 0.05 for a, b in zip(ref, ps)) and all((a - b).abs().max() < 0.05 for a, b in zip(ref, hs))
    assert graphs.captures > 0 and not graphs.pending

def test_init_from_warm_start_and_compatibility_checks(tmp_path):
    """PR #9: --init_from loads an existing adapter + pointer head before training and refuses incompatible sources.
    Two tiny runs on Qwen2.5-0.5B: the second warm-starts from the first and must start with identical head weights."""
    import subprocess, sys, json, torch
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    base = [sys.executable, "-m", "kev.train", "--n_per_source", "3", "--epochs", "1", "--accum", "1", "--batch", "1", "--device", "cpu", "--lr", "1e-12", "--base", "Qwen/Qwen2.5-0.5B"]
    subprocess.run(base + ["--out", str(tmp_path / "a")], check=True, capture_output=True, env=env)
    r = subprocess.run(base + ["--out", str(tmp_path / "b"), "--init_from", str(tmp_path / "a")], check=True, capture_output=True, text=True, env=env)
    assert "delta: warm start" in r.stdout
    from kev.checkpoint import read_meta
    from kev.suite import read_json
    ha, hb = read_meta(tmp_path / "a"), read_meta(tmp_path / "b")
    assert all((ha.head[k] - hb.head[k]).abs().max() < 1e-6 for k in ha.head), "a warm start at a negligible lr must keep the source head"
    assert hb.extra["init_source"]["adapter_sha256"] and read_json(tmp_path / "b/training_config.json")["init_source"]["resolved"] == str(tmp_path / "a")
    bad = subprocess.run(base + ["--out", str(tmp_path / "c"), "--init_from", str(tmp_path / "a"), "--lora", "8"], capture_output=True, text=True, env=env)
    assert bad.returncode != 0 and "lora is 16 there and 8 here" in bad.stderr
