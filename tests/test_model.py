"""Numerical parity of the model's serving paths, on real weights: merged vs unmerged LoRA, prefix cache vs full pass,
shape-bucket padding, row form vs packed mask, hybrid isolation, and the --init_from warm start end to end.
Needs the smoke checkpoint (runs/smoke-hl/00-trial-0/checkpoint) and downloads Qwen/Qwen2.5-0.5B (the hybrid test also
Qwen/Qwen3.5-0.8B-Base); not run in CI.
Run: uv run --extra serve python -m pytest tests/test_model.py -q
"""
import os

import pytest

SMOKE = "runs/smoke-hl/00-trial-0/checkpoint"
KEV_08B = "jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e"   # pinned (round 15): the bf16 bars below are per checkpoint


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


@pytest.mark.parametrize("device,dtype", [("cpu", "float32"), ("cuda", "float32"), ("cuda", "bfloat16")])
def test_shared_prefix_matches_rows(device, dtype):
    """kev.shared_prefix on Qwen3.5-0.8B-Base, gradient checkpointing on: over records with 1-4 questions and states of
    unequal length (so states are left-padded), the logits and every parameter's gradient equal the row form's, as closely
    as the row form agrees with itself when only the batch changes (the same records one at a time). On CUDA (run it on
    Modal, modal_app.py::gpu_tests) fla's Triton kernels differentiate through the prefix's `initial_state` and their fp32
    dots round like TF32, so that noise is measured, not assumed; on the CPU transformers' reference kernels are exact to
    fp32 and the floor applies."""
    import importlib.util
    import torch
    from kev.data import materialize
    from kev.model import DecisionModel, load_tokenizer
    from kev.suite import load_split
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("needs CUDA")
    if device == "cpu" and importlib.util.find_spec("causal_conv1d"): pytest.skip("transformers sends CPU tensors to causal-conv1d's CUDA kernel when it is installed")
    tok = load_tokenizer("Qwen/Qwen3.5-0.8B-Base")
    m = DecisionModel("Qwen/Qwen3.5-0.8B-Base", tok, device, dtype=getattr(torch, dtype)).train()
    m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dev = load_split("evals/v7/decision-v7", "development")
    recs = [materialize(r) for r in dev if len(r["questions"]) > 1][:2] + [materialize(dev[0])]
    long = materialize(dev[5]); long["state"] = " ".join(materialize(r)["state"] for r in dev[:20])
    long["questions"] = [q for r in recs for q in r["questions"]][:4]
    encs = [m.encode(tok, r) for r in [*recs, long]]

    def run(shared, batches):
        m.zero_grad(); logits = []
        for batch in batches:
            zs = [z for zz in m.forward_batch(batch, shared) for z in zz]
            sum(torch.log_softmax(z.float(), -1)[(len(logits) + i) % len(z)] for i, z in enumerate(zs)).backward(); logits += zs
        return torch.cat(logits).float().detach(), {n: p.grad.float().clone() for n, p in m.named_parameters() if p.grad is not None}

    (rows, g_rows), (alone, g_alone), (prefix, g_prefix) = run(False, [encs]), run(False, [[e] for e in encs]), run(True, [encs])
    big = [k for k in g_rows if g_rows[k].norm() > 1e-3 * max(g.norm() for g in g_rows.values())]
    rel = lambda g: max(float((g_rows[k] - g[k]).norm() / g_rows[k].norm()) for k in big)
    noise, diff = (float((rows - alone).abs().max()), rel(g_alone)), (float((rows - prefix).abs().max()), rel(g_prefix))
    print(f"{device} {dtype}: max |logit diff| rows vs rows one record at a time {noise[0]:.2e}, rows vs shared prefix {diff[0]:.2e}; "
          f"worst relative gradient diff {noise[1]:.2e} vs {diff[1]:.2e}")
    assert g_rows.keys() == g_prefix.keys() and diff[0] <= max(3e-4, 3 * noise[0]) and diff[1] <= max(1e-3, 3 * noise[1])
    # Magnitudes alone would let a small systematic shift pass as noise. Signed checks: (1) the shared prefix is as close
    # to the second batching of the row form as to the first; (2) the mean signed logit difference sits within the
    # batching noise's own mean plus 3 standard errors; (3) the gradient difference along the gradient itself (a
    # systematic scaling) is no larger than the batching's and well under the whole difference's size.
    assert float((alone - prefix).abs().max()) <= max(3e-4, 3 * noise[0])
    d_noise, d_prefix = alone - rows, prefix - rows
    bias_bound = float(d_noise.mean().abs() + 3 * d_noise.std() / len(d_noise) ** 0.5) + 3e-5   # floor: a tenth of the magnitude floor
    flat = lambda g: torch.cat([g[k].flatten() for k in big])
    along = lambda g: float(torch.dot(flat(g) - flat(g_rows), flat(g_rows)) / flat(g_rows).square().sum())
    print(f"  mean signed logit diff {float(d_prefix.mean()):+.2e} (bound {bias_bound:.2e}); gradient shift along itself {along(g_prefix):+.2e} (batching {along(g_alone):+.2e})")
    assert float(d_prefix.mean().abs()) <= bias_bound
    assert abs(along(g_prefix)) <= max(1e-4, 3 * abs(along(g_alone)), 0.3 * noise[1])


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
    tok, m = Checkpoint(KEV_08B).load("cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True, fused=True))
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


def test_server_recovers_when_a_pass_runs_out_of_memory():
    """kev.serve.Server._run on CUDA, served as kev.serve serves it (bf16, CUDA graphs, fused kernels): a new state whose
    pass cannot fit beside a full prefix cache raises a genuine torch.OutOfMemoryError inside the model, the server drops
    the cache and the retry answers as the unpressured server did (#132, #75). Afterwards a cache hit and a new state still
    match (the graphs and fused kernels survived the failed pass). Under a cap an empty cache cannot meet either, the error
    reaches the caller, the cache is left empty and the failed pass's tensors are freed (the model thread used to keep them
    until its next batch). The pressure is a memory cap (set_per_process_memory_fraction) placed halfway across what
    dropping the cache frees, so both outcomes have that half as margin (H100: 652 MiB freed, 326 each side; the pass adds 158)."""
    import torch
    if not torch.cuda.is_available(): pytest.skip("needs CUDA")
    from types import MethodType
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev import cuda_graphs
    from kev.device import empty_cache, sync
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    from kev.serve import Server
    ck = Checkpoint(KEV_08B)
    tok, m = ck.load("cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True, fused=True))
    qs = [{"instr": "Which team should handle this?", "options": ["returns", "shipping", "billing", "other"], "label": 0},
          {"instr": "Is a refund owed?", "options": ["yes", "no"], "label": 0}]
    # states past the graphed state pass (an eager state pass, the path a long document takes) that still fit a bank
    # entry (graphed question rows); ~55 MiB of prefix each on Kev-0.8B
    rec = lambda i: {"state": f"Ticket {i}. " + f"Order {4400 + i} arrived late and the box was crushed. Two charges appear on the card. " * 170, "questions": qs}
    fills, target, after = [rec(i) for i in range(12)], rec(100), rec(101)
    enc = lambda r: m.encode(tok, r, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
    n = enc(target)["seg"].count(0)
    assert cuda_graphs.GRAPH_STATE < n <= cuda_graphs.BANK_WIDTH, n
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    s = Server(ck, tok, m, "cuda")
    attempts = []   # per probs_batch call: [bytes allocated when it starts, the type of the exception it raised or None]
    def probs_batch(self, *a):   # keeps only the type: an exception kept would keep its pass's tensors
        attempts.append([torch.cuda.memory_allocated(), None])
        try: return type(self).probs_batch(self, *a)
        except Exception as e: attempts[-1][1] = type(e); raise
    m.probs_batch = MethodType(probs_batch, m)

    def settle():   # model thread idle, freed blocks returned: -> bytes reserved
        s.wait_idle()
        with s.lock: sync("cuda"); empty_cache("cuda"); return torch.cuda.memory_reserved()

    def fill():
        s.prefix_cache.clear()
        for r in fills: s.probs(r)
        assert len(s.prefix_cache.entries) == len(fills)
        return settle()

    close = lambda a, b: all((x - y).abs().max() < 0.05 for x, y in zip(map(torch.tensor, a), map(torch.tensor, b)))
    cap = lambda b: torch.cuda.set_per_process_memory_fraction(b / total)
    key = lambda r: s.prefix_cache.plan([enc(r)])[0][0]
    try:
        s.prefix_cache.size = len(fills)
        refs = {id(r): s.probs(r)[0] for r in fills + [after]}   # the unpressured answers (graphs captured along the way)
        s.prefix_cache.clear(); empty = settle(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        refs[id(target)] = s.probs(target)[0]
        need = torch.cuda.max_memory_reserved() - empty            # what the target's pass adds with nothing cached
        full = fill(); freed = full - empty                        # what dropping the cache gives back
        mib = lambda b: f"{b / 2**20:.0f} MiB"
        print(f"\nreserved {mib(empty)} with an empty cache ({mib(base)} allocated), {mib(full)} with {len(fills)} states cached; "
              f"the target's {n}-token pass adds {mib(need)}; cap {mib(full + need - freed // 2)} of {mib(total)}")
        assert freed > 2**29 and need > 0, (freed, need)            # a quarter of a GiB of margin on each side at least
        attempts.clear(); cap(full + need - freed // 2)            # the pass cannot fit beside the cache, and fits without it
        ps, stats = s.probs(target)
        print(f"attempts (allocated at start, error): {[(mib(b), e and e.__name__) for b, e in attempts]}")
        assert [e for _, e in attempts] == [torch.OutOfMemoryError, None], attempts   # a genuine OOM, then the retry
        assert attempts[1][0] - base < 2**24, "the retry started with the cache or the failed pass still resident"
        assert s.prefix_cache.oom_retries == 1 and list(s.prefix_cache.entries) == [key(target)] and not stats["prefix_cache_hit"]
        assert close(ps, refs[id(target)])
        cap(total)
        done = [s.submit(r) for r in (target, after)]              # after the OOM: a cache hit and a new state
        for r, (ps, stats), hit in zip((target, after), [d.result() for d in done], (True, False)):
            assert stats["prefix_cache_hit"] is hit and close(ps, refs[id(r)])
        assert list(s.prefix_cache.entries) == [key(target), key(after)]
        s.prefix_cache.clear(); settle(); base = torch.cuda.memory_allocated()
        full = fill(); attempts.clear(); cap(full - freed + need // 2)   # half the room even with the cache dropped: the error reaches the caller
        with pytest.raises(torch.OutOfMemoryError): s.probs(target)
        s.wait_idle(); held = torch.cuda.memory_allocated() - base
        print(f"failed twice: attempts {[(mib(b), e and e.__name__) for b, e in attempts]}, {mib(held)} still allocated past the empty server")
        assert [e for _, e in attempts] == [torch.OutOfMemoryError] * 2, attempts
        assert s.prefix_cache.entries == {} and s.prefix_cache.oom_retries == 2
        assert held < 2**20, "the failed batch's tensors outlive it"
        attempts.clear(); cap(total)
        ps, stats = s.probs(after)                                 # and the server still answers
        assert close(ps, refs[id(after)]) and list(s.prefix_cache.entries) == [key(after)]
        assert m.graphs.captures > 0 and not m.graphs.failed, m.graphs.stats()
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        s.close()


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
    assert hb.extra["init_source"]["weights_sha256"] and read_json(tmp_path / "b/training_config.json")["init_source"]["resolved"] == str(tmp_path / "a")
    bad = subprocess.run(base + ["--out", str(tmp_path / "c"), "--init_from", str(tmp_path / "a"), "--lora", "8"], capture_output=True, text=True, env=env)
    assert bad.returncode != 0 and "lora is 16 there and 8 here" in bad.stderr
