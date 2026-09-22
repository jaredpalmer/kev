"""Equivalence checks of kev.cached_state against the packed forward, on a real backbone (the probe's three checks):

(1) cached state, adapter disabled everywhere: branch pass over the base state K/V == packed forward at the readout positions
(2) chunked branches, adapter enabled (lora_B perturbed): gradients == packed gradients; without the push-back through the
    state graph they are wrong (control)
(3) frozen state mode: every LoRA tensor trains, the cache carries no grad; wall time of state pass vs branch pass vs packed,
    at the record's own state and at a long (repeated) state

Run: cd <repo> && PYTHONPATH=. uv run python scripts/exp_equivalence.py --device cpu --base Qwen/Qwen3-0.6B-Base --out runs/equivalence.json
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
from kev.model import DecisionModel, load_tokenizer, encode
LONG_STATE, LONG_BRANCH = 4096, 8192   # the long-state probe deliberately exceeds the training context (kev.model.MAX_STATE)
from kev.cached_state import branch_hidden, branch_logits, chunked_loss_backward, size_bytes, state_kv, state_len, sync

STATE = ("Incident report INC-4471. At 02:13 UTC the checkout service began returning HTTP 503 for roughly 40% of requests "
         "in the eu-west region. The on-call engineer observed that the primary Postgres instance had hit its connection limit "
         "after a deploy of the order-history worker opened one pool per process instead of one pool per host. Latency on the "
         "load balancer stayed normal and the cache hit rate was unchanged. The worker was rolled back at 02:41 UTC and "
         "connections drained within six minutes. No data was lost. Customers who retried succeeded; about 1,900 orders were "
         "delayed. A follow-up task was filed to add a connection-count alert and to cap the pool size in the worker's configuration.")
REC = {"state": STATE, "questions": [
    {"instr": "Is this incident resolved?", "options": ["yes", "no", "unclear"], "label": 0},
    {"instr": "Which component failed?", "options": ["database", "load balancer", "cache", "network"], "label": 0},
    {"instr": "How severe was the incident?", "options": ["low", "medium", "high", "critical"], "label": 0}]}


def grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def grad_gap(ref, other, min_scale=1e-6):
    """Worst per-tensor max|diff|/max|ref| (tensors whose reference gradient is ~0, e.g. head.k.bias, are skipped) and the
    global relative L2 over all tensors."""
    worst, name, num, den = 0.0, None, 0.0, 0.0
    for n, g in ref.items():
        o = other.get(n)
        if o is None: return float("inf"), n, float("inf")
        num += (g - o).pow(2).sum().item(); den += g.pow(2).sum().item()
        scale = g.abs().max().item()
        if scale >= min_scale and (g - o).abs().max().item() / scale > worst: worst, name = (g - o).abs().max().item() / scale, n
    return worst, name, (num ** 0.5) / (den ** 0.5 + 1e-30)


def timed(fn, device, n):
    """Best-of-n and median wall seconds (device synchronized); the last result."""
    fn(); ts = []
    for _ in range(n):
        sync(device); t = time.perf_counter(); r = fn(); sync(device); ts.append(time.perf_counter() - t)
    return min(ts), sorted(ts)[n // 2], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", default="runs/equivalence.json")
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--long_repeat", type=int, default=7, help="the state repeated this many times for the long-state timing (0 = skip)")
    ap.add_argument("--timing_n", type=int, default=3)
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)
    tok = load_tokenizer(a.base)
    t0 = time.perf_counter(); model = DecisionModel(a.base, tok, dev, lora=a.lora); model.eval()   # dropout off: same function on both paths
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n: p.normal_(0, 0.02)   # PEFT starts lora_B at 0: the adapter would be the identity
    enc = encode(tok, REC, strict=True); S = state_len(enc); Q = len(enc["labels"])
    sel = enc["decide_idx"] + [i for oi in enc["opt_idx"] for i in oi]
    res = {"env": {"device": dev, "base": a.base, "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                   "attn": model.lm.config._attn_implementation, "load_s": time.perf_counter() - t0},
           "encoding": {"S": S, "L": len(enc["ids"]), "Q": Q, "readout_positions": len(sel)}}
    rows = []

    # (1) cached state == packed, adapter disabled everywhere
    with torch.no_grad(), model.lm.disable_adapter():
        model.state_mode = "adapted"; h_ref = model.hidden(enc); z_ref = model(enc)
        model.state_mode = "frozen"; h_frozen = model.hidden(enc); z_frozen = model(enc)
        kv = state_kv(model, enc["ids"][:S], adapter=False)
        h_one = torch.cat([branch_hidden(model, enc, kv, [q])[0] for q in range(Q)])
        z_zero = branch_logits(model, enc, [(torch.zeros_like(k), torch.zeros_like(v)) for k, v in kv])
    model.state_mode = "adapted"
    mag = h_ref[sel].abs().max().item()
    r1 = {"ref_absmax_at_readout": mag,
          "maxabs_all_branches_one_pass": (h_frozen[sel] - h_ref[sel]).abs().max().item(),
          "maxabs_one_branch_at_a_time": (h_one[[i - S for i in sel]] - h_ref[sel]).abs().max().item(),
          "maxabs_all_branch_tokens": (h_frozen[S:] - h_ref[S:]).abs().max().item(),
          "logits_maxabs": max((x - y).abs().max().item() for x, y in zip(z_frozen, z_ref)),
          "control_logits_maxabs_zero_state": max((x - y).abs().max().item() for x, y in zip(z_zero, z_ref))}
    res["1_cached_state_equivalence"] = r1
    rows += [("(1) hidden max-abs diff, all branches one pass", r1["maxabs_all_branches_one_pass"]),
             ("(1) hidden max-abs diff, one branch at a time", r1["maxabs_one_branch_at_a_time"]),
             ("(1) hidden ref max-abs at readout", mag), ("(1) logits max-abs diff", r1["logits_maxabs"]),
             ("(1) control: zeroed state K/V, logits max-abs diff", r1["control_logits_maxabs_zero_state"])]

    # (2) chunked gradients == packed gradients, adapter enabled
    def q_loss(z, q): return F.cross_entropy(z.float()[None], torch.tensor([enc["labels"][q]], device=dev)) / Q
    model.zero_grad(set_to_none=True)
    z_packed = model(enc); loss_ref = sum(q_loss(z, q) for q, z in enumerate(z_packed)); loss_ref.backward(); g_ref = grads(model)
    r2 = {"loss_packed": loss_ref.item()}
    rows.append(("(2) packed loss", loss_ref.item()))
    for chunk in (1, 2, Q):
        model.zero_grad(set_to_none=True); zs = {}
        def q_loss_keep(z, q): zs[q] = z.detach(); return q_loss(z, q)
        loss = chunked_loss_backward(model, enc, q_loss_keep, chunk)
        worst, name, rel_l2 = grad_gap(g_ref, grads(model))
        r2[f"chunk{chunk}"] = {"loss": loss, "logits_maxabs": max((zs[q] - z_packed[q].detach()).abs().max().item() for q in range(Q)),
                               "grad_max_rel_inf": worst, "grad_worst_tensor": name, "grad_global_rel_l2": rel_l2}
        rows += [(f"(2) chunk={chunk} loss", loss), (f"(2) chunk={chunk} grad max per-tensor rel-inf", worst), (f"(2) chunk={chunk} grad global rel L2", rel_l2)]
    model.zero_grad(set_to_none=True)
    kv_g = state_kv(model, enc["ids"][:S], adapter=True, grad=True)
    leaves = [(k.detach().requires_grad_(True), v.detach().requires_grad_(True)) for k, v in kv_g]
    for q in range(Q): q_loss(branch_logits(model, enc, leaves, [q])[0], q).backward()
    lora_ref = {n: g for n, g in g_ref.items() if "lora" in n}
    worst, _, rel_l2 = grad_gap(lora_ref, grads(model))
    r2["control_no_pushback"] = {"lora_grad_max_rel_inf": worst, "lora_grad_global_rel_l2": rel_l2}
    rows.append(("(2) control: no push-back, LoRA grad global rel L2", rel_l2))
    res["2_chunked_grad_equivalence"] = r2

    # (3) frozen mode: trains every LoRA tensor; timings
    def frozen_kv(): return state_kv(model, enc["ids"][:S], adapter=False)
    def frozen_fwd_bwd():
        model.zero_grad(set_to_none=True); l = sum(q_loss(z, q) for q, z in enumerate(branch_logits(model, enc, kv_f))); l.backward(); return l
    def packed_fwd_bwd():
        model.zero_grad(set_to_none=True); l = sum(q_loss(z, q) for q, z in enumerate(model(enc))); l.backward(); return l
    def state_grad(): return state_kv(model, enc["ids"][:S], adapter=True, grad=True)
    t_state, t_state_med, kv_f = timed(frozen_kv, dev, a.timing_n)
    t_fb, t_fb_med, loss_f = timed(frozen_fwd_bwd, dev, a.timing_n)
    g_f = {n: g for n, g in grads(model).items() if "lora" in n}
    n_lora = sum(1 for n, p in model.named_parameters() if p.requires_grad and "lora" in n)
    t_pb, t_pb_med, _ = timed(packed_fwd_bwd, dev, a.timing_n)
    t_sg, t_sg_med, _ = timed(state_grad, dev, a.timing_n)
    r3 = {"loss": loss_f.item(), "lora_tensors": n_lora, "lora_nonzero_grad": sum(int(g.abs().max() > 0) for g in g_f.values()),
          "cache_requires_grad": any(k.requires_grad or v.requires_grad for k, v in kv_f), "cache_bytes_fp32": size_bytes(kv_f),
          "t_state_pass_base_nograd_s": [t_state, t_state_med], "t_frozen_branch_fwd_bwd_s": [t_fb, t_fb_med],
          "t_packed_fwd_bwd_s": [t_pb, t_pb_med], "t_state_pass_adapter_grad_s": [t_sg, t_sg_med]}
    rows += [("(3) frozen loss", loss_f.item()), ("(3) LoRA tensors with nonzero grad / total", f"{r3['lora_nonzero_grad']}/{n_lora}"),
             ("(3) cache requires grad", r3["cache_requires_grad"]), ("(3) cache bytes fp32", size_bytes(kv_f)),
             (f"(3) S={S}: base state pass (no grad) s", t_state), (f"(3) S={S}: frozen branch fwd+bwd s", t_fb),
             (f"(3) S={S}: packed fwd+bwd s", t_pb), (f"(3) S={S}: adapted state pass (grad) s", t_sg)]
    res["3_frozen_state_mode"] = r3
    if a.long_repeat:
        enc_l = encode(tok, {**REC, "state": " ".join([STATE] * a.long_repeat)}, max_state=LONG_STATE, max_branch=LONG_BRANCH, strict=True)
        S_l = state_len(enc_l)
        def frozen_kv_l(): return state_kv(model, enc_l["ids"][:S_l], adapter=False)
        def frozen_fwd_bwd_l():
            model.zero_grad(set_to_none=True); l = sum(q_loss(z, q) for q, z in enumerate(branch_logits(model, enc_l, kv_l))); l.backward(); return l
        def packed_fwd_bwd_l():
            model.zero_grad(set_to_none=True); l = sum(q_loss(z, q) for q, z in enumerate(model(enc_l))); l.backward(); return l
        t1, t1m, kv_l = timed(frozen_kv_l, dev, a.timing_n); t2, t2m, _ = timed(frozen_fwd_bwd_l, dev, a.timing_n); t3, t3m, _ = timed(packed_fwd_bwd_l, dev, a.timing_n)
        res["3b_long_state"] = {"S": S_l, "L": len(enc_l["ids"]), "t_state_pass_base_nograd_s": [t1, t1m], "t_frozen_branch_fwd_bwd_s": [t2, t2m],
                                "t_packed_fwd_bwd_s": [t3, t3m], "cache_bytes_fp32": size_bytes(kv_l)}
        rows += [(f"(3b) S={S_l}: base state pass (no grad) s", t1), (f"(3b) S={S_l}: frozen branch fwd+bwd s", t2), (f"(3b) S={S_l}: packed fwd+bwd s", t3),
                 (f"(3b) S={S_l}: cache bytes fp32", size_bytes(kv_l))]

    width = max(len(r[0]) for r in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value:.6g}" if isinstance(value, float) else f"{name:<{width}}  {value}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f: json.dump(res, f, indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
