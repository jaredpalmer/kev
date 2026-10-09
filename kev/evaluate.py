"""Tests of the architectural hypotheses on held-out data (all through the serving-format path).

1. accuracy + ECE per source / per question type (Score also reports MAE of the expected level)
   vs a zero-shot letter-logit baseline from the same base model (K<=8)
2. permutation stability (argmax flip rate, prob spread of correct option) on Choice
3. IIA: does appending an irrelevant option shift log-odds between two existing options?
4. isolation: can question B read a secret placed in sibling question A? (must not) / in state (should)
5. latency + equality: packed N questions vs N separate calls
"""
import argparse, json, math, random, time, string
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from .checkpoint import Checkpoint, LoadOptions
from .data import NONE_OPTIONS, build, augment, materialize, DISTRACTORS
from .device import DEVICE_HELP, DEVICES, empty_cache, select, sync
from .metrics import ece
from .suite import write_json


def _probs(tok, model, req):
    rec = materialize(req)
    return rec, model.probs(model.encode(tok, rec, strict=True))


def _one(req, qid):
    return {"state": req["state"], "questions": {qid: req["questions"][qid]}}


def test_accuracy(tok, model, reqs, rng):
    by = defaultdict(lambda: {"conf": [], "ok": [], "nll": [], "mae": []})
    for r in reqs:
        try: rec, ps = _probs(tok, model, augment(r, rng, p_none=0, p_none_distract=0, p_distract=0))
        except ValueError as exc: raise ValueError("Evaluation rejected an example; refusing partial metrics") from exc
        for q, p in zip(rec["questions"], ps):
            p = p.numpy(); y = q["label"]; d = by[q["src"]]
            d["conf"].append(float(p.max())); d["ok"].append(int(p.argmax() == y)); d["nll"].append(-math.log(max(p[y], 1e-9)))
            if q["qtype"] == "score": d["mae"].append(abs(float((np.arange(len(p)) * p).sum()) - y))
    res = {}
    for s, d in by.items():
        res[s] = {"n": len(d["ok"]), "acc": float(np.mean(d["ok"])), "ece": ece(d["conf"], d["ok"]), "nll": float(np.mean(d["nll"])), "mean_conf": float(np.mean(d["conf"]))}
        if d["mae"]: res[s]["score_mae_levels"] = float(np.mean(d["mae"]))
    allc = sum((d["conf"] for d in by.values()), []); allo = sum((d["ok"] for d in by.values()), [])
    res["ALL"] = {"n": len(allo), "acc": float(np.mean(allo)), "ece": ece(allc, allo), "mean_conf": float(np.mean(allc))}
    return res


def test_permutation(tok, model, reqs, rng, n_perm=4):
    flips, spreads, n = 0, [], 0
    for r in reqs:
        for qid, q in r["questions"].items():
            if q["type"] != "choice" or len(q["criteria"]) < 3: continue
            argmaxes, pcorrect = [], []
            for _ in range(n_perm):
                keys = list(q["criteria"]); rng.shuffle(keys)
                q2 = {**q, "criteria": {k: q["criteria"][k] for k in keys}}
                try: _, ps = _probs(tok, model, {"state": r["state"], "questions": {qid: q2}})
                except ValueError as exc: raise ValueError("Permutation evaluation rejected an example") from exc
                p = ps[0].numpy(); argmaxes.append(keys[int(p.argmax())]); pcorrect.append(float(p[keys.index(q["label"])]))
            if len(pcorrect) == n_perm:
                n += 1; flips += int(len(set(argmaxes)) > 1); spreads.append(max(pcorrect) - min(pcorrect))
    return {"n": n, "argmax_flip_rate": flips / max(n, 1), "mean_prob_spread": float(np.mean(spreads)), "p90_prob_spread": float(np.percentile(spreads, 90))}


def test_iia(tok, model, reqs, rng):
    """Log-odds between the top-2 original options, before vs after appending an irrelevant option."""
    shifts = []
    for r in reqs:
        for qid, q in r["questions"].items():
            if q["type"] != "choice" or not 3 <= len(q["criteria"]) <= 10: continue
            try:
                _, ps0 = _probs(tok, model, {"state": r["state"], "questions": {qid: q}})
                k = rng.choice(list(DISTRACTORS))
                _, ps1 = _probs(tok, model, {"state": r["state"], "questions": {qid: {**q, "criteria": {**q["criteria"], k: DISTRACTORS[k]}}}})
            except ValueError as exc: raise ValueError("Evaluation rejected an example; refusing partial metrics") from exc
            p0, p1 = ps0[0].numpy(), ps1[0].numpy()
            a, b = np.argsort(-p0)[:2]
            shifts.append(math.log(max(p1[a], 1e-6) / max(p1[b], 1e-6)) - math.log(max(p0[a], 1e-6) / max(p0[b], 1e-6)))
    s = np.array(shifts)
    return {"n": len(s), "mean_abs_logodds_shift": float(np.abs(s).mean()), "mean_shift": float(s.mean()), "p90_abs_shift": float(np.percentile(np.abs(s), 90))}


def test_none_of_the_above(tok, model, reqs, rng):
    """Add a 'none of the above' option. When the true option is still present it should get little mass;
    when the true option is removed it should be chosen. A shortcut model picks it in both cases."""
    p_present, hit_present, hit_absent, n = [], 0, 0, 0
    for r in reqs:
        for qid, q in r["questions"].items():
            if q["type"] != "choice" or len(q["criteria"]) < 3: continue
            nk, nd = rng.choice(NONE_OPTIONS)
            if nk in q["criteria"]: continue
            try:
                present = {**q, "criteria": {**q["criteria"], nk: nd}}
                _, ps = _probs(tok, model, {"state": r["state"], "questions": {qid: present}})
                keys = list(present["criteria"]); p = ps[0].numpy()
                none_probability = float(p[keys.index(nk)])
                present_correct = int(keys[int(p.argmax())] == q["label"])
                absent = {**q, "label": nk, "criteria": {k: v for k, v in present["criteria"].items() if k != q["label"]}}
                _, ps = _probs(tok, model, {"state": r["state"], "questions": {qid: absent}})
                keys = list(absent["criteria"]); p = ps[0].numpy()
                p_present.append(none_probability); hit_present += present_correct
                hit_absent += int(keys[int(p.argmax())] == nk); n += 1
            except ValueError as exc: raise ValueError("Evaluation rejected an example; refusing partial metrics") from exc
    return {"n": n, "true_option_present": {"mean_p_none": float(np.mean(p_present)), "p_none_over_0.5_rate": float(np.mean(np.array(p_present) > 0.5)), "acc": hit_present / max(n, 1)},
            "true_option_removed": {"picks_none_rate": hit_absent / max(n, 1)}}


def test_isolation(tok, model, rng, n=20):
    """Secret code in sibling question vs in state. Probe asks which code was mentioned."""
    res = {"sibling": [], "state": [], "absent": []}
    weather = "The weather is nice today and the park is full of people."
    for _ in range(n):
        codes = ["".join(rng.choices(string.ascii_uppercase, k=5)) + "-" + str(rng.randint(1000, 9999)) for _ in range(3)]
        keys = ["c0", "c1", "c2", "none"]; rng.shuffle(keys)
        crit = {k: (codes[int(k[1])] if k != "none" else "None of these codes appears") for k in keys}
        probe = {"type": "choice", "instructions": "Which secret code is mentioned in this request?", "criteria": crit, "label": "c0", "src": "probe"}
        sib = {"type": "noul", "instructions": f"The secret code for this request is {codes[0]}. Is the weather described as nice?", "label": True, "src": "x"}
        sib0 = {"type": "noul", "instructions": "Is the weather described as nice?", "label": True, "src": "x"}
        for cond, state, a in [("sibling", weather, sib), ("absent", weather, sib0), ("state", weather + f" The secret code for this request is {codes[0]}.", sib0)]:
            _, ps = _probs(tok, model, {"state": state, "questions": {"a": a, "probe": probe}})
            res[cond].append(float(ps[1].numpy()[keys.index("c0")]))
    return {k: {"mean_p_code": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in res.items()}


def test_packed_vs_separate(tok, model, reqs, rng, n=30):
    diffs, t_pack, t_sep, nq = [], 0.0, 0.0, 0
    for r in [x for x in reqs if len(x["questions"]) >= 2][:n]:
        enc = model.encode(tok, materialize(r))   # legacy prototype eval (not used for releases): truncates at the default training context by design
        sync(model.device); t = time.time(); pp = model.probs(enc); sync(model.device); t_pack += time.time() - t
        for qi, qid in enumerate(r["questions"]):
            e1 = model.encode(tok, materialize(_one(r, qid)))
            sync(model.device); t = time.time(); p1 = model.probs(e1)[0]; sync(model.device); t_sep += time.time() - t
            diffs.append(float((pp[qi] - p1).abs().max())); nq += 1
    return {"n_questions": nq, "max_abs_prob_diff": float(np.max(diffs)), "mean_abs_prob_diff": float(np.mean(diffs)), "packed_s": t_pack, "separate_s": t_sep, "speedup": t_sep / t_pack}


def baseline_letter_logits(base, reqs, dev, rng, chat=False):
    """Zero-shot MCQ baseline on the same rendered text: read next-token logits over option letters. K<=8.
    chat=True wraps the prompt in the model's chat template (for -Instruct models) and asks for the letter only."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base); lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(dev).eval()
    letters = list("ABCDEFGH")
    letter_ids = [tok(("" if chat else " ") + L, add_special_tokens=False).input_ids[0] for L in letters]
    by = defaultdict(lambda: {"conf": [], "ok": []})
    with torch.no_grad():
        for r in reqs:
            rec = materialize(augment(r, rng, p_none=0, p_none_distract=0, p_distract=0))
            for q in rec["questions"]:
                K = len(q["options"])
                if K > 8: continue
                body = f"{rec['state']}\n\nQuestion: {q['instr']}\n" + "".join(f"{letters[i]}. {o}\n" for i, o in enumerate(q["options"]))
                if chat:
                    prompt = tok.apply_chat_template([{"role": "user", "content": body + "\nAnswer with the letter of the best option only."}], tokenize=False, add_generation_prompt=True)
                else:
                    prompt = body + "Answer:"
                ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1200).to(dev)
                p = F.softmax(lm(**ids).logits[0, -1, letter_ids[:K]], -1).cpu().numpy()
                by[q["src"]]["conf"].append(float(p.max())); by[q["src"]]["ok"].append(int(p.argmax() == q["label"]))
    del lm
    return {s: {"n": len(d["ok"]), "acc": float(np.mean(d["ok"])), "ece": ece(d["conf"], d["ok"]), "mean_conf": float(np.mean(d["conf"]))} for s, d in by.items()}


def test_temperature(tok, model, reqs, rng):
    """Fit one global temperature on even-indexed records (by NLL), report NLL/ECE on odd-indexed ones before and after."""
    fit, held = [], []
    with torch.no_grad():
        for i, r in enumerate(reqs):   # legacy prototype eval (not used for releases): encode truncates at the default training context by design
            try: rec = materialize(augment(r, rng, p_none=0, p_none_distract=0, p_distract=0)); zs = model(model.encode(tok, rec))
            except ValueError as exc: raise ValueError("Evaluation rejected an example; refusing partial metrics") from exc
            for z, q in zip(zs, rec["questions"]):
                (fit if i % 2 == 0 else held).append((z.cpu(), q["label"]))
    logT = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)
    def nll(pairs, T): return torch.stack([F.cross_entropy((z / T)[None], torch.tensor([y])) for z, y in pairs]).mean()
    def closure():
        opt.zero_grad(); l = nll(fit, logT.exp()); l.backward(); return l
    opt.step(closure); T = float(logT.exp())
    def stats(pairs, T):
        ps = [F.softmax(z / T, -1) for z, _ in pairs]
        return {"nll": float(nll(pairs, T)), "ece": ece([float(p.max()) for p in ps], [int(p.argmax() == y) for p, (_, y) in zip(ps, pairs)])}
    return {"T": T, "n_fit": len(fit), "n_held": len(held), "before": stats(held, 1.0), "after": stats(held, T)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kev")
    ap.add_argument("--n_per_source", type=int, default=150)
    ap.add_argument("--baseline", action="store_true", help="zero-shot letter-logit baseline from the base model")
    ap.add_argument("--baseline_instruct", default="", help="e.g. Qwen/Qwen2.5-0.5B-Instruct: chat-template letter-logit baseline")
    ap.add_argument("--device", choices=("auto", *DEVICES), default="auto", help=DEVICE_HELP)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    opts = LoadOptions.from_env()
    dev = select(a.device) if opts.dtype is None else select(a.device, dtype=opts.dtype)
    rng = random.Random(a.seed)
    reqs = build(a.n_per_source, "test", a.seed)
    ck = Checkpoint(a.run)
    out = {"run": a.run, "holdout_sources": ck.meta.holdout}
    if a.baseline:
        out["baseline_zero_shot_base"] = baseline_letter_logits(ck.meta.base, reqs, dev, random.Random(a.seed)); print(json.dumps(out, indent=1), flush=True)
    if a.baseline_instruct:
        out["baseline_zero_shot_instruct"] = baseline_letter_logits(a.baseline_instruct, reqs, dev, random.Random(a.seed), chat=True); print(json.dumps(out["baseline_zero_shot_instruct"], indent=1), flush=True)
    empty_cache(dev)
    tok, model = ck.load(dev, opts)
    for name, fn in [("accuracy_calibration", lambda: test_accuracy(tok, model, reqs, random.Random(a.seed))),
                     ("temperature_scaling", lambda: test_temperature(tok, model, reqs, random.Random(a.seed))),
                     ("permutation", lambda: test_permutation(tok, model, reqs[:150], rng)),
                     ("iia", lambda: test_iia(tok, model, reqs[:250], rng)),
                     ("none_of_the_above", lambda: test_none_of_the_above(tok, model, reqs[:200], rng)),
                     ("isolation", lambda: test_isolation(tok, model, rng)),
                     ("packed_vs_separate", lambda: test_packed_vs_separate(tok, model, reqs, rng))]:
        t = time.time(); out[name] = fn(); print(f"== {name} ({time.time()-t:.0f}s)\n{json.dumps(out[name], indent=1)}", flush=True)
    if out["holdout_sources"]:
        acc = out["accuracy_calibration"]
        out["held_out_sources"] = {s: acc[s] for s in acc if any(s.startswith(h) for h in out["holdout_sources"])}
        print("== held_out_sources (never seen in training)\n" + json.dumps(out["held_out_sources"], indent=1))
    write_json(f"{a.run}/eval.json", out)


if __name__ == "__main__":
    main()
