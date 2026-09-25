"""Agentic RL after SFT: group-relative policy gradient on multi-step episodes (kev.envs), from a trained checkpoint.

    uv run python -m kev.rl --init_from jaredpalmer/kev-4b --out runs/rl/kev-4b-inv --device cuda

Why not REINFORCE on single decisions: its reward is a proper score of one answer, whose optimum is the log loss SFT
already minimises; sampling only adds variance. Here the reward is the episode's: which records Kev opened, whether it
answered or escalated, and whether the answer was right. That credit assignment across steps is what SFT has no label for.

Policy: Kev's served distribution over the step's actions, softmax(z / tau) with tau the parent's calibration temperature
(unless --tau), so rollouts sample and the gradient moves the distribution that is actually served. Per iteration:
- `--episodes` worlds x `--group` rollouts each; advantage = (return - group mean) / (group std + eps) (GRPO);
- loss = mean over steps of -A log pi(a) + kl_w KL(pi || pi_sft) (exact over the options, at the same tau),
  plus replay_w x the SFT loss (kev.train.question_loss, raw logits as in SFT) on `--replay` records.
The KL anchor and the replay are the calibration guards: policy gradient sharpens a distribution toward whatever was
rewarded, and the SFT head's probabilities are the thing Kev sells. Evaluation (held-out seeds and template) reports
return, outcomes, the confidence of answer steps against their correctness (ECE, Brier) and the policy entropy; the
checkpoint keeps the parent's temperature and must be refitted (scripts/calibrate_checkpoint.py) before any served read.
"""
import argparse
import contextlib
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .api import SystemOneRequest, to_record
from .checkpoint import Checkpoint, LoadOptions, Meta, write_meta
from .data import load_records, materialize
from .device import default_device
from .envs import STEP_COST, investigation
from .metrics import ece
from .model import DecisionModel, fits, load_tokenizer
from .suite import write_json
from .train import MAX_GRAD_NORM, question_loss


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init_from", required=True, help="the SFT checkpoint (run directory or Hub id) the policy starts from and is anchored to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=16, help="worlds per iteration")
    ap.add_argument("--group", type=int, default=8, help="rollouts per world (the group the advantage is relative to)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=0.0, help="pointer head learning rate; 0 = --lr")
    ap.add_argument("--kl_w", type=float, default=0.05, help="weight of KL(pi || pi_sft) per step")
    ap.add_argument("--tau", type=float, default=0.0, help="policy temperature; 0 = the parent's calibration temperature")
    ap.add_argument("--replay", default="", help="JSONL of labelled requests (kev.data.load_records) replayed with the SFT loss")
    ap.add_argument("--replay_w", type=float, default=0.5)
    ap.add_argument("--replay_batch", type=int, default=8, help="replay records per iteration")
    ap.add_argument("--microbatch", type=int, default=8, help="step encodings per forward")
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--n_people", type=int, default=6)
    ap.add_argument("--max_hops", type=int, default=2)
    ap.add_argument("--p_redact", type=float, default=0.25)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16", help="autocast dtype on CUDA")
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=1)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def step_record(ep):
    """The episode's current step as an internal record, and the action keys in option order."""
    rec, meta = to_record(SystemOneRequest.model_validate(ep.request()))
    return rec, meta[0]["keys"]


def logits_of(model, encs, microbatch, autocast):
    out = []
    for i in range(0, len(encs), microbatch):
        with autocast: out += [z[0].float() for z in model.forward_batch(encs[i:i + microbatch])]
    return out


@torch.no_grad()
def rollout(policy, ref, tok, episodes, tau, rng, greedy=False, microbatch=8, autocast=contextlib.nullcontext()):
    """Run every episode to its end. -> per episode a list of steps {"enc", "action", "ref", "p"} (ref: the SFT policy's
    log-probabilities at tau; p: the policy's probabilities the action was drawn from)."""
    policy.eval()
    trajs = [[] for _ in episodes]
    while active := [i for i, ep in enumerate(episodes) if not ep.done]:
        steps = [step_record(episodes[i]) for i in active]
        encs = [policy.encode(tok, rec, strict=True) for rec, _ in steps]
        zs = logits_of(policy, encs, microbatch, autocast)
        zr = logits_of(ref, encs, microbatch, autocast) if ref is not None else [None] * len(encs)
        for i, (_, keys), enc, z, r in zip(active, steps, encs, zs, zr):
            p = F.softmax(z / tau, -1).cpu()
            a = int(p.argmax()) if greedy else rng.choices(range(len(p)), weights=p.tolist())[0]
            trajs[i].append({"enc": enc, "action": a, "keys": keys, "p": p, "ref": None if r is None else F.log_softmax(r / tau, -1).cpu()})
            episodes[i].step(keys[a])
    return trajs


def advantages(returns, group):
    """Group-relative advantage: each rollout's return against the other rollouts of the same world."""
    out = []
    for g in range(0, len(returns), group):
        rs = returns[g:g + group]; mu = statistics.fmean(rs); sd = statistics.pstdev(rs)
        out += [(r - mu) / (sd + 1e-6) if sd > 0 else 0.0 for r in rs]
    return out


def policy_loss(z, step, advantage, tau, kl_w):
    """-A log pi(a) + kl_w KL(pi || pi_sft) for one step, pi = softmax(z / tau). -> (loss, kl, entropy)"""
    logp = F.log_softmax(z / tau, -1)
    ref = step["ref"].to(z.device)
    kl = (logp.exp() * (logp - ref)).sum()
    entropy = -(logp.exp() * logp).sum()
    return -advantage * logp[step["action"]] + kl_w * kl, kl, entropy


def episode_return(ep, traj):
    """What the episode earned: the terminal reward less the cost of every record opened."""
    return ep.reward - STEP_COST * sum(s["keys"][s["action"]].startswith("open ") for s in traj)


def evaluate(policy, tok, a, tau, greedy, autocast):
    """Held-out worlds (the eval seed namespace and template). Answer-step confidence is the policy's probability of the
    chosen answer renormalised over the answer actions: how sure Kev is of the value, apart from whether to answer."""
    eps = [investigation(s, "eval", a.n_people, a.max_hops, a.p_redact) for s in range(a.eval_episodes)]
    trajs = rollout(policy, None, tok, eps, tau, random.Random(a.seed + 1), greedy, a.microbatch, autocast)
    outcomes = Counter((ep.answer() is not None, ep.outcome) for ep in eps)
    conf, ok, ent = [], [], []
    for ep, traj in zip(eps, trajs):
        ent += [float(-(s["p"] * s["p"].clamp_min(1e-12).log()).sum()) for s in traj]
        last = traj[-1]
        if ep.outcome in ("correct", "wrong"):
            answer = [i for i, k in enumerate(last["keys"]) if k.startswith("answer ")]
            conf.append(float(last["p"][last["action"]] / last["p"][answer].sum())); ok.append(ep.outcome == "correct")
    knowable = sum(ep.answer() is not None for ep in eps)
    return {"mode": "greedy" if greedy else "sampled", "n": len(eps), "return": statistics.fmean(episode_return(e, t) for e, t in zip(eps, trajs)),
            "knowable_correct": outcomes[(True, "correct")] / max(knowable, 1),
            "knowable_escalated": outcomes[(True, "escalate")] / max(knowable, 1),
            "unknowable_escalated": outcomes[(False, "escalate")] / max(len(eps) - knowable, 1),
            "out_of_budget": sum(v for (_, o), v in outcomes.items() if o == "out_of_budget") / len(eps),
            "opens": statistics.fmean(sum(s["keys"][s["action"]].startswith("open ") for s in t) for t in trajs),
            "answer_n": len(ok), "answer_acc": float(np.mean(ok)) if ok else None, "answer_conf": float(np.mean(conf)) if conf else None,
            "answer_ece": ece(conf, ok) if ok else None, "answer_brier": float(np.mean((np.asarray(conf) - np.asarray(ok)) ** 2)) if ok else None,
            "entropy": statistics.fmean(ent)}


def load_policy(a, dev):
    """(tokenizer, trainable policy warm-started from --init_from, frozen SFT reference, the parent's Meta)."""
    ck = Checkpoint(a.init_from)
    parent = ck.meta
    if parent.weights != "lora": raise ValueError("kev.rl trains a LoRA on the parent's adapter; full-weight parents are not supported yet")
    tok = load_tokenizer(parent.base, revision=parent.base_revision)
    targets = parent.extra.get("args", {}).get("lora_targets", "all")
    policy = DecisionModel(parent.base, tok, dev, lora=parent.lora, revision=parent.base_revision, head_dim=parent.head_dim,
                           special_embeddings=parent.special_embeddings, lora_targets=targets)
    ours = Meta(base=parent.base, base_revision=parent.base_revision, lora=parent.lora, head_dim=parent.head_dim,
                special_embeddings=parent.special_embeddings, holdout=parent.holdout, temperature=parent.temperature)
    init_source = ck.warm_start(policy, ours)
    if a.checkpointing: policy.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.lm.config.use_cache = False
    _, ref = ck.load(dev, LoadOptions(temperature=1.0))
    for p in ref.parameters(): p.requires_grad_(False)
    return tok, policy, ref, ours, init_source


def main(argv=None):
    a = parse_args(argv)
    dev = a.device or default_device()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" and a.dtype == "bf16" else contextlib.nullcontext()
    tok, policy, ref, meta, init_source = load_policy(a, dev)
    policy.head.temperature = 1.0   # the policy applies tau itself; the served temperature goes back into head.pt
    tau = a.tau or meta.temperature
    replay = [r for r in (materialize(q) for q in load_records(a.replay, "replay")) if fits(r, tok)] if a.replay else []
    write_json(out / "rl_config.json", {"args": vars(a), "tau": tau, "init_source": init_source, "replay_records": len(replay)})
    print(f"kev.rl: tau={tau:.3f}, {len(replay)} replay records", flush=True)

    head_ids = {id(p) for p in policy.head.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in policy.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
                             {"params": list(policy.head.parameters()), "lr": a.head_lr or a.lr}], weight_decay=0.0)
    log, evals, t0 = [], [], time.time()

    def run_eval(it):
        for greedy in (True, False):
            evals.append({"iter": it, **evaluate(policy, tok, a, tau, greedy, autocast)}); print(evals[-1], flush=True)
        write_json(out / "evals.json", evals)

    run_eval(0)
    for it in range(1, a.iters + 1):
        seeds = [(it - 1) * a.episodes + k for k in range(a.episodes)]
        eps = [investigation(s, "train", a.n_people, a.max_hops, a.p_redact) for s in seeds for _ in range(a.group)]
        trajs = rollout(policy, ref, tok, eps, tau, rng, False, a.microbatch, autocast)
        returns = [episode_return(ep, t) for ep, t in zip(eps, trajs)]
        adv = advantages(returns, a.group)
        steps = [(s, A) for t, A in zip(trajs, adv) for s in t]
        policy.train(); opt.zero_grad()
        terms, replayed = Counter(), 0
        for i in range(0, len(steps), a.microbatch):
            chunk = steps[i:i + a.microbatch]
            zs = logits_of(policy, [s["enc"] for s, _ in chunk], a.microbatch, autocast)
            loss = 0.0
            for z, (s, A) in zip(zs, chunk):
                l, kl, h = policy_loss(z, s, A, tau, a.kl_w); loss = loss + l; terms["kl"] += kl.item(); terms["entropy"] += h.item()
            (loss / len(steps)).backward()
        if replay:
            batch = rng.sample(replay, min(a.replay_batch, len(replay))); replayed = len(batch)
            for i in range(0, len(batch), a.microbatch):
                recs = batch[i:i + a.microbatch]
                with autocast: logits = policy.forward_batch([policy.encode(tok, r) for r in recs])
                ce = sum(sum(question_loss(z.float(), q, dev, 0.0) for z, q in zip(zz, r["questions"])) / len(zz) for zz, r in zip(logits, recs))
                (a.replay_w * ce / len(batch)).backward(); terms["replay_ce"] += ce.item()
        norm = float(torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), MAX_GRAD_NORM))
        opt.step()
        outcomes = Counter(ep.outcome for ep in eps)
        log.append({"iter": it, "return": statistics.fmean(returns), "steps": len(steps), "kl": terms["kl"] / len(steps), "entropy": terms["entropy"] / len(steps),
                    "replay_ce": terms["replay_ce"] / max(replayed, 1), "grad_norm": norm,
                    "outcomes": {k: v / len(eps) for k, v in outcomes.items()}, "seconds": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
        if it % a.eval_every == 0 or it == a.iters: run_eval(it)
        write_json(out / "rl_log.json", log)
        if not all(math.isfinite(log[-1][k]) for k in ("return", "kl", "entropy")): raise ValueError("non-finite RL statistics")

    policy.lm.save_pretrained(out)
    meta.head = policy.head.state_dict()
    meta.extra = {"args": vars(a), "init_source": init_source, "rl": {"tau": tau, "temperature_refit_needed": True}}
    write_meta(out, meta); tok.save_pretrained(out)
    write_json(out / "report.json", {"clean": {"parent": evals[:2], "final": evals[-2:]}, "wall_seconds": time.time() - t0})


if __name__ == "__main__":
    main()
