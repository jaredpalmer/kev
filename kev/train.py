"""LoRA fine-tune of the decision model on labelled requests (a frozen suite's training partition, records built on the
fly from the public sources, or your own JSONL), with the pointer head trained from scratch. `--full_ft 1` trains the
whole backbone instead (kev.full_ft: bf16 weights, fp32 masters; several GPUs through torchrun + FSDP2).

    uv run python -m kev.train --suite evals/v7/decision-v7 --out runs/<name>          # what studies run
    uv run python -m kev.train --n_per_source 40 --accum 4 --out runs/smoke               # ~1 min smoke test
    uv run python -m kev.train --data mine.jsonl --init_from jaredpalmer/kev-4b --lr 2e-5 --out runs/mine   # delta
    torchrun --standalone --nproc_per_node 8 -m kev.train --full_ft 1 --weights_dtype bf16 --dtype bf16 --checkpointing 1 ...

Batch size is small (variable-length records with custom masks) and gradients are accumulated over --accum micro-batches
(per rank: a step sees accum x batch x world size records).
"""
import argparse, contextlib, json, math, os, random, resource, shutil, sys, time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from . import full_ft
from .checkpoint import Checkpoint, Meta, write_meta
from .device import allocated_bytes, default_device, empty_cache, sync
from .data import EVAL_ONLY, build, augment, load_records, materialize, none_pair, source_seed
from .suite import SYNTHETIC_SOURCES, digest, load_split, read_json, read_manifest, validate_training, write_json
from .model import MAX_STATE, MAX_TRAIN_STATE, DecisionModel, fits, load_tokenizer, rows_of, training_context


# --- losses -----------------------------------------------------------------------------------------------------------

def permuted_copy(rec, rng):
    """Re-shuffle options of every Choice question with K>=3; return (record, perms) with perms[q] = new->old index or None."""
    out, perms = {"state": rec["state"], "questions": []}, []
    for q in rec["questions"]:
        if q["qtype"] == "choice" and len(q["options"]) >= 3:
            perm = list(range(len(q["options"]))); rng.shuffle(perm)
            out["questions"].append({**q, "options": [q["options"][j] for j in perm], "label": perm.index(q["label"])}); perms.append(perm)
        else:
            out["questions"].append(q); perms.append(None)
    return out, perms


def question_loss(z, q, dev, ord_w, label_smoothing=0.0, brier_w=0.0, focal_gamma=0.0):
    """Cross-entropy (or cross-entropy against a soft target when the question carries one), optionally plus the
    normalized ranked probability score for ordered levels."""
    options = (label_smoothing, brier_w, focal_gamma)
    if not all(math.isfinite(v) and v >= 0 for v in options) or label_smoothing > 1 or sum(v > 0 for v in options) > 1:
        raise ValueError("choose at most one finite, nonnegative loss modifier; smoothing must be <= 1")
    if q.get("target") is not None:
        t = torch.tensor(q["target"], device=dev, dtype=z.dtype)
        return -(t * F.log_softmax(z, -1)).sum()
    y = torch.tensor([q["label"]], device=dev)
    loss = F.cross_entropy(z[None], y, label_smoothing=label_smoothing)
    if brier_w:
        target = F.one_hot(y[0], len(z)).to(z.dtype)
        loss = loss + brier_w * (F.softmax(z, -1) - target).square().sum()
    if focal_gamma:
        loss = (1 - torch.exp(-loss)).pow(focal_gamma) * loss
    if q["qtype"] == "score" and ord_w > 0:
        p = F.softmax(z, -1)
        observed_cdf = (torch.arange(len(p) - 1, device=dev) >= q["label"]).to(p.dtype)
        loss = loss + ord_w * (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


def anchor_loss(z, q, target, dev):
    """KL(teacher || student) for one question, teacher = frozen base zero-shot distribution keyed by option key.
    Skips (returns None) when the current option set is not exactly the teacher's (e.g. a none-option was inserted)."""
    if target is None or set(target) != set(q["keys"]): return None
    t = torch.tensor([target[k] for k in q["keys"]], device=dev, dtype=torch.float32).clamp_min(1e-6); t = t / t.sum()
    return F.kl_div(F.log_softmax(z, -1), t, reduction="sum")


def permutation_kl(z1, z2, perm, dev):
    """Symmetric KL between one question's predictions under two option orders; perm maps the second order's positions
    back to the first (perms from permuted_copy)."""
    lp1 = F.log_softmax(z1, -1); lp2 = F.log_softmax(z2, -1)[torch.tensor([perm.index(j) for j in range(len(perm))], device=dev)]
    return 0.5 * (F.kl_div(lp2, lp1, log_target=True, reduction="sum") + F.kl_div(lp1, lp2, log_target=True, reduction="sum"))


def accumulation_records(n, batch, accum, microbatch):
    start = (microbatch // accum) * accum * batch
    return min(accum * batch, n - start)


# --- data -------------------------------------------------------------------------------------------------------------

def training_requests(a, tok, manifest, holdout):
    """The labelled requests one run trains on: the suite's training partition, records built from the public sources,
    or the user's own file (optionally with a replay sample from the suite); filtered to the training context, checked
    against the eval-only policy, then the ablation knobs (--train_sources, --public_frac, --synthetic_repeat)."""
    # the suite's rules (declared trainable sources, no held-out structures) apply to every record taken from it
    if a.data:
        reqs = load_records(a.data)
        if a.replay:
            pool = load_split(a.suite, "train"); replay = random.Random(f"replay:{a.seed}").sample(pool, min(a.replay, len(pool)))
            validate_training(replay, manifest)
            print(f"replay: {len(replay)} of {len(pool)} suite training records mixed with {len(reqs)} from {a.data}", flush=True)
            reqs = reqs + replay
    elif manifest:
        reqs = load_split(a.suite, "train"); validate_training(reqs, manifest)
    else:
        reqs = build(a.n_per_source, "train", a.seed, exclude=holdout)
    if not manifest or a.data:
        # frozen suites are filtered to the training context when they are frozen (kev.suite.select_unique); records built
        # on the fly here are not, so apply the same rule instead of letting the strict encoder abort the run (issue #5)
        kept = [r for r in reqs if fits(materialize(r), tok, **training_context(a.max_state))]
        if len(kept) < len(reqs):
            c = training_context(a.max_state)
            print(f"dropped {len(reqs) - len(kept)} of {len(reqs)} records that exceed the training context "
                  f"({c['max_state']} state / {c['max_branch']} branch / {c['max_packed']} packed tokens)", flush=True)
        reqs = kept
    if not reqs:
        raise ValueError("empty training set")
    eval_only = set(EVAL_ONLY) | set(manifest.get("eval_only_sources", []) if manifest else [])
    forbidden = {r["_meta"]["source"] for r in reqs} & eval_only
    if forbidden:
        raise ValueError(f"training data contains eval-only sources: {sorted(forbidden)}")
    if a.train_sources:
        wanted = set(a.train_sources.split(","))
        unknown = wanted - {r["_meta"]["source"] for r in reqs}
        if unknown: raise ValueError(f"--train_sources not in the training partition: {sorted(unknown)}")
        reqs = [r for r in reqs if r["_meta"]["source"] in wanted]
        print(f"ablation: training on {sorted(wanted)} -> {len(reqs)} records", flush=True)
    if a.public_frac < 1:
        mix_rng = random.Random(source_seed(a.seed, "public_frac"))
        public = [r for r in reqs if r["_meta"]["source"] not in SYNTHETIC_SOURCES]; synth = [r for r in reqs if r["_meta"]["source"] in SYNTHETIC_SOURCES]
        keep = sorted(mix_rng.sample(range(len(public)), int(round(a.public_frac * len(public)))))
        reqs = [public[i] for i in keep] + synth
        print(f"mix: public_frac {a.public_frac} -> {len(keep)} public + {len(synth)} synthetic records", flush=True)
    if a.synthetic_repeat > 1:
        extra = [r for r in reqs if r["_meta"]["source"] in SYNTHETIC_SOURCES] * (a.synthetic_repeat - 1)
        reqs = reqs + extra
        print(f"mix: synthetic_repeat {a.synthetic_repeat} -> +{len(extra)} records", flush=True)
    return reqs


@dataclass(eq=False)   # identity, so batch.index(v) finds this very variant
class Variant:
    """One encoded training example: an augmented copy of a source request, with the request's id and source kept for
    the anchor lookup, and optionally the same record under a second option order for the permutation KL."""
    rec: dict
    enc: dict
    request_id: str
    source: str
    permuted: tuple | None = None   # (encoding under the other order, perms from permuted_copy)
    share: float = 1.0              # the part of its variant this is, when --row_budget split the variant's questions (row_passes)

    @property
    def tokens(self):
        return len(self.enc["ids"]) + (len(self.permuted[0]["ids"]) if self.permuted else 0)


def shape(enc):
    """(state tokens, branch tokens of each question) of an encoding (kev.model.rows_of)."""
    state, _, rows = rows_of(enc)
    return len(state), [len(r["ids"]) for r in rows]


def pass_tokens(shapes, shared):
    """Padded tokens of one forward pass over records of these shapes: in the row form, rows x the longest row (the state
    once per question); with a shared prefix (kev.shared_prefix), states x the longest state + branches x the longest branch."""
    if shared: return len(shapes) * max(s for s, _ in shapes) + sum(len(b) for _, b in shapes) * max(x for _, b in shapes for x in b)
    rows = [s + x for s, branches in shapes for x in branches]
    return len(rows) * max(rows)


def question_parts(enc, budget, shared):
    """--row_budget: a record's questions in consecutive groups whose pass fits `budget` tokens (a single question always
    fits: its row is at most kev.model.MAX_TRAIN_STATE + a branch); one group without a budget."""
    state, branches = shape(enc)
    if not budget: return [list(range(len(branches)))]
    parts = [[]]
    for q in range(len(branches)):
        if parts[-1] and pass_tokens([(state, [branches[i] for i in parts[-1] + [q]])], shared) > budget: parts.append([])
        parts[-1].append(q)
    return parts


def microbatch_plan(reqs, a, world, rank):
    """This rank's micro-batches for one epoch of `reqs` (already shuffled), in order: (records, records in its optimizer
    step over all ranks, whether that step ends after it). Every rank gets the same number of micro-batches (FSDP2's
    collectives line up) and each record is seen once per epoch (the last step wraps around to fill every rank, as
    full_ft.rank_share does).
    Plain: `--batch` consecutive records of this rank's share per micro-batch, `--accum` micro-batches per step.
    --length_sort 1: each step's records (batch x accum x world) are cut, in length order, into accum x world runs of
    neighbours whose largest padded cost (pass_tokens) is as small as possible (sizes vary: one long record, or many short
    ones), and the k-th micro-batch of every rank is one of the k-th costliest runs, so micro-batches pad little and the
    ranks, which wait for the slowest at every layer, run similar loads. A step still sees the same records (the same
    gradient up to summation order). Characters stand in for tokens (known before encoding). On the SFT corpus's shapes
    (8 ranks, 128 records per step) the ranks' critical path is 1.4x the real tokens; plain length order, a fixed --batch
    of neighbours, left it 4.3x (a few 6k-token states among many short public records)."""
    if not a.length_sort:
        mine = full_ft.rank_share(reqs, rank, world)
        n = math.ceil(len(mine) / a.batch)
        return [(mine[mb * a.batch:(mb + 1) * a.batch], world * accumulation_records(len(mine), a.batch, a.accum, mb), (mb + 1) % a.accum == 0 or mb + 1 == n)
                for mb in range(n)]
    shapes = {id(r): (len(json.dumps(r["state"], ensure_ascii=False)), [len(json.dumps(q, ensure_ascii=False)) for q in r["questions"].values()]) for r in reqs}
    cost = lambda rs: pass_tokens([shapes[id(r)] for r in rs], a.shared_prefix)
    plan, per_step = [], a.batch * a.accum * world
    for start in range(0, len(reqs), per_step):
        step = reqs[start:start + per_step]
        m = math.ceil(len(step) / (world * a.batch))   # micro-batches per rank: accum, fewer in a short last step
        step += step[:max(0, world * m - len(step))]    # so no micro-batch is empty; the repeats count in len(step), the
                                                         # step's loss normaliser, as rank_share's do in the plain path
        runs = sorted(balanced_runs(sorted(step, key=lambda r: cost([r]), reverse=True), world * m, cost), key=cost, reverse=True)
        plan += [(runs[k * world + rank], len(step), k == m - 1) for k in range(m)]
    return plan


def balanced_runs(items, n, cost):
    """`items` cut into exactly n consecutive non-empty runs, the costliest as cheap as a greedy cut allows (binary search
    on the cap); when the cap leaves fewer runs, the costliest splittable run is halved until there are n."""
    def cut(cap):
        runs = [[]]
        for item in items:
            if runs[-1] and cost(runs[-1] + [item]) > cap: runs.append([])
            runs[-1].append(item)
        return runs
    low, high = max(cost([item]) for item in items), cost(items)
    while low < high:
        mid = (low + high) // 2
        if len(cut(mid)) <= n: high = mid
        else: low = mid + 1
    runs = cut(low)
    while len(runs) < n:
        run = max((r for r in runs if len(r) > 1), key=cost)
        at = runs.index(run); runs[at:at + 1] = [run[:len(run) // 2], run[len(run) // 2:]]
    return runs


def row_passes(batch, budget, shared):
    """--row_budget: a micro-batch's variants in forward/backward passes of at most `budget` padded tokens (pass_tokens),
    longest first; [batch] without a budget. The loss is a sum over variants (a split record's parts carry their share of
    its mean), so the gradient is the same sum, but only one pass's activations are alive at a time: on one H200 a 27B's
    bf16 weights and gradients leave ~35 GB; the first probe, with passes of up to 16k tokens, ran out of memory."""
    if not budget: return [batch]
    passes = []   # [variants, their shapes]
    for v in sorted(batch, key=lambda v: pass_tokens([shape(v.enc)], shared), reverse=True):
        if passes and pass_tokens(passes[-1][1] + [shape(v.enc)], shared) <= budget:
            passes[-1][0].append(v); passes[-1][1].append(shape(v.enc))
        else:
            passes.append([[v], [shape(v.enc)]])
    return [p[0] for p in passes]


def encode_batch(model, tok, a, chunk, epoch):
    """Augment each request (fresh permutation / none option / distractor per epoch), optionally add its none-pair
    siblings and a permuted copy for the KL term, and encode strictly."""
    out, c = [], training_context(a.max_state)
    limits = {"max_state": c["max_state"], "max_branch": c["max_branch"]}
    for req in chunk:
        item_rng = random.Random(source_seed(a.seed, f"{epoch}:{req['_meta']['id']}"))
        variants = [augment(req, item_rng, p_none=a.p_none, p_none_distract=a.p_none_distract, p_distract=a.p_distract)]
        if a.p_none_pair > 0 and item_rng.random() < a.p_none_pair:
            variants += none_pair(req, item_rng)
        for v in variants:
            rec = materialize(v)
            enc = model.encode(tok, rec, strict=True, **limits)
            if len(enc["ids"]) > c["max_packed"]:
                raise ValueError(f"training request exceeds {c['max_packed']} packed tokens")
            parts = question_parts(enc, a.row_budget, a.shared_prefix)
            for part in parts:   # one part unless --row_budget splits a record whose rows do not fit one pass
                sub = rec if len(parts) == 1 else {**rec, "questions": [rec["questions"][q] for q in part]}
                out.append(Variant(sub, enc if sub is rec else model.encode(tok, sub, strict=True, **limits), req["_meta"]["id"], req["_meta"]["source"],
                                   share=len(part) / len(rec["questions"])))
        if a.perm_kl > 0 and item_rng.random() < a.perm_frac and any(q["qtype"] == "choice" and len(q["options"]) >= 3 for q in rec["questions"]):
            rec2, perms = permuted_copy(rec, item_rng)
            out[-1].permuted = (model.encode(tok, rec2, strict=True, **limits), perms)
    return out


def batch_loss(model, a, batch, dev, anchors, anchor_sources, autocast):
    """Forward the variants and sum the loss terms: mean question loss per variant, the anchor KL per anchored variant,
    the permutation KL per permuted variant. Returns (loss, terms) with the summed term values for logging."""
    terms = Counter()
    permuted = [v for v in batch if v.permuted]
    with autocast:
        logits_b = model.forward_batch([v.enc for v in batch], a.shared_prefix)
        logits2_b = model.forward_batch([v.permuted[0] for v in permuted], a.shared_prefix) if permuted else []
    loss = 0.0
    for v, logits in zip(batch, logits_b):
        ce = sum(question_loss(z.float(), q, dev, a.ord_w, a.label_smoothing, a.brier_w, a.focal_gamma)
                 for z, q in zip(logits, v.rec["questions"])) / len(logits) * v.share
        terms["ce"] += ce.item(); loss = loss + ce
        if anchors and v.request_id in anchors and (anchor_sources is None or v.source in anchor_sources):
            kls = [t for t in (anchor_loss(z.float(), q, anchors[v.request_id].get(q["qid"]), dev) for z, q in zip(logits, v.rec["questions"])) if t is not None]
            if kls:
                kl_a = sum(kls) / len(kls) * v.share; loss = loss + a.anchor_w * kl_a; terms["anchor"] += kl_a.item(); terms["anchor_n"] += v.share
    for v, logits2 in zip(permuted, logits2_b):
        logits = logits_b[batch.index(v)]
        kls = [permutation_kl(z1.float(), z2.float(), perm, dev) for z1, z2, perm in zip(logits, logits2, v.permuted[1]) if perm is not None]
        kl = sum(kls) / len(kls); loss = loss + a.perm_kl * kl; terms["kl"] += kl.item(); terms["kl_n"] += 1
    if not torch.isfinite(loss):
        raise ValueError("non-finite training loss")
    return loss, terms


# --- run --------------------------------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n_per_source", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head_lr", type=float, default=0.0, help="separate learning rate for the pointer head (0 = same as --lr); the head trains from scratch")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay on LoRA and head parameters")
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--holdout", default="", help="comma-separated sources excluded from training (evaluated as out-of-source)")
    ap.add_argument("--perm_kl", type=float, default=0.0, help="weight of symmetric KL between predictions under two option orders")
    ap.add_argument("--perm_frac", type=float, default=0.3, help="fraction of records that get the second permuted forward pass")
    ap.add_argument("--ord_w", type=float, default=0.0, help="weight of ranked probability score for Score questions")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="hard-label CE smoothing; existing soft targets are unchanged")
    ap.add_argument("--brier_w", type=float, default=0.0, help="weight of sum-squared probability error added to hard-label CE")
    ap.add_argument("--focal_gamma", type=float, default=0.0, help="hard-label CE multiplier (1-p_y)^gamma; 0 is ordinary CE")
    ap.add_argument("--suite", help="frozen suite directory; train only on its training partition")
    ap.add_argument("--train_sources", default="", help="comma-separated subset of the suite's trainable sources (ablations); default all")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--batch", type=int, default=1, help="records per forward pass (padded batch); optimizer step every --accum micro-batches")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="bf16 = autocast forward with fp32 master weights (CUDA only)")
    ap.add_argument("--weights_dtype", choices=["fp32", "bf16"], default="fp32", help="dtype of the frozen backbone weights. bf16 halves memory and is required by the fused MoE experts "
                                                                                        "(torch._grouped_mm wants bf16); LoRA and head stay fp32 (peft upcasts adapters). The checkpoint records it and is loaded the same way.")
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=0)
    ap.add_argument("--option_isolation", type=int, choices=[0, 1], default=0, help="option spans are isolated sub-branches with shared positions (exact permutation invariance)")
    ap.add_argument("--special_embeddings", type=int, choices=[0, 1], default=0, help="also train the embeddings of the 5 delimiter tokens")
    ap.add_argument("--head_dim", type=int, default=256, help="pointer head dimension")
    ap.add_argument("--lora_targets", choices=["all", "dense", "attn", "qv"], default="all", help="LoRA module set; fewer modules = less drift from the base; dense = all minus the DeltaNet projections on hybrid bases")
    ap.add_argument("--base_revision", default="", help="pin the base commit when the suite manifest does not pin this base")
    ap.add_argument("--p_none", type=float, default=0.1)
    ap.add_argument("--p_none_distract", type=float, default=0.12)
    ap.add_argument("--p_distract", type=float, default=0.15)
    ap.add_argument("--p_none_pair", type=float, default=0.0, help="fraction of Choice records that additionally emit a none-present/none-absent minimal pair")
    ap.add_argument("--synthetic_repeat", type=int, default=1, help="oversample synthetic policy sources (legacy_policy, compositional, contrastive) this many times per epoch")
    ap.add_argument("--public_frac", type=float, default=1.0, help="deterministic subsample of public-source training records (mix ablations)")
    ap.add_argument("--anchor", default="", help="JSON of frozen-base zero-shot distributions {record_id: {qid: {key: p}}} (kev.anchors); enables the anchoring loss")
    ap.add_argument("--anchor_w", type=float, default=0.0, help="weight of KL(base || model) toward the frozen base model's zero-shot distribution, per anchored question")
    ap.add_argument("--anchor_sources", default="", help="comma-separated sources to anchor (default: every record with a target)")
    ap.add_argument("--out", default="runs/kev")
    ap.add_argument("--data", default="", help="your own labelled requests, one JSON object per line (see kev.data.load_records); an alternative to --suite for fine-tuning, or combined with --suite and --replay")
    ap.add_argument("--max_state", type=int, default=MAX_STATE, help=f"state tokens per training record (default {MAX_STATE}); raising it admits long-state --data records, the packed limit grows by the same amount")
    ap.add_argument("--replay", type=int, default=0, help="with --data and --suite: mix in this many records sampled (by --seed) from the suite's training partition, so a delta fine-tune does not forget the released recipe")
    ap.add_argument("--init_from", default="", help="delta mode: warm-start LoRA and the pointer head from an existing run "
                                                   "(local directory or hub id) instead of starting from the base model; keeps the "
                                                   "released model's in-domain skill while adapting to a new domain")
    ap.add_argument("--full_ft", type=int, choices=[0, 1], default=0, help="train every backbone weight (bf16, fp32 masters in kev.full_ft.MasterAdamW) instead of a LoRA; "
                                                                          "needs --weights_dtype bf16. One GPU: masters in host memory. Under torchrun: FSDP2 across the GPUs")
    ap.add_argument("--row_budget", type=int, default=0, help="padded row tokens per forward/backward pass (0 = the whole micro-batch at once); a micro-batch over it "
                                                              "runs in several passes, a record whose rows do not fit is split by question (long states x many questions)")
    ap.add_argument("--shared_prefix", type=int, choices=[0, 1], default=None, help="hybrid backbones: run each record's state once and its question branches from it "
                                                                                  "(kev.shared_prefix; exact) instead of one row per question; default on with --full_ft 1, off otherwise")
    ap.add_argument("--length_sort", type=int, choices=[0, 1], default=0, help="deal each optimizer step's records into micro-batches balanced by padded length "
                                                                               "(--batch becomes the average; same records per step; see microbatch_plan)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (0 = every epoch); the lr schedule spans them")
    ap.add_argument("--save_every_steps", type=int, default=0, help="full-weight: write a resume point (<out>/resume) every N optimizer steps")
    ap.add_argument("--save_every_minutes", type=float, default=0, help="full-weight: write a resume point once this many minutes have passed since the last")
    ap.add_argument("--resume", type=int, choices=[0, 1], default=0, help="full-weight: continue from <out>/resume if it holds a resume point (same arguments), else start")
    ap.add_argument("--stop_after", type=int, default=0, help="full-weight: exit after this optimizer step without saving the checkpoint (a run split across containers; tests)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.shared_prefix is None: a.shared_prefix = a.full_ft
    if min(a.epochs, a.accum, a.n_per_source, a.lora, a.batch, a.synthetic_repeat) < 1 or not 0 < a.public_frac <= 1:
        ap.error("epochs, accum, n_per_source, lora, batch and synthetic_repeat must be positive; 0 < public_frac <= 1")
    if a.dtype == "bf16" and a.device != "cuda":
        ap.error("--dtype bf16 requires --device cuda")
    if a.lr <= 0 or a.head_lr < 0 or a.weight_decay < 0 or min(a.ord_w, a.perm_kl, a.anchor_w) < 0 or not 0 <= a.perm_frac <= 1:
        ap.error("invalid learning rate or loss weights")
    loss_options = (a.label_smoothing, a.brier_w, a.focal_gamma)
    if not all(math.isfinite(v) and v >= 0 for v in loss_options) or a.label_smoothing > 1 or sum(v > 0 for v in loss_options) > 1:
        ap.error("use at most one finite, nonnegative loss modifier; smoothing must be <= 1")
    if bool(a.anchor) != (a.anchor_w > 0):
        ap.error("--anchor and --anchor_w > 0 go together")
    if not MAX_STATE <= a.max_state <= MAX_TRAIN_STATE:
        ap.error(f"--max_state must be in [{MAX_STATE}, {MAX_TRAIN_STATE}]")
    if a.replay and not (a.data and a.suite):
        ap.error("--replay needs both --data and --suite")
    if a.full_ft and (problem := full_ft.unsupported_torch()):
        ap.error(problem)
    if a.full_ft and (a.weights_dtype != "bf16" or a.special_embeddings):
        ap.error("--full_ft 1 trains bf16 weights (--weights_dtype bf16) and every embedding already (no --special_embeddings)")
    if a.row_budget < 0 or a.max_steps < 0:
        ap.error("--row_budget and --max_steps are >= 0")
    if a.row_budget and (a.perm_kl > 0 or a.anchor_w > 0 or int(os.environ.get("WORLD_SIZE", "1")) > 1):
        ap.error("--row_budget splits micro-batches into passes: not with --perm_kl (a record and its permuted copy share a loss term), "
                 "nor --anchor_w (a split record's parts would weight its anchored questions by their part's share of all its questions, "
                 "not 1 / anchored questions), nor under torchrun (FSDP2 ranks must run the same number of backward passes; sharded "
                 "ranks have the memory without it)")
    if (a.save_every_steps or a.save_every_minutes or a.resume or a.stop_after) and not a.full_ft:
        ap.error("resume points are for full-weight runs (--full_ft 1)")
    if Path(a.out).exists() and not a.resume and os.environ.get("RANK", "0") == "0":   # under torchrun rank 0 creates it; the others would race it
        ap.error("refusing to overwrite an existing run")
    return a


RESUME_KNOBS = ("resume", "save_every_steps", "save_every_minutes", "stop_after")   # may differ between a run and its continuation
RESUMED = ("step", "seen", "tokens_seen", "peak_mem", "optimizer_seconds", "step_seconds", "elapsed", "epoch", "microbatch", "grad_norms")   # counters a resume point carries


MAX_GRAD_NORM = 1.0   # both optimizers clip the global gradient norm to this (full_ft.MasterAdamW's default)


def grad_norm_summary(grad_norms):
    """Per epoch: mean and max of the steps' global gradient norms before clipping, and how many steps were clipped."""
    return [{"epoch": ep, "steps": len(norms), "mean": sum(norms) / len(norms), "max": max(norms), "clipped_steps": sum(n > MAX_GRAD_NORM for n in norms)}
            for ep, norms in enumerate(grad_norms) if norms]


def pinned_revision(a, manifest):
    """The base commit this run trains against: the suite's pin, or --base_revision when the suite has none."""
    revision = manifest["base_revisions"].get(a.base) if manifest else None
    if a.base_revision:
        if revision and revision != a.base_revision: raise ValueError("--base_revision conflicts with the suite's pinned revision")
        revision = a.base_revision
    if manifest and not revision:
        raise ValueError("base not pinned by the suite; pass --base_revision")
    return revision


def main():
    a = parse_args()
    dev = a.device or default_device()
    rank, world = full_ft.init_distributed(dev) if a.full_ft else (0, 1)   # torchrun: each rank's "cuda" is its own GPU
    out_dir = Path(a.out)
    if rank: sys.stdout = open(os.devnull, "w", encoding="utf-8")   # one log: rank 0's (errors still reach stderr)
    else: out_dir.mkdir(parents=True, exist_ok=bool(a.resume))
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.dtype == "bf16" else contextlib.nullcontext()
    manifest = read_manifest(a.suite) if a.suite else None
    revision = pinned_revision(a, manifest)
    holdout = manifest["holdout_sources"] if manifest else [s for s in a.holdout.split(",") if s]
    anchors = read_json(a.anchor).get("targets", {}) if a.anchor else {}
    if a.anchor: print(f"anchor targets: {len(anchors)} records from {a.anchor}", flush=True)
    anchor_sources = set(a.anchor_sources.split(",")) if a.anchor_sources else None

    tok = load_tokenizer(a.base, revision=revision)
    model = DecisionModel(a.base, tok, dev, lora=None if a.full_ft else a.lora, revision=revision, head_dim=a.head_dim, lora_targets=a.lora_targets,
                          option_isolation=bool(a.option_isolation), special_embeddings=bool(a.special_embeddings),
                          dtype=torch.bfloat16 if a.weights_dtype == "bf16" else torch.float32, direct_load=bool(a.full_ft))
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    # what this run will save as head.pt; also the architecture a warm start must match
    meta = Meta(base=a.base, base_revision=revision, lora=0 if a.full_ft else a.lora, head_dim=a.head_dim, option_isolation=bool(a.option_isolation),
                special_embeddings=bool(a.special_embeddings), weights_dtype=a.weights_dtype, holdout=holdout, weights="full" if a.full_ft else "lora")
    init_source = None
    if a.init_from:
        # delta mode (PR #9, Radexito): start from an already trained adapter (or full backbone) + pointer head instead of the
        # base model, so a fine-tune on new data keeps what the released checkpoint knows
        init_source = Checkpoint(a.init_from).warm_start(model, meta)
        print(f"delta: warm start from {init_source['resolved']}: {init_source['tensors']} {meta.weights} tensors and the pointer head loaded", flush=True)
    if world > 1: full_ft.shard(model)
    print(f"device={dev} world={world} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M", flush=True)

    reqs = training_requests(a, tok, manifest, holdout)
    suite_hash = digest(Path(a.suite) / "manifest.json") if manifest else None
    if not rank:
        write_json(out_dir / "training_config.json", {"args": vars(a), "suite_sha256": suite_hash, "base_revision": revision, "init_source": init_source,
                                                    "ordinal_objective": "ranked_probability_score", "holdout": holdout})
    print(f"{len(reqs)} training requests (holdout={holdout}), questions by type "
          f"{dict(Counter(q['qtype'] for r in reqs for q in materialize(r)['questions']))}")

    head_params = list(model.head.parameters()); head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": head_params, "lr": a.head_lr or a.lr}]
    # full weights: one GPU keeps the fp32 masters and moments in host memory; FSDP2 ranks keep their shard's on the GPU
    opt = full_ft.MasterAdamW(groups, lr=a.lr, weight_decay=a.weight_decay, offload=world == 1, max_grad_norm=MAX_GRAD_NORM) if a.full_ft else torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    per_epoch = microbatch_plan(reqs, a, world, rank)   # counts only: they depend on len(reqs), not on the shuffle
    steps = a.epochs * sum(ends for _, _, ends in per_epoch)
    steps = min(steps, a.max_steps) if a.max_steps else steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr], total_steps=max(steps, 1), pct_start=0.1)
    step = seen = tokens_seen = peak_mem = optimizer_seconds = elapsed = start_epoch = start_mb = 0; step_seconds, resume_seconds = [], []; run = Counter()
    grad_norms = []   # per epoch, each optimizer step's global gradient norm before clipping
    resume_dir, resume_args = out_dir / "resume", {k: v for k, v in vars(a).items() if k not in RESUME_KNOBS}
    position = full_ft.load_resume(resume_dir, opt, sched, resume_args) if a.resume else None
    if position:
        step, seen, tokens_seen, peak_mem, optimizer_seconds, step_seconds, elapsed, start_epoch, start_mb, grad_norms = (position[k] for k in RESUMED)
        seen, tokens_seen = seen / world, tokens_seen / world   # saved as sums over the ranks (global_sum below adds them back)
        run = Counter(position["run"])
        print(f"resumed from {resume_dir / position['dir']}: step {step}, epoch {start_epoch}, micro-batch {start_mb}", flush=True)
    writer = full_ft.ResumeWriter(resume_dir, background=world > 1) if a.full_ft else None   # FSDP2: state on the GPUs, written from a host copy
    model.train(); t0, last, saved_at, stopped = time.time() - elapsed, time.time(), time.time(), False
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        if ep < start_epoch: continue   # the finished epochs' shuffles are replayed, so the interrupted epoch's order returns
        plan = microbatch_plan(reqs, a, world, rank)   # every record once per epoch across the ranks (all of them on one GPU)
        for mb in range(start_mb if ep == start_epoch else 0, len(plan)):
            chunk, step_records, ends_step = plan[mb]
            batch = encode_batch(model, tok, a, chunk, ep)
            variants = sum(v.share for v in batch)   # a record split by --row_budget counts once
            # weight by source records in the accumulation group (over all ranks) so none-pair siblings do not inflate a record's share
            group_records = step_records * (variants / len(chunk))
            for part in row_passes(batch, a.row_budget, a.shared_prefix):
                loss, terms = batch_loss(model, a, part, dev, anchors, anchor_sources, autocast)
                (loss / group_records).backward()
                run += terms
            run["n"] += variants; seen += round(variants); tokens_seen += sum(v.tokens for v in batch)
            peak_mem = max(peak_mem, allocated_bytes(dev))
            if ends_step:
                if not a.full_ft: norm = float(torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), MAX_GRAD_NORM))   # MasterAdamW clips by the global norm itself
                started = time.time(); opt.step(); sync(dev); optimizer_seconds += time.time() - started
                grad_norms += [[] for _ in range(ep + 1 - len(grad_norms))]
                grad_norms[ep].append(round(opt.grad_norm if a.full_ft else norm, 6))
                sched.step(); opt.zero_grad(); step += 1
                step_seconds.append(round(time.time() - last, 3)); last = time.time()
                if dev == "mps": empty_cache(dev)   # MPS only: per-step cache release keeps the unified-memory footprint down; on CUDA it would just slow the step
                if step % 10 == 0:
                    print(f"ep{ep} step {step}/{steps} loss {run['ce']/run['n']:.3f} kl {run['kl']/max(run['kl_n'],1):.3f} anchor {run['anchor']/max(run['anchor_n'],1):.3f} {(time.time()-t0)/seen:.3f}s/rec", flush=True)
                    run = Counter()
                if step == steps: break
                if a.full_ft and full_ft.save_due(step, a.save_every_steps, a.save_every_minutes, saved_at, max(resume_seconds, default=0)):
                    values = (step, *full_ft.global_sum([seen, tokens_seen]), peak_mem, optimizer_seconds, step_seconds, time.time() - t0, ep, mb + 1, grad_norms)
                    writer.save(step, opt, sched, {**dict(zip(RESUMED, values)), "run": dict(run), "world": world, "args": resume_args})
                    resume_seconds.append(round(time.time() - last, 3)); saved_at = last = time.time()   # the time training blocked, not part of the next step's
                if step == a.stop_after: stopped = True; break
        if step == steps or stopped: break
    if writer: writer.wait()   # a resume point being written in the background is finished (and then superseded, or continued from)
    if stopped:
        print(f"stopped after step {step}; continue with --resume 1", flush=True); return
    seen, tokens_seen = full_ft.global_sum([seen, tokens_seen])   # ranks' micro-batches differ in size under --length_sort

    wall = time.time() - t0
    if a.full_ft: full_ft.save_backbone(model.lm, a.out)   # every rank: FSDP2 gathers to rank 0
    else: model.lm.save_pretrained(a.out)
    if rank: return
    shutil.rmtree(resume_dir, ignore_errors=True)   # the checkpoint supersedes it
    meta.head, meta.extra = model.head.state_dict(), {"args": vars(a), "suite_sha256": suite_hash, "init_source": init_source}
    write_meta(a.out, meta)
    tok.save_pretrained(a.out)
    write_json(out_dir / "training_metrics.json", {"wall_seconds": wall, "records_seen": round(seen),
               "requested_records": a.epochs * len(reqs), "truncated_records": 0, "rejected_records": 0,
               "optimizer_steps": step, "forward_tokens": round(tokens_seen), "step_seconds": step_seconds, "optimizer_seconds": optimizer_seconds, "resume_seconds": resume_seconds, "resume_write_seconds": writer.seconds if writer else [], "world_size": world,
               "grad_norm": grad_norm_summary(grad_norms),
               "weights": meta.weights, "peak_device_bytes": peak_mem, "device": dev, "dtype": a.dtype, "batch": a.batch,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)})
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
