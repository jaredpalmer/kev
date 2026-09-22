"""LoRA fine-tune of the decision model on labelled requests (a frozen suite's training partition, records built on the
fly from the public sources, or your own JSONL), with the pointer head trained from scratch.

    uv run python -m kev.train --suite evals/v7/decision-v7 --out runs/<name>          # what studies run
    uv run python -m kev.train --n_per_source 40 --accum 4 --out runs/smoke               # ~1 min smoke test
    uv run python -m kev.train --data mine.jsonl --init_from jaredpalmer/kev-4b --lr 2e-5 --out runs/mine   # delta
    uv run python -m kev.train --suite evals/v3/decision-v3 --state_mode frozen --state_cache 1 --cache_dtype bf16 \
        --branch_chunk 8 --out runs/frozen   # the state through the base weights once, its K/V cached across epochs

Batch size is small (variable-length records with custom masks) and gradients are accumulated over --accum micro-batches.

Frozen state mode (--state_mode frozen): the state tokens run through the base weights with the adapter off and only the
question branches see the LoRA (kev.cached_state). The state's K/V never change during training, so --state_cache keeps
them across epochs and augmentation variants (host RAM, the device, or disk); --branch_chunk runs the branches a few
questions at a time against the K/V with a backward per chunk, so activation memory is the state plus one chunk.
"""
import argparse, contextlib, hashlib, json, math, os, random, resource, shutil, sys, time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from .checkpoint import Checkpoint, Meta, write_meta
from .cached_state import StateCache, chunked_loss_backward, frozen_loss_backward, kv_bytes_per_token, state_kv, state_len
from .device import allocated_bytes, default_device, empty_cache, sync
from .data import EVAL_ONLY, build, augment, load_records, materialize, none_pair, source_seed
from .suite import SYNTHETIC_SOURCES, digest, load_split, read_json, read_manifest, validate_training, write_json
from .model import MAX_BRANCH, MAX_PACKED, MAX_STATE, DecisionModel, fits, load_tokenizer, user_tokens


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


# --- frozen state: base K/V of the state, cached ------------------------------------------------------------------------

def state_key(req, rec):
    """StateCache key of a materialized record: its source id plus a hash of the rendered state text."""
    return f"{req['_meta']['id']}:{hashlib.sha1(rec['state'].encode()).hexdigest()[:16]}"


def frozen_kv(model, enc, key, cache, stats):
    """Base state K/V of one encoded record (frozen state mode): from the StateCache when it holds the key, otherwise a timed
    base pass over the state tokens, stored when a cache is given. stats counts the seconds ("state") and tokens
    ("state_tokens") of the passes actually run."""
    kv = cache.get(key) if cache is not None else None
    if kv is None:
        t = time.perf_counter(); S = state_len(enc)
        kv = state_kv(model, enc["ids"][:S], adapter=False, grad=False)
        sync(model.device); stats["state"] += time.perf_counter() - t; stats["state_tokens"] += S
        if cache is not None: kv = cache.put(key, kv)
    return kv


def cache_projection(tok, reqs, per_token):
    """(distinct records, state tokens, bytes) a full-epoch StateCache would hold: one entry per source record (the state
    text is the same across epochs and augmentation variants), per_token bytes per state token."""
    tokens = {r["_meta"]["id"]: min(len(user_tokens(tok, materialize(r)["state"])) + 1, MAX_STATE) for r in reqs}
    n = sum(tokens.values())
    return len(tokens), n, n * per_token


def physical_memory_bytes():
    """RAM available to this process: the machine's, capped by the container's cgroup limit when there is one (inside a
    container sysconf reports the host's RAM, so the default cache cap would otherwise exceed what the container may use)."""
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    for limit in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(limit).read_text(encoding="utf-8").strip()
            if text.isdigit(): total = min(total, int(text))
        except OSError:
            pass
    return total


def tier_free_bytes(cache_device, cache_dir, dev):
    """Free bytes of the tier a state cache would live on: the disk under cache_dir, the training device's memory (free HBM
    on cuda; the driver's recommended working set minus the current allocation on mps), or host RAM for the cpu tier."""
    if cache_device == "disk":
        return shutil.disk_usage(cache_dir).free
    if cache_device == "device" and dev == "cuda":
        return torch.cuda.mem_get_info()[0]
    if cache_device == "device" and dev == "mps":
        return max(torch.mps.recommended_max_memory() - allocated_bytes(dev), 0)
    return physical_memory_bytes()


def build_state_cache(a, tok, model, reqs, dev):
    """The StateCache of a --state_cache run, sized against the tier it lives on: the projected full-epoch size is printed,
    refused above the default cap (a quarter of what the tier has free) and merely warned about above an explicit cap."""
    cache_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "int8": "int8"}[a.cache_dtype]
    disk = a.cache_device == "disk"
    budget = tier_free_bytes(a.cache_device, a.cache_dir, dev)
    cap = int((budget / 4 if a.cache_max_gb is None else a.cache_max_gb * 2**30))
    n_rec, n_tok, projected = cache_projection(tok, reqs, kv_bytes_per_token(model, cache_dtype))
    print(f"state cache: {n_rec} records, {n_tok} state tokens -> {projected/2**30:.2f} GB {a.cache_dtype} on {a.cache_device}{f' ({a.cache_dir})' if disk else ''} "
          f"(cap {'unlimited' if cap == 0 else f'{cap/2**30:.2f} GB'})", flush=True)
    if cap and projected > cap:
        if a.cache_max_gb is None:
            raise ValueError(f"projected state cache {projected/2**30:.2f} GB exceeds a quarter of {'the free disk space' if disk else 'physical memory'}; "
                             "pass --cache_dtype bf16, an explicit --cache_max_gb (partial cache), or --state_cache 0")
        print("state cache: cap below one epoch; entries are evicted oldest-first, so only within-batch reuse is expected", flush=True)
    return StateCache(cache_dtype, {"cpu": "cpu", "device": dev, "disk": "disk"}[a.cache_device], cap, a.cache_dir if disk else None)


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
        kept = [r for r in reqs if fits(materialize(r), tok)]
        if len(kept) < len(reqs):
            print(f"dropped {len(reqs) - len(kept)} of {len(reqs)} records that exceed the training context "
                  f"({MAX_STATE} state / {MAX_BRANCH} branch / {MAX_PACKED} packed tokens)", flush=True)
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
    key: str = ""                   # StateCache key (frozen state mode): the source record and its rendered state

    @property
    def tokens(self):
        return len(self.enc["ids"]) + (len(self.permuted[0]["ids"]) if self.permuted else 0)

    def forward_tokens(self, frozen=False):
        """Tokens the forward passes run: in frozen state mode the branches only (state tokens are counted by frozen_kv,
        on a cache miss); the permuted copy shares its source record's state."""
        skip = state_len(self.enc) if frozen else 0
        return len(self.enc["ids"]) - skip + (len(self.permuted[0]["ids"]) - skip if self.permuted else 0)


def encode_batch(model, tok, a, chunk, epoch):
    """Augment each request (fresh permutation / none option / distractor per epoch), optionally add its none-pair
    siblings and a permuted copy for the KL term, and encode strictly."""
    out = []
    for req in chunk:
        item_rng = random.Random(source_seed(a.seed, f"{epoch}:{req['_meta']['id']}"))
        variants = [augment(req, item_rng, p_none=a.p_none, p_none_distract=a.p_none_distract, p_distract=a.p_distract)]
        if a.p_none_pair > 0 and item_rng.random() < a.p_none_pair:
            variants += none_pair(req, item_rng)
        for v in variants:
            rec = materialize(v)
            enc = model.encode(tok, rec, strict=True)
            if len(enc["ids"]) > MAX_PACKED:
                raise ValueError(f"training request exceeds {MAX_PACKED} packed tokens")
            out.append(Variant(rec, enc, req["_meta"]["id"], req["_meta"]["source"], key=state_key(req, rec)))
        if a.perm_kl > 0 and item_rng.random() < a.perm_frac and any(q["qtype"] == "choice" and len(q["options"]) >= 3 for q in rec["questions"]):
            rec2, perms = permuted_copy(rec, item_rng)
            out[-1].permuted = (model.encode(tok, rec2, strict=True), perms)
    return out


def batch_loss(model, a, batch, dev, anchors, anchor_sources, autocast, caches=None):
    """Forward the variants and sum the loss terms: mean question loss per variant, the anchor KL per anchored variant,
    the permutation KL per permuted variant. Returns (loss, terms) with the summed term values for logging.
    caches (frozen state mode): each variant's base state K/V, in batch order; a permuted copy shares its source's."""
    terms = Counter()
    permuted = [v for v in batch if v.permuted]
    with autocast:
        logits_b = model.forward_batch([v.enc for v in batch], caches)
        logits2_b = model.forward_batch([v.permuted[0] for v in permuted], None if caches is None else [caches[batch.index(v)] for v in permuted]) if permuted else []
    loss = 0.0
    for v, logits in zip(batch, logits_b):
        ce = sum(question_loss(z.float(), q, dev, a.ord_w, a.label_smoothing, a.brier_w, a.focal_gamma)
                 for z, q in zip(logits, v.rec["questions"])) / len(logits)
        terms["ce"] += ce.item(); loss = loss + ce
        if anchors and v.request_id in anchors and (anchor_sources is None or v.source in anchor_sources):
            kls = [t for t in (anchor_loss(z.float(), q, anchors[v.request_id].get(q["qid"]), dev) for z, q in zip(logits, v.rec["questions"])) if t is not None]
            if kls:
                kl_a = sum(kls) / len(kls); loss = loss + a.anchor_w * kl_a; terms["anchor"] += kl_a.item(); terms["anchor_n"] += 1
    for v, logits2 in zip(permuted, logits2_b):
        logits = logits_b[batch.index(v)]
        kls = [permutation_kl(z1.float(), z2.float(), perm, dev) for z1, z2, perm in zip(logits, logits2, v.permuted[1]) if perm is not None]
        kl = sum(kls) / len(kls); loss = loss + a.perm_kl * kl; terms["kl"] += kl.item(); terms["kl_n"] += 1
    if not torch.isfinite(loss):
        raise ValueError("non-finite training loss")
    return loss, terms


def record_loss_fn(v, a, dev, anchors, anchor_sources, terms):
    """Per-question loss of one variant for the chunked backward: its question loss / Q plus the anchor KL / (anchored
    questions), the same terms batch_loss sums, accumulated into `terms` as the chunks run."""
    qs = v.rec["questions"]
    targets = anchors[v.request_id] if anchors and v.request_id in anchors and (anchor_sources is None or v.source in anchor_sources) else {}
    n_anchor = sum(1 for q in qs if targets.get(q["qid"]) is not None and set(targets[q["qid"]]) == set(q["keys"]))
    if n_anchor: terms["anchor_n"] += 1
    def fn(z, qi):
        q = qs[qi]; z = z.float()
        loss = question_loss(z, q, dev, a.ord_w, a.label_smoothing, a.brier_w, a.focal_gamma) / len(qs); terms["ce"] += loss.item()
        kl = anchor_loss(z, q, targets.get(q["qid"]), dev) if n_anchor else None
        if kl is not None:
            kl = kl / n_anchor; loss = loss + a.anchor_w * kl; terms["anchor"] += kl.item()
        return loss
    return fn


def chunked_batch_backward(model, a, batch, dev, anchors, anchor_sources, autocast, cache, stats, scale):
    """--branch_chunk: every variant's question branches run `branch_chunk` at a time against its state K/V and each chunk's
    loss (times `scale`) is backpropagated right away, so at most one chunk's activations are alive. Adapted mode: the K/V
    gradients are pushed back through the state graph (gradient-exact, scripts/exp_equivalence.py). Frozen mode: against
    the cached base K/V, nothing to push back. Returns the summed loss terms for logging."""
    terms = Counter()
    for v in batch:
        fn = record_loss_fn(v, a, dev, anchors, anchor_sources, terms)
        kw = dict(scale=scale, times=stats, forward_ctx=autocast)
        if a.state_mode == "frozen":
            with autocast: kv = frozen_kv(model, v.enc, v.key, cache, stats)   # cache hit or a timed base pass, as in the one-pass path
            frozen_loss_backward(model, v.enc, kv, fn, a.branch_chunk, **kw)
        else:
            chunked_loss_backward(model, v.enc, fn, a.branch_chunk, **kw)
    return terms


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
    ap.add_argument("--state_mode", choices=["adapted", "frozen"], default="adapted", help="frozen = the state runs once through the base weights (its K/V cached), only the question branches see the adapter")
    ap.add_argument("--state_cache", type=int, choices=[0, 1], default=0, help="frozen mode: keep each record's base state K/V in memory across epochs and augmentation variants "
                    "(2 x layers x kv_heads x head_dim x itemsize per state token: ~230 KB fp32 for Qwen3-0.6B; the projected full-epoch size is printed before training)")
    ap.add_argument("--cache_dtype", choices=["fp32", "bf16", "int8"], default="fp32", help="storage dtype of the state K/V cache (bf16 halves it, int8 quarters it: int8 codes "
                    "plus one fp16 scale per layer, head and token; the branches then train against rounded K/V from the first step, which is recorded in head.pt so "
                    "evaluation rounds the same way)")
    ap.add_argument("--cache_max_gb", type=float, default=None, help="cap on the state K/V cache, oldest entries evicted (default: a quarter of what the tier has free: host RAM, "
                    "the device's free memory, or the disk; the run is refused when the projected size exceeds it; 0 = unlimited). A cap below one epoch only serves reuse within a batch "
                    "(none-pair siblings, permuted copies)")
    ap.add_argument("--cache_device", choices=["cpu", "device", "disk"], default="cpu", help="where the state K/V cache lives: host memory (entries are moved to the training device on use), "
                    "the training device, or disk (one file per record under --cache_dir, loaded on use; the default cap is a quarter of the free disk space)")
    ap.add_argument("--cache_dir", default="", help="--cache_device disk: directory of the state K/V files (default <out>/state_cache). Kept after training; a later run of "
                    "the same base and --cache_dtype can point at it and reuse its entries")
    ap.add_argument("--lora_layers", type=int, default=0, help="LoRA in the top M transformer layers only (0 = all layers); reloads from adapter_config.json")
    ap.add_argument("--state_grad", type=int, choices=[0, 1], default=1, help="adapted mode: 0 = the state tokens still run through the adapter but their keys/values are detached in every "
                    "layer, so no gradient flows into the state (same forward as 1; needs --batch 1 and no --branch_chunk)")
    ap.add_argument("--branch_chunk", type=int, default=0, help="run the question branches this many at a time against the state K/V, backward per chunk (adapted mode: gradient-exact, "
                    "the K/V gradients are pushed back through the state; frozen mode: against the cached base K/V, nothing to push back); 0 = all branches in one pass")
    ap.add_argument("--option_isolation", type=int, choices=[0, 1], default=0, help="option spans are isolated sub-branches with shared positions (exact permutation invariance)")
    ap.add_argument("--special_embeddings", type=int, choices=[0, 1], default=0, help="also train the embeddings of the 5 delimiter tokens (frozen state mode: the state runs "
                    "with the adapter off, so <state> keeps its base embedding and only the 4 branch delimiters train)")
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
    ap.add_argument("--replay", type=int, default=0, help="with --data and --suite: mix in this many records sampled (by --seed) from the suite's training partition, so a delta fine-tune does not forget the released recipe")
    ap.add_argument("--init_from", default="", help="delta mode: warm-start LoRA and the pointer head from an existing run "
                                                   "(local directory or hub id) instead of starting from the base model; keeps the "
                                                   "released model's in-domain skill while adapting to a new domain")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
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
    if a.replay and not (a.data and a.suite):
        ap.error("--replay needs both --data and --suite")
    if a.branch_chunk < 0 or (a.cache_max_gb is not None and a.cache_max_gb < 0):
        ap.error("--branch_chunk and --cache_max_gb must be non-negative")
    if a.branch_chunk and a.perm_kl > 0:
        ap.error("--branch_chunk does not support --perm_kl (the permuted copy's logits must be alive with the original's)")
    if a.checkpointing and (a.branch_chunk or a.state_mode == "frozen"):
        ap.error("--checkpointing drops the state K/V cache in training mode; not with --branch_chunk or --state_mode frozen")
    if a.state_cache and a.state_mode != "frozen":
        ap.error("--state_cache applies to --state_mode frozen only")
    if a.state_cache and a.cache_device == "disk" and not a.cache_dir and os.environ.get("KEV_CACHE_DIR"):
        a.cache_dir = os.environ["KEV_CACHE_DIR"]   # set by modal_app.run_trial so the trial config stays the plan's (recorded below as used)
    if a.cache_dir and a.cache_device != "disk":
        ap.error("--cache_dir applies to --cache_device disk only")
    if a.state_cache and a.dtype == "bf16" and a.cache_dtype == "fp32":
        ap.error("a bf16 run must use --cache_dtype bf16 or int8: a fp32 cache would store upcast bf16 K/V and record fp32 for evaluation")
    if a.lora_layers < 0:
        ap.error("--lora_layers must be non-negative")
    if not a.state_grad and (a.state_mode != "adapted" or a.branch_chunk or a.batch != 1):
        ap.error("--state_grad 0 needs --state_mode adapted, --branch_chunk 0 and --batch 1")
    if Path(a.out).exists():
        ap.error("refusing to overwrite an existing run")
    return a


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
    out_dir = Path(a.out); out_dir.mkdir(parents=True)
    if a.state_cache and a.cache_device == "disk":   # resolved here so training_config.json records the directory actually used
        a.cache_dir = str(Path(a.cache_dir) if a.cache_dir else out_dir / "state_cache"); Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
    frozen = a.state_mode == "frozen"
    state_kv_dtype = a.cache_dtype if a.state_cache else None   # frozen mode: the dtype the branches saw the state K/V in
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = a.device or default_device()
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
    model = DecisionModel(a.base, tok, dev, lora=a.lora, revision=revision, head_dim=a.head_dim, lora_targets=a.lora_targets,
                          option_isolation=bool(a.option_isolation), special_embeddings=bool(a.special_embeddings),
                          dtype=torch.bfloat16 if a.weights_dtype == "bf16" else torch.float32,
                          state_mode=a.state_mode, lora_layers=a.lora_layers, state_grad=bool(a.state_grad))
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    # what this run will save as head.pt; also the architecture a warm start must match
    meta = Meta(base=a.base, base_revision=revision, lora=a.lora, head_dim=a.head_dim, option_isolation=bool(a.option_isolation),
                special_embeddings=bool(a.special_embeddings), weights_dtype=a.weights_dtype, holdout=holdout,
                state_mode=a.state_mode, state_kv_dtype=state_kv_dtype, lora_layers=a.lora_layers, state_grad=bool(a.state_grad))
    init_source = None
    if a.init_from:
        # delta mode (PR #9, Radexito): start from an already trained adapter + pointer head instead of the base model, so a
        # fine-tune on new data keeps what the released checkpoint knows
        init_source = Checkpoint(a.init_from).warm_start(model, meta)
        print(f"delta: warm start from {init_source['resolved']}: {init_source['adapter_tensors']} adapter tensors and the pointer head loaded", flush=True)
    print(f"device={dev} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M", flush=True)

    reqs = training_requests(a, tok, manifest, holdout)
    suite_hash = digest(Path(a.suite) / "manifest.json") if manifest else None
    write_json(out_dir / "training_config.json", {"args": vars(a), "suite_sha256": suite_hash, "base_revision": revision, "init_source": init_source,
                                                "ordinal_objective": "ranked_probability_score", "holdout": holdout, "state_mode": a.state_mode,
                                                "state_kv_dtype": state_kv_dtype, "lora_layers": a.lora_layers, "state_grad": bool(a.state_grad)})
    print(f"{len(reqs)} training requests (holdout={holdout}), questions by type "
          f"{dict(Counter(q['qtype'] for r in reqs for q in materialize(r)['questions']))}")

    head_params = list(model.head.parameters()); head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": head_params, "lr": a.head_lr or a.lr}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    micro_per_epoch = math.ceil(len(reqs) / a.batch)
    steps = a.epochs * math.ceil(micro_per_epoch / a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr], total_steps=max(steps, 1), pct_start=0.1)
    cache = build_state_cache(a, tok, model, reqs, dev) if a.state_cache else None
    stats = Counter()   # seconds in state passes ("state") vs branch passes ("branch") and state tokens run ("state_tokens"); the packed forward is one pass
    model.train(); t0 = time.time(); run = Counter(); step = seen = tokens_seen = peak_mem = 0
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        for mb in range(micro_per_epoch):
            chunk = reqs[mb * a.batch : (mb + 1) * a.batch]
            batch = encode_batch(model, tok, a, chunk, ep)
            # weight by source records in the accumulation group so none-pair siblings do not inflate a record's share
            group_records = accumulation_records(len(reqs), a.batch, a.accum, mb) * (len(batch) / len(chunk))
            if a.branch_chunk:
                terms = chunked_batch_backward(model, a, batch, dev, anchors, anchor_sources, autocast, cache, stats, 1 / group_records)
            else:
                caches = None
                if frozen:
                    with autocast: caches = [frozen_kv(model, v.enc, v.key, cache, stats) for v in batch]   # hits, or timed base passes
                    t_branch = time.perf_counter()
                loss, terms = batch_loss(model, a, batch, dev, anchors, anchor_sources, autocast, caches)
                (loss / group_records).backward()
                if frozen: sync(dev); stats["branch"] += time.perf_counter() - t_branch
            run += terms; run["n"] += len(batch); seen += len(batch); tokens_seen += sum(v.forward_tokens(frozen) for v in batch)
            peak_mem = max(peak_mem, allocated_bytes(dev))
            if (mb + 1) % a.accum == 0 or mb + 1 == micro_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if dev == "mps": empty_cache(dev)   # MPS only: per-step cache release keeps the unified-memory footprint down; on CUDA it would just slow the step
                if step % 10 == 0:
                    print(f"ep{ep} step {step}/{steps} loss {run['ce']/run['n']:.3f} kl {run['kl']/max(run['kl_n'],1):.3f} anchor {run['anchor']/max(run['anchor_n'],1):.3f} {(time.time()-t0)/seen:.3f}s/rec", flush=True)
                    run = Counter()

    model.lm.save_pretrained(a.out)
    meta.head, meta.extra = model.head.state_dict(), {"args": vars(a), "suite_sha256": suite_hash, "init_source": init_source}
    write_meta(a.out, meta)
    tok.save_pretrained(a.out)
    write_json(out_dir / "training_metrics.json", {"wall_seconds": time.time() - t0, "records_seen": seen,
               "requested_records": a.epochs * len(reqs), "truncated_records": 0, "rejected_records": 0,
               "optimizer_steps": step, "forward_tokens": tokens_seen + stats["state_tokens"], "state_tokens_computed": stats["state_tokens"] if frozen else None,
               "peak_device_bytes": peak_mem, "device": dev, "dtype": a.dtype, "batch": a.batch,
               # the activation peak alone: a device-tier cache is resident in the same allocator, so its bytes are subtracted here
               "peak_device_bytes_without_cache": peak_mem - (cache.bytes if cache is not None and a.cache_device == "device" else 0),
               "state_mode": a.state_mode, "branch_chunk": a.branch_chunk, "state_seconds": stats["state"], "branch_seconds": stats["branch"],
               "state_cache": cache.stats() if cache is not None else None,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)})
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
