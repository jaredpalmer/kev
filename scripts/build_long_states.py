"""Long-state records (PLAN.md round 4, item 4.12; PLAN_27b A3): a decision-v7 record's state buried among unrelated
states from the same partition until the state reaches about 1k, 2k or 4k tokens. The question and label are unchanged
and a note names the primary record, as in kev.transfer_v9.buried (which stops at three neighbours and ~400 tokens).

    uv run python scripts/build_long_states.py --out evals/round4/longstate-v1

Writes
    train.jsonl        long-state records from decision-v7/train: delta training data (kev.train --data ... --max_state 4608)
    train_control.jsonl  the same primaries unburied, one per long record: the matched short-state continuation control
    development.jsonl  long-state records from decision-v7/development, plus each primary unburied (source
                       longstate_control), so a paired read gives the cost of burial per length; eval-only, scored under
                       the serving context
    manifest.json      sha256 per file, seed, lengths, tokenizer, and the context the development partition is scored in

Sources: every trainable decision-v7 source; neighbours never share the primary's id. Lengths are targeted in tokens of
the JSON serialisation (longstate-v1 was built that way and is kept byte-identical); the model sees kev.api.render's
text, so the manifest also reports rendered-state token counts per length (`rendered_state_tokens`), and every record is
checked against the serving context with kev.model.fits.
"""
import argparse, copy, hashlib, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.api import render  # noqa: E402
from kev.data import materialize  # noqa: E402
from kev.model import MAX_TRAIN_STATE, fits, load_tokenizer  # noqa: E402
from kev.suite import ADMISSION_TOKENIZER as TOKENIZER, SERVING_CONTEXT, digest, load_split, read_manifest, record_digest, write_json, write_jsonl  # noqa: E402

SUITE = "evals/v7/decision-v7"
LENGTHS = (1024, 2048, 4096)
NOTE = "Answer about the primary record only; the other records are unrelated."
PAIRING = ("pair_id", "sibling", "control_id")   # the parent's minimal-pair links; copies at several lengths must not claim them


def meta(record, **fields):
    return {**{k: v for k, v in record["_meta"].items() if k not in PAIRING}, "parent_source": record["_meta"]["source"],
            "parent_id": record["_meta"]["id"], "variant": "clean", **fields}


def as_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def bury(record, pool, length, rng, count):
    """The record with its state placed among neighbours until the serialised state reaches `length` tokens (at most
    1.25 * length); None when the draw overshoots."""
    records, slot = [record["state"]], 0
    while True:
        state = {"records": records, "primary_record": slot + 1, "note": NOTE}
        n = count(state)
        if n >= length:
            break
        other = rng.choice(pool)
        if other["_meta"]["id"] == record["_meta"]["id"]:
            continue
        position = rng.randrange(len(records) + 1)
        records.insert(position, other["state"]); slot += position <= slot
    if n > min(1.25 * length, MAX_TRAIN_STATE - 64):   # every record stays trainable (kev.model.training_context) and servable
        return None
    rec = copy.deepcopy(record); rec["state"] = state
    for q in rec["questions"].values(): q["src"] = f"longstate_{length}_{q['src']}"
    parent = record["_meta"]["id"]
    rec["_meta"] = meta(record, source="longstate", id=f"longstate/{length}/{parent}", group_id=f"longstate/{parent}", length=length, state_tokens=n,
                        primary_slot=slot, text_sha256=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest())
    rec["_meta"]["row_sha256"] = record_digest({k: v for k, v in rec.items() if k != "_meta"})
    return rec


def control(record):
    rec = copy.deepcopy(record); parent = record["_meta"]["id"]
    for q in rec["questions"].values(): q["src"] = f"longstate_control_{q['src']}"
    rec["_meta"] = meta(record, source="longstate_control", id=f"longstate_control/{parent}", group_id=f"longstate/{parent}")
    return rec


def build(records, counts, rng, count, with_controls):
    """`counts`: {length: records to make}, drawn in that order."""
    out, primaries = [], set()
    for length, per_length in counts.items():
        made = 0
        for record in rng.sample(records, len(records)):
            if made == per_length: break
            rec = bury(record, records, length, rng, count)
            if rec is None: continue
            out.append(rec); primaries.add(record["_meta"]["id"]); made += 1
    if with_controls:
        out += [control(r) for r in records if r["_meta"]["id"] in primaries]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_per_length", type=int, default=600)
    ap.add_argument("--dev_per_length", type=int, default=100)
    ap.add_argument("--seed", default="round4-longstate-v1")
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)), help="comma-separated target state lengths in tokens")
    ap.add_argument("--train_counts", default="", help="comma-separated training records per length (default --train_per_length each)")
    ap.add_argument("--panel_partition", default="development", choices=["development", "calibration"], help="decision-v7 partition the scored panel is built from")
    ap.add_argument("--version", default="longstate-v1")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    counts = [int(x) for x in a.train_counts.split(",")] if a.train_counts else [a.train_per_length] * len(lengths)
    if len(counts) != len(lengths):
        raise SystemExit("--train_counts needs one count per --lengths entry")
    train_counts = dict(zip(lengths, counts))
    panel_counts = {length: a.dev_per_length for length in lengths}
    tok = load_tokenizer(TOKENIZER[0], revision=TOKENIZER[1])
    count = lambda state: len(tok.encode(as_text(state), add_special_tokens=False))
    trainable = set(read_manifest(SUITE)["trainable_sources"])
    parts = {part: [r for r in load_split(SUITE, part) if r["_meta"]["source"] in trainable] for part in ("train", a.panel_partition)}
    train = build(parts["train"], train_counts, random.Random(f"{a.seed}:train"), count, with_controls=False)
    dev = build(parts[a.panel_partition], panel_counts, random.Random(f"{a.seed}:development"), count, with_controls=True)   # RNG key kept for every partition: v1 and v2 were built with it
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    parents = {r["_meta"]["id"]: r for r in parts["train"]}
    train_control = [parents[r["_meta"]["parent_id"]] for r in train]
    write_jsonl(out / "train.jsonl", train); write_jsonl(out / "train_control.jsonl", train_control); write_jsonl(out / "development.jsonl", dev)
    tokens = lambda rows, length: sorted(r["_meta"]["state_tokens"] for r in rows if r["_meta"].get("length") == length)
    rendered = lambda rows, length: sorted(len(tok.encode(render(r["state"]), add_special_tokens=False)) for r in rows if r["_meta"].get("length") == length)
    quantiles = lambda xs: {"min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]}
    serving = {k: v for k, v in SERVING_CONTEXT.items() if k != "truncate"}
    if not all(fits(materialize(r), tok, **serving) for r in dev):
        raise SystemExit("a development record exceeds the serving context")
    write_json(out / "manifest.json", {
        "version": a.version, "parent": SUITE, "panel_partition": a.panel_partition, "train_counts": {str(k): v for k, v in train_counts.items()}, "parent_manifest_sha256": digest(Path(SUITE) / "manifest.json"), "seed": a.seed,
        "tokenizer": {"model": TOKENIZER[0], "revision": TOKENIZER[1]}, "lengths": lengths,
        "state_tokens": {str(L): {"train_median": (t := tokens(train, L))[len(t) // 2], "dev_median": (d := tokens(dev, L))[len(d) // 2]} for L in lengths},
        "rendered_state_tokens": {str(L): {"train": quantiles(rendered(train, L)), "development": quantiles(rendered(dev, L))} for L in lengths},
        "eval_only": True, "context": SERVING_CONTEXT, "holdout_sources": [], "base_revisions": read_manifest(SUITE)["base_revisions"],
        "files": {name: {"sha256": digest(out / name), "records": len(rows)} for name, rows in (("train.jsonl", train), ("train_control.jsonl", train_control), ("development.jsonl", dev))},
        "protocol": f"development = decision-v7/{a.panel_partition} primaries buried at each length + the same primaries unburied (longstate_control); "
                    "train = decision-v7/train primaries buried (delta data, never scored). Scored under the serving context."})
    print(f"train {len(train)} records, development {len(dev)} records -> {out}")


if __name__ == "__main__":
    main()
