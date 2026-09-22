"""Bounded self-improvement loop for kev.

    uv run python -m kev.autoresearch leaderboard                  # rebuild runs/leaderboard.{jsonl,md} from every study
    uv run python -m kev.autoresearch propose --base Qwen/Qwen3-0.6B-Base --n 8 --out experiments/auto/r1.json
    uv run python -m kev.autoresearch round --base Qwen/Qwen3-0.6B-Base --n 8 --name auto-r1 [--seeds 0,1]
    uv run python -m kev.autoresearch loop --base Qwen/Qwen3-0.6B-Base --rounds 4 --n 8 --spend-cap 120

What it may change: the allowlisted trial parameters in kev.experiment (optimizer, LoRA rank, epochs, effective batch,
augmentation rates, loss weights, architecture switches). What it may never change: the evaluator, the frozen suites,
the gates, or the locked test. Every trial still goes through kev.experiment.execute_trial with full provenance.

Score: transfer development accuracy (out-of-domain), among trials that pass the correctness gates; ties broken by
transfer Brier, then dev NLL. Selection happens on development partitions only. A round's best trial replaces the
incumbent only through next_incumbent(): a replication of the incumbent's config joins it; another config needs a
positive record-clustered paired-bootstrap delta on per-task macro transfer accuracy against the incumbent's best seed
with a 95% lower bound >= -1 pp (PLAN.md round 4, item 4.3). The decision and its interval are written to the round
ledger runs/autoresearch.jsonl (leaderboard.jsonl is regenerated from result.json files on every refresh), whose latest
`incumbent_after` is the champion the next round mutates. A config is never re-run with the same seed.

Budget: before each round, `modal billing summary` is read; the loop stops when metered spend since `--spend-start`
exceeds `--spend-cap`. Per-round admission bounds still apply in modal_app.admit_study.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kev.data import source_seed
from kev.experiment import CHOICE_DEFAULTS, DEFAULTS, validated_trial
from kev.metrics import paired_bootstrap
from kev.suite import ENCODING, digest, read_json, read_jsonl, read_manifest, write_json, write_jsonl


def config_digest(value):
    """Canonical (key-sorted) digest for config identity; kev.suite.record_digest keeps insertion order for provenance."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# per-backbone execution shape, not part of a recipe: hidden from knob summaries
INFRA_KEYS = ("base", "seed", "base_revision", "dtype", "checkpointing", "batch", "accum", "perm_frac")
ALL_DEFAULTS = {**DEFAULTS, **CHOICE_DEFAULTS}   # what kev.train does when a knob is not given


def knobs(cfg):
    """The recipe a config expresses: every non-default, non-infrastructure parameter."""
    return {k: v for k, v in sorted(cfg.items()) if k not in INFRA_KEYS and v != ALL_DEFAULTS.get(k)}

ROOT = Path(__file__).resolve().parents[1]
SUITE, TRANSFER = "evals/v4/decision-v4", "evals/v4/transfer-v4"
H100_RATE = 3.95

# Mutation space: each entry is (knob, candidate values). One knob changes per proposal, plus a few two-knob combos.
SPACE = {
    "lr": [2e-5, 3e-5, 5e-5, 1e-4, 2e-4], "lora": [4, 8, 16, 32], "epochs": [1, 2, 3],
    "accum": [1, 2],                                  # with batch 8 -> effective 8 or 16
    "p_none_pair": [0.0, 0.25, 0.5], "p_none": [0.05, 0.1, 0.2], "p_none_distract": [0.06, 0.12, 0.25], "p_distract": [0.0, 0.15, 0.3],
    "perm_kl": [0.0, 0.2, 0.5], "ord_w": [0.0, 0.25, 0.5],
    "option_isolation": [0, 1], "special_embeddings": [0, 1], "head_dim": [128, 256, 512, 1024],
    "synthetic_repeat": [1, 2], "public_frac": [0.33, 0.5, 1.0], "lora_targets": ["all", "attn", "qv"],
    "head_lr": [0.0, 5e-4, 2e-3], "weight_decay": [0.0, 0.01, 0.1],
}
BASE_DEFAULTS = {  # per-backbone memory-safe batch shape; effective batch stays 8 unless accum is mutated
    "Qwen/Qwen3-0.6B-Base": {"batch": 8, "accum": 1, "dtype": "bf16"},
    "Qwen/Qwen3-4B-Base": {"batch": 4, "accum": 2, "dtype": "bf16", "checkpointing": 1},
    "Qwen/Qwen3-8B-Base": {"batch": 2, "accum": 4, "dtype": "bf16", "checkpointing": 1, "base_revision": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"},
}
TRIAL_MINUTES = {"Qwen/Qwen3-0.6B-Base": 14, "Qwen/Qwen3-4B-Base": 50, "Qwen/Qwen3-8B-Base": 80}


def metered_spend():
    out = subprocess.run([sys.executable, "-m", "modal", "billing", "summary", "--json"], capture_output=True, text=True, cwd=ROOT)
    return float(json.loads(out.stdout)["metered_cost"]) if out.returncode == 0 else None


def collect():
    """Every completed trial in runs/*/ with its provenance, as flat leaderboard rows."""
    rows = []
    for result in sorted(ROOT.glob("runs/*/*/result.json")):
        r = read_json(result)
        prov = r.get("provenance", {}); cfg = prov.get("config") or {}
        tr = r.get("transfer") or {}
        row = {"study": result.parent.parent.name, "trial": result.parent.name, "legacy": prov.get("legacy_checkpoint", False),
               "base": cfg.get("base"), "seed": cfg.get("seed"), "config": cfg, "config_sha256": prov.get("config_sha256"),
               "suite_sha256": prov.get("suite_sha256"), "transfer_suite_sha256": tr.get("suite_sha256"), "git": prov.get("git_commit", "")[:8],
               "dev_acc": r["clean"]["acc"], "dev_nll": r["clean"]["nll"], "dev_ece": r["clean"]["ece"],
               "transfer_acc": tr.get("clean", {}).get("acc"), "transfer_brier": tr.get("clean", {}).get("brier"),
               "transfer_conf_err": tr.get("clean", {}).get("confident_error_rate"),
               "heldout_pairs": (tr.get("paired_flip") or {}).get("both_correct_rate"),
               "none_present": r["variants"].get("none_present", {}).get("acc"), "perm_flip": r["permutation"]["flip_rate"],
               "gates": r["gates"]["checks"], "gates_passed": r["gates"]["passed"],
               "train_s": (r.get("training_resources") or {}).get("wall_seconds"), "wall_s": r.get("wall_seconds"),
               "est_usd": round(H100_RATE * r.get("wall_seconds", 0) / 3600, 2) if prov.get("device") == "cuda" else None}
        rows.append(row)
    return rows


def dev_partition_hashes():
    """manifest sha -> development.jsonl sha, for every frozen suite. Suites that differ only in training data (v4/v5/v6)
    share development bytes and are comparable."""
    out = {}
    for m in ROOT.glob("evals/**/manifest.json"):
        try: out[digest(m)] = read_json(m)["files"]["development.jsonl"]["sha256"]
        except (KeyError, ValueError): pass
    return out


DEV_HASHES = dev_partition_hashes()


def eligible(row, suite_hash=None, transfer_hash=None):
    """Comparable and correct: same development bytes, same transfer suite, complete coverage, isolation, transfer scored."""
    g = row["gates"]
    ok = g.get("complete_coverage") and g.get("isolation_and_packing") and g.get("transfer_complete", True) and row["transfer_acc"] is not None
    if suite_hash and DEV_HASHES.get(row["suite_sha256"]) != DEV_HASHES.get(suite_hash): return False
    if transfer_hash and DEV_HASHES.get(row["transfer_suite_sha256"]) != DEV_HASHES.get(transfer_hash): return False
    return bool(ok) and not row["legacy"]


def score(row):
    return (row["transfer_acc"], -(row["transfer_brier"] or 1), -(row["dev_nll"] or 9))


def trial_id(row):
    return f"{row['study']}/{row['trial']}"


def strip_seed(cfg):
    return {k: v for k, v in cfg.items() if k != "seed"}


def recipe(row):
    """A trial's recipe identity: its config without the seed (config_sha256 hashes the seed too, so seeds of one
    recipe never share it)."""
    return config_digest(strip_seed(row["config"]))


def as_incumbent(group):
    """Incumbent summary of one config's eligible trials (seeds averaged)."""
    return {"config": group[0]["config"], "recipe": recipe(group[0]), "seeds": sorted({r["seed"] for r in group}),
            "transfer_acc": sum(r["transfer_acc"] for r in group) / len(group), "dev_acc": sum(r["dev_acc"] for r in group) / len(group),
            "trials": [trial_id(r) for r in group]}


def ledger():
    path = ROOT / "runs/autoresearch.jsonl"
    return read_jsonl(path) if path.exists() else []


def incumbent(rows, base, suite_hash, transfer_hash, history=()):
    """The backbone's champion: the latest ledger round's `incumbent_after` (only the latest is authoritative; champions
    change only through next_incumbent), plus any later eligible replications of the same config. If that champion's
    trials are no longer all eligible (other suites, pulled elsewhere), or there is no ledger entry, the loop is seeded
    with the best eligible config, preferring >= 2 seeds by mean transfer accuracy, then the best single seed."""
    pool = [r for r in rows if r["base"] == base and eligible(r, suite_hash, transfer_hash)]
    by_trial = {trial_id(r): r for r in pool}
    latest = next((e["incumbent_after"] for e in reversed(history) if e["base"] == base and e.get("incumbent_after")), None)
    if latest and all(t in by_trial for t in latest["trials"]):
        champion = recipe(by_trial[latest["trials"][0]])
        return as_incumbent([r for r in pool if recipe(r) == champion])
    by_cfg = defaultdict(list)
    for r in pool: by_cfg[recipe(r)].append(r)
    if not by_cfg: return None
    def agg(group):
        n = len(group)
        return (len({r["seed"] for r in group}) >= 2, sum(r["transfer_acc"] for r in group) / n, -sum(r["transfer_brier"] for r in group) / n)
    return as_incumbent(max(by_cfg.values(), key=agg))


NONINFERIORITY = 0.01   # PLAN.md round 4, item 4.3: a challenger may be at most 1 pp worse at the paired CI's lower bound


def transfer_rows(trial):
    return read_json(ROOT / "runs" / trial / "transfer/rows.json")


def challenge(champion, challenger, rows, samples=1000):
    """Whether `challenger` (a leaderboard row of another config) replaces `champion` (an incumbent summary): the
    record-clustered paired bootstrap of per-task macro transfer accuracy (the aggregation compare() and kev.compare
    print) against the champion's best seed must have a positive point estimate and a 95% lower bound >= -NONINFERIORITY.
    A one-seed point-estimate lead on ~650 questions is mostly noise. A pair that cannot be bootstrapped (missing or
    mismatched transfer rows) is recorded as not accepted rather than raised: the round's spend is already metered."""
    by_trial = {trial_id(r): r for r in rows}
    reference = max(champion["trials"], key=lambda t: by_trial[t]["transfer_acc"])
    decision = {"candidate": trial_id(challenger), "reference": reference, "aggregation": "macro", "noninferiority": NONINFERIORITY}
    try:
        b = paired_bootstrap(transfer_rows(decision["candidate"]), transfer_rows(reference), samples=samples, metric="acc", aggregation="macro")
    except (ValueError, OSError) as error:
        return {**decision, "accepted": False, "error": f"{type(error).__name__}: {error}"}
    delta, ci = b["macro_acc_delta"], b["ci95"]
    return {**decision, "delta": delta, "ci95": ci, "accepted": delta > 0 and ci[0] >= -NONINFERIORITY}


def next_incumbent(champion, best, rows):
    """(incumbent after a round, the challenge record or None). A replication of the champion's own config joins it
    (seeds are averaged; a config does not challenge itself); another config must win challenge()."""
    if best is None:
        return champion, None
    if champion is None:
        return as_incumbent([best]), None
    if recipe(best) == champion["recipe"]:
        by_trial = {trial_id(r): r for r in rows}
        return as_incumbent([by_trial[t] for t in champion["trials"]] + [best]), None
    decision = challenge(champion, best, rows)
    return (as_incumbent([best]) if decision["accepted"] else champion), decision




def propose(rows, base, n, seed, suite_manifest, incumbent_cfg=None, rng_seed=0):
    """n one-knob mutations of the incumbent (plus a few two-knob combos), deduplicated against every tried config."""
    rng = random.Random(rng_seed)
    parent = dict(incumbent_cfg or {**DEFAULTS, **BASE_DEFAULTS[base], "base": base, "epochs": 2})
    parent.pop("seed", None); parent.pop("train_sources", None)
    tried = {config_digest(validated_trial({**strip_seed(r["config"]), "seed": seed}, suite_manifest)) for r in rows if r["config"] and r["base"] == base}
    candidates, seen = [], set()
    for _ in range(400):
        if len(candidates) >= n: break
        k = rng.sample(list(SPACE), rng.choice([1, 1, 1, 2]))
        cfg = dict(parent)
        for knob in k:
            current = parent.get(knob, ALL_DEFAULTS.get(knob))
            choices = [v for v in SPACE[knob] if v != current]
            if not choices: continue
            cfg[knob] = rng.choice(choices)
        if cfg.get("accum", 1) > 1 and base == "Qwen/Qwen3-0.6B-Base": cfg["batch"] = 8      # effective batch 16 via accum
        cfg["seed"] = seed
        try: full = validated_trial(cfg, suite_manifest)
        except ValueError: continue
        h = config_digest(full)
        if h in tried or h in seen: continue
        seen.add(h); candidates.append(cfg)
    # always include the incumbent itself at this seed if it has not been run at this seed (replication)
    inc = {**parent, "seed": seed}
    if config_digest(validated_trial(inc, suite_manifest)) not in tried and len(candidates) < n + 1:
        candidates.insert(0, inc)
    return candidates[: n + 1]


def leaderboard_md(rows, incumbents):
    lines = ["# Leaderboard", "", f"Generated {datetime.now(timezone.utc).isoformat(timespec='minutes')} from runs/*/result.json. "
             "Selection on development partitions only; the locked test is never read here.", ""]
    for base, inc in incumbents.items():
        if inc: lines.append(f"- **{base}** incumbent: transfer {inc['transfer_acc']:.3f}, dev {inc['dev_acc']:.3f}, seeds {inc['seeds']} ({', '.join(inc['trials'])})")
    lines += ["", "| study/trial | base | seed | dev acc | transfer acc | Brier | conf-err | held-out pairs | none_present | perm flip | gates | $ | knobs |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["transfer_acc"] is None, -(r["transfer_acc"] or 0))):
        b = (r["base"] or "legacy").split("/")[-1]
        f = lambda x, p=3: "" if x is None else f"{x:.{p}f}"
        lines.append(f"| {r['study']}/{r['trial']} | {b} | {r['seed'] if r['seed'] is not None else ''} | {f(r['dev_acc'])} | {f(r['transfer_acc'])} | {f(r['transfer_brier'])} | {f(r['transfer_conf_err'])} | {f(r['heldout_pairs'],2)} | {f(r['none_present'],2)} | {f(r['perm_flip'],2)} | {'pass' if r['gates_passed'] else 'fail'} | {r['est_usd'] or ''} | {', '.join(f'{k}={v}' for k, v in knobs(r['config']).items()) or 'defaults'} |")
    return "\n".join(lines) + "\n"


def refresh_leaderboard():
    rows = collect()
    dv4, tv4 = digest(ROOT / SUITE / "manifest.json"), digest(ROOT / TRANSFER / "manifest.json")
    history = ledger()
    incumbents = {b: incumbent(rows, b, dv4, tv4, history) for b in BASE_DEFAULTS}
    write_jsonl(ROOT / "runs/leaderboard.jsonl", rows)
    (ROOT / "runs/leaderboard.md").write_text(leaderboard_md(rows, incumbents), encoding=ENCODING)
    return rows, incumbents, dv4, tv4


def update_plan(section_text):
    """Rewrite the '## Autoresearch log' section of PLAN.md (created if absent)."""
    plan = ROOT / "PLAN.md"; text = plan.read_text(encoding=ENCODING)
    marker = "## Autoresearch log"
    head = text.split(marker)[0].rstrip() + "\n\n"
    plan.write_text(head + marker + "\n\n" + section_text.strip() + "\n", encoding=ENCODING)


def run_round(base, n, name, seeds, spend_start, spend_cap, timeout=None):
    rows, incumbents, dv4, tv4 = refresh_leaderboard()
    manifest = read_manifest(ROOT / SUITE)
    inc = incumbents.get(base)
    plan = []
    for seed in seeds:
        plan += propose(rows, base, n, seed, manifest, inc and inc["config"], rng_seed=source_seed(seed, name))   # reproducible from the round's name and seed
    plan = plan[:8]
    if not plan: raise SystemExit("nothing new to try in the space at these seeds")
    plan_path = ROOT / "experiments/auto" / f"{name}.json"; plan_path.parent.mkdir(exist_ok=True)
    write_json(plan_path, plan)
    minutes = TRIAL_MINUTES[base]; timeout = timeout or min(7200, int(minutes * 60 * 2.2))
    bound = len(plan) * timeout / 3600 * (H100_RATE + 0.5)
    spent = metered_spend()
    if spent is not None and spent - spend_start + bound > spend_cap:
        raise SystemExit(f"round would exceed spend cap: spent ${spent - spend_start:.2f} + bound ${bound:.2f} > ${spend_cap}")
    print(f"[{name}] {len(plan)} trials on {base}; incumbent {inc and inc['transfer_acc']}; bound ${bound:.2f}; spent so far ${0 if spent is None else spent - spend_start:.2f}", flush=True)
    cmd = [sys.executable, "-m", "modal", "run", "modal_app.py::study", "--suite", SUITE, "--plan", str(plan_path.relative_to(ROOT)),
           "--name", name, "--transfer", TRANSFER, "--budget", f"{bound + 1:.2f}", "--timeout", str(timeout)]
    log = ROOT / "runs" / f"{name}.log"
    with log.open("w", encoding=ENCODING) as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    rows, _, _, _ = refresh_leaderboard()
    new = [r for r in rows if r["study"] == name]
    best = max((r for r in new if eligible(r, dv4, tv4)), key=score, default=None)
    after, decision = next_incumbent(inc, best, rows)
    summary = {"round": name, "base": base, "trials": len(plan), "completed": len(new), "rc": rc,
               "best": best and {"trial": best["trial"], "transfer_acc": best["transfer_acc"], "dev_acc": best["dev_acc"], "knobs": strip_seed(best["config"])},
               "challenge": decision,
               "incumbent_after": after and {k: after[k] for k in ("transfer_acc", "dev_acc", "seeds", "trials")},
               "spend_since_start": None if (s := metered_spend()) is None else round(s - spend_start, 2), "at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    (ROOT / "runs/autoresearch.jsonl").open("a", encoding=ENCODING).write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def plan_section():
    rows, incumbents, dv4, tv4 = refresh_leaderboard()
    log = read_jsonl(ROOT / "runs/autoresearch.jsonl") if (ROOT / "runs/autoresearch.jsonl").exists() else []
    lines = ["Maintained by `kev.autoresearch`; full table in [`runs/leaderboard.md`](runs/leaderboard.md). Selection uses development partitions only.", ""]
    for b, inc in incumbents.items():
        lines.append(f"- **{b.split('/')[-1]}** incumbent (v4 suites): transfer {inc['transfer_acc']:.3f}, dev {inc['dev_acc']:.3f}, seeds {inc['seeds']}, knobs `{json.dumps(knobs(inc['config']))}`" if inc else f"- **{b.split('/')[-1]}**: no eligible trial yet")
    lines += ["", "| round | base | trials | best transfer | best knobs | challenge (macro delta, 95% CI) | incumbent after | spend |", "|---|---|---|---|---|---|---|---|"]
    for e in log:
        b = e["best"] or {}; ia = e["incumbent_after"] or {}; c = e.get("challenge")
        verdict = "" if not c else c.get("error") or f"{c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] {'accepted' if c['accepted'] else 'kept'}"
        lines.append(f"| {e['round']} | {e['base'].split('/')[-1]} | {e['completed']}/{e['trials']} | {b.get('transfer_acc', float('nan')):.3f} | `{json.dumps(knobs(b.get('knobs') or {}))}` | {verdict} | {ia.get('transfer_acc', float('nan')):.3f} | ${e['spend_since_start']} |")
    return "\n".join(lines)


def compare(studies, reference, tasks=("mmlu", "paws", "qnli", "emotion", "tweet_offensive", "contrastive_deadline")):
    """Print every trial of the given studies with a record-clustered paired bootstrap on transfer accuracy vs `reference`
    (a runs/<study>/<trial> path). Development-set selection only."""
    def rows(p): return read_json(ROOT / p / "transfer/rows.json")
    ref_rows = rows(reference)
    print(f"reference: {reference}")
    print(f"{'trial':34} {'dev':>6} {'trf':>6} {'brier':>6} {'cerr':>6} {'pairs':>6} {'d trf':>7} {'ci95':>18}  knobs / tasks")
    for study in studies:
        for d in sorted((ROOT / "runs" / study).iterdir()):
            if not (d / "result.json").exists(): continue
            r = read_json(d / "result.json"); tr = r.get("transfer")
            if not tr: continue
            cfg = r["provenance"]["config"]
            try:
                b = paired_bootstrap(rows(f"runs/{study}/{d.name}"), ref_rows, metric="acc"); delta, ci = b["macro_acc_delta"], [round(x, 3) for x in b["ci95"]]
            except ValueError:
                delta, ci = float("nan"), "n/a (different suite)"
            print(f"{study + '/' + d.name:34} {r['clean']['acc']:6.3f} {tr['clean']['acc']:6.3f} {tr['clean']['brier']:6.3f} {tr['clean']['confident_error_rate']:6.3f} {tr['paired_flip']['both_correct_rate']:6.2f} {delta:+7.3f} {str(ci):>18}  {knobs(cfg)} {({k: round(tr['tasks'][k]['acc'], 2) for k in tasks if k in tr['tasks']})}")


def release_check(study):
    """Release screen across seeds: every trial of the same config in the study must pass its gates (including the 70%
    held-out-pair screen) - a single seed clearing the bar is not enough. Prints the verdict per config."""
    rows = [r for r in collect() if r["study"] == study]
    by_cfg = defaultdict(list)
    for r in rows: by_cfg[recipe(r)].append(r)
    verdicts = {}
    for h, group in by_cfg.items():
        seeds = sorted(r["seed"] for r in group)
        passed = all(r["gates_passed"] for r in group)
        pairs = [round(r["heldout_pairs"], 2) for r in group]
        failing = sorted({k for r in group for k, v in r["gates"].items() if not v})
        base = group[0]["base"].split("/")[-1]
        verdicts[base] = {"seeds": seeds, "all_gates_passed": passed and len(seeds) >= 2, "heldout_pairs": pairs,
                          "transfer_acc": [round(r["transfer_acc"], 3) for r in group], "failing_gates": failing,
                          "candidate": passed and len(seeds) >= 2, "trials": [f"{r['study']}/{r['trial']}" for r in group]}
        print(f"{base}: seeds {seeds} pairs {pairs} transfer {verdicts[base]['transfer_acc']} -> {'RELEASE CANDIDATE (gated locked read allowed)' if verdicts[base]['candidate'] else 'not a candidate: ' + ', '.join(failing or ['fewer than two seeds'])}")
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("leaderboard")
    p = sub.add_parser("release-check"); p.add_argument("--study", required=True)
    p = sub.add_parser("compare"); p.add_argument("--studies", required=True); p.add_argument("--reference", default="runs/v4-4b-baseline/01-trial-1")
    p = sub.add_parser("propose"); p.add_argument("--base", required=True); p.add_argument("--n", type=int, default=8); p.add_argument("--seed", type=int, default=0); p.add_argument("--out", required=True)
    for cmd in ("round", "loop"):
        p = sub.add_parser(cmd); p.add_argument("--base", required=True); p.add_argument("--n", type=int, default=7)
        p.add_argument("--seeds", default="0"); p.add_argument("--spend-start", type=float, required=True); p.add_argument("--spend-cap", type=float, required=True)
        p.add_argument("--timeout", type=int)
        if cmd == "round": p.add_argument("--name", required=True)
        else: p.add_argument("--rounds", type=int, default=3); p.add_argument("--prefix", default="auto")
    a = ap.parse_args()
    if a.cmd == "release-check":
        release_check(a.study); return
    if a.cmd == "compare":
        compare(a.studies.split(","), a.reference); return
    if a.cmd == "leaderboard":
        rows, inc, _, _ = refresh_leaderboard(); update_plan(plan_section()); print(f"{len(rows)} trials;", {b.split('/')[-1]: (i and round(i['transfer_acc'], 3)) for b, i in inc.items()})
    elif a.cmd == "propose":
        rows, inc, dv4, tv4 = refresh_leaderboard(); manifest = read_manifest(ROOT / SUITE)
        write_json(a.out, propose(rows, a.base, a.n, a.seed, manifest, inc[a.base] and inc[a.base]["config"])); print(Path(a.out).read_text(encoding=ENCODING))
    else:
        seeds = [int(x) for x in a.seeds.split(",")]
        names = [a.name] if a.cmd == "round" else [f"{a.prefix}-{a.base.split('/')[-1].split('-')[1].lower()}-r{i}-{int(time.time()) % 100000}" for i in range(a.rounds)]
        for name in names:
            run_round(a.base, a.n, name, seeds, a.spend_start, a.spend_cap, a.timeout)
            update_plan(plan_section())
            subprocess.run(["git", "add", "-A"], cwd=ROOT); subprocess.run(["git", "commit", "-qm", f"autoresearch: {name}"], cwd=ROOT)
            subprocess.run(["git", "push", "-q"], cwd=ROOT)


if __name__ == "__main__":
    main()
