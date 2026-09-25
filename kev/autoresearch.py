"""Unattended research sessions on the round harness, plus the study leaderboard.

    uv run python -m kev.autoresearch session experiments/rounds/r19.json [...] --spend-start 1377 --spend-cap 300
    uv run python -m kev.autoresearch leaderboard                 # runs/leaderboard.{jsonl,md} from every trial's result.json
    uv run python -m kev.autoresearch release-check --study X     # every seed of a config must pass its gates
    uv run python -m kev.autoresearch compare --studies a,b --reference runs/<study>/<trial>

A session runs registered rounds in order through kev.rounds (validate, launch the studies, watch: pull and read each
finished trial, then the read-out). What it may run is what the specs register: config-only trials through
kev.experiment.execute_trial with full provenance, and the reads the rule names. It never changes the evaluator, the
frozen suites or the gates, and it never confirms: test partitions and the locked read are read once, by a deliberate
`python -m kev.rounds confirm` after a read-out names a candidate (the session prints those commands).

Budget: before each round, `modal billing summary` is read; the session stops when metered spend since --spend-start
plus the round's study budgets (each at least its admission bound, checked by kev.rounds.validate) would pass
--spend-cap. Each round appends a line to runs/autoresearch-sessions.jsonl.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kev import rounds
from kev.experiment import CHOICE_DEFAULTS, DEFAULTS
from kev.metrics import paired_bootstrap
from kev.suite import ENCODING, read_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
H100_RATE = 3.95
SNAPSHOT_KEYS = ("snapshot_fractions", "snapshot_every_steps")   # when a full-weight trial saves intermediate checkpoints: not what it trains
INFRA_KEYS = ("base", "seed", "base_revision", "dtype", "checkpointing", "batch", "accum", "perm_frac", "shared_prefix", *SNAPSHOT_KEYS)   # execution shape, not recipe
ALL_DEFAULTS = {**DEFAULTS, **CHOICE_DEFAULTS}   # what kev.train does when a knob is not given


def config_digest(value):
    """Canonical (key-sorted) digest for config identity; kev.suite.record_digest keeps insertion order for provenance."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def knobs(cfg):
    """The recipe a config expresses: every non-default, non-infrastructure parameter."""
    return {k: v for k, v in sorted(cfg.items()) if k not in INFRA_KEYS and v != ALL_DEFAULTS.get(k)}


def recipe(row):
    """A trial's recipe identity: its config without the seed (config_sha256 hashes the seed too) and snapshot schedule."""
    return config_digest({k: v for k, v in row["config"].items() if k != "seed" and k not in SNAPSHOT_KEYS})


# --- sessions --------------------------------------------------------------------------------------------------------

def metered_spend():
    out = subprocess.run([sys.executable, "-m", "modal", "billing", "summary", "--json"], capture_output=True, text=True, cwd=ROOT)
    return float(json.loads(out.stdout)["metered_cost"]) if out.returncode == 0 else None


def session(paths, spend_start, spend_cap, spend=metered_spend, log=print):
    """Run each registered round to its read-out while the spend cap allows; returns the ledger entries written."""
    entries = []
    for path in paths:
        spec = rounds.load(path)
        problems = rounds.launchable(spec)
        if problems: log(f"{path}: does not validate, session stops:\n" + "\n".join(problems)); break
        spent, bound = spend(), sum(s["budget"] for s in spec["studies"].values())
        if spent is None: log("cannot read `modal billing summary`; session stops"); break
        if spent - spend_start + bound > spend_cap:
            log(f"round {spec['round']}: spent ${spent - spend_start:.2f} + budgets ${bound:.2f} > cap ${spend_cap:.2f}; session stops"); break
        rounds.launch_studies(spec)
        report = rounds.watch(spec)
        entry = {"round": spec["round"], "spec": str(path), "spend_before": round(spent - spend_start, 2), "budgets": bound, "candidates": report["candidates"],
                 "at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
        with (ROOT / "runs/autoresearch-sessions.jsonl").open("a", encoding=ENCODING) as f: f.write(json.dumps(entry) + "\n")
        entries.append(entry)
        for arm in filter(None, report["candidates"].values()):
            for stage in spec.get("confirm", {}):
                log(f"candidate {arm}: uv run python -m kev.rounds launch-reads {path} --stage {stage} --arm {arm}  then  confirm {path} --stage {stage} --arm {arm}")
    return entries


# --- leaderboard -----------------------------------------------------------------------------------------------------

def collect():
    """Every completed trial in runs/*/ with its provenance, as flat leaderboard rows."""
    rows = []
    for result in sorted(ROOT.glob("runs/*/*/result.json")):
        r = read_json(result)
        prov = r.get("provenance", {}); cfg = prov.get("config") or {}
        tr = r.get("transfer") or {}
        rows.append({"study": result.parent.parent.name, "trial": result.parent.name, "legacy": prov.get("legacy_checkpoint", False),
                     "base": cfg.get("base"), "seed": cfg.get("seed"), "config": cfg, "config_sha256": prov.get("config_sha256"),
                     "suite_sha256": prov.get("suite_sha256"), "transfer_suite_sha256": tr.get("suite_sha256"), "git": prov.get("git_commit", "")[:8],
                     "dev_acc": r["clean"]["acc"], "dev_nll": r["clean"]["nll"], "dev_ece": r["clean"]["ece"],
                     "transfer_acc": tr.get("clean", {}).get("acc"), "transfer_brier": tr.get("clean", {}).get("brier"),
                     "transfer_conf_err": tr.get("clean", {}).get("confident_error_rate"),
                     "heldout_pairs": (tr.get("paired_flip") or {}).get("both_correct_rate"),
                     "none_present": r["variants"].get("none_present", {}).get("acc"), "perm_flip": r["permutation"]["flip_rate"],
                     "gates": r["gates"]["checks"], "gates_passed": r["gates"]["passed"],
                     "train_s": (r.get("training_resources") or {}).get("wall_seconds"), "wall_s": r.get("wall_seconds"),
                     "est_usd": round(H100_RATE * r.get("wall_seconds", 0) / 3600, 2) if prov.get("device") == "cuda" else None})
    return rows


def leaderboard_md(rows):
    lines = ["# Leaderboard", "", f"Generated {datetime.now(timezone.utc).isoformat(timespec='minutes')} from runs/*/result.json. "
             "Selection on development partitions only; the locked test is never read here. Rounds are decided by their "
             "registered rules (`python -m kev.rounds readout`), not by this table.", "",
             "| study/trial | base | seed | dev acc | transfer acc | Brier | conf-err | held-out pairs | none_present | perm flip | gates | $ | knobs |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["transfer_acc"] is None, -(r["transfer_acc"] or 0))):
        b = (r["base"] or "legacy").split("/")[-1]
        f = lambda x, p=3: "" if x is None else f"{x:.{p}f}"
        lines.append(f"| {r['study']}/{r['trial']} | {b} | {r['seed'] if r['seed'] is not None else ''} | {f(r['dev_acc'])} | {f(r['transfer_acc'])} | {f(r['transfer_brier'])} | {f(r['transfer_conf_err'])} | {f(r['heldout_pairs'],2)} | {f(r['none_present'],2)} | {f(r['perm_flip'],2)} | {'pass' if r['gates_passed'] else 'fail'} | {r['est_usd'] or ''} | {', '.join(f'{k}={v}' for k, v in knobs(r['config']).items()) or 'defaults'} |")
    return "\n".join(lines) + "\n"


def refresh_leaderboard():
    rows = collect()
    write_jsonl(ROOT / "runs/leaderboard.jsonl", rows)
    (ROOT / "runs/leaderboard.md").write_text(leaderboard_md(rows), encoding=ENCODING)
    return rows


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
    for group in by_cfg.values():
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
    p = sub.add_parser("session"); p.add_argument("specs", nargs="+"); p.add_argument("--spend-start", type=float, required=True); p.add_argument("--spend-cap", type=float, required=True)
    sub.add_parser("leaderboard")
    p = sub.add_parser("release-check"); p.add_argument("--study", required=True)
    p = sub.add_parser("compare"); p.add_argument("--studies", required=True); p.add_argument("--reference", required=True, help="runs/<study>/<trial>")
    a = ap.parse_args()
    if a.cmd == "session": session(a.specs, a.spend_start, a.spend_cap)
    elif a.cmd == "leaderboard": print(f"{len(refresh_leaderboard())} trials -> runs/leaderboard.md")
    elif a.cmd == "release-check": release_check(a.study)
    else: compare(a.studies.split(","), a.reference)


if __name__ == "__main__":
    main()
