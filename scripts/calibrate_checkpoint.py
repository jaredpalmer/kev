"""Fit one temperature on held-out rows and write it into a checkpoint's head.pt, so every loader (kev.serve,
kev.benchmark, the Space, third-party harnesses) serves calibrated probabilities by default. It also reports an
out-of-fold, group-disjoint cross-validated calibration estimate with bootstrap intervals, so the in-sample fit can be
checked against held-out records; the value written is always the full fit.

The fit rows must be held-out DATASETS, not held-out items of what the checkpoint trained on. The script refuses rows that
share data with the checkpoint's training (kev.rounds.pool_conflicts, the check a round's temperature pool gets): rows of
its own training suite or of a component of it, rows of sources it trained on, or the calibration / development partition
of any training corpus. That is round 19's failure mode: its SFT arms were served at T 0.955 fitted on sft-v1 development
rows (held-out items of the training sources, in distribution) and missed breadth-v1 calibration (ECE 0.059); a pool of
held-out datasets gave 0.0085 on the same checkpoint (round 20). The checkpoint's training comes from head.pt (kev.train
records the suite's manifest hash, its args and `data`) or the trial's provenance.json beside the checkpoint; each rows
file's suite and partition from the kev.benchmark report.json beside it, or from the trial it belongs to
(<trial>/{calibration,development,transfer}/rows.json). Rows or training this checkout cannot place are refused too.
`--allow-in-distribution` fits anyway, prints a warning and records it (with every conflict) in
head.pt["temperature_fit"]["in_distribution"]; head.pt["temperature_fit"]["fit_rows"] records each rows file's suite,
partition, sources and question count either way. A `path:source,...` allowlist must name sources the rows' suite lists
(or, for rows it cannot place, sources present in them): a typo is refused. `--temperature T` writes a value without
fitting and needs `--reason`, recorded in head.pt["temperature_fit"].

A round that registers a temperature pool (kev.rounds, spec `temperature`: round 20) ships the temperature fitted on that
pool, selected the same way (kev.rounds.select_rows): repeat --rows, limit a file to some sources with `path:source,...`,
and drop the records of --exclude_rows, e.g. for an interpolated checkpoint with no trial of its own

    uv run python scripts/calibrate_checkpoint.py --run <checkpoint> --rows runs/r20-27b-a-w70-r3cal/rows.json:composition_holdout,emotion,legacy_holdout,mmlu,paws,qnli,sciq,tweet_offensive \
        --rows runs/r20-27b-a-w70-v9/rows.json:mmlu_pro --exclude_rows runs/r20-27b-a-w70-transfer4/rows.json [--transfer runs/.../rows.json]

The optional --transfer rows are reported, never fitted. Argmax never changes; accuracy is identical before and after.
KEV_TEMPERATURE=1.0 restores raw logits at load time. Rows saved at a temperature are restored to raw logits first
(kev.metrics.raw_row), so served reads can be pooled. Every temperature shipped before round 20 was fitted on the
checkpoint's own decision-v7 (or suite) development rows, which this script now refuses without --allow-in-distribution.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import TEMPERATURE_FIT as FIT, TEMPERATURE_FIT_METHOD, cross_validated_temperature, fit_temperature, metrics, raw_row, recorded, scored_rows  # noqa: E402
from kev.checkpoint import read_meta, write_meta  # noqa: E402
from kev.rounds import ROUND_19, listed_sources, pool_conflicts, recorded_training, select_rows, suite_by_digest, suite_manifest  # noqa: E402
from kev.suite import read_json  # noqa: E402


def _reads(rows):
    """--rows values (`path` or `path:source,...`) as kev.rounds.select_rows reads: (path, sources or None)."""
    return [(path, sources.split(",") if sources else None) for path, _, sources in (r.partition(":") for r in rows)]


def fit_rows(rows, exclude=()):
    """The clean rows a temperature is fitted on, at raw logits: --rows files (each `path` or `path:source,...`) pooled and
    filtered by kev.rounds.select_rows, as a round's temperature pool is."""
    return [raw_row(recorded(r)) for r in select_rows(_reads(rows), exclude)[0] if r["variant"] == "clean"]


def allowlist_typos(rows):
    """Each `path:source,...` whose allowlist names a source its suite does not contain (a typo would silently drop out): the
    sources its suite's manifest lists (kev.rounds.listed_sources, as for a round's pool) when rows_origin places the rows,
    else the sources present in the rows file."""
    out = []
    for path, sources in _reads(rows):
        if not sources: continue
        suite = rows_origin(path)[0]
        known = listed_sources(suite_manifest(suite)) if suite else {r["source"] for r in read_json(path)}
        if missing := sorted(set(sources) - known): out.append(f"{path}: {missing} not among the sources of {suite or 'these rows'} {sorted(known)[:10]}")
    return out


def rows_origin(path):
    """(suite dir, partition) a rows.json was scored on: the kev.benchmark report.json beside it (suite_sha256, split), or,
    for a trial's own reads (<trial>/{calibration,development,transfer}/rows.json), the trial's provenance.json and
    result.json; (None, None) when neither says (custom --data rows, a suite this checkout lacks)."""
    here = Path(path).parent
    report = here / "report.json"
    if report.exists() and "suite_sha256" in (r := read_json(report)):
        return suite_by_digest(r["suite_sha256"]), r.get("split", "development")
    trial = here.parent
    if here.name in ("calibration", "development") and (trial / "provenance.json").exists():
        return suite_by_digest(read_json(trial / "provenance.json").get("suite_sha256")), here.name
    if here.name == "transfer" and (trial / "result.json").exists():
        return suite_by_digest((read_json(trial / "result.json").get("transfer") or {}).get("suite_sha256")), "development"
    return None, None


def checkpoint_training(run):
    """kev.rounds.Training of a checkpoint: what kev.train recorded in head.pt (suite_sha256, args.suite, args.data; kept
    by scripts/interpolate_checkpoint.py), else the trial's provenance.json beside the checkpoint; None when neither names
    a suite of this checkout."""
    extra = read_meta(run).extra
    args = extra.get("args") or {}
    training = recorded_training(extra.get("suite_sha256"), args.get("suite"), args.get("data"))
    provenance = Path(run).parent / "provenance.json"
    if training is None and provenance.exists():
        p = read_json(provenance)
        training = recorded_training(p.get("suite_sha256"), data=p.get("config", {}).get("data"))
    return training


def fit_report(run, rows, exclude=()):
    """(fit_rows record, training suites or None, problems): each --rows file's suite, partition, sources and knowable
    question count, and every reason the fit set is in distribution for the checkpoint (empty = held out)."""
    fitted = []
    for path, sources in _reads(rows):
        suite, split = rows_origin(path)
        fitted.append({"rows": path, "suite": suite, "split": split, **({"sources": sources} if sources else {}),
                       "questions": len(scored_rows(select_rows([(path, sources)], exclude)[0]))})
    training = checkpoint_training(run)
    problems = [f"{f['rows']}: cannot tell which suite these rows were scored on (no kev.benchmark report.json with a suite of this checkout beside them, not a trial's read)" for f in fitted if f["suite"] is None]
    if training is None:
        problems.append(f"{run}: cannot tell what this checkpoint was trained on (head.pt records no suite of this checkout and there is no provenance.json beside it)")
    else:
        problems += [f"{label}: {why}" for label, why in pool_conflicts([(f["rows"], f["suite"], f["split"], f.get("sources")) for f in fitted if f["suite"]], training)]
        problems += [f"cannot list the sources of {d}, training data of {run}" for d in training.unlisted]
    return fitted, sorted(training.suites) if training else None, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rows", required=True, action="append", help="fit set: a rows.json, optionally path:source,... to keep only those sources; repeat to pool")
    ap.add_argument("--exclude_rows", action="append", default=[], help="rows.json whose record ids are dropped from the fit set; repeatable")
    ap.add_argument("--transfer", help="out-of-domain rows.json, reported before/after (never fitted)")
    ap.add_argument("--temperature", type=float, help="skip fitting (and cross-validation, and the in-distribution check: nothing is fitted) and write this value; needs --reason")
    ap.add_argument("--reason", help="with --temperature: where the value comes from, recorded in head.pt (e.g. 'copied from the pool fit of runs/r20-readout')")
    ap.add_argument("--allow-in-distribution", action="store_true", help="fit even on rows that share data with the checkpoint's training "
                    "(round 19's failure mode); warned about and recorded in head.pt")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.temperature is not None and not (a.reason or "").strip(): ap.error("--temperature needs --reason: where the value comes from (recorded in head.pt)")
    if typos := allowlist_typos(a.rows): raise SystemExit("refusing a sources allowlist that names sources its rows do not contain (a typo shrinks the fit set):\n  " + "\n  ".join(typos))
    fitted, training_suites, problems = fit_report(a.run, a.rows, a.exclude_rows) if a.temperature is None else ([], None, [])
    if problems and not a.allow_in_distribution:
        raise SystemExit("refusing to fit a temperature on these rows:\n  " + "\n  ".join(problems) + f"\nThis is {ROUND_19}. "
                         "Pass --allow-in-distribution to fit anyway (recorded in head.pt).")
    for line in problems:
        print(f"!!! IN DISTRIBUTION (--allow-in-distribution): {line}", flush=True)
    if problems: print(f"!!! This temperature is not held out: {ROUND_19}.", flush=True)
    dev = fit_rows(a.rows, a.exclude_rows)
    T = a.temperature or fit_temperature(dev, **FIT)
    for name, rows in (("fit rows", dev), *((("transfer", [r for r in read_json(a.transfer) if r["variant"] == "clean"]),) if a.transfer else ())):
        raw, cal = metrics(rows), metrics(rows, T)
        print(f"{name:12} T={T:.2f}  acc {raw['acc']:.3f} -> {cal['acc']:.3f} | brier {raw['brier']:.3f} -> {cal['brier']:.3f} | ece {raw['ece']:.3f} -> {cal['ece']:.3f} | conf-err {raw['confident_error_rate']:.3f} -> {cal['confident_error_rate']:.3f} | cov@5% {raw['coverage_at_5pct_error']:.2f} -> {cal['coverage_at_5pct_error']:.2f}")
    meta = read_meta(a.run)
    meta.temperature = T
    if a.temperature is not None:
        meta.extra["temperature_fit"] = {"method": "manual", "reason": a.reason.strip()}
        write_meta(a.run, meta); print(f"wrote temperature {T:.2f} to {a.run}/head.pt")
        return
    cv = cross_validated_temperature(dev, folds=a.folds, seed=a.seed, **FIT)
    ci = cv["ece_ci95"]
    temperatures = ", ".join(f"{t:.2f}" for t in cv["temperatures"])
    print(f"fit rows     OOF T=[{temperatures}] ece raw {cv['raw']['ece']:.3f} [{ci['raw'][0]:.3f}, {ci['raw'][1]:.3f}]"
          f" -> oof {cv['out_of_fold']['ece']:.3f} [{ci['out_of_fold'][0]:.3f}, {ci['out_of_fold'][1]:.3f}]"
          f"  delta [{ci['delta'][0]:.3f}, {ci['delta'][1]:.3f}] separated={cv['separated']}")
    meta.extra["temperature_fit"] = {"rows": a.rows[0] if len(a.rows) == 1 else a.rows, **({"exclude_rows": a.exclude_rows} if a.exclude_rows else {}),
                                     "n": len(dev), "method": TEMPERATURE_FIT_METHOD, "cross_validation": cv,
                                     "fit_rows": fitted, "training_suites": training_suites,
                                     **({"in_distribution": {"allowed": True, "problems": problems}} if problems else {})}
    write_meta(a.run, meta); print(f"wrote temperature {T:.2f} to {a.run}/head.pt")


if __name__ == "__main__":
    main()
