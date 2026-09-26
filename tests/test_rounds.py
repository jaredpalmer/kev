"""The round harness (kev.rounds): spec validation, the benchmark job codec, the watcher's resume and network handling, and
reproduction of the committed read-outs and verdicts of rounds 5-20 from saved rows, and round 20's temperature pools,
transfer reads and checkpoint arms on a synthetic round; the calibration guards (a temperature pool that shares data with
an arm's training is refused, every arm's temperature source is recorded, scripts/calibrate_checkpoint.py refuses
in-distribution rows) and calibration by state length.

Offline vs archive. Rounds 5-18 ran on the research branch; their trial rows, reads and most committed outputs live on the
git tag `research-archive-2026-09-24`, not on main. This checkout carries everything the round-5 read-out and the round-15
locked verdict need (the released Kev-0.8B's confirmation), so those two run everywhere, CI included. Round 19's read-out
runs wherever its trials' private development rows can be fetched (test_readout_reproduces_round_19); round 20's six
interpolations reproduce everywhere, its whole read-out where those rows can be fetched. Every other
reproduction skips unless KEV_ROUNDS_ROOT points at a checkout that has the rows and the outputs: a worktree of the tag
(`git worktree add /tmp/kev-archive research-archive-2026-09-24`, whose gitignored trial rows are not in git either) or the
checkout the rounds ran in:
    KEV_ROUNDS_ROOT=/path/to/kev uv run --extra serve python -m pytest tests/test_rounds.py -q
The per-round scripts (on the tag) wrote different key names for the same numbers; the legacy_* functions below map them.
Every number is compared: bit for bit on macOS, floats to 1e-12 relative elsewhere (same()).
"""
import math
import os
import socket
from pathlib import Path

import pytest

from kev import rounds
from kev.suite import ADMISSION_TOKENIZER, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("KEV_ROUNDS_ROOT", ROOT))
SPECS = sorted((ROOT / "experiments/rounds").glob("r*.json"), key=lambda p: int(p.stem[1:]))


# --- spec validation -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_round_spec_is_well_formed(path):
    spec = rounds.load(path)
    problems, archived = rounds.validate(spec, ROOT, rows=False)   # structure, suites, plans against their manifests, budgets
    assert problems == []
    assert all("not in this checkout" in a for a in archived)       # plans and suites a recorded round names, kept on the tag


def test_validation_names_what_is_wrong():
    spec = rounds.load(ROOT / "experiments/rounds/r10.json")
    spec["arms"]["9b-x"] = {"trial": "runs/nowhere/00-trial-0", "parent": "missing"}
    spec["rule"]["panels"]["primary"]["reads"].append("nope")
    spec["rule"]["criteria"]["bad"] = {"left": "primary.acc.middle", "op": "~", "right": 0}
    problems = "\n".join(rounds.validate(spec, ROOT, rows=False, plans=False).problems)
    for expected in ("arm 9b-x: unknown parent", "unknown read tags ['nope']", "criterion bad: op '~'", "criterion bad: unknown path 'primary.acc.middle'"):
        assert expected in problems


def test_a_recorded_round_lists_what_is_archived_and_a_new_round_fails(tmp_path):
    """A read-out round names runs this checkout may not carry: listed, not fatal. The same spec as a new round (no
    `archive`) fails on the missing parent reads, and a recorded round is never launched again."""
    spec = rounds.load(ROOT / "experiments/rounds/r14.json")
    problems, archived = rounds.validate(spec, tmp_path, plans=False)   # an empty checkout: no rows
    assert problems == [] and any("parent 4b-r10" in a and "development/rows.json not in this checkout" in a for a in archived)
    assert "is a record" in rounds.launchable(spec)[0]
    new = {k: v for k, v in spec.items() if k != "archive"}
    problems, archived = rounds.validate(new, tmp_path, plans=False)
    assert archived == [] and any("parent read hard not in this checkout" in p for p in problems)


def test_validation_checks_budgets():
    spec = {k: v for k, v in rounds.load(ROOT / "experiments/rounds/r5.json").items() if k != "archive"}   # its plan and suite are on main
    assert rounds.validate(spec, ROOT, rows=False).problems == []
    spec["studies"]["r5-combined"]["budget"] = 1
    assert any("exceeds budget" in p for p in rounds.validate(spec, ROOT, rows=False).problems)


# --- benchmark jobs --------------------------------------------------------------------------------------------------

def test_benchmark_jobs_keep_a_pinned_hub_revision():
    """Round 10's parent test reads failed because `repo@sha@suite@name` shifted every field (PLAN.md at research-archive-2026-09-24, Night 3 incidents)."""
    from modal_app import parse_jobs
    jobs = ",".join([rounds.bench_job("jaredpalmer/kev-4b@957b91e762e883935830246eeb02381f9d2694b6", "evals/hard-v1", "r10c-4b-parent-hardtest", "--allow-test"),
                     rounds.bench_job("/runs/r10-skills/00-trial-0/checkpoint", "evals/external/semif-v1", "r10-4b-skills-semif"),
                     "jaredpalmer/kev-9b@evals/v9/transfer-v9@x-v9@--date_facts --rotations 4"])
    assert parse_jobs(jobs) == [("jaredpalmer/kev-4b@957b91e762e883935830246eeb02381f9d2694b6", "evals/hard-v1", "r10c-4b-parent-hardtest", "--allow-test"),
                                ("/runs/r10-skills/00-trial-0/checkpoint", "evals/external/semif-v1", "r10-4b-skills-semif", ""),
                                ("jaredpalmer/kev-9b", "evals/v9/transfer-v9", "x-v9", "--date_facts --rotations 4")]
    for bad in ("run@suite", "a@b@c@d@e", "run@@name", "a@b@c@d@e@--flag"):
        with pytest.raises(ValueError):
            parse_jobs(bad)
    with pytest.raises(ValueError):
        rounds.bench_job("run", "evals/x", "name@2")


def test_read_commands_batch_one_arm_and_skip_existing_reads(tmp_path):
    spec = rounds.load(ROOT / "experiments/rounds/r17.json")
    (tmp_path / "runs/r17-27b-r10k-lr2e5-hard").mkdir(parents=True)
    write_json(tmp_path / "runs/r17-27b-r10k-lr2e5-hard/rows.json", [])
    [bench] = rounds.read_commands(spec, "27b-r10k-lr2e5", root=tmp_path)
    assert bench[:4] == ["modal", "run", "--detach", "modal_app.py::benchmarks"] and bench[-4:] == ["--gpu", "H200", "--timeout", "14400"]
    jobs = bench[bench.index("--jobs") + 1].split(",")
    assert len(jobs) == 7 and not any(j.endswith("-hard") for j in jobs)                          # hard exists locally
    assert "/runs/r17-27b/00-trial-0/checkpoint@evals/devtools-v1@r17-27b-r10k-lr2e5-devtools" in jobs
    [locked] = rounds.read_commands(spec, "27b-r10k-lr2e5", stage="locked", root=tmp_path)
    assert locked[:3] == ["modal", "run", "modal_app.py::locked_test"] and locked[locked.index("--name") + 1] == "kev-27b-r17-ungated"
    assert locked[-4:] == ["--timeout", "14400", "--memory-mb", "131072"]


def test_launches_are_staggered(tmp_path, monkeypatch):
    monkeypatch.setattr(rounds, "ROOT", tmp_path); (tmp_path / "runs").mkdir()
    waits, started = [], []
    rounds.launch_commands([["modal", "run", "a"], ["modal", "run", "b"], ["modal", "run", "c"]], {"round": 1}, "log", stagger=75,
                           run=lambda cmd, **kw: started.append(cmd[2:]), sleep=waits.append)
    assert waits == [75, 75] and [c[2] for c in started] == ["a", "b", "c"]


# --- watcher ---------------------------------------------------------------------------------------------------------

def _spawned(tmp_path, calls):
    (tmp_path / "runs").mkdir(exist_ok=True)
    write_json(tmp_path / "runs/s.spawn.json", {"name": "s", "calls": calls})


def test_watcher_retries_network_errors_and_launches_reads_once(tmp_path):
    _spawned(tmp_path, {"trial-0": "fc-0", "trial-1": "fc-1"})
    script = {"fc-0": [socket.gaierror(8, "nodename nor servname provided"), ConnectionResetError(), "running", "done"],
              "fc-1": [FileExistsError("refusing to overwrite remote trial")]}   # the trial's own exception: failed, not retried
    def poll(cid):
        step = script[cid].pop(0)
        if isinstance(step, BaseException): raise step
        return step
    launched, logs = [], []
    rounds.watch_studies(["s"], lambda study, label: launched.append((study, label)), poll=poll, sleep=lambda s: None, log=logs.append, root=tmp_path)
    assert launched == [("s", "trial-0")]
    state = read_json(tmp_path / "runs/s.watch.json")["calls"]
    assert state["trial-0"]["status"] == "done" and state["trial-0"]["launched"] and state["trial-1"]["status"] == "failed"
    assert sum("network error" in line for line in logs) == 2


def test_watcher_resumes_without_relaunching(tmp_path):
    _spawned(tmp_path, {"trial-0": "fc-0", "trial-1": "fc-1"})
    write_json(tmp_path / "runs/s.watch.json", {"calls": {"trial-0": {"status": "done", "launched": True, "transient": 0}}})
    polled, launched = [], []
    def poll(cid):
        polled.append(cid); return "done"
    rounds.watch_studies(["s"], lambda study, label: launched.append(label), poll=poll, sleep=lambda s: None, log=lambda m: None, root=tmp_path)
    assert polled == ["fc-1"] and launched == ["trial-1"]


def test_watcher_retries_a_failed_launch_on_the_next_pass(tmp_path):
    _spawned(tmp_path, {"trial-0": "fc-0"})
    attempts = []
    def on_done(study, label):
        attempts.append(label)
        if len(attempts) == 1: raise OSError("pull lost the network")
    rounds.watch_studies(["s"], on_done, poll=lambda cid: "done", sleep=lambda s: None, log=lambda m: None, root=tmp_path)
    assert attempts == ["trial-0", "trial-0"] and read_json(tmp_path / "runs/s.watch.json")["calls"]["trial-0"]["launched"]


def test_watcher_gives_up_after_max_transient(tmp_path):
    _spawned(tmp_path, {"trial-0": "fc-0"})
    def poll(cid): raise ConnectionRefusedError()
    rounds.watch_studies(["s"], lambda *a: None, poll=poll, sleep=lambda s: None, log=lambda m: None, root=tmp_path, max_transient=3)
    assert read_json(tmp_path / "runs/s.watch.json")["calls"]["trial-0"]["status"] == "failed"


def test_transient_errors_are_network_errors_only():
    assert rounds.transient(socket.gaierror(8, "nodename nor servname provided")) and rounds.transient(ConnectionResetError())
    assert not rounds.transient(FileExistsError("refusing to overwrite")) and not rounds.transient(RuntimeError("CUDA out of memory"))
    assert not rounds.transient(OSError("disk full"))


def test_arm_of_maps_spawn_labels_to_arms():
    spec = rounds.load(ROOT / "experiments/rounds/r10.json")
    assert rounds.arm_of(spec, "r10-skills", "trial-1") == "4b-hard" and rounds.arm_of(spec, "r10-skills-27b", "trial-0") == "27b-skills"
    assert rounds.arm_of(spec, "r10-skills", "trial-9") is None


# --- reproduction ----------------------------------------------------------------------------------------------------

B = lambda e: {"delta": e["delta"], "ci95": e["ci95"]}   # a bootstrapped panel entry as the scripts wrote it


def legacy_documents(arm, rep, jev):
    """rounds 7, 8, 9, 11 (scripts/round7_readout.py, round8_readout.py)."""
    p = rep["panels"]
    yield from [("trial", arm["trial"], rep["trial"]), ("parent", arm["parent"], rep["parent"]), ("temperature", arm["temperature"], rep["temperature"]),
                ("docs", arm["docs"], B(p["docs"]["acc"])), ("docs_acc", arm["docs_acc"], p["docs"]["acc"]["candidate"]),
                ("parent_docs_acc", arm["parent_docs_acc"], p["docs"]["acc"]["parent"]), ("docs_vs_jev", arm["docs_vs_jev"], B(p["docs"]["versus"]["jev"]["acc"])),
                ("jev_docs_acc", jev, p["docs"]["versus"]["jev"]["acc"]["reference"]), ("pooled_external", arm["pooled_external"], B(p["pooled_external"]["acc"])),
                ("unknowable_share", arm["unknowable_share"], rep["unknowable"]["candidate"]), ("criteria", arm["criteria"], rep["criteria"]), ("passed", arm["passed"], rep["passed"])]
    if "parent_temperature" in arm: yield "parent_temperature", arm["parent_temperature"], rep["parent_temperature"]
    if "docs_brier" in arm: yield "docs_brier", arm["docs_brier"], p["docs"]["brier"]["candidate"]
    for m, e in arm["short"].items(): yield f"short.{m}", e, B(p["short"][m])
    for s, e in arm["externals"].items(): yield f"externals.{s}", (B(e), e["n"]), (B(p[s]["acc"]), p[s]["n"])


def legacy_skills(arm, rep):
    """rounds 10, 12-18 (scripts/round10_readout.py)."""
    p = rep["panels"]
    yield from [("trial", arm["trial"], rep["trial"]), ("parent", arm["parent"], rep["parent"]), ("temperature", arm["temperature"], rep["temperature"]),
                ("parent_temperature", arm["parent_temperature"], rep["parent_temperature"]), ("primary", arm["primary"], B(p["primary"]["acc"])),
                ("pooled_external", arm["pooled_external"], B(p["pooled_external"]["acc"])), ("unknowable_share", arm["unknowable_share"], rep["unknowable"]["candidate"]),
                ("hard_ece", arm["hard_ece"], {"candidate": p["hard"]["ece"]["candidate"], "parent": p["hard"]["ece"]["parent"], "delta": B(p["hard"]["ece"])}),
                ("criteria", arm["criteria"], rep["criteria"]), ("passed", arm["passed"], rep["passed"])]
    for s in ("hard", "devtools", "docs"):
        yield f"{s}_acc", list(arm[f"{s}_acc"]), [p[s]["acc"]["candidate"], p[s]["acc"]["parent"]]
        yield f"{s}_delta", arm[f"{s}_delta"], B(p[s]["acc"])
    for m, e in arm["short"].items(): yield f"short.{m}", e, B(p["short"][m])
    for s, e in arm["externals"].items(): yield f"externals.{s}", e, B(p[s]["acc"])


def legacy_round6(arm, rep):
    """scripts/round6_readout.py: arms of one size; a missing read is 'unread' and its criteria are absent."""
    p = rep["panels"]
    unread = lambda name, f: f(p[name]) if name in p else "unread"
    yield from [("trial", arm["trial"], rep["trial"]), ("temperature", arm["temperature"], rep["temperature"]), ("complete", arm["complete"], rep["complete"]),
                ("passed", arm["passed"], rep["passed"]), ("criteria", arm["criteria"], {k: v for k, v in rep["criteria"].items() if v is not None}),
                ("long", arm["long"], unread("long", lambda x: B(x["acc"]))), ("pooled_external", arm["pooled_external"], unread("pooled_external", lambda x: B(x["acc"]))),
                ("unknowable_share", arm["unknowable_share"], "unread" if rep["unknowable"]["candidate"] is None else rep["unknowable"]["candidate"])]
    if "long_acc" in arm: yield "long_acc", arm["long_acc"], p["long"]["acc"]["candidate"]
    for m, e in arm["short"].items(): yield f"short.{m}", e, B(p["short"][m])
    for m, v in arm["short_metrics"].items(): yield f"short_metrics.{m}", v, p["short"][m]["candidate"]
    for s, e in arm["externals"].items():
        yield f"externals.{s}", e, unread(s, lambda x: {**B(x["acc"]), "n": x["n"], "margin": e["margin"], "acc": x["acc"]["candidate"]})


def value(panel, metric, side):
    return panel["n"] if metric == "n" else panel[metric][side]   # paired panels have one n: the bootstrap requires identical examples


def legacy_round5(size_file, report, spec):
    """scripts/round5_confirm.py: one file per size, the candidate's criteria plus the attribution arms' deltas."""
    cand = f"{size_file['size'].replace('.', '')}-{size_file['candidate']}"
    parent = size_file["parent"]
    rep = report["arms"][cand]
    for name, t in size_file["temperature"].items():
        arm = next(a for a in report["arms"] if a.split("-", 1)[1] == name) if name != parent else None
        yield f"temperature.{name}", t, report["arms"][arm]["temperature"] if arm else rep["parent_temperature"]
        if arm is None: continue
        r = report["arms"][arm]["panels"]
        yield f"long_minus_parent.{name}", size_file["long_minus_parent"][name], B(r["long"]["acc"])
        yield f"short_minus_parent.{name}", size_file["short_minus_parent"][name], {m: B(r["short"][m]) for m in ("acc", "brier", "confident_error_rate")}
        yield f"long.{name}", size_file["long"][name], {k: value(r["long"], k, "candidate") for k in ("n", "acc", "brier")}
        yield f"short.{name}", size_file["short"][name], {k: value(r["short"], k, "candidate") for k in size_file["short"][name]}
    p = rep["panels"]
    yield "long.parent", size_file["long"][parent], {k: value(p["long"], k, "parent") for k in ("n", "acc", "brier")}
    yield "short.parent", size_file["short"][parent], {k: value(p["short"], k, "parent") for k in size_file["short"][parent]}
    yield "v9_unknowable_share", list(size_file["v9_unknowable_share"].values()), [rep["unknowable"]["candidate"], rep["unknowable"]["parent"]]
    for e, v in size_file["externals"].items(): yield f"externals.{e}", list(v.values()), [p[e]["acc"]["candidate"], p[e]["acc"]["parent"]]
    yield "criteria", size_file["criteria"], rep["criteria"]
    yield "passed", size_file["passed"], rep["passed"]


def legacy_stage(verdict, rep):
    """scripts/round8_confirm.py, round10_confirm.py, round15_confirm.py."""
    p = rep["panels"]
    yield "temperature", [verdict["temperature"]["cand"], verdict["temperature"]["parent"]], [rep["temperature"], rep["parent_temperature"]]
    yield "criteria", verdict["criteria"], rep["criteria"]
    yield "passed", verdict["passed"], rep["passed"]
    if "locked" in verdict:
        for side, key in (("cand", "candidate"), ("parent", "parent")):
            for m, v in verdict["locked"][side].items(): yield f"locked.{side}.{m}", v, p["locked"]["n"] if m == "n" else p["locked"][m][key]
        yield "acc_delta", verdict["acc_delta"], B(p["locked"]["acc"])
        if "brier_delta" in verdict: yield "brier_delta", verdict["brier_delta"], B(p["locked"]["brier"])
    elif "deltas" in verdict:   # round 15
        for s, e in verdict["deltas"].items(): yield f"deltas.{s}", e, B(p[s]["acc"])
        for s, v in verdict["acc"].items(): yield f"acc.{s}", [v["cand"], v["parent"]], [p[s]["acc"]["candidate"], p[s]["acc"]["parent"]]
        yield "pooled_skills", verdict["pooled_skills"], B(p["pooled_skills"]["acc"])
    elif "pooled" in verdict:   # round 10 family
        yield "pooled", verdict["pooled"], B(p["pooled"]["acc"])
        for s in ("hardtest", "devtest"):
            v = verdict[s]
            yield s, [v["cand"], v["parent"], v["delta"], v["n"]], [p[s]["acc"]["candidate"], p[s]["acc"]["parent"], B(p[s]["acc"]), p[s]["n"]]
        yield "hard_ece", [verdict["hard_ece"]["cand"], verdict["hard_ece"]["parent"]], [p["hardtest"]["ece"]["candidate"], p["hardtest"]["ece"]["parent"]]
    else:   # round 8 family: documents-v1 test, documents-v2
        [name] = p
        yield "acc", [verdict["acc"]["cand"], verdict["acc"]["parent"]], [p[name]["acc"]["candidate"], p[name]["acc"]["parent"]]
        yield "n", verdict["n"], p[name]["n"]
        yield "acc_delta", verdict["acc_delta"], B(p[name]["acc"])
        yield "brier_delta", verdict["brier_delta"], B(p[name]["brier"])


def same(a, b):
    """Equal, with floats allowed a 1e-12 relative difference. On macOS every number reproduces bit for bit; on some Linux
    CI runners numpy's SIMD exp/log (CPU-dispatched) moves the last digit of a Brier or a bootstrap bound. Anything that
    is not a float (a verdict, a count, a name) must match exactly."""
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(a, dict) and isinstance(b, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)): return len(a) == len(b) and all(map(same, a, b))
    return a == b


def reproduce(round_number, root=DATA):
    """(checked numbers, differences) between the committed read-out of a round and kev.rounds.readout on the same rows."""
    spec = rounds.load(ROOT / f"experiments/rounds/r{round_number}.json")
    report = rounds.readout(spec, root)
    pairs = []
    if round_number == 5:
        for size in ("9b", "4b", "0.8b"): pairs += [(f"{size}.{k}", a, b) for k, a, b in legacy_round5(read_json(root / f"runs/r5-verdict/{size}.json"), report, spec)]
    elif round_number == 6:
        for size in ("9b", "4b", "08b"):
            old = read_json(root / f"runs/r6-readout/{size}.json")
            pairs.append((f"{size}.ranking", old["ranking"], [a.split("-", 1)[1] for a in report["ranking"][size]]))
            for arm in old["arms"]: pairs += [(f"{size}-{arm['arm']}.{k}", a, b) for k, a, b in legacy_round6(arm, report["arms"][f"{size}-{arm['arm']}"])]
    else:
        old = read_json(root / f"runs/r{round_number}-readout/round{round_number}.json")
        for arm, a in old["arms"].items():
            rep = report["arms"][arm]
            if a == "not read yet": pairs.append((f"{arm}.status", False, rep["complete"])); continue
            rows = legacy_skills(a, rep) if "primary" in a else legacy_documents(a, rep, old["jev_docs_acc"])
            pairs += [(f"{arm}.{k}", x, y) for k, x, y in rows]
        pairs += [(k, v, report["candidates"].get(k.removeprefix("candidate_"))) for k, v in old.items() if k.startswith("candidate_")]
        if "excluded_duplicate_ids" in old: pairs.append(("drop_ids", old["excluded_duplicate_ids"], report["drop_ids"]))
    return len(pairs), [(k, a, b) for k, a, b in pairs if not same(a, b)]


VERDICTS = [(r, f"{size}-{stage}") for r, size, stages in ((8, "4b", ("docs", "docs2", "locked")), (10, "4b", ("tests", "locked")), (11, "08b", ("docs", "locked")),
                                                          (12, "08b", ("tests", "locked")), (15, "08b", ("tests", "locked"))) for stage in stages]


def candidate_arm(round_number, size, root=DATA):
    """The arm the committed read-out chose (round 8's read-out predates candidate_<size>: its passing arm)."""
    old = read_json(root / f"runs/r{round_number}-readout/round{round_number}.json")
    return old.get(f"candidate_{size}") or next(a for a, v in old["arms"].items() if a.startswith(size + "-") and v["passed"])


def reproduce_verdict(round_number, name, root=DATA):
    spec = rounds.load(ROOT / f"experiments/rounds/r{round_number}.json")
    size, stage = name.split("-", 1)
    pairs = list(legacy_stage(read_json(root / f"runs/r{round_number}-verdict/{name}.json"), rounds.confirm(spec, stage, candidate_arm(round_number, size, root), root)))
    return len(pairs), [(k, a, b) for k, a, b in pairs if not same(a, b)]


def absent(paths, root=DATA):
    return [p for p in paths if not (root / p).exists()]


READ_OUT = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]   # round 17's reads had not landed when this harness replaced the scripts


@pytest.mark.parametrize("round_number", READ_OUT)
def test_readout_reproduces_the_committed_round(round_number):
    spec = rounds.load(ROOT / f"experiments/rounds/r{round_number}.json")
    output = {5: "runs/r5-verdict/9b.json", 6: "runs/r6-readout/9b.json"}.get(round_number, f"runs/r{round_number}-readout/round{round_number}.json")
    missing = absent([output] + [f"{x['trial']}/development/rows.json" for x in (*spec["arms"].values(), *spec["parents"].values())])
    if missing: pytest.skip(f"archived, not in {DATA} (set KEV_ROUNDS_ROOT): {missing[:2]}")
    checked, diffs = reproduce(round_number)
    assert checked > 20 and diffs == [], diffs[:5]


@pytest.mark.parametrize("round_number,name", VERDICTS, ids=lambda x: str(x))
def test_confirm_reproduces_the_committed_verdicts(round_number, name):
    spec = rounds.load(ROOT / f"experiments/rounds/r{round_number}.json")
    size, stage = name.split("-", 1)
    missing = absent([f"runs/r{round_number}-verdict/{name}.json", f"runs/r{round_number}-readout/round{round_number}.json"])
    if not missing:
        arm = candidate_arm(round_number, size)
        sides = rounds.arm_side(spec, arm, DATA, stage), rounds.parent_side(spec, arm, DATA, stage)
        missing = absent([f"{s.trial}/development/rows.json" for s in sides]) + [s.dirs[t] for s in sides for p in spec["confirm"][stage]["panels"].values() for t in p["reads"] if not s.has(t)]
    if missing: pytest.skip(f"archived, not in {DATA} (set KEV_ROUNDS_ROOT): {missing[:2]}")
    checked, diffs = reproduce_verdict(round_number, name)
    assert checked >= 4 and diffs == [], diffs[:5]


def as_committed(report):
    """A read-out without `temperature_source` / `parent_temperature_source` (recorded per arm since the calibration guards;
    the committed read-outs of rounds 19 and 20 predate them, and every other number must reproduce as committed)."""
    new = ("temperature_source", "parent_temperature_source")
    return {**report, "arms": {a: {k: v for k, v in r.items() if k not in new} for a, r in report["arms"].items()}}


def test_readout_reproduces_round_19():
    """Round 19 was read out by this harness. Its read-out and every read it scores are committed; the arms' development
    rows (their temperatures' fit set) carry sft-v1 record ids, so they come from the private dataset the manifest names
    (scripts/private_rows.py), fetched into this checkout's gitignored runs/: the test skips for an account without access.
    Every arm now records that its temperature came from sft-v1 development rows, a training corpus (round 19's mistake)."""
    from scripts.private_rows import restore
    try:
        restore("runs/r19-readout/private-rows.json", ROOT)
    except PermissionError as error:
        pytest.skip(str(error))
    spec = rounds.load(ROOT / "experiments/rounds/r19.json")
    report = rounds.readout(spec, ROOT)
    assert same(as_committed(report), read_json(ROOT / "runs/r19-readout/round19.json"))
    for arm, r in report["arms"].items():
        assert r["temperature_source"] == {"kind": "trial development rows", "rows": f"{spec['arms'][arm]['trial']}/development", "suite": "evals/sft-v1", "training_corpus": True}
    assert rounds.table(report).count("!!!") == 3 and "round 19's failure mode" in rounds.table(report)


def test_readout_reproduces_round_20():
    """Round 20's arms are served at a pooled temperature fitted on public reads, so the six interpolations (checkpoints
    without a trial) reproduce from committed rows everywhere. The two finals count as finished only once their trials'
    development rows are present, and those are round 19's private rows: with access the whole read-out is compared, and
    without it the interpolations, the ranking and the verdict are compared."""
    from scripts.private_rows import restore
    try:
        restore("runs/r19-readout/private-rows.json", ROOT)
        private = True
    except PermissionError:
        private = False
    spec = rounds.load(ROOT / "experiments/rounds/r20.json")
    report, committed = rounds.readout(spec, ROOT), read_json(ROOT / "runs/r20-readout/round20.json")
    interpolated = [a for a, x in spec["arms"].items() if not x.get("trial")]
    for arm in interpolated + (list(spec["arms"]) if private else []):
        source = report["arms"][arm]["temperature_source"]
        assert source["kind"] == "pool" and source["questions"] == 648 and source["reads"] == report["arms"][arm]["temperature_fit"]["reads"]
    assert "!!!" not in rounds.table(report)
    report = as_committed(report)
    if private:
        assert same(report, committed)
        return
    assert len(interpolated) == 6
    assert same({a: report["arms"][a] for a in interpolated}, {a: committed["arms"][a] for a in interpolated})
    assert report["ranking"] == committed["ranking"] and report["candidates"] == committed["candidates"] == {"27b": None}


# --- temperature pools, transfer reads and checkpoints without a trial (round 20) -------------------------------------

def _rows(root, d, ids, seed, scale=1.0, source="s"):
    """Clean rows with raw logits (3 options), one question per record id; the label follows the id, so sides pair."""
    import numpy as np
    rng, rows = np.random.default_rng(seed), []
    for i in ids:
        z = rng.normal(size=3) * scale
        z[int(i.split("/")[1]) % 3] += 2.0   # informative: the fitted temperature stays inside the grid
        p = np.exp(z - z.max()); p /= p.sum()
        rows.append({"id": i, "group": i, "question": "q", "source": source, "task": source, "type": "choice", "variant": "clean",
                     "keys": ["a", "b", "c"], "label": int(i.split("/")[1]) % 3, "p": p.tolist(), "logits": z.tolist(), "inference_temperature": 1.0})
    (root / d).mkdir(parents=True, exist_ok=True); write_json(root / d / "rows.json", rows)
    return rows


def _trained_on(root, trial, suite, data=None):
    """A trial's provenance.json as kev.experiment writes it: the manifest hash of the suite it trained on (and a `data` file)."""
    from kev.suite import digest
    (root / trial).mkdir(parents=True, exist_ok=True)
    write_json(root / trial / "provenance.json", {"config": {**({"data": data} if data else {})}, "suite_sha256": digest(ROOT / suite / "manifest.json")})


def _pool_round(root):
    """A two-arm round: a trained arm and an interpolated checkpoint (no trial), both served at a pooled temperature. The
    pool's first read carries ten rows of another source and the second repeats five transfer records, all confidently
    wrong: the sources allowlist and exclude_reads remove them."""
    write_json(root / "r99.json", {
        "round": 99, "registered": "test", "gpu": "H200",
        "reads": {tag: {"suite": "evals/v4/transfer-v4"} for tag in ("main", "cal", "v9", "transfer4")} | {"locked": {"entrypoint": "locked_test", "decision": "evals/v7/decision-v7"}},
        "temperature": {"reads": ["cal", "v9"], "sources": {"cal": ["mmlu"]}, "exclude_reads": ["transfer"]},   # mmlu: a transfer-v4 source
        "parents": {"p": {"trial": "runs/p/00-trial-0", "reads": {"main": "runs/p-main"}}},
        "arms": {"x-trained": {"trial": "runs/s/00-trial-0", "parent": "p"},
                 "x-wise": {"checkpoint": "/runs/wise/x-w50/checkpoint", "parent": "p", "transfer_read": "transfer4"}},
        "rule": {"panels": {"main": {"reads": ["main", "transfer"], "metrics": ["acc", "ece"]}},
                 "criteria": {"acc": {"left": "main.acc.lower", "op": ">", "right": -1}}, "rank": [{"by": "main.acc.delta"}]},
        "confirm": {"locked": {"candidate_reads": {"locked": "runs/locked/kev-{size}-r99-ungated/transfer"}, "panels": {"locked": {"reads": ["locked"], "metrics": ["acc"]}},
                               "criteria": {"acc": {"left": "locked.acc.candidate", "op": ">=", "right": 0}}}}})
    spec = rounds.load(root / "r99.json")
    _trained_on(root, "runs/s/00-trial-0", "evals/v7/decision-v7")   # what the pool is checked against (pool_training_problems)
    transfer_ids = [f"t/{i}" for i in range(40)]
    for d in ("runs/p/00-trial-0/development", "runs/s/00-trial-0/development"): _rows(root, d, [f"d/{i}" for i in range(60)], 1)
    for d, seed in (("runs/p/00-trial-0/transfer", 2), ("runs/s/00-trial-0/transfer", 3), ("runs/r99-x-wise-transfer4", 4)): _rows(root, d, transfer_ids, seed)
    for d, seed in (("runs/p-main", 5), ("runs/r99-x-trained-main", 6), ("runs/r99-x-wise-main", 7)): _rows(root, d, [f"m/{i}" for i in range(50)], seed)
    for arm in ("x-trained", "x-wise"):
        cal = _rows(root, f"runs/r99-{arm}-cal", [f"c/{i}" for i in range(80)], 8, source="mmlu") + _rows(root, f"runs/r99-{arm}-other", [f"o/{i}" for i in range(10)], 11, source="other")
        for r in cal[-10:]: r.update(logits=[20.0, 0.0, 0.0], p=[1.0, 0.0, 0.0], label=1)   # another source, confidently wrong
        write_json(root / f"runs/r99-{arm}-cal/rows.json", cal)
        v9 = _rows(root, f"runs/r99-{arm}-v9", [f"v/{i}" for i in range(30)] + transfer_ids[:5], 9)
        for r in v9[-5:]: r.update(logits=[20.0, 0.0, 0.0], p=[1.0, 0.0, 0.0], label=2)   # transfer records, confidently wrong
        v9 += _rows(root, f"runs/r99-{arm}-unk", [f"u/{i}" for i in range(10)], 10, scale=9, source="unknowable")
        write_json(root / f"runs/r99-{arm}-v9/rows.json", v9)
    return spec


def test_arms_are_served_at_the_pooled_temperature_and_parents_at_their_own(tmp_path):
    from kev.metrics import served
    spec = _pool_round(tmp_path)
    assert rounds.validate(spec, tmp_path, plans=False).problems == []
    report = rounds.readout(spec, tmp_path)
    pooled = [r for d in ("cal", "v9") for r in read_json(tmp_path / f"runs/r99-x-wise-{d}/rows.json") if r["id"][0] not in "to"]
    unfiltered = [r for d in ("cal", "v9") for r in read_json(tmp_path / f"runs/r99-x-wise-{d}/rows.json")]
    wise, trained = report["arms"]["x-wise"], report["arms"]["x-trained"]
    assert wise["temperature"] == served(pooled, [])[0] != served(unfiltered, [])[0]   # the filtered rows would move the fit
    assert trained["temperature"] == wise["temperature"]                               # same pool rows here, different arm
    assert wise["parent_temperature"] == rounds.temperature("runs/p/00-trial-0", tmp_path)
    assert wise["temperature_fit"] == {"reads": ["runs/r99-x-wise-cal", "runs/r99-x-wise-v9"], "sources": {"runs/r99-x-wise-cal": ["mmlu"]},
                                       "exclude_reads": ["runs/r99-x-wise-transfer4"], "questions": 110, "excluded_questions": 5}   # 80 + 30 knowable; the 10 unknowable never count
    assert trained["temperature_fit"]["exclude_reads"] == ["runs/s/00-trial-0/transfer"] and wise["checkpoint"] == "/runs/wise/x-w50/checkpoint" and wise["trial"] is None
    assert wise["complete"] and wise["panels"]["main"]["n"] == 90                     # main (50) + transfer4 standing in for transfer (40)
    side = rounds.arm_side(spec, "x-wise", tmp_path, stage="locked")                   # a stage keeps the rule-stage pool
    assert side.pool.reads == ["runs/r99-x-wise-cal", "runs/r99-x-wise-v9"] and side.dirs["locked"] == "runs/locked/kev-x-r99-ungated/transfer"


def test_calibrate_checkpoint_ships_the_pooled_temperature(tmp_path):
    """scripts/calibrate_checkpoint.py --rows path:sources ... --exclude_rows selects what the round's pool selects, so the
    temperature a release writes into head.pt is the one its round served it at."""
    from kev.metrics import TEMPERATURE_FIT, fit_temperature
    from scripts.calibrate_checkpoint import fit_rows
    spec = _pool_round(tmp_path)
    rows = fit_rows([f"{tmp_path}/runs/r99-x-wise-cal/rows.json:mmlu", f"{tmp_path}/runs/r99-x-wise-v9/rows.json"], [tmp_path / "runs/r99-x-wise-transfer4/rows.json"])
    assert fit_temperature(rows, **TEMPERATURE_FIT) == rounds.arm_side(spec, "x-wise", tmp_path).t


def test_a_missing_pool_read_leaves_the_arm_incomplete(tmp_path):
    spec = _pool_round(tmp_path)
    (tmp_path / "runs/r99-x-wise-cal/rows.json").unlink()
    wise = rounds.readout(spec, tmp_path)["arms"]["x-wise"]
    assert (wise["complete"], wise["passed"], wise["temperature"], wise["missing"]) == (False, None, None, ["candidate:runs/r99-x-wise-cal"])


def test_reads_of_a_checkpoint_without_a_trial(tmp_path):
    """Its benchmarks run on the declared checkpoint, "transfer" is read as its transfer_read, the pool reads are launched
    with the rule's, and the locked read names the checkpoint's directory as the trial (modal_app.run_locked_test reads a
    checkpoint without result.json only under an -ungated name)."""
    spec = _pool_round(tmp_path)
    for arm in ("x-trained", "x-wise"):
        for tag in ("main", "cal", "v9", "transfer4"): (tmp_path / f"runs/r99-{arm}-{tag}/rows.json").unlink(missing_ok=True)
    [bench] = rounds.read_commands(spec, "x-wise", root=tmp_path)
    jobs = bench[bench.index("--jobs") + 1].split(",")
    assert jobs == [f"/runs/wise/x-w50/checkpoint@evals/v4/transfer-v4@r99-x-wise-{t}" for t in ("cal", "main", "transfer4", "v9")]
    [bench] = rounds.read_commands(spec, "x-trained", root=tmp_path)
    assert [j.rsplit("@", 1)[1] for j in bench[bench.index("--jobs") + 1].split(",")] == [f"r99-x-trained-{t}" for t in ("cal", "main", "v9")]   # its transfer is in-trial
    [locked] = rounds.read_commands(spec, "x-wise", stage="locked", root=tmp_path)
    assert locked[locked.index("--trial") + 1] == "wise/x-w50" and locked[locked.index("--name") + 1] == "kev-x-r99-ungated"


def test_validation_of_pools_transfer_reads_and_trialless_arms(tmp_path):
    spec = _pool_round(tmp_path)
    spec["rule"]["panels"]["cal"] = {"reads": ["cal"], "metrics": ["acc", "brier"]}
    spec["rule"]["criteria"]["cal_acc"] = {"left": "cal.acc.lower", "op": ">", "right": -1}        # accuracy: the temperature cannot move it
    assert rounds.validate(spec, tmp_path, rows=False, plans=False).problems == []
    spec["rule"]["criteria"]["cal_brier"] = {"left": "cal.brier.upper", "op": "<=", "right": 0.01}
    spec["temperature"]["reads"].append("nope")
    spec["temperature"]["sources"]["main"] = ["s"]
    spec["arms"]["x-bare"] = {"checkpoint": "jaredpalmer/kev-27b", "parent": "p"}
    spec["arms"]["x-lost"] = {"trial": "runs/s/01-trial-1", "parent": "p", "transfer_read": "locked", "reads": {"main": "runs/x"}}
    problems = "\n".join(rounds.validate(spec, tmp_path, rows=False, plans=False).problems)
    for expected in ("['cal'] pooled, but rule criterion cal_brier reads cal.brier.upper", "temperature: 'nope' is not a read tag", "temperature: sources for 'main', which is not pooled",
                     "arm x-bare: an arm without a trial needs a checkpoint on the runs volume", "arm x-bare: no trial, so no in-trial transfer read",
                     "arm x-lost: transfer_read 'locked' is not a benchmarks read", "arm x-lost: its reads do not locate its transfer_read",
                     "arm x-lost: its reads do not locate the temperature pool's ['cal', 'v9', 'nope']"):
        assert expected in problems, expected
    del spec["temperature"]
    assert "arm x-wise: no trial, so no development rows to fit its temperature on" in "\n".join(rounds.validate(spec, tmp_path, rows=False, plans=False).problems)


# --- the temperature pool against training data (round 19's failure mode) ---------------------------------------------

def _r20_pooling(tag, read, sources=None):
    """Round 20's spec with its pool replaced by one read (a tag every arm locates)."""
    spec = rounds.load(ROOT / "experiments/rounds/r20.json")
    spec["reads"][tag] = read
    spec["temperature"] = {"reads": [tag], **({"sources": {tag: sources}} if sources else {})}
    for a in spec["arms"].values():
        if isinstance(a.get("reads"), dict): a["reads"][tag] = f"runs/x-{tag}"
    return spec


def _pool_refusals(spec, root=ROOT):
    return [p for p in rounds.validate(spec, root, rows=False, plans=False).problems if p.startswith("temperature: pooled read")]


def test_round_20s_pool_is_disjoint_from_its_arms_training():
    """Round 20's arms are round 19's sft-v1 trials (their training suite from the committed provenance.json, sft-v1's
    components included) and its interpolations of them; its pool (transfer-r3 calibration, eight held-out sources;
    transfer-v9 MMLU-Pro) shares nothing with that training."""
    spec = rounds.load(ROOT / "experiments/rounds/r20.json")
    assert rounds.pool_training_problems(spec, ROOT) == ([], [])
    training = rounds.trial_training(spec, "runs/r19-27b-lr2e6/00-trial-0", ROOT)
    assert {"evals/sft-v1", "evals/documents-v1", "evals/hard-v1", "evals/devtools-v1", "evals/round6/b1v2"} <= training.suites
    assert {"boolq", "cfpb", "hard_judge", "synthetic-v1/intent"} <= training.sources and training.unlisted == ()
    olddata = rounds.trial_training(rounds.load(ROOT / "experiments/rounds/r19.json"), "runs/r19-27b-olddata/00-trial-0", ROOT)   # its plan's data file
    assert "evals/round6/b1v2" in olddata.suites and "yelp" in olddata.sources


@pytest.mark.parametrize("read,sources,expected", [
    ({"suite": "evals/sft-v1"}, None, ["evals/sft-v1 is training data", "training source(s)", "development partition of evals/sft-v1"]),        # round 19's pool: (a) (b) (c)
    ({"suite": "evals/sft-v1", "flags": "--split calibration"}, ["boolq"], ["evals/sft-v1 is training data", "['boolq']", "calibration partition"]),   # one of its partitions
    ({"suite": "evals/documents-v1", "flags": "--allow-test"}, ["not-a-source"], ["evals/documents-v1 is training data"]),                      # (a) a component of sft-v1
    ({"suite": "evals/v7/decision-v7", "flags": "--allow-test"}, None, ["training source(s) ['agnews', 'amazon'"]),                               # (b) only: shared sources
    ({"suite": "evals/breadth-v1"}, ["musr"], []),                                                                                             # held-out datasets: fine
], ids=["sft-v1-dev", "sft-v1-calibration", "component", "shared-sources", "held-out"])
def test_a_pool_that_shares_data_with_training_is_refused(read, sources, expected):
    refusals = _pool_refusals(_r20_pooling("fit", read, sources))
    assert len(refusals) == (1 if expected else 0), refusals
    for text in expected:
        assert text in refusals[0], (text, refusals[0])
    if refusals:
        assert "arm(s) 27b-a, 27b-b" in refusals[0] and "round 19's failure mode" in refusals[0] and "T 0.955" in refusals[0]


def test_a_training_corpus_development_split_is_refused_even_when_not_trained_on(tmp_path):
    """(c): the pool reads hard-v1 development, the arm trained on decision-v7 only (no shared suite, no shared source): a
    training corpus's calibration and development partitions are never a temperature pool."""
    spec = _pool_round(tmp_path)
    spec["reads"]["hard"] = {"suite": "evals/hard-v1"}
    spec["temperature"]["reads"] = ["hard"]; spec["temperature"]["sources"] = {"hard": ["hard_judge"]}
    [refusal] = _pool_refusals(spec, tmp_path)
    assert "it reads the development partition of evals/hard-v1, a training corpus" in refusal and "training data" not in refusal and "training source" not in refusal
    spec["reads"]["hard"]["flags"] = "--allow-test"
    assert _pool_refusals(spec, tmp_path) == []


def test_a_pool_cannot_be_checked_without_the_arms_training(tmp_path):
    """A trial arm with no provenance and no study of the spec: a new round fails, a recorded round lists it as archived.
    A round of checkpoints only must say what they trained on (`trained_on`), which is then checked like a trial's."""
    spec = _pool_round(tmp_path)
    (tmp_path / "runs/s/00-trial-0/provenance.json").unlink()
    problems, archived = rounds.validate(spec, tmp_path, rows=False, plans=False)
    assert any("arm x-trained: what runs/s/00-trial-0 trained on is unknown" in p for p in problems) and archived == []
    problems, archived = rounds.validate({**spec, "archive": "tag"}, tmp_path, rows=False, plans=False)
    assert problems == [] and any("trained on is unknown" in a for a in archived)
    del spec["arms"]["x-trained"]
    assert any("no arm names its training" in p for p in rounds.validate(spec, tmp_path, rows=False, plans=False).problems)
    spec["arms"]["x-wise"]["trained_on"] = ["evals/v4/transfer-v4"]   # a checkpoint trained on the pool's own suite
    assert "evals/v4/transfer-v4 is training data" in "\n".join(_pool_refusals(spec, tmp_path))


def test_a_trial_served_at_its_training_corpus_rows_is_flagged(tmp_path):
    """Without a pool an arm is served at its trial's development rows; the read-out records their suite, and the table
    warns when that suite is a training corpus (decision-v7 here, sft-v1 in round 19)."""
    spec = _pool_round(tmp_path)
    del spec["temperature"], spec["arms"]["x-wise"]
    spec["round"] = 20   # the last round that only warns (from 21 a missing pool is refused)
    report = rounds.readout(spec, tmp_path)
    assert report["arms"]["x-trained"]["temperature_source"] == {"kind": "trial development rows", "rows": "runs/s/00-trial-0/development",
                                                                 "suite": "evals/v7/decision-v7", "training_corpus": True}
    [warning] = [line for line in rounds.table(report).splitlines() if "!!!" in line]
    assert "x-trained is served at T" in warning and "evals/v7/decision-v7" in warning and "round 19's failure mode" in warning
    assert rounds.calibration_warnings(spec, tmp_path) == []                      # its only criterion is on accuracy
    spec["rule"]["criteria"]["ece"] = {"left": "main.ece.candidate", "op": "<=", "right": 1}
    [warning] = rounds.calibration_warnings(spec, tmp_path)                        # validate/launch print it before training
    assert warning.startswith("arm x-trained will be served at a temperature fitted on its trial's evals/v7/decision-v7 development rows") and "rule ece" in warning
    r19 = rounds.load(ROOT / "experiments/rounds/r19.json")
    assert len(rounds.calibration_warnings(r19)) == 3 and rounds.calibration_warnings(rounds.load(ROOT / "experiments/rounds/r20.json")) == []


def test_from_round_21_a_temperature_criterion_needs_a_pool(tmp_path):
    """Jared: make sure we don't botch calibration. From round 21 a rule or confirmation criterion the temperature moves,
    without a registered `temperature` pool, is a problem (so launch refuses too), with the round-19 message; up to round 20
    it stays a warning, so rounds 5-20 still validate. Accuracy-only criteria need no pool."""
    spec = _pool_round(tmp_path)
    del spec["temperature"], spec["arms"]["x-wise"]
    assert rounds.pool_required_problems(spec) == []                                  # round 99, accuracy criteria only
    spec["confirm"]["locked"]["panels"]["locked"]["metrics"].append("brier")
    spec["confirm"]["locked"]["criteria"]["brier"] = {"left": "locked.brier.candidate", "op": "<=", "right": 0.2}   # a confirmation criterion counts too
    [problem] = rounds.pool_required_problems(spec)
    assert problem.startswith("temperature: round 99 registers no `temperature` pool, but 1 (confirm.locked brier) criteria") and "round 19's failure mode" in problem
    assert problem in rounds.validate(spec, tmp_path, rows=False, plans=False).problems
    assert problem in rounds.validate({**spec, "archive": "tag"}, tmp_path, rows=False, plans=False).problems   # the number decides, not the record
    for number in (20, 19):
        assert rounds.pool_required_problems({**spec, "round": number}) == []
        assert any(w.startswith("arm x-trained will be served") for w in rounds.calibration_warnings({**spec, "round": number}, tmp_path))
    assert not any(w.startswith("arm ") for w in rounds.calibration_warnings(spec, tmp_path))   # round 99: a problem, not a warning
    spec["temperature"] = {"reads": ["cal"], "sources": {"cal": ["mmlu"]}}
    spec["arms"]["x-trained"]["reads"] = {"cal": "runs/r99-x-trained-cal", "main": "runs/r99-x-trained-main"}
    assert rounds.pool_required_problems(spec) == [] and rounds.validate(spec, tmp_path, rows=False, plans=False).problems == []


def test_launch_refuses_a_new_round_without_a_pool(tmp_path, monkeypatch):
    spec = _pool_round(tmp_path)
    del spec["temperature"], spec["arms"]["x-wise"]
    spec["rule"]["criteria"]["ece"] = {"left": "main.ece.upper", "op": "<=", "right": 0.01}
    monkeypatch.setattr(rounds, "ROOT", tmp_path)
    assert any("registers no `temperature` pool" in p for p in rounds.launchable(spec, rows=False))
    write_json(tmp_path / "r99.json", spec)
    with pytest.raises(SystemExit, match="registers no `temperature` pool"):
        rounds.main(["launch", str(tmp_path / "r99.json"), "--dry-run"])


def test_a_parent_served_far_from_its_shipped_temperature_is_warned_about(tmp_path):
    """Parents are served at their trial's development rows; the read-out records that and the head.pt temperature its
    checkpoint ships. When those rows are a training corpus's and the two differ by more than 0.05, validate warns (report only)."""
    from kev.checkpoint import Meta, write_meta
    spec = _pool_round(tmp_path)
    _trained_on(tmp_path, "runs/p/00-trial-0", "evals/v7/decision-v7")
    served_t = rounds.temperature("runs/p/00-trial-0", tmp_path)
    assert rounds.calibration_warnings(spec, tmp_path) == []                           # no head.pt here: nothing to compare
    (tmp_path / "runs/p/00-trial-0/checkpoint").mkdir()
    for shipped, warned in ((served_t + 0.04, False), (served_t + 0.2, True)):
        write_meta(tmp_path / "runs/p/00-trial-0/checkpoint", Meta(base="b", temperature=shipped))
        warnings = rounds.calibration_warnings(spec, tmp_path)
        assert bool(warnings) is warned
    [warning] = warnings
    assert warning.startswith(f"parent p is served at T {served_t:.3f} fitted on runs/p/00-trial-0/development (evals/v7/decision-v7, a training corpus), 0.200 from")
    assert rounds.validate(spec, tmp_path, rows=False, plans=False).problems == []   # report only
    source = rounds.readout(spec, tmp_path)["arms"]["x-wise"]["parent_temperature_source"]
    assert source == {"kind": "trial development rows", "rows": "runs/p/00-trial-0/development", "suite": "evals/v7/decision-v7", "training_corpus": True, "shipped": served_t + 0.2}


def test_a_data_file_outside_evals_cannot_be_checked(tmp_path):
    """A trial trained with `data` outside evals/ (kev.train --data): its sources cannot be listed, so the pool cannot be
    checked against it. A problem for a new round, reported (archived) for a recorded one; never silently dropped."""
    spec = _pool_round(tmp_path)
    _trained_on(tmp_path, "runs/s/00-trial-0", "evals/v7/decision-v7", data="/data/mine.jsonl")
    assert rounds.trial_training(spec, "runs/s/00-trial-0", tmp_path).unlisted == ("data /data/mine.jsonl (outside evals/)",)
    problems, archived = rounds.validate(spec, tmp_path, rows=False, plans=False)
    assert any("arm x-trained: cannot list the sources of data /data/mine.jsonl (outside evals/)" in p for p in problems) and archived == []
    problems, archived = rounds.validate({**spec, "archive": "tag"}, tmp_path, rows=False, plans=False)
    assert problems == [] and any("data /data/mine.jsonl (outside evals/)" in a for a in archived)
    _trained_on(tmp_path, "runs/s/00-trial-0", "evals/v7/decision-v7", data="evals/round6/b1v2/train.jsonl")   # inside evals/: its suite counts
    training = rounds.trial_training(spec, "runs/s/00-trial-0", tmp_path)
    assert "evals/round6/b1v2" in training.suites and training.unlisted == ()


def test_a_sources_allowlist_typo_is_a_problem(tmp_path):
    """A pool read's allowlist names only sources its suite lists: a typo would silently shrink the pool. Round 20's passes."""
    assert rounds.validate(rounds.load(ROOT / "experiments/rounds/r20.json"), ROOT, rows=False, plans=False).problems == []
    spec = _pool_round(tmp_path)
    spec["temperature"]["sources"]["cal"] = ["mmlu", "emotoin"]
    assert "temperature: sources for 'cal' name ['emotoin'] which evals/v4/transfer-v4 does not contain" in rounds.validate(spec, tmp_path, rows=False, plans=False).problems
    spec["reads"]["nosrc"] = {"suite": "evals/round15/joint"}                                                  # a manifest listing no sources
    assert rounds.allowlist_problems("nosrc", "evals/round15/joint", ["x"]) == ["temperature: cannot check the sources allowlist of 'nosrc': 'evals/round15/joint' lists no sources in this checkout"]


def test_calibrate_checkpoint_manual_temperature_needs_a_reason(tmp_path, monkeypatch):
    from kev.checkpoint import read_meta
    run = _checkpoint(tmp_path)
    _rows(tmp_path, "loose", [f"d/{i}" for i in range(30)], 1)
    with pytest.raises(SystemExit):   # argparse error: no --reason
        _calibrate(monkeypatch, "--run", run, "--rows", tmp_path / "loose/rows.json", "--temperature", "1.41")
    _calibrate(monkeypatch, "--run", run, "--rows", tmp_path / "loose/rows.json", "--temperature", "1.41", "--reason", "copied from the pool fit of runs/r20-readout")
    meta = read_meta(run)
    assert meta.temperature == 1.41 and meta.extra["temperature_fit"] == {"method": "manual", "reason": "copied from the pool fit of runs/r20-readout"}
    with pytest.raises(SystemExit, match=r"\['mmluu'\] not among the sources of these rows \['s'\]"):   # an allowlist typo is refused before anything else
        _calibrate(monkeypatch, "--run", run, "--rows", f"{tmp_path / 'loose/rows.json'}:mmluu")


def test_state_lengths_count_the_encoded_state_segment(monkeypatch):
    """One definition of a state's token count: kev.model.encode's state segment, <state> token included (what serve reports)."""
    from kev.data import materialize
    from kev.model import encode
    records = [{"state": "Refund window: 30 days. Order placed 12 days ago.", "questions": {"q": {"type": "noul", "instructions": "Refundable?", "label": True, "src": "x"}},
                "_meta": {"id": "r/0"}}]
    class Tok:   # characters as tokens, plus the special tokens encode asks for
        def __call__(self, text, add_special_tokens=False): return type("E", (), {"input_ids": [ord(c) for c in text]})()
        def convert_tokens_to_ids(self, t): return 1
    monkeypatch.setattr(rounds, "load_split", lambda *a, **k: records)
    monkeypatch.setattr("kev.model.load_tokenizer", lambda *a: Tok())
    rounds.state_lengths.cache_clear()
    try:
        n = rounds.state_lengths("evals/x", "development", ("t", "r"))["r/0"]
    finally:
        rounds.state_lengths.cache_clear()
    assert n == encode(Tok(), materialize(records[0]))["seg"].count(0) == len(materialize(records[0])["state"]) + 1


def test_a_malformed_by_length_option_is_a_problem_not_a_crash(tmp_path):
    spec = _pool_round(tmp_path)
    for option in ({"edges": 8192}, {"tokenizer": 5}, {"edges": [8192, "16k"]}):
        spec["rule"]["panels"]["long"] = {"reads": ["main"], "metrics": ["acc"], "by_length": option}
        assert any("panel long: by_length" in p for p in rounds.validate(spec, tmp_path, rows=False, plans=False).problems), option


# --- calibration by state length -------------------------------------------------------------------------------------

def test_calibration_by_length_on_synthetic_rows(tmp_path):
    from kev.metrics import calibration_by_length, length_buckets, metrics
    assert [b[0] for b in length_buckets()] == ["under_8k", "8k_16k", "16k_32k", "32k_64k", "64k_plus", "8k_plus", "16k_plus", "32k_plus"]
    assert [b[0] for b in length_buckets((4096,))] == ["under_4k", "4k_plus"]
    with pytest.raises(ValueError): length_buckets((8192, 4096))
    with pytest.raises(ValueError): length_buckets((5000,))
    rows = _rows(tmp_path, "r", [f"d/{i}" for i in range(40)], 0)
    lengths = {f"d/{i}": [100, 9000, 20000, 40000][i % 4] for i in range(40)}
    rows[0]["state_tokens"] = 70000                                        # a row's own count wins over lengths
    out = calibration_by_length(rows, lengths)
    assert {b: out[b]["n"] for b in out} == {"under_8k": 9, "8k_16k": 10, "16k_32k": 10, "32k_64k": 10, "64k_plus": 1, "8k_plus": 31, "16k_plus": 21, "32k_plus": 11}
    long = [r for i, r in enumerate(rows) if i == 0 or i % 4 in (2, 3)]   # row order kept: the sums match bit for bit
    assert out["16k_plus"] == {"n": 21, **{m: metrics(long)[m] for m in ("acc", "ece", "brier", "confident_error_rate")}}
    assert calibration_by_length(rows[1:4], lengths, (65536,))["64k_plus"] == {"n": 0, "acc": None, "ece": None, "brier": None, "confident_error_rate": None}
    with pytest.raises(KeyError): calibration_by_length(rows, {})


def test_a_by_length_panel_reports_buckets_and_takes_a_criterion(tmp_path, monkeypatch):
    """A panel with `by_length` reports <metric>_<bucket> for both sides (one set of token counts for both); a criterion on
    a bucket's ECE is a valid path and decides the arm; the table prints the buckets."""
    spec = _pool_round(tmp_path)
    spec["rule"]["panels"]["long"] = {"reads": ["main"], "metrics": ["acc"], "by_length": True}
    spec["rule"]["criteria"]["long_ece"] = {"left": "long.ece_16k_plus.candidate", "op": "<=", "right": 0.5}
    assert rounds.validate(spec, tmp_path, rows=False, plans=False).problems == []
    asked = []
    monkeypatch.setattr(rounds, "panel_lengths", lambda s, panel: asked.append(panel["reads"]) or {f"m/{i}": 1000 * (i % 2) + 17000 * (i % 3 == 0) for i in range(50)})
    report = rounds.readout(spec, tmp_path)
    wise = report["arms"]["x-wise"]["panels"]["long"]
    assert wise["by_length"] == {"edges": [8192, 16384, 32768, 65536], "tokens": {"records": "suite", "tokenizer": list(ADMISSION_TOKENIZER)}}
    assert wise["ece_16k_plus"]["n"] == 17 and wise["acc_under_8k"]["n"] == 33 and wise["ece_32k_64k"] == {"candidate": None, "parent": None, "n": 0}
    assert report["arms"]["x-wise"]["criteria"]["long_ece"] is (wise["ece_16k_plus"]["candidate"] <= 0.5) and asked
    assert "long by state length (candidate/parent): under_8k n=33" in rounds.table(report)
    spec["rule"]["criteria"]["long_ece"]["left"] = "long.ece_16k_plus.lower"          # value only: no interval
    spec["rule"]["criteria"]["bucket"] = {"left": "long.ece_12k_plus.candidate", "op": "<=", "right": 1}
    spec["rule"]["panels"]["bad"] = {"reads": ["main", "transfer"], "metrics": ["acc"], "by_length": {"edges": [5000]}}
    problems = "\n".join(rounds.validate(spec, tmp_path, rows=False, plans=False).problems)
    for expected in ("unknown path 'long.ece_16k_plus.lower'", "unknown path 'long.ece_12k_plus.candidate'", "panel bad: by_length: length edges", "panel bad: by_length counts state tokens from a read's suite; 'transfer'"):
        assert expected in problems, expected


def test_a_pooled_read_under_a_by_length_criterion(tmp_path):
    """A bucket's accuracy is temperature-free; its ECE is not, so a pooled read may not sit under a criterion on it."""
    spec = _pool_round(tmp_path)
    spec["rule"]["panels"]["long"] = {"reads": ["cal"], "metrics": ["acc"], "by_length": True}
    spec["rule"]["criteria"]["long_acc"] = {"left": "long.acc_16k_plus.candidate", "op": ">=", "right": 0}
    assert rounds.validate(spec, tmp_path, rows=False, plans=False).problems == []
    spec["rule"]["criteria"]["long_ece"] = {"left": "long.ece_16k_plus.candidate", "op": "<=", "right": 0.05}
    assert any("criterion long_ece reads long.ece_16k_plus.candidate, which the temperature moves" in p for p in rounds.validate(spec, tmp_path, rows=False, plans=False).problems)


# --- scripts/calibrate_checkpoint.py: the shipped temperature -------------------------------------------------------

def _checkpoint(root, suite="evals/sft-v1"):
    from kev.checkpoint import Meta, write_meta
    from kev.suite import digest
    run = root / "runs/t/00-trial-0/checkpoint"; run.mkdir(parents=True)
    write_meta(run, Meta(base="b", extra={"suite_sha256": digest(ROOT / suite / "manifest.json"), "args": {"suite": f"/root/kev/{suite}"}}))
    return run


def _calibrate(monkeypatch, *args):
    import sys
    from scripts import calibrate_checkpoint
    monkeypatch.setattr(sys, "argv", ["calibrate_checkpoint.py", *map(str, args)])
    calibrate_checkpoint.main()


def test_calibrate_checkpoint_refuses_its_own_training_suite(tmp_path, monkeypatch):
    """Round 19's fit set: the trial's own development rows of sft-v1. Refused; --allow-in-distribution fits, warns and
    records it; every fit records the rows' suites, partitions and question counts."""
    from kev.checkpoint import read_meta
    from kev.suite import digest
    run = _checkpoint(tmp_path)
    _trained_on(tmp_path, "runs/t/00-trial-0", "evals/sft-v1")
    own = _rows(tmp_path, "runs/t/00-trial-0/development", [f"d/{i}" for i in range(60)], 1, source="boolq")
    with pytest.raises(SystemExit) as refused:
        _calibrate(monkeypatch, "--run", run, "--rows", tmp_path / "runs/t/00-trial-0/development/rows.json")
    message = str(refused.value)
    assert "evals/sft-v1 is training data" in message and "development partition of evals/sft-v1" in message and "round 19's failure mode" in message
    assert read_meta(run).temperature == 1.0                                                                  # nothing written
    _calibrate(monkeypatch, "--run", run, "--rows", tmp_path / "runs/t/00-trial-0/development/rows.json", "--allow-in-distribution")
    fit = read_meta(run).extra["temperature_fit"]
    assert fit["in_distribution"]["allowed"] and any("evals/sft-v1 is training data" in p for p in fit["in_distribution"]["problems"])
    assert fit["fit_rows"] == [{"rows": str(tmp_path / "runs/t/00-trial-0/development/rows.json"), "suite": "evals/sft-v1", "split": "development", "questions": len(own)}]
    assert "evals/sft-v1" in fit["training_suites"] and "evals/documents-v1" in fit["training_suites"]
    # held-out datasets: a kev.benchmark read of transfer-r3's calibration partition, two sources kept
    bench = tmp_path / "runs/x-r3cal"
    _rows(tmp_path, "runs/x-r3cal", [f"c/{i}" for i in range(50)], 2, source="mmlu")
    write_json(bench / "report.json", {"suite_sha256": digest(ROOT / "evals/round3/transfer-r3/manifest.json"), "split": "calibration"})
    _calibrate(monkeypatch, "--run", run, "--rows", f"{bench / 'rows.json'}:mmlu,emotion")   # emotion: a transfer-r3 source, absent from these rows
    with pytest.raises(SystemExit, match=r"\['emotoin'\] not among the sources of evals/round3/transfer-r3"):
        _calibrate(monkeypatch, "--run", run, "--rows", f"{bench / 'rows.json'}:mmlu,emotoin")
    fit = read_meta(run).extra["temperature_fit"]
    assert "in_distribution" not in fit and fit["fit_rows"] == [{"rows": str(bench / "rows.json"), "suite": "evals/round3/transfer-r3", "split": "calibration", "sources": ["mmlu", "emotion"], "questions": 50}]


def test_calibrate_checkpoint_refuses_what_it_cannot_place(tmp_path, monkeypatch):
    """Rows with no suite (no report.json, not a trial's read) and a checkpoint whose training is unknown are refused too."""
    from kev.checkpoint import Meta, write_meta
    run = _checkpoint(tmp_path)
    _rows(tmp_path, "loose", [f"d/{i}" for i in range(30)], 1)
    with pytest.raises(SystemExit, match="cannot tell which suite these rows were scored on"):
        _calibrate(monkeypatch, "--run", run, "--rows", tmp_path / "loose/rows.json")
    write_meta(run, Meta(base="b"))
    with pytest.raises(SystemExit, match="cannot tell what this checkpoint was trained on"):
        _calibrate(monkeypatch, "--run", run, "--rows", tmp_path / "loose/rows.json")


def test_concurrent_pulls_of_one_study_run_one_at_a_time(tmp_path, monkeypatch):
    """Two watchers' pulls of the same study collided (PLAN.md at research-archive-2026-09-24, Night 3 incidents): modal_app.pull_lock serialises them."""
    import threading, time
    import modal_app
    inside, overlaps = [], []

    def pull_volume(remote, local_parent, weights=True):
        inside.append(remote)
        if len(inside) > 1: overlaps.append(list(inside))
        time.sleep(0.05)
        (local_parent / remote.rsplit("/", 1)[1]).mkdir(exist_ok=True)
        inside.remove(remote)

    monkeypatch.setattr(modal_app, "ROOT", tmp_path)
    monkeypatch.setattr(modal_app, "pull_volume", pull_volume)
    monkeypatch.setattr(modal_app, "volume_names", lambda path: ({"00-trial-0"}, set()))
    monkeypatch.setattr(modal_app.subprocess, "run", lambda cmd, **kw: None)
    threads = [threading.Thread(target=modal_app.pull_study, args=("s",)) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert overlaps == [] and (tmp_path / "runs/s").is_dir()


def test_session_stops_at_the_spend_cap_and_never_confirms(tmp_path, monkeypatch):
    from kev import autoresearch
    monkeypatch.setattr(autoresearch, "ROOT", tmp_path); (tmp_path / "runs").mkdir()
    ran = []
    monkeypatch.setattr(rounds, "launchable", lambda spec: [])   # the recorded specs refuse a launch; the cap logic is under test
    monkeypatch.setattr(rounds, "launch_studies", lambda spec: ran.append(("launch", spec["round"])))
    monkeypatch.setattr(rounds, "watch", lambda spec: ran.append(("watch", spec["round"])) or {"candidates": {"9b": "9b-joint-lr2e5"}})
    monkeypatch.setattr(rounds, "confirm", lambda *a, **kw: pytest.fail("a session must not confirm"))
    spends, logs = iter([100.0, 150.0]), []
    specs = [ROOT / "experiments/rounds/r18.json", ROOT / "experiments/rounds/r16.json"]   # budgets $51 each
    entries = autoresearch.session(specs, spend_start=100.0, spend_cap=100.0, spend=lambda: next(spends), log=logs.append)
    assert ran == [("launch", 18), ("watch", 18)] and [e["round"] for e in entries] == [18]   # 50 spent + 51 > 100: round 16 never launches
    assert any("--stage locked --arm 9b-joint-lr2e5" in line for line in logs) and "session stops" in logs[-1]
    assert read_json(tmp_path / "runs/autoresearch-sessions.jsonl")["round"] == 18


def test_watch_pulls_reads_and_reads_out_each_finished_trial_once(monkeypatch, tmp_path):
    """The wiring of `watch`: a finished call -> its arm -> one pull of its study -> that arm's read commands -> read-out
    once every benchmarks process has exited (a failed read never lands)."""
    monkeypatch.setattr(rounds, "ROOT", tmp_path)   # the per-arm lock and launch intents are written under runs/
    spec = rounds.load(ROOT / "experiments/rounds/r16.json")
    events = []
    monkeypatch.setattr(rounds, "watch_studies", lambda studies, on_done, **kw: [on_done("r16-9b", label) for label in ("trial-1", "trial-0")] and [])   # no unmapped calls
    monkeypatch.setattr(rounds, "pull", lambda study, spec: events.append(("pull", study)))
    class Done:
        def poll(self): return 0
    monkeypatch.setattr(rounds, "launch_commands", lambda commands, spec, stem, stagger: events.append(("reads", commands[0][2])) or [Done()])
    monkeypatch.setattr(rounds, "read_commands", lambda spec, arm, *rest: [["modal", "run", arm]])
    monkeypatch.setattr(rounds, "write_readout", lambda spec: events.append(("readout", spec["round"])) or {"candidates": {}})
    monkeypatch.setattr(rounds.time, "sleep", lambda s: None)
    rounds.watch(spec, stagger=0)
    assert events == [("pull", "r16-9b"), ("reads", "9b-r10k-lr2e5"), ("pull", "r16-9b"), ("reads", "9b-r10k-lr1e5"), ("readout", 16)]


# --- review follow-ups: atomic state, interrupted launches, entrypoints, unmapped calls ------------------------------

def test_an_atomic_write_interrupted_mid_file_leaves_the_old_state_readable(tmp_path, monkeypatch):
    """The watcher's state is replaced, not rewritten in place: a crash while the new text is being written leaves the
    previous state intact (the in-place writer, for contrast, leaves a torn file)."""
    from pathlib import Path
    from kev import suite
    path = tmp_path / "s.watch.json"
    write_json(path, {"calls": {"trial-0": {"status": "running"}}})
    real = Path.write_text
    def crash(self, text, **kw):   # half the new text reaches the disk, then the process dies
        real(self, text[: len(text) // 2], **kw); raise KeyboardInterrupt
    monkeypatch.setattr(Path, "write_text", crash)
    with pytest.raises(KeyboardInterrupt):
        suite.write_json(path, {"calls": {"trial-0": {"status": "done", "launched": True}}}, atomic=True)
    monkeypatch.setattr(Path, "write_text", real)
    assert read_json(path) == {"calls": {"trial-0": {"status": "running"}}}
    monkeypatch.setattr(Path, "write_text", crash)
    with pytest.raises(KeyboardInterrupt):
        suite.write_json(path, {"calls": {"trial-0": {"status": "done", "launched": True}}})
    monkeypatch.setattr(Path, "write_text", real)
    with pytest.raises(ValueError):
        read_json(path)


def test_a_launch_interrupted_after_its_intent_is_not_repeated_while_the_reads_may_run(tmp_path, monkeypatch):
    """Crash between recording the launch and finishing it; the restarted watcher logs the interrupted launch, sees the
    fresh intent and holds off (InFlight) until the reads' timeout has passed, then launches exactly once."""
    monkeypatch.setattr(rounds, "ROOT", tmp_path)
    _spawned(tmp_path, {"trial-0": "fc-0"})
    spec = rounds.load(ROOT / "experiments/rounds/r16.json")
    clock, launched, logs = [1000.0], [], []
    def on_done(launch):
        return lambda study, label: rounds.launch_arm_reads(spec, "9b-r10k-lr1e5", stagger=0, launch=launch, now=lambda: clock[0], log=logs.append)
    def killed(commands, spec, tag, stagger): raise KeyboardInterrupt   # the process dies while spawning
    with pytest.raises(KeyboardInterrupt):
        rounds.watch_studies(["s"], on_done(killed), poll=lambda cid: "done", sleep=lambda s: None, log=logs.append, root=tmp_path, now=lambda: clock[0])
    state = read_json(tmp_path / "runs/s.watch.json")["calls"]["trial-0"]
    assert state["launching_at"] == 1000.0 and not state["launched"]
    def tick(seconds): clock[0] += 3 * 3600   # each pass is three hours later
    clock[0] += 60
    rounds.watch_studies(["s"], on_done(lambda commands, spec, tag, stagger: launched.append(tag) or []), poll=lambda cid: "done",
                         sleep=tick, log=logs.append, root=tmp_path, now=lambda: clock[0])
    assert launched == ["r16-reads-9b-r10k-lr1e5"]
    assert any("interrupted" in line for line in logs) and any("may still be running" in line for line in logs)
    assert read_json(tmp_path / "runs/s.watch.json")["calls"]["trial-0"]["launched"]


def test_reads_that_landed_are_never_relaunched(tmp_path, monkeypatch):
    monkeypatch.setattr(rounds, "ROOT", tmp_path)
    spec = rounds.load(ROOT / "experiments/rounds/r16.json")
    for tag in rounds.rule_tags(spec["rule"]) - {"transfer"}:
        d = tmp_path / rounds.arm_side(spec, "9b-r10k-lr1e5", tmp_path).dirs[tag]; d.mkdir(parents=True); write_json(d / "rows.json", [])
    assert rounds.launch_arm_reads(spec, "9b-r10k-lr1e5", launch=lambda *a: pytest.fail("relaunched"), log=lambda m: None) == []


def test_a_watcher_and_launch_reads_cannot_both_launch_one_arm(tmp_path, monkeypatch):
    import threading, time
    monkeypatch.setattr(rounds, "ROOT", tmp_path)
    spec = rounds.load(ROOT / "experiments/rounds/r16.json")
    launched, refused = [], []
    def slow(commands, spec, tag, stagger): time.sleep(0.05); launched.append(tag); return []
    def go():
        try: rounds.launch_arm_reads(spec, "9b-r10k-lr1e5", stagger=0, launch=slow, log=lambda m: None)
        except rounds.InFlight: refused.append(1)
    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(launched) == 1 and len(refused) == 1


def test_a_misspelled_entrypoint_fails_validation():
    spec = rounds.load(ROOT / "experiments/rounds/r10.json")
    spec["reads"]["locked"]["entrypoint"] = "locked-test"
    problems = rounds.validate(spec, ROOT, rows=False, plans=False).problems
    assert any("read locked: entrypoint 'locked-test' is not one of ('benchmarks', 'locked_test')" in p for p in problems)


def test_a_finished_call_with_no_arm_is_not_marked_launched(tmp_path, monkeypatch):
    """A spawned call the spec cannot map stays unlaunched and flagged, the watch still ends, and `watch` exits non-zero."""
    _spawned(tmp_path, {"trial-0": "fc-0", "trial-7": "fc-7"})
    logs = []
    def on_done(study, label):
        if label == "trial-7": raise rounds.Unmapped("no arm has a trial runs/s/<NN>-trial-7")
    unmapped = rounds.watch_studies(["s"], on_done, poll=lambda cid: "done", sleep=lambda s: None, log=logs.append, root=tmp_path)
    state = read_json(tmp_path / "runs/s.watch.json")["calls"]
    assert unmapped == ["s/trial-7"] and state["trial-0"]["launched"] and not state["trial-7"]["launched"] and state["trial-7"]["unmapped"]
    assert any(line.startswith("!!! s/trial-7") for line in logs)
    spec = rounds.load(ROOT / "experiments/rounds/r16.json")
    monkeypatch.setattr(rounds, "watch_studies", lambda studies, on_done, **kw: pytest.raises(rounds.Unmapped, on_done, "r16-9b", "trial-9") and ["r16-9b/trial-9"])
    monkeypatch.setattr(rounds, "write_readout", lambda spec: {"candidates": {}})
    with pytest.raises(SystemExit, match="map to no arm"):
        rounds.watch(spec, stagger=0)
