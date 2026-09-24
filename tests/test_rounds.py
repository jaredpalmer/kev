"""The round harness (kev.rounds): spec validation, the benchmark job codec, the watcher's resume and network handling, and
reproduction of the committed read-outs and verdicts of rounds 5-18 from saved rows.

Offline vs archive. Rounds 5-18 ran on the research branch; their trial rows, reads and most committed outputs live on the
git tag `research-archive-2026-09-24`, not on main. This checkout carries everything the round-5 read-out and the round-15
locked verdict need (the released Kev-0.8B's confirmation), so those two run everywhere, CI included. Every other
reproduction skips unless KEV_ROUNDS_ROOT points at a checkout that has the rows and the outputs: a worktree of the tag
(`git worktree add /tmp/kev-archive research-archive-2026-09-24`, whose gitignored trial rows are not in git either) or the
checkout the rounds ran in:
    KEV_ROUNDS_ROOT=/path/to/kev uv run --extra serve python -m pytest tests/test_rounds.py -q
The per-round scripts (on the tag) wrote different key names for the same numbers; the legacy_* functions below map them,
and every number is compared for exact equality.
"""
import os
import socket
from pathlib import Path

import pytest

from kev import rounds
from kev.suite import read_json, write_json

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
    """Round 10's parent test reads failed because `repo@sha@suite@name` shifted every field (PLAN.md, Night 3 incidents)."""
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
    tests, locked = rounds.read_commands(spec, "27b-r10k-lr2e5", stage="locked", root=tmp_path), None
    [locked] = tests
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
    return len(pairs), [(k, a, b) for k, a, b in pairs if a != b]


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
    return len(pairs), [(k, a, b) for k, a, b in pairs if a != b]


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


def test_concurrent_pulls_of_one_study_run_one_at_a_time(tmp_path, monkeypatch):
    """Two watchers' pulls of the same study collided (PLAN.md, Night 3 incidents): modal_app.pull_lock serialises them."""
    import threading, time
    import modal_app
    inside, overlaps = [], []

    def pull_volume(remote, local_parent):
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


def test_watch_pulls_reads_and_reads_out_each_finished_trial_once(monkeypatch):
    """The wiring of `watch`: a finished call -> its arm -> one pull of its study -> that arm's read commands -> read-out
    once every benchmarks process has exited (a failed read never lands)."""
    spec = rounds.load(ROOT / "experiments/rounds/r16.json")
    events = []
    monkeypatch.setattr(rounds, "watch_studies", lambda studies, on_done, **kw: [on_done("r16-9b", label) for label in ("trial-1", "trial-0")])
    monkeypatch.setattr(rounds, "pull", lambda study, spec: events.append(("pull", study)))
    monkeypatch.setattr(rounds, "read_commands", lambda spec, arm: [["modal", "run", arm]])
    class Done:
        def poll(self): return 0
    monkeypatch.setattr(rounds, "launch_commands", lambda commands, spec, stem, stagger: events.append(("reads", commands[0][2])) or [Done()])
    monkeypatch.setattr(rounds, "write_readout", lambda spec: events.append(("readout", spec["round"])) or {"candidates": {}})
    monkeypatch.setattr(rounds.time, "sleep", lambda s: None)
    rounds.watch(spec, stagger=0)
    assert events == [("pull", "r16-9b"), ("reads", "9b-r10k-lr2e5"), ("pull", "r16-9b"), ("reads", "9b-r10k-lr1e5"), ("readout", 16)]
