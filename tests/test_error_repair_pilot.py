import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from kev.benchmark import prediction_rows
from kev.rounds import validate
from kev.suite import digest, read_json, record_digest, write_json
from scripts import error_repair_pilot as pilot
from scripts.head_weight_edit import identities


def record(i, source="a"):
    return {"state": f"State {i}", "questions": {"q": {"type": "choice", "instructions": "Pick one",
            "criteria": {"a": "A", "b": "B"}, "label": "a", "src": source}},
            "_meta": {"id": str(i), "group_id": str(i), "source": source, "variant": "clean", "split": "train"}}


def test_singleton_is_label_independent_and_does_not_mutate():
    r = record(0)
    r["questions"]["other"] = deepcopy(r["questions"]["q"])
    selected = pilot.singleton(r)
    assert list(selected["questions"]) == [min(r["questions"], key=record_digest)]
    selected["questions"][next(iter(selected["questions"]))]["label"] = "b"
    assert all(q["label"] == "a" for q in r["questions"].values())


def test_mine_only_errors_one_question_per_record_source_round_robin():
    rs = [record(i, str(i % 2)) for i in range(6)]
    rs[0]["questions"]["other"] = deepcopy(rs[0]["questions"]["q"])
    rows = []
    for r in rs:
        p = {"a": .8, "b": .2} if r is rs[-1] else {"a": .2, "b": .8}
        rows += prediction_rows(r, {"probabilities": {qid: p for qid in r["questions"]}})
    picked = pilot.error_candidates(rs, rows)
    assert len(picked) == 5
    assert len({r["_meta"]["id"] for r in picked}) == 5
    assert all(len(r["questions"]) == 1 for r in picked)
    assert [r["_meta"]["source"] for r in picked[:4]] == ["0", "1", "0", "1"]
    assert picked == pilot.error_candidates(rs[::-1], rows[::-1])
    for row in rows:
        row["label"] = None
    assert pilot.error_candidates(rs, rows) == []


def test_teacher_blind_to_labels_metadata_and_student():
    r = record(0)
    body = pilot.teacher_body(r)
    r["questions"]["q"]["label"] = "b"
    r["_meta"]["student_prediction"] = "a"
    assert body == pilot.teacher_body(r)
    sent = json.loads(body["messages"][1]["content"])
    assert set(sent) == {"state", "question", "options"}
    assert body["model"] == pilot.TEACHER
    assert body["max_tokens"] == pilot.MAX_OUTPUT_TOKENS
    assert 0 < pilot.request_bound(body) < pilot.TEACHER_CAP


@pytest.mark.parametrize("text,agree", [(' {"label":"a"}', True), ('{"label":"b"}', False),
                                          ('{"label":"unknown"}', False), ('[]', False), ('invalid', False)])
def test_teacher_answer_requires_exact_known_key(text, agree):
    response = {"usage": {"cost": "0.002"}, "choices": [{"message": {"content": text}}]}
    a = pilot.teacher_answer(response, record(0))
    assert a["agrees"] == agree
    assert a["cost"] == .002


@pytest.mark.parametrize("cost", [None, -1, float("nan"), float("inf")])
def test_unknown_teacher_charge_fails_closed(cost):
    with pytest.raises(ValueError, match="cost"):
        pilot.teacher_answer({"usage": {"cost": cost}}, record(0))


@pytest.mark.parametrize("answer,passed", [("a", True), ("b", False)])
def test_label_phase_reserves_before_call_and_checks_teacher_pilot(tmp_path, monkeypatch, answer, passed):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(pilot, "POOL_RECORDS", 20)
    monkeypatch.setattr(pilot, "REPAIR_RECORDS", 20)
    pools = {"hard": [record(i) for i in range(40)], "development": [record(41)], "documents": [record(42)]}
    monkeypatch.setattr(pilot, "train_pools", lambda: pools)
    mining, out = tmp_path / "mining", tmp_path / "labels"
    mining.mkdir(); out.mkdir()
    write_json(mining / "candidates.json", [pilot.selection_record(r) for r in pools["hard"][:20]])
    write_json(mining / "report.json", {"checkpoint": pilot.CHECKPOINT, "candidates_sha256": digest(mining / "candidates.json")})
    import io
    calls = []
    def call(req, timeout):
        ledger = read_json(out / "teacher-ledger.json")
        assert ledger["calls"][-1]["status"] == "pending"
        assert ledger["total"] <= pilot.TEACHER_CAP
        assert ledger["total"] >= len(calls) * .001 + ledger["calls"][-1]["reserved_usd"] - 1e-12
        calls.append(req)
        return io.BytesIO(json.dumps({"usage": {"cost": .001}, "choices": [{"message": {"content": json.dumps({"label": answer})}}]}).encode())
    monkeypatch.setattr(pilot.urllib.request, "urlopen", call)
    if passed:
        pilot.label(out, mining)
        assert read_json(out / "teacher-ledger.json")["complete"]
        assert (out / "ordinary.jsonl").exists()
    else:
        with pytest.raises(ValueError, match="85%"):
            pilot.label(out, mining)
        assert not (out / "ordinary.jsonl").exists()
    assert len(calls) == 20
    assert read_json(out / "teacher-ledger.json")["total"] == pytest.approx(.02)


def test_api_failure_keeps_reservation_and_never_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-placeholder")
    r = record(0)
    monkeypatch.setattr(pilot, "train_pools", lambda: {"hard": [r]})
    mining, out = tmp_path / "mining", tmp_path / "labels"
    mining.mkdir(); out.mkdir()
    write_json(mining / "candidates.json", [pilot.selection_record(r)])
    write_json(mining / "report.json", {"checkpoint": pilot.CHECKPOINT, "candidates_sha256": digest(mining / "candidates.json")})
    calls = []
    def call(req, timeout):
        calls.append(req)
        raise TimeoutError("simulated ambiguous charge")
    monkeypatch.setattr(pilot.urllib.request, "urlopen", call)
    with pytest.raises(TimeoutError):
        pilot.label(out, mining)
    ledger = read_json(out / "teacher-ledger.json")
    assert len(calls) == 1 and ledger["total"] > 0
    assert ledger["calls"][0]["status"] == "pending" and not ledger["complete"]


def test_control_matches_sources_and_identical_replay(monkeypatch):
    monkeypatch.setattr(pilot, "POOL_RECORDS", 4)
    monkeypatch.setattr(pilot, "REPAIR_RECORDS", 3)
    pool = {"hard": [record(i, str(i % 2)) for i in range(8)],
            "development": [record(9)], "documents": [record(10)]}
    accepted = pool["hard"][:3]
    arms = pilot.matched_data(pool, accepted)
    assert Counter(r["_meta"]["source"] for r in arms["ordinary"][:3]) == Counter(r["_meta"]["source"] for r in accepted)
    assert arms["ordinary"][3:] == arms["targeted"][3:]
    assert not {k for r in arms["ordinary"][:3] for k in identities(r)} & {k for r in accepted for k in identities(r)}
    with pytest.raises(ValueError, match="not enough"):
        pilot.matched_data(pool, accepted[:2])
    pool["hard"][4:] = [record(20, "absent")]
    with pytest.raises(ValueError, match="source counts"):
        pilot.matched_data(pool, accepted)


def test_primary_requires_improvement_over_ordinary_and_calibration_guards():
    c = {"targeted": {tag: {"acc": {"ci95": [.001, .03]}, "brier": {"ci95": [-.01, -.001]},
                            "ece": {"ci95": [-.01, .005]}} for tag in ("hard", "transfer", "documents")}}
    assert pilot.success(c, {"ci95": [.001, .02]}, True)
    assert not pilot.success(c, {"ci95": [0, .02]}, True)
    assert not pilot.success(c, {"ci95": [.001, .02]}, False)
    c["targeted"]["hard"]["brier"]["ci95"][1] = .001
    assert not pilot.success(c, {"ci95": [.001, .02]}, True)


def test_registration_is_report_only_and_fits_additional_cap():
    spec = read_json("experiments/rounds/r32.json")
    assert not validate(spec, rows=False).problems
    assert all(a["select"] is False for a in spec["arms"].values())
    assert spec["probe"]["budget_usd"] == 5
    assert (600 + 1800) / 3600 * 5.6752 + pilot.TEACHER_CAP < 5
    assert pilot.TRAINING["init_from"] == pilot.CHECKPOINT
    assert pilot.REPAIR_RECORDS + 2 * pilot.REPLAY_PER_SUITE == 256


def test_real_training_pools_are_disjoint_and_exclude_all_available_eval_partitions():
    needed = [Path(path) / f"{split}.jsonl" for path in pilot.SUITES.values()
              for split in ("train", "calibration", "development")
              if f"{split}.jsonl" in pilot.read_manifest(path)["files"]]
    if not all(p.exists() for p in needed):
        pytest.skip("real-pool audit requires locally cached frozen partitions; unit CI does not fetch training corpora")
    pools = pilot.train_pools()
    seen = set()
    for records in pools.values():
        for r in records:
            keys = identities(r)
            assert not keys & seen
            seen |= keys
    for path in pilot.SUITES.values():
        manifest = pilot.read_manifest(path)
        for split in ("calibration", "development"):
            if f"{split}.jsonl" in manifest["files"]:
                assert not seen & {k for r in pilot.load_split(path, split) for k in identities(r)}
