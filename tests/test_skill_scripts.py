"""The kev-finetune skill's local scripts (skills/kev-finetune/scripts): record validation, state-grouped splitting,
generator prompt/balancing and response parsing; the kev-deploy Modal configuration. No network.
Run: uv run python -m pytest tests/test_skill_scripts.py -q
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/kev-finetune"
sys.path.insert(0, str(SKILL / "scripts"))
import convert_data  # noqa: E402
import extract_workload  # noqa: E402
import generate_data  # noqa: E402
import plan_size  # noqa: E402
import split_data  # noqa: E402

SPEC = json.loads((SKILL / "assets/workload.example.json").read_text(encoding="utf-8"))


def modal_deployment(monkeypatch, **settings):
    import modal
    import runpy
    for key in ("KEV_MODEL", "KEV_APP_NAME", "KEV_MIN_CONTAINERS", "KEV_GPU", "KEV_REGION", "KEV_FLASH",
                "KEV_TRUNCATE_STATES", "KEV_API_KEY", "KEV_HF_SECRET"):
        monkeypatch.delenv(key, raising=False)
    for key, value in settings.items(): monkeypatch.setenv(key, value)
    options = {}
    for method in ("cls", "server"):
        original = getattr(modal.App, method)
        def register(self, _method=method, _original=original, **kwargs):
            options[_method] = kwargs
            return _original(self, **kwargs)
        monkeypatch.setattr(modal.App, method, register)
    concurrent = modal.concurrent
    def concurrency(**kwargs):
        options["concurrent"] = kwargs
        return concurrent(**kwargs)
    monkeypatch.setattr(modal, "concurrent", concurrency)
    module = runpy.run_path(str(ROOT / "skills/kev-deploy/scripts/kev_serve.py"))
    return module, options


def test_modal_web_deployment_preserves_concurrency_and_scale_to_zero(monkeypatch):
    import modal
    module, options = modal_deployment(monkeypatch, KEV_API_KEY="pilot-test-key")
    assert isinstance(module["Kev"], modal.Cls) and "server" not in options
    assert options["concurrent"] == {"max_inputs": 64, "target_inputs": 32}
    assert options["cls"]["min_containers"] == 0 and options["cls"]["region"] is None
    assert options["cls"]["timeout"] == 600 and options["cls"]["startup_timeout"] == 1200
    assert len(options["cls"]["secrets"]) == 1
    assert module["SECRET"]["KEV_API_KEY"] == "pilot-test-key"


@pytest.mark.parametrize("minimum, expected", [("0", 1), ("2", 2)])
def test_modal_direct_server_uses_native_concurrency_and_application_auth(monkeypatch, minimum, expected):
    import modal
    module, options = modal_deployment(monkeypatch, KEV_FLASH="1", KEV_REGION="us-east,us-west",
                                       KEV_MIN_CONTAINERS=minimum, KEV_API_KEY="pilot-test-key", KEV_TRUNCATE_STATES="1")
    assert isinstance(module["Kev"], modal.Server)
    assert "cls" not in options and "concurrent" not in options
    server = options["server"]
    assert server["target_concurrency"] == 64 and server["min_containers"] == expected
    assert server["compute_region"] == ["us-east", "us-west"] and server["routing_region"] == "us-east"
    assert server["port"] == 8000 and server["startup_timeout"] == 1200 and server["exit_grace_period"] == 120
    assert server["unauthenticated"] is True and len(server["secrets"]) == 1   # Kev, not Modal's proxy, checks the bearer
    assert module["SECRET"]["KEV_API_KEY"] == "pilot-test-key" and module["SETTINGS"]["KEV_TRUNCATE_STATES"] == "1"


def test_modal_direct_server_requires_a_routing_region(monkeypatch):
    with pytest.raises(SystemExit, match="set KEV_REGION"):
        modal_deployment(monkeypatch, KEV_FLASH="1")


def record(state, dept="billing", urgent=True, level=1):
    return {"state": state, "questions": {
        "department": {**SPEC["questions"]["department"], "label": dept},
        "escalate": {**SPEC["questions"]["escalate"], "label": urgent},
        "frustration": {**SPEC["questions"]["frustration"], "label": level}}}


def test_check_question_accepts_valid_and_names_the_problem():
    assert split_data.check_record(record("charged twice")) == []
    bad = record("x"); bad["questions"]["department"]["label"] = "legal"
    assert any("not one of the criteria" in p for p in split_data.check_record(bad))
    bad = record("x"); bad["questions"]["escalate"]["label"] = "yes"
    assert any("true or false" in p for p in split_data.check_record(bad))
    bad = record("x"); bad["questions"]["frustration"]["label"] = 3
    assert any("level index" in p for p in split_data.check_record(bad))
    bad = record("x"); del bad["questions"]["escalate"]["label"]
    assert any("no label" in p for p in split_data.check_record(bad))
    soft = record("x"); soft["questions"]["escalate"]["target"] = {"true": 0.5, "false": 0.5}
    assert split_data.check_record(soft) == []
    soft["questions"]["escalate"]["target"] = {"maybe": 1}
    assert any("target" in p for p in split_data.check_record(soft))


def test_dedupe_drops_exact_copies_and_conflicting_states():
    rows = [record("late order"), record("late order"), record("LATE  order", dept="shipping"), record("login broken", dept="account")]
    kept, dupes, conflicts = split_data.dedupe(rows)
    assert dupes == 1 and conflicts == 1
    assert [r["state"] for r in kept] == ["login broken"]


@pytest.mark.parametrize("left,right", [
    ({"type": "score", "criteria": ["bad", "good"], "label": 1},
     {"type": "score", "criteria": ["bad", "neutral", "good"], "label": 2}),
    ({"type": "score", "instructions": "Rate anger", "criteria": ["bad", "good"], "label": 0},
     {"type": "score", "instructions": "Rate joy", "criteria": ["bad", "good"], "label": 1}),
    ({"type": "score", "criteria": ["bad", "good"], "label": 0},
     {"type": "score", "criteria": ["good", "bad"], "label": 1}),
    ({"type": "choice", "criteria": {"a": "billing", "b": "shipping"}, "label": "a"},
     {"type": "choice", "criteria": {"a": "shipping", "b": "billing"}, "label": "b"}),
    ({"type": "noul", "label": True},
     {"type": "choice", "criteria": {"yes": "yes", "no": "no"}, "label": "yes"}),
    ({"type": "noul", "label": True},
     {"type": "noul", "criteria": {"true": "yes", "false": "no"}, "label": False}),
])
def test_dedupe_preserves_different_question_definitions(left, right):
    rows = [{"state": "same input", "questions": {"answer": {"instructions": "Assess this input", **q}}} for q in (left, right)]
    assert all(split_data.check_record(row) == [] for row in rows)
    assert split_data.dedupe(rows) == (rows, 0, 0)


@pytest.mark.parametrize("criteria", [
    {"a": "billing", "b": "shipping"},
    {"é": "naïve", "b": "autre"},
])
def test_dedupe_still_detects_conflicts_with_reordered_criteria_keys(criteria):
    first = {"type": "choice", "instructions": "Route this", "criteria": criteria, "label": next(iter(criteria))}
    second = {**first, "criteria": {k: criteria[k] for k in reversed(criteria)}, "label": next(reversed(criteria))}
    rows = [{"state": "same input", "questions": {"answer": q}} for q in (first, second)]
    assert split_data.dedupe(rows) == ([], 0, 1)


def test_dedupe_treats_missing_and_null_noul_criteria_as_one_form():
    bare = {"type": "noul", "instructions": "Urgent?", "label": True}
    null = {"type": "noul", "instructions": "Urgent?", "criteria": None, "label": False}
    rows = [{"state": "same input", "questions": {"urgent": q}} for q in (bare, null)]
    assert all(split_data.check_record(row) == [] for row in rows)
    assert split_data.dedupe(rows) == ([], 0, 1)


def test_label_table_counts_each_question_definition():
    short = {"type": "score", "instructions": "Rate it", "criteria": ["bad", "good"], "label": 1}
    long = {"type": "score", "instructions": "Rate it", "criteria": ["bad", "neutral", "good"], "label": 1}
    rows = [{"state": "same", "questions": {"rating": q}} for q in (short, long)]
    table = split_data.label_table(rows)
    assert table["rating (2 levels)"] == Counter({"1": 1})
    assert table["rating (3 levels)"] == Counter({"1": 1})
    warnings = {w.split(":")[0]: w for w in split_data.warnings_for(rows)}
    assert "['0']" in warnings["question 'rating (2 levels)'"]
    assert "['0', '2']" in warnings["question 'rating (3 levels)'"]
    assert set(split_data.label_table([record("x")])) == {"department", "escalate", "frustration"}
    plain = {"type": "noul", "instructions": "Is it urgent?", "label": True}
    described = {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "yes", "false": "no"}, "label": False}
    noul_rows = [{"state": f"n{i}", "questions": {"answer": q}} for i, q in enumerate((plain, described))]
    assert set(split_data.label_table(noul_rows)) == {"answer (noul)", "answer (noul) #2"}


def test_form_names_stay_with_the_definition_when_the_file_is_reversed():
    forms = [{"type": "choice", "instructions": instr, "criteria": {"a": "1", "b": "2"}, "label": "a"} for instr in ("Alpha", "Beta", "Gamma")]
    rows = [{"state": f"s{i}", "questions": {"answer": q}} for i, q in enumerate(forms)]
    def named(seq):
        return {sample["instructions"]: name for name, sample, _ in split_data.label_groups(seq)}
    assert named(rows) == named(list(reversed(rows)))
    assert set(named(rows).values()) == {"answer (2 options)", "answer (2 options) #2", "answer (2 options) #3"}


def test_form_name_does_not_replace_another_question_id():
    rows = [
        {"state": "a", "questions": {
            "answer": {"type": "choice", "instructions": "One", "criteria": {"a": "1", "b": "2"}, "label": "a"},
            "answer (2 options)": {"type": "noul", "instructions": "Other?", "label": True}}},
        {"state": "b", "questions": {
            "answer": {"type": "choice", "instructions": "Two", "criteria": {"c": "1", "d": "2"}, "label": "c"}}}]
    table = split_data.label_table(rows)
    assert table["answer (2 options)"] == Counter({"true": 1})
    assert sum(sum(counts.values()) for counts in table.values()) == 3
    assert Counter({"a": 1}) in table.values() and Counter({"c": 1}) in table.values()


def test_summary_keeps_question_ids_in_first_seen_order(tmp_path, monkeypatch):
    rows = [{"state": f"s{i}", "questions": {
        "zebra": {"type": "noul", "instructions": "Z?", "label": i % 2 == 0},
        "apple": {"type": "noul", "instructions": "A?", "label": True}}} for i in range(30)]
    src, out = tmp_path / "ids.jsonl", tmp_path / "out"
    src.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["split_data.py", str(src), "--out", str(out)])
    assert split_data.main() == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert list(split_data.label_table(rows)) == ["zebra", "apple"]
    for part in summary["partitions"].values():
        assert list(part["labels"]) == ["zebra", "apple"]


def test_summary_uses_one_name_for_a_form_in_every_partition(tmp_path, monkeypatch):
    short = {"type": "score", "instructions": "Rate", "criteria": ["bad", "good"], "label": 1}
    long = {"type": "score", "instructions": "Rate", "criteria": ["bad", "neutral", "good"], "label": 2}
    rows = [{"state": f"short {i}", "questions": {"rating": short}} for i in range(2)]
    rows.append({"state": "long 0", "questions": {"rating": long}})
    src, out = tmp_path / "scales.jsonl", tmp_path / "out"
    src.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    # 3 states at 0.34/0.34 puts one state in each partition, so two partitions contain only the short scale.
    monkeypatch.setattr(sys, "argv", ["split_data.py", str(src), "--out", str(out), "--calibration", "0.34", "--development", "0.34"])
    assert split_data.main() == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    keys = [set(part["labels"]) for part in summary["partitions"].values()]
    assert {"rating (3 levels)"} in keys and {"rating (2 levels)"} in keys
    assert all("rating" not in part["labels"] for part in summary["partitions"].values())


def test_holdout_scale_keeps_its_own_summary_name(tmp_path, monkeypatch):
    short = {"type": "score", "instructions": "Rate", "criteria": ["bad", "good"], "label": 1}
    long = {"type": "score", "instructions": "Rate", "criteria": ["bad", "neutral", "good"], "label": 2}
    synthetic = [{"state": f"gen {i}", "questions": {"rating": short}} for i in range(10)]
    real = [{"state": f"real {i}", "questions": {"rating": long}} for i in range(6)]
    for name, group in (("gen.jsonl", synthetic), ("real.jsonl", real)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in group), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["split_data.py", str(tmp_path / "gen.jsonl"), "--out", str(tmp_path / "out"), "--holdout", str(tmp_path / "real.jsonl")])
    assert split_data.main() == 0
    summary = json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8"))
    assert set(summary["partitions"]["train"]["labels"]) == {"rating (2 levels)"}
    assert set(summary["partitions"]["calibration"]["labels"]) == {"rating (3 levels)"}
    assert set(summary["partitions"]["development"]["labels"]) == {"rating (3 levels)"}


def test_invalid_criteria_is_rejected_before_the_histogram(tmp_path, monkeypatch):
    rows = [record(f"case {i}") for i in range(30)]
    bad = [{"state": "no criteria", "questions": {"rating": {"type": "score", "instructions": "Rate", "label": 1}}},
           {"state": "list criteria", "questions": {"team": {"type": "choice", "instructions": "Who?", "criteria": ["a", "b"], "label": "a"}}}]
    src = tmp_path / "d.jsonl"
    src.write_text("".join(json.dumps(row) + "\n" for row in rows + bad), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["split_data.py", str(src), "--out", str(tmp_path / "out")])
    assert split_data.main() == 0
    summary = json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8"))
    assert summary["invalid_lines"] == 2 and summary["records"] == 30
    assert all(set(part["labels"]) == {"department", "escalate", "frustration"} for part in summary["partitions"].values())


def test_split_cli_keeps_question_variants_in_the_same_partition(tmp_path, monkeypatch):
    rows = [{"state": f"case {i}", "questions": {"rating": {
        "type": "score", "instructions": "Rate the experience", "criteria": levels, "label": len(levels) - 1}}}
        for i in range(60) for levels in (["bad", "good"], ["bad", "neutral", "good"])]
    src, out = tmp_path / "variants.jsonl", tmp_path / "out"
    src.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["split_data.py", str(src), "--out", str(out)])
    assert split_data.main() == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["records"] == 120 and summary["conflicting_states_dropped"] == 0
    for part in summary["partitions"].values():
        assert set(part["labels"]) == {"rating (2 levels)", "rating (3 levels)"}
    states = []
    for name in ("train", "calibration", "development"):
        part = [json.loads(line) for line in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        counts = Counter(split_data.state_key(row["state"]) for row in part)
        assert set(counts.values()) == {2}
        states.append(set(counts))
    assert sum(map(len, states)) == 60
    assert not (states[0] & states[1] or states[0] & states[2] or states[1] & states[2])


def test_split_keeps_states_together_and_is_deterministic():
    rows = [record(f"ticket {i % 40}", dept=["returns", "shipping", "billing", "account"][i % 4]) for i in range(80)]
    parts = split_data.split(rows, {"calibration": 0.2, "development": 0.2}, seed=3)
    assert sum(len(v) for v in parts.values()) == 80
    states = {name: {split_data.state_key(r["state"]) for r in rs} for name, rs in parts.items()}
    assert not (states["train"] & states["calibration"]) and not (states["train"] & states["development"]) and not (states["calibration"] & states["development"])
    assert parts == split_data.split(rows, {"calibration": 0.2, "development": 0.2}, seed=3)
    assert len(parts["calibration"]) == 16 and len(parts["development"]) == 16


def test_cli_writes_partitions_and_summary(tmp_path, monkeypatch):
    src = tmp_path / "d.jsonl"
    src.write_text("".join(json.dumps(record(f"case {i}", dept=["returns", "shipping", "billing", "account"][i % 4], level=i % 3)) + "\n" for i in range(60)) + "not json\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["split_data.py", str(src), "--out", str(tmp_path / "out")])
    assert split_data.main() == 0
    summary = json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8"))
    assert summary["records"] == 60 and summary["invalid_lines"] == 1
    assert set(summary["partitions"]) == {"train", "calibration", "development"}
    assert all((tmp_path / f"out/{p}.jsonl").exists() for p in summary["partitions"])


def test_warnings_flag_rare_and_missing_labels():
    rows = [record(f"s{i}", dept="billing", urgent=True) for i in range(50)] + [record("t", dept="returns")]
    text = " ".join(split_data.warnings_for(rows))
    assert "'department'" in text and "shipping" in text and "under 5%" in text
    assert "'escalate'" in text and "false" in text


def test_batch_targets_fill_deficits_then_even_out():
    keys = ["a", "b", "c"]
    assert generate_data.batch_targets(Counter(), keys, 0, 9) == {"a": 3, "b": 3, "c": 3}
    alloc = generate_data.batch_targets(Counter(a=20, b=2), keys, 22, 10)
    assert sum(alloc.values()) == 10 and "a" not in alloc and alloc["c"] >= alloc["b"]


def test_prompt_lists_every_question_and_targets():
    counts = {qid: Counter() for qid in SPEC["questions"]}
    prompt = generate_data.build_prompt(SPEC, counts, 0, 20, [], generate_data.random.Random(0))
    for qid in SPEC["questions"]: assert f"- {qid} (" in prompt and f"- {qid}: " in prompt
    assert "exactly 20 records" in prompt and "Levels: 0 = Calm or neutral" in prompt


def test_parse_and_coerce_generated_records():
    text = '```json\n{"records": [{"state": "charged twice", "labels": {"department": "Billing", "escalate": "true", "frustration": "2"}}]}\n```'
    items = generate_data.parse_records(text)
    rec, why = generate_data.to_record(SPEC, items[0])
    assert why is None and rec["questions"]["department"]["label"] == "billing"
    assert rec["questions"]["escalate"]["label"] is True and rec["questions"]["frustration"]["label"] == 2
    assert generate_data.to_record(SPEC, {"state": "x", "labels": {"department": "legal", "escalate": True, "frustration": 0}})[0] is None
    assert generate_data.to_record(SPEC, {"state": "x", "labels": {"department": "billing"}})[1] == "no label for escalate"
    assert generate_data.parse_records("no json here") == []


def test_example_spec_is_valid_and_skill_files_exist():
    assert generate_data.load_spec(SKILL / "assets/workload.example.json")["questions"].keys() == {"department", "escalate", "frustration"}
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: kev-finetune\n")
    for ref in ("references/data-format.md", "references/hill-climbing.md", "references/deploy.md", "scripts/kev_modal.py"):
        assert (SKILL / ref).exists() and ref in text, ref


@pytest.mark.parametrize("value,expected", [("yes", "yes"), ("TRUE", True), (1.0, 1.0)])
def test_coerce_noul_only_accepts_booleans(value, expected):
    q = SPEC["questions"]["escalate"]
    assert generate_data.coerce_label(q, value) == expected


def test_split_holdout_keeps_real_records_out_of_train(tmp_path, monkeypatch):
    synthetic = [record(f"gen {i}", dept=["returns", "shipping", "billing", "account"][i % 4]) for i in range(100)]
    real = [record(f"real {i}") for i in range(40)] + [record("gen 3")]   # one real state also appears in the synthetic file
    for name, rows in (("gen.jsonl", synthetic), ("real.jsonl", real)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["split_data.py", str(tmp_path / "gen.jsonl"), "--out", str(tmp_path / "out"), "--holdout", str(tmp_path / "real.jsonl")])
    assert split_data.main() == 0
    parts = {p: [json.loads(l) for l in (tmp_path / f"out/{p}.jsonl").read_text(encoding="utf-8").splitlines()] for p in ("train", "calibration", "development")}
    assert len(parts["train"]) == 99 and all(r["state"].startswith("gen") for r in parts["train"])
    assert len(parts["calibration"]) + len(parts["development"]) == 41 and all(not r["state"].startswith("gen ") or r["state"] == "gen 3" for r in parts["calibration"] + parts["development"])


def test_convert_maps_columns_and_labels(tmp_path, monkeypatch):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text('body,team,urgent,prio\n"charged twice",Billing Ops,yes,2\n"where is it",shipping,no,1\n"login",Legal,no,1\n,billing,no,1\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(sys, "argv", ["convert_data.py", str(SKILL / "assets/workload.example.json"), str(csv_path), "--state", "body", "--label", "department=team", "--label", "escalate=urgent",
                "--label", "frustration=prio", "--score-offset", "1", "--map", "department=Billing Ops:billing", "--out", str(out)])
    assert convert_data.main() == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [r["questions"]["department"]["label"] for r in rows] == ["billing", "shipping"]
    assert rows[0]["questions"]["escalate"]["label"] is True and rows[0]["questions"]["frustration"]["label"] == 1
    assert split_data.check_record(rows[0]) == []


def test_convert_label_matching():
    q = SPEC["questions"]["frustration"]
    assert convert_data.map_label(q, "Annoyed", {}, 0) == (1, None)
    assert convert_data.map_label(q, "3", {}, 1) == (2, None)
    assert convert_data.map_label(q, "5", {}, 1)[0] is None
    assert convert_data.map_label(SPEC["questions"]["department"], "RETURNS", {}, 0) == ("returns", None)
    assert convert_data.map_label(SPEC["questions"]["escalate"], "N", {}, 0) == (False, None)
    assert convert_data.get({"meta": {"team": "x"}}, "meta.team") == "x"


@pytest.mark.parametrize("value", ["1.9", "-0.5", "0.99999999999999999", "NaN", "Infinity", "-Infinity", "1e9999999"])
def test_convert_rejects_fractional_and_nonfinite_score_indices(value):
    label, reason = convert_data.map_label(SPEC["questions"]["frustration"], value, {}, 0)
    assert label is None and reason


@pytest.mark.parametrize("value,offset,expected", [("1", 0, 1), ("1.0", 0, 1), ("2e0", 1, 1), ("-0.0", 0, 0), ("Annoyed", 0, 1)])
def test_convert_preserves_integer_scores_offsets_and_level_text(value, offset, expected):
    assert convert_data.map_label(SPEC["questions"]["frustration"], value, {}, offset) == (expected, None)


def test_convert_skips_invalid_scores_and_continues(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"questions": {"rating": SPEC["questions"]["frustration"]}}), encoding="utf-8")
    src, out = tmp_path / "ratings.csv", tmp_path / "out.jsonl"
    src.write_text("body,rating\nfirst,1.9\nsecond,Infinity\nthird,2.0\nfourth,medium\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["convert_data.py", str(spec), str(src), "--state", "body", "--score-offset", "1",
                                    "--map", "rating=medium:2.0", "--out", str(out)])
    assert convert_data.main() == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["state"] for r in rows] == ["third", "fourth"]
    assert all(r["questions"]["rating"]["label"] == 1 for r in rows)
    assert "2 rows skipped" in capsys.readouterr().out


def test_extract_finds_typescript_and_python_questions(tmp_path):
    ts = ['import { experimental_evaluate as evaluate } from "ai";', 'const r = await evaluate({ model: "typesafe-ai/jev", state, questions: {',
          '  urgent: { type: "boolean", instructions: "Is this urgent?" },',
          '  team: { type: "choice", instructions: "Which team?", criteria: { billing: "Money", shipping: "Boxes that don\'t arrive" } },', "}});"]
    py = ["from typesafe_sdk import Choice, Noul, Score, TypeSafeClient", "client = TypeSafeClient(api_key=KEY)",
          'r = client.system_one(state=text, questions={"mood": Score(instructions="How angry?", criteria=["calm", "angry"]), "spam": Noul(instructions="Is it spam?")})']
    (tmp_path / "app.ts").write_text("\n".join(ts), encoding="utf-8")
    (tmp_path / "svc.py").write_text("\n".join(py), encoding="utf-8")
    (tmp_path / "node_modules").mkdir(); (tmp_path / "node_modules/x.ts").write_text('type: "choice", instructions: "ignored"', encoding="utf-8")
    found = {}
    for p in extract_workload.files_under(tmp_path):
        found.update(extract_workload.extract_questions(extract_workload.read(p), p.name))
    assert found["urgent"]["type"] == "noul" and found["urgent"]["instructions"] == "Is this urgent?"
    assert found["team"]["criteria"] == {"billing": "Money", "shipping": "Boxes that don't arrive"}
    assert found["mood"] == {"type": "score", "instructions": "How angry?", "criteria": ["calm", "angry"], "_source": "svc.py:3"}
    assert found["spam"]["type"] == "noul" and "ignored" not in json.dumps(found)


def test_extract_cli_writes_draft(tmp_path, capsys, monkeypatch):
    (tmp_path / "a.py").write_text('q = {"ok": {"type": "noul", "instructions": "Fine?"}}\nrequests.post(url + "/v1/systemone", json=q)\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["extract_workload.py", str(tmp_path), "--out", str(tmp_path / "wl.json")])
    assert extract_workload.main() == 0
    spec = json.loads((tmp_path / "wl.json").read_text(encoding="utf-8"))
    assert spec["questions"]["ok"]["instructions"] == "Fine?" and spec["domain"].startswith("_todo")
    assert "a.py" in capsys.readouterr().out


def test_plan_size_numbers():
    p = plan_size.plan(3, 0.75, 0.05, 0.05, 0.8, 0.15, 0.15)
    assert 400 < p["development_questions"] < 550 and p["development_questions_unpaired_bound"] > p["development_questions"]
    assert p["total_records"] >= p["development_records"] / 0.15 and p["calibration_records"] * 3 >= 100
    assert plan_size.plan(1, 0.7, 0.1, 0.05, 0.8, 0.15, 0.15)["development_questions"] < p["development_questions"]
