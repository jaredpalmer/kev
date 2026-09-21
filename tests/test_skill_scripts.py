"""The kev-finetune skill's local scripts (skills/kev-finetune/scripts): record validation, state-grouped splitting,
generator prompt/balancing and response parsing. Standard library only, no network.
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
import generate_data  # noqa: E402
import split_data  # noqa: E402

SPEC = json.loads((SKILL / "assets/workload.example.json").read_text(encoding="utf-8"))


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


def test_split_keeps_states_together_and_is_deterministic():
    rows = [record(f"ticket {i % 40}", dept=["returns", "shipping", "billing", "account"][i % 4]) for i in range(80)]
    parts = split_data.split(rows, {"calibration": 0.2, "development": 0.2}, seed=3)
    assert sum(len(v) for v in parts.values()) == 80
    states = {name: {split_data.state_key(r["state"]) for r in rs} for name, rs in parts.items()}
    assert not (states["train"] & states["calibration"]) and not (states["train"] & states["development"]) and not (states["calibration"] & states["development"])
    assert parts == split_data.split(rows, {"calibration": 0.2, "development": 0.2}, seed=3)
    assert len(parts["calibration"]) == 16 and len(parts["development"]) == 16


def test_cli_writes_partitions_and_summary(tmp_path):
    src = tmp_path / "d.jsonl"
    src.write_text("".join(json.dumps(record(f"case {i}", dept=["returns", "shipping", "billing", "account"][i % 4], level=i % 3)) + "\n" for i in range(60)) + "not json\n", encoding="utf-8")
    sys.argv = ["split_data.py", str(src), "--out", str(tmp_path / "out")]
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
