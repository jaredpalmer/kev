"""hard-v1 generators (scripts/build_hard_v1.py): labels come from the rule engines, option order is randomised, partitions
use disjoint templates, the build is deterministic. No weights, no network, no tokenizer (a tiny build without the
context check). Run: uv run python -m pytest tests/test_hard_v1.py -q
"""
import json
from collections import Counter, defaultdict
from datetime import date

import pytest

from kev.data import materialize
from scripts.build_hard_v1 import TEMPLATE_SPLITS, build, family_counts, normalised
from scripts.hard_v1_common import business_days_after
from scripts.hard_v1_families import FAMILIES, amb_decide, labels, solve_judge, solve_temporal
from scripts.hard_v1_policy import solve as solve_policy

SIZES = {"train": 70, "development": 28, "test": 28}


@pytest.fixture(scope="module")
def parts():
    return build(SIZES)[0]


def test_sizes_and_families(parts):
    for split, n in SIZES.items():
        assert len(parts[split]) == n
        assert Counter(r["_meta"]["family"] for r in parts[split]) == Counter(family_counts(n))
    assert sum(family_counts(6000).values()) == 6000 and family_counts(6000)["ambiguous"] % 2 == 0


def test_every_label_is_what_the_rule_engine_computes(parts):
    for split, records in parts.items():
        for r in records:
            fam, facts = r["_meta"]["family"], json.loads(json.dumps(r["_meta"]["facts"]))
            assert labels(fam, facts, r["questions"]) == {qid: q["label"] for qid, q in r["questions"].items()}, r["_meta"]["id"]
            truth = FAMILIES[fam][1](facts)
            for qid, q in r["questions"].items():
                if q["type"] == "choice":
                    assert facts["options_q"][qid][q["label"]] == truth[qid]
            materialize(r)   # a valid request with a label of the right type


def test_option_order_is_randomised(parts):
    positions = defaultdict(Counter)
    for records in parts.values():
        for r in records:
            for q in r["questions"].values():
                if q["type"] == "choice":
                    positions[r["_meta"]["family"]][list(q["criteria"]).index(q["label"])] += 1
    for fam, c in positions.items():
        assert len(c) >= 3, (fam, c)                      # the answer is not parked in one slot
        assert max(c.values()) <= 0.6 * sum(c.values()), (fam, c)


def test_partitions_use_disjoint_templates_and_states(parts):
    templates = {split: {r["_meta"]["template"].split("/t")[1] for r in recs} for split, recs in parts.items()}
    for split, used in templates.items():
        assert used <= {str(t) for t in TEMPLATE_SPLITS[split]}
    assert not templates["train"] & templates["development"] and not templates["train"] & templates["test"] and not templates["development"] & templates["test"]
    states = [normalised(r["state"]) for recs in parts.values() for r in recs]
    assert len(states) == len(set(states))
    ids = [r["_meta"]["id"] for recs in parts.values() for r in recs]
    assert len(ids) == len(set(ids))


def test_ambiguous_twins_share_a_group(parts):
    for records in parts.values():
        groups = defaultdict(list)
        for r in records:
            if r["_meta"]["family"] == "ambiguous": groups[r["_meta"]["group_id"]].append(r)
        for twins in groups.values():
            assert sorted(r["_meta"]["twin"] for r in twins) == ["absent", "intact"]
            assert len({r["_meta"]["template"] for r in twins}) == 1


def test_build_is_deterministic(parts):
    again = build(SIZES)[0]
    for split in SIZES:
        assert json.dumps(parts[split], ensure_ascii=False) == json.dumps(again[split], ensure_ascii=False)


def test_rule_engines_on_hand_built_cases():
    # business days: Friday + 3, over a weekend and a Monday holiday -> Thursday
    assert business_days_after(date(2026, 3, 6), 3, {date(2026, 3, 9)}) == date(2026, 3, 12)
    assert solve_temporal({"kind": "deadline", "received": "2026-03-07", "n": 1, "holidays": []})["due"] == "2026-03-09"   # received Saturday
    assert solve_temporal({"kind": "tz", "start": "2026-07-11T03:00", "off_a": 12, "off_b": -10})["local"] == "2026-07-10T05:00"
    assert solve_temporal({"kind": "prorata", "start": "2027-03-01", "end": "2028-03-01", "cancel": "2027-03-01", "price_cents": 36600})["refund"] == 36500   # leap plan year
    assert solve_judge({"kind": "fence", "args": {"length": "20", "gap": "5"}, "dp": 0, "proposed": 4}) == {"correct": False, "value": 5}
    # ambiguous: a missing Friday timesheet does not matter once Monday-Thursday already exceed 40 hours
    assert amb_decide({"scenario": "overtime", "params": {"hours": [11, 11, 11, 11, None], "max_shift": 12}}) == "overtime_due"
    assert amb_decide({"scenario": "overtime", "params": {"hours": [8, 8, 8, 8, None], "max_shift": 12}}) is None
    # long_policy: sublimit applied before the deductible vs after
    f = {"fee_order": "before", "notice_days": 30, "categories": {"baggage": {"limit": 2000, "fee": 100, "waiting": 0, "purchased": True}},
         "classes": {"valuables": {"category": "baggage", "cap": 500}}, "exclusions": [],
         "case": {"category": "baggage", "item": "the laptop", "item_class": "valuables", "loss_cents": 150000, "start": "2026-01-01",
                  "incident": "2026-02-01", "reported": "2026-02-10", "conditions": {}, "values": {}, "exceptions": {}}}
    assert solve_policy(f) == {"outcome": "pay_sublimit", "amount": 50000}
    assert solve_policy({**f, "fee_order": "after"}) == {"outcome": "pay_sublimit", "amount": 40000}
    late = {**f, "case": {**f["case"], "reported": "2026-03-15"}}
    assert solve_policy(late) == {"outcome": "deny_late", "amount": 0}
    excluded = {**f, "exclusions": [{"key": "unattended", "categories": ["baggage"], "kind": "flag", "thr": None, "cmp": None, "has_exception": True}],
                "case": {**f["case"], "conditions": {"unattended": True}, "exceptions": {"unattended": True}}}
    assert solve_policy(excluded)["outcome"] == "pay_sublimit"                      # the exception saves the claim
    excluded["case"]["exceptions"]["unattended"] = False
    assert solve_policy(excluded) == {"outcome": "deny_unattended", "amount": 0}
