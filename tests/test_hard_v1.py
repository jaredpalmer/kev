"""hard-v1 generators (scripts/build_hard_v1.py): labels come from the rule engines, option order is randomised, partitions
use disjoint templates, the build is deterministic, and every fact a solver decides on is stated in the frozen records'
states. No weights, no network, no tokenizer (a tiny build without the context check).
Run: uv run python -m pytest tests/test_hard_v1.py -q
"""
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path

import pytest

from kev.api import render
from kev.data import materialize
from kev.suite import read_jsonl
from scripts.build_hard_v1 import TEMPLATE_SPLITS, build, family_counts, normalised
from scripts.hard_v1_common import business_days_after, day, money, words
from scripts.hard_v1_families import FAMILIES, UNITS, amb_decide, labels
from scripts.hard_v1_numeric import LOGIC_ATOMS, TZ, solve_judge, solve_temporal
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


def test_family_counts_never_leave_a_family_empty():
    for total in range(len(FAMILIES) + 1, 300):
        counts = family_counts(total)
        assert sum(counts.values()) == total and min(counts.values()) >= 1 and counts["ambiguous"] % 2 == 0, (total, counts)
    for total in range(len(FAMILIES) + 1):
        with pytest.raises(ValueError, match="at least"):
            family_counts(total)


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


# ---------------------------------------------------------------- deciding facts are stated
# The labels are computed from `_meta.facts`, so a fact the solver reads but the state never states would make a label
# unanswerable from the text. For each family, DECIDING lists the values its solver reads and the surface forms the
# generators may write them in; every value must appear, as a whole token in one of its forms, in kev.api.render(state).
# Runs over the frozen development and test partitions (in git).

SUITE = Path(__file__).resolve().parents[1] / "evals" / "hard-v1"
DATE_STYLES = ("us", "iso", "eu", "weekday", "weekday_eu")


def number(v):
    """A number as the generators write it: digits with or without thousands separators, 0-3 decimals, words below 1,000."""
    v = abs(Fraction(v))
    out = {f"{float(v):,.{k}f}" for k in range(4)} | {f"{float(v):.{k}f}" for k in range(4)}
    if v.denominator == 1 and v < 1000: out.add(words(int(v)))
    return out


def cents(c):
    return {money(abs(c)), money(abs(c), "whole_ok"), money(abs(c), "code")}


def dollars(n):
    return cents(100 * n)


def percent(p):
    """A probability or share (Fraction) in any of the templates' number styles (hard_v1_numeric.prob_phrases)."""
    p = Fraction(p)
    pc = p * 100
    out = {f"{float(pc):g}%", f"{float(pc):.1f}%", f"{float(p):g}", f"{p.numerator}-in-{p.denominator}", f"{p.numerator} in {p.denominator}",
           f"{p.numerator} times in {p.denominator}"}
    if pc.denominator == 1:
        n = int(pc)
        out |= {f"{n} percent", f"{n}-in-100", f"{n} out of every 100", f"{n} out of 100", f"{n}% "}
        if 0 <= n < 1000: out.add(f"{words(n)} percent")
    return out


def ordinal(n):
    return {f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"}


def when(iso):
    d = date.fromisoformat(iso)
    return {day(d, s) for s in DATE_STYLES}


def clock(iso):
    t = datetime.fromisoformat(iso)
    return {t.strftime("%H:%M"), t.strftime("%I:%M %p").lstrip("0")}


def name(s):
    return {s}


def tradeoff(f):
    out = [name(n) for n in f["options"]]
    for o in f["options"].values():
        out += [{fmt(v) for fmt in UNITS.values()} for v in o.values() if not isinstance(v, bool)]
    out += [{fmt(c["value"]) for fmt in UNITS.values()} for c in f["constraints"] if c["op"] != "is"]
    out += [{f"{w}%"} for w in f["weights"].values()]
    if f["years"]: out.append(number(f["years"]))
    return out


def multi_hop(f):
    k = f["kind"]
    if k == "org":
        chain, who = [f["requester"]], f["requester"]
        while who in f["reports"]:
            who = f["reports"][who]; chain.append(who)
        return [name(p) for p in chain] + [name(f["titles"][p]) for p in chain] + ([name(f["min_title"])] if f.get("min_title") else [])
    if k == "deps":
        return [name(s) for a, b, _ in f["edges"] for s in (a, b)] + [name(f["down"])] + [name(c) for c in f["candidates"]]
    if k == "ownership":
        return ([name(e) for e in f["stakes"]] + [name(h) for hs in f["stakes"].values() for h in hs] + [percent(p) for hs in f["stakes"].values() for p in hs.values()]
                + [name(c) for c in f["candidates"]])
    team = f["routes"][f["component"]]
    people = [f[role][team] for role in ("primary", "secondary", "manager")]
    return ([name(f["component"]), name(team), when(f["date"])] + [name(p) for p in people]
            + [when(d) for p in people for span in f["leave"].get(p, []) for d in span])


def ambiguous(f):
    out = []
    for k, v in f["params"].items():
        if v is None or isinstance(v, bool) or k == "cause": continue
        if k == "hours": out += [number(h) for h in v if h is not None]
        elif k.endswith("_cents"): out.append(cents(v))
        elif k == "title": out.append(name(v))
        elif k == "max_gap_pct": out.append({f"{v}%", f"{v} percent"})
        elif isinstance(v, str): out.append(when(v))
        else: out.append(number(v))
    return out


def long_policy(f):
    case, cat = f["case"], f["categories"][f["case"]["category"]]
    out = [when(case[k]) for k in ("start", "incident", "reported")] + [cents(case["loss_cents"]), number(f["notice_days"]), dollars(cat["limit"]), dollars(cat["fee"])]
    if cat["waiting"]: out.append(number(cat["waiting"]))
    if case["item_class"]: out.append(dollars(f["classes"][case["item_class"]]["cap"]))
    for ex in f["exclusions"]:
        if case["category"] in ex["categories"] and ex["kind"] != "flag":
            out.append(number(ex["thr"]))
            if case["values"].get(ex["key"]) is not None: out.append(number(case["values"][ex["key"]]))
    return out


def probability(f):
    k = f["kind"]
    if k == "ev":
        return ([name(n) for n in f["projects"]] + [percent(p) for o in f["projects"].values() for p, _ in o["outcomes"]]
                + [dollars(v) for o in f["projects"].values() for _, v in o["outcomes"]] + [dollars(o["cost"]) for o in f["projects"].values()])
    if k == "bayes": return [percent(f["p"][x]) for x in ("base", "sens", "fpr")]
    if k == "independent": return [percent(p) for p in f["ps"]]
    if k == "compare": return [percent(f["a"]["p1"]), percent(f["a"]["p2"]), percent(f["b"])]
    return [name(f["row"]), name(f["col"])] + [number(n) for cols in f["counts"].values() for n in cols.values()]


def temporal_numeric(f):
    k = f["kind"]
    if k == "deadline": return [when(f["received"]), number(f["n"])] + [when(h) for h in f["holidays"]]
    if k == "diff": return [when(f["invoice"]), when(f["paid"]), number(f["limit"])]
    if k == "tz":
        city = {off: c for c, off in TZ}
        return [clock(f["start"]), name(city[f["off_a"]]), name(city[f["off_b"]])]
    if k == "prorata": return [when(f["start"]), when(f["end"]), when(f["cancel"]), cents(f["price_cents"])]
    if k == "shift":
        brk = number(f["break_min"]) | ({"no break"} if f["break_min"] == 0 else set())
        return [clock(f["start"]), clock(f["end"]), brk, cents(f["rate_cents"]), number(f["ot_after_h"])]
    p = f["pattern"]
    if p in ("every_n_weeks", "business_days"):
        return [when(f["first"]), number(f["n"]) | ({"every other"} if f["n"] == 2 else set()), ordinal(f["k"]) | {f"number {f['k']}"}]
    month = f"{date(f['year'], f['month'], 1):%B}"
    return [name(month), number(f["year"])] + ([number(f["dom"])] if p == "monthly_rollback" else [])


def judge(f):
    a = f["args"]
    if f["kind"] == "logic":
        phrase = {key: text for text, key in LOGIC_ATOMS}
        return [name(phrase[x]) for x in a["atoms"]]
    out = []
    for k, v in a.items():
        if k == "factor": continue                       # stated as the conversion hint, not as the factor
        if f["kind"] in ("order", "reverse_pct") and k in ("p1", "p2", "after"): out.append(cents(int(Fraction(v) * 100)))
        elif k in ("d", "tax", "p") and f["kind"] in ("order", "reverse_pct"): out.append(percent(v))
        elif k == "offset" and Fraction(v) == 0: continue
        else: out.append(number(v))
    return out


DECIDING = {"tradeoff": tradeoff, "multi_hop": multi_hop, "ambiguous": ambiguous, "long_policy": long_policy, "probability": probability,
            "temporal_numeric": temporal_numeric, "judge": judge}


def stated(form, text):
    """`form` occurs in `text` as a whole token: not inside a longer word or number (so "4" does not match "14" or "4.5")."""
    return re.search(r"(?<![\w.,])" + re.escape(form.casefold()) + r"(?!\w|[.,]\d)", text) is not None


def records():
    return [r for split in ("development", "test") for r in read_jsonl(SUITE / f"{split}.jsonl")]


def test_every_family_has_a_deciding_facts_projection():
    assert set(DECIDING) == {r["_meta"]["family"] for r in records()}


def test_stated_matches_whole_tokens_only():
    assert stated("4", "every 4 business days.") and stated("$18.00", "the rate is $18.00.")
    assert not stated("4", "every 14 days") and not stated("4", "4.5 hours") and not stated("4", "40 hours")


@pytest.mark.parametrize("family", sorted(DECIDING))
def test_deciding_facts_are_stated_in_the_state(family):
    missing = []
    for r in records():
        if r["_meta"]["family"] != family: continue
        text = render(r["state"]).casefold()
        for forms in DECIDING[family](r["_meta"]["facts"]):
            if not any(stated(str(x), text) for x in forms):
                missing.append((r["_meta"]["id"], sorted(forms)[:4]))
    assert not missing, f"{len(missing)} deciding facts not in the state, e.g. {missing[:5]}"
