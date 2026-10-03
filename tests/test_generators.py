"""The programmatic data generators: contrastive policy pairs (kev.contrastive), compositional rule records (kev.composition)
and the v3 suite builder's grouping (kev.study_v3). No weights, no network.
Run: uv run python -m pytest tests/test_generators.py -q
"""
import pytest

from kev.contrastive import generate, paired_flip


def test_rendered_pairs_change_one_sentence_not_order():
    rows, _ = generate(20, 17)
    for a, b in zip(rows[::2], rows[1::2]):
        left = a["state"]["case"].split(". ")
        right = b["state"]["case"].split(". ")
        assert len(left) == len(right)
        assert sum(x != y for x, y in zip(left, right)) == 1

def test_dates_render_like_the_frozen_suites_on_every_os():
    """kev.contrastive wrote the frozen suites' dates with strftime("%B %-d, %Y"), which Windows rejects; the portable form
    renders every day the same."""
    import sys
    from datetime import date, timedelta
    from kev.contrastive import _day
    assert [_day(date(2026, 3, 5)), _day(date(2026, 11, 28))] == ["March 5, 2026", "November 28, 2026"]
    if sys.platform != "win32":
        days = [date(2026, 1, 1) + timedelta(n) for n in range(365)]
        assert [_day(d) for d in days] == [d.strftime("%B %-d, %Y") for d in days]

def test_pair_metric_compares_semantics_not_indices():
    rows = [
        {"pair_id": "p", "sibling": "a", "keys": ["deny", "allow"], "label": 1, "p": [0, 1]},
        {"pair_id": "p", "sibling": "b", "keys": ["allow", "deny"], "label": 1, "p": [0, 1]},
    ]
    assert paired_flip(rows)["flip_rate"] == 1
    assert paired_flip(rows)["both_correct_rate"] == 1

def test_pair_metric_refuses_missing_sibling():
    with pytest.raises(ValueError, match="incomplete"):
        paired_flip([{"pair_id": "p", "sibling": "a", "keys": ["x", "y"], "label": 0, "p": [1, 0]}])

def test_compositional_truth_tables_and_unknowns():
    from itertools import product
    from kev.composition import evaluate_rule
    atoms = [{"kind": "flag", "fields": [k], "threshold": 0} for k in ("a", "b", "c")]
    for a, b, c in product((True, False), repeat=3):
        facts = dict(a=a, b=b, c=c)
        assert evaluate_rule(("and", ("or", 0, 1), 2), atoms, facts) == ((a or b) and c)
        assert evaluate_rule(("unless", 0, 1), atoms, facts) == (a and not b)
        assert evaluate_rule(("if", 0, 1, 2), atoms, facts) == (b if a else c)
    assert evaluate_rule(("and", 0, 1), atoms, {"a": False}) is False
    assert evaluate_rule(("and", 0, 1), atoms, {"a": True}) is None
    assert evaluate_rule(("or", 0, 1), atoms, {"a": False}) is None

@pytest.mark.parametrize("kind,threshold,value,expected", [
    ("lt", 10, 10, False), ("le", 10, 10, True), ("gt", 10, 10, False),
    ("ge", 10, 10, True), ("eq", 10, 11, False), ("range", 10, 20, True), ("range", 10, 21, False),
])
def test_rule_boundary_labels(kind, threshold, value, expected):
    from kev.composition import atom_value
    assert atom_value({"kind": kind, "threshold": threshold, "fields": ["x"]}, {"x": value}) == expected

def test_compositional_pairs_validate_and_keep_invariance():
    from kev.composition import SHAPES, TRAIN_SHAPES, DEV_SHAPES, TEST_SHAPES, check_group, generate as compose
    from kev.benchmark import labels, prediction_rows
    assert not set(TRAIN_SHAPES) & (set(DEV_SHAPES) | set(TEST_SHAPES))
    records = compose(4, "test", tuple(SHAPES))
    rows = []
    for i in range(0, len(records), 4):
        group = records[i:i + 4]
        assert check_group(group)
        for r in group:
            q = r["questions"]["decision"]
            keys, y = labels(q)
            rows += prediction_rows(r, {"probabilities": {"decision": {k: int(i == y) for i, k in enumerate(keys)}}})
    summary = paired_flip(rows)
    assert summary["both_correct_rate"] == 1
    assert summary["invariance_rate"] == 1
    assert summary["invariant_both_correct_rate"] == 1
    records[0]["state"]["case"] = "The facts were changed."
    with pytest.raises(ValueError, match="rendered facts"):
        check_group(records[:4])

def test_calibration_covers_every_family_without_splitting_groups():
    from kev.study_v3 import grouped_split, legacy
    train, calibration = grouped_split(legacy(10, "split-test"), 2)
    assert {r["_meta"]["family"] for r in train} == {r["_meta"]["family"] for r in calibration}
    assert not {r["_meta"]["group_id"] for r in train} & {r["_meta"]["group_id"] for r in calibration}
    assert len(calibration) == 16

def test_date_and_entity_atoms():
    from kev.composition import atom_value
    a = {"kind": "elapsed", "fields": ["start", "end"], "threshold": 2}
    assert atom_value(a, {"start": "2028-02-28", "end": "2028-03-01"}) is True
    assert atom_value(a, {"start": "2028-02-28", "end": "2028-03-02"}) is False
    a = {"kind": "match", "fields": ["signer", "approver"], "threshold": 0}
    assert atom_value(a, {"signer": "Mira", "approver": "Mira"}) is True
    assert atom_value(a, {"signer": "Mira", "approver": "Noah"}) is False
    assert atom_value(a, {"signer": "Mira"}) is None

def test_random_rule_structures_exclude_heldout_and_cover_negation():
    from kev.composition import SHAPES, DEV_SHAPES, TEST_SHAPES, canonical, push_negation, sample_trees, generate as compose, check_group
    assert canonical(("or", ("not", 0), ("and", 1, 2))) == canonical(SHAPES["held_or_not"])       # order/numbering-independent
    assert canonical(("not", ("and", 0, 1))) != canonical(("or", ("not", 0), ("not", 1)))
    assert canonical(push_negation(("not", ("and", 0, 1)))) == canonical(("or", ("not", 0), ("not", 1)))   # De Morgan
    trees = sample_trees(30, "t")
    held = {canonical(SHAPES[s]) for s in DEV_SHAPES + TEST_SHAPES}
    assert len({canonical(t) for t in trees}) == 30 and not any(canonical(t) in held or canonical(push_negation(t)) in held for t in trees)
    assert any("not(" in canonical(t) for t in trees)
    recs = compose(1, "t", styles=(3, 4), trees={f"rand{i}": t for i, t in enumerate(trees[:5])})
    for i in range(0, len(recs), 4):
        assert check_group(recs[i:i + 4])

def test_ordinal_threshold_families_are_balanced_minimal_pairs():
    import collections
    from kev.contrastive import ORDINAL_FAMILIES, generate
    from kev.data import materialize
    recs, rep = generate(30, "t", families=list(ORDINAL_FAMILIES))
    assert all(v["pairs"] == 30 for v in rep.values())
    for a, b in zip(recs[::2], recs[1::2]):
        assert a["questions"]["decision"]["type"] == "score" and a["questions"]["decision"]["label"] != b["questions"]["decision"]["label"]
        assert sum(x != y for x, y in zip(a["state"]["case"].split(". "), b["state"]["case"].split(". "))) == 1
        materialize(a)
    counts = collections.Counter((r["_meta"]["family"], r["questions"]["decision"]["label"]) for r in recs)
    assert all(counts[(f, level)] >= 8 for f in ORDINAL_FAMILIES for level in (0, 1, 2))   # every level appears in every family
