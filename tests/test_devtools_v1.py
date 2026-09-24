"""The devtools-v1 builder's label mappings and selection rules (scripts/build_devtools_v1.py) on small synthetic inputs.
No weights, no network. Run: uv run python -m pytest tests/test_devtools_v1.py -q
"""
import random

import pytest

from kev.data import materialize
from scripts.build_devtools_v1 import (COMMIT_TYPES, Components, assign_match, balanced_pairs, check_invariants, codereviewer_state,
                                       commit_type, deal_groups, is_balanced, message_negatives, q_choice, q_noul, round_robin, text_key)


@pytest.mark.parametrize("subject,expected", [
    ("Add support for YAML configs", "feature"),
    ("Implement retry with backoff", "feature"),
    ("Fix off-by-one in pager", "fix"),
    ("Fixed crash when list is empty", "fix"),
    ("Correct typo in variable name", "fix"),
    ("Remove unused imports", "remove"),
    ("Delete legacy handler", "remove"),
    ("Drop Python 2 support", "remove"),
    ("Refactor parser into modules", "change"),
    ("Update dependencies", "change"),
    ("Rename foo to bar", "change"),
    ("Use pathlib instead of os.path", "change"),
    ("Bump version to 1.2.3", "change"),
    ("Document the public API", "docs"),
    ("Test the empty-input case", "test"),
    ('Revert "Add caching layer"', "revert"),
    ("Add tests for the parser", "test"),
    ("Add unit tests for pager", "test"),
    ("Add missing docstrings", "docs"),
    ("Update README with install steps", "docs"),
    ("Update the changelog", "docs"),
    ("fix(cli): handle empty args", "fix"),
    ("feat: add dark mode", "feature"),
    ("feat(api)!: add tests endpoint", "feature"),
    ("docs: explain flags", "docs"),
    ("refactor: split module", "change"),
    ("chore: bump deps", None),
    ("Make the build faster", None),
    ("Handle None in parser", None),
    ("", None),
    ("1.2.3 release", None),
])
def test_commit_type_from_first_verb(subject, expected):
    assert commit_type(subject) == expected
    assert expected is None or expected in COMMIT_TYPES


def test_commit_type_ignores_case_and_whitespace():
    assert commit_type("  FIX   Crash ") == "fix"
    assert commit_type("ADD Tests") == "test"


def items(n_lang=2, per_repo=3, repos=3):
    return [{"id": f"{lang}/{r}/{i}", "lang": lang, "group": f"repo{r}", "subject": f"Subject {lang} {r} {i}"}
            for lang in ("python", "go")[:n_lang] for r in range(repos) for i in range(per_repo)]


def test_message_negatives_same_language_other_repo_other_subject():
    its = items()
    by_id = {it["id"]: it for it in its}
    negatives = message_negatives(its, random.Random(0))
    assert set(negatives) == set(by_id)
    for i, subject in negatives.items():
        partner = next(o for o in its if o["subject"] == subject)
        assert partner["lang"] == by_id[i]["lang"]
        assert partner["group"] != by_id[i]["group"]
        assert subject != by_id[i]["subject"]


def test_message_negatives_skip_items_without_partner():
    its = [{"id": "a", "lang": "go", "group": "r1", "subject": "Fix x"}, {"id": "b", "lang": "go", "group": "r1", "subject": "Fix y"},
           {"id": "c", "lang": "rust", "group": "r2", "subject": "Fix z"},
           {"id": "d", "lang": "c", "group": "r3", "subject": "Fix  X"}, {"id": "e", "lang": "c", "group": "r4", "subject": "fix x"}]
    assert message_negatives(its, random.Random(0)) == {}   # same repo, lone language, same normalised subject


def test_message_negatives_are_deterministic():
    assert message_negatives(items(), random.Random("s")) == message_negatives(items(), random.Random("s"))


@pytest.mark.parametrize("n", [0, 1, 2, 7, 150])
def test_assign_match_exact_half(n):
    flags = assign_match(n, random.Random(n))
    assert len(flags) == n and sum(flags) == n // 2


def test_balanced_pairs_exact_balance_and_admission():
    trues = [("t", i) for i in range(10)]
    falses = [("f", i) for i in range(4)]
    rejected = {("t", 1), ("f", 2)}
    pairs = balanced_pairs([(trues, falses)], 100, lambda x: x not in rejected)
    assert len(pairs) == 3                                 # falses 0, 1, 3 admissible
    assert all(t[0] == "t" and f[0] == "f" for t, f in pairs)
    assert not rejected & {x for p in pairs for x in p}


def test_balanced_pairs_round_robin_over_strata_and_cap():
    a = ([("a", "t", i) for i in range(5)], [("a", "f", i) for i in range(5)])
    b = ([("b", "t", i) for i in range(5)], [("b", "f", i) for i in range(1)])
    pairs = balanced_pairs([a, b], 4, lambda x: True)
    assert [t[0] for t, _ in pairs] == ["a", "b", "a", "a"]  # b runs out of falses after one pair
    assert all(t[0] == f[0] for t, f in pairs)               # a pair never mixes strata


def test_round_robin_spreads_labels_and_continues_when_one_runs_out():
    classes = {"fix": list(range(10)), "docs": [100], "feature": list(range(200, 210))}
    out = round_robin(classes, 7, lambda x: x != 201)
    assert out == [100, 200, 0, 202, 1, 203, 2]


def test_is_balanced():
    assert is_balanced([True] * 55 + [False] * 45)
    assert is_balanced([True] * 60 + [False] * 40)
    assert not is_balanced([True] * 61 + [False] * 39)
    assert not is_balanced([])


def test_deal_groups_follows_shares_and_is_deterministic():
    weights = {f"g{i}": 1 for i in range(100)}
    shares = {"test": 0.1, "development": 0.1, "train": 0.8}
    split = deal_groups(weights, shares, "seed")
    assert split == deal_groups(weights, shares, "seed")
    counts = {s: sum(v == s for v in split.values()) for s in shares}
    assert counts == {"test": 10, "development": 10, "train": 80}
    assert "train" not in deal_groups(weights, {"test": 0.5, "development": 0.5, "train": 0.0}, "seed").values()


def test_components_merge_forks():
    c = Components()
    c.union(["a/x", "b/x"]); c.union(["c/y"]); c.union(["b/x", "d/x"])
    assert c.find("a/x") == c.find("d/x") != c.find("c/y")


def test_codereviewer_state_takes_lines_before_hunk():
    oldf = "\n".join(f"line {i}" for i in range(1, 31))
    state = codereviewer_state("@@ -20,3 +20,4 @@ def f():\n line 20\n+new\n line 21", oldf)
    assert state["lines_before_hunk"].split("\n") == [f"line {i}" for i in range(10, 20)]
    assert list(state) == ["lines_before_hunk", "diff"]
    assert codereviewer_state("@@ -0,0 +1,2 @@\n+a\n+b", "") == {"diff": "@@ -0,0 +1,2 @@\n+a\n+b"}
    assert codereviewer_state("@@ -3,2 +3,2 @@\n-a\n+b", "\n\n\n") == {"diff": "@@ -3,2 +3,2 @@\n-a\n+b"}   # blank lines only


def record(split_group, key, label, src="x_yes"):
    return {"state": key, "questions": {"q": q_noul("Is it?", label, src)}, "_meta": {"group_id": split_group, "text_sha256": text_key(key)}}


def test_check_invariants_catches_group_leak_duplicates_and_imbalance():
    ok = {"train": [record("g1", "a", True), record("g1", "b", False)], "test": [record("g2", "c", True), record("g2", "d", False)]}
    check_invariants(ok)
    with pytest.raises(AssertionError, match="spans"):
        check_invariants({"train": [record("g1", "a", True), record("g1", "b", False)], "test": [record("g1", "c", True), record("g1", "d", False)]})
    with pytest.raises(AssertionError, match="duplicate"):
        check_invariants({"train": [record("g1", "a", True), record("g1", "b", False)], "test": [record("g2", " A ", True), record("g2", "d", False)]})
    with pytest.raises(AssertionError, match="unbalanced"):
        check_invariants({"train": [record("g1", "a", True), record("g1", "b", True), record("g1", "c", False)]})


def test_questions_materialize():
    rec = {"state": "diff --git a/x b/x", "questions": {
        "change_type": q_choice("What kind of change is this?", COMMIT_TYPES, "fix", "commitpackft_type"),
        "message_match": q_noul('Does this commit message describe this diff? Message: "Fix x"', False, "commitpackft_message")}}
    out = materialize(rec)
    assert [q["label"] for q in out["questions"]] == [list(COMMIT_TYPES).index("fix"), 0]
    with pytest.raises(AssertionError):
        q_choice("?", COMMIT_TYPES, "chore", "x")
