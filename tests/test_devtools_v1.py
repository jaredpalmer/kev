"""The devtools-v1 builder's label mappings and selection rules (scripts/build_devtools_v1.py) on small synthetic inputs.
No weights, no network. Run: uv run python -m pytest tests/test_devtools_v1.py -q
"""
import os, random
from collections import Counter
from pathlib import Path

import pytest

from kev.data import materialize
from kev.suite import read_jsonl, text_digest
from scripts.build_devtools_v1 import (COMMIT_TYPES, Components, assign_match, balanced_pairs, check_invariants, choose, codereviewer_candidates,
                                       codereviewer_state, commit_type, deal_groups, is_balanced, line_safe, message_negatives, q_choice, q_noul,
                                       round_robin, state_key)

SUITE = Path(__file__).resolve().parents[1] / "evals" / "devtools-v1"


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
    return {"state": key, "questions": {"q": q_noul("Is it?", label, src)}, "_meta": {"id": f"x/{key}", "group_id": split_group, "text_sha256": text_digest(key)}}


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


def codereviewer_rows():
    """Four rows as the dataset has them: its `id` field (7) is shared by rows of different projects and files."""
    where = [("cls-test.jsonl", 1, "a-x"), ("cls-test.jsonl", 2, "b-y"), ("cls-valid.jsonl", 1, "a-x"), ("cls-valid.jsonl", 2, "b-y")]
    return [(member, line, {"id": 7, "proj": proj, "patch": f"@@ -1 +1 @@\n-v{i}\n+w{i}", "oldf": "", "y": i % 2, "lang": "go"})
            for i, (member, line, proj) in enumerate(where)]


def test_a_new_build_gives_codereviewer_records_unique_ids():
    licences = {"codereviewer": {"a-x": {"repo": "a/x", "spdx": "MIT"}, "b-y": {"repo": "b/y", "spdx": "Apache-2.0"}}}
    fresh = codereviewer_candidates(codereviewer_rows(), licences, {}, v1=False)
    assert [x["_meta"]["id"] for x in fresh] == ["codereviewer/cls-test/L1", "codereviewer/cls-test/L2", "codereviewer/cls-valid/L1", "codereviewer/cls-valid/L2"]
    check_invariants({"train": fresh})
    legacy = codereviewer_candidates(codereviewer_rows(), licences, {}, v1=True)   # devtools-v1's ids
    assert Counter(x["_meta"]["id"] for x in legacy) == {"codereviewer/cls-test/7": 2, "codereviewer/cls-valid/7": 2}
    with pytest.raises(AssertionError, match="duplicate id"):
        check_invariants({"train": legacy})
    check_invariants({"train": legacy}, unique_ids=False)


def test_devtools_v1_shares_exactly_the_documented_ids_inside_its_evaluation_partitions():
    """The frozen defect paired comparisons must drop (AGENTS.md, scripts/build_devtools_v1.py): one id per partition."""
    for split, expected in (("development", {"codereviewer/cls-test/13657"}), ("test", {"codereviewer/cls-test/19245"})):
        ids = Counter(r["_meta"]["id"] for r in read_jsonl(SUITE / f"{split}.jsonl"))
        assert {i for i, n in ids.items() if n > 1} == expected


def test_a_new_build_keys_codereviewer_records_on_the_whole_state():
    """devtools-v1 hashed only CodeReviewer's diff, so the same hunk under different context lines collided."""
    hunk = "@@ -3,1 +3,1 @@\n-a\n+b"
    licences = {"codereviewer": {"a-x": {"repo": "a/x", "spdx": "MIT"}}}
    rows = [("cls-test.jsonl", i + 1, {"id": i, "proj": "a-x", "patch": hunk, "oldf": oldf, "y": 1, "lang": "go"}) for i, oldf in enumerate(["x = 1\ny = 2", "p = 1\nq = 2"])]
    new = codereviewer_candidates(rows, licences, {}, v1=False)
    assert new[0]["state"] != new[1]["state"]
    assert [x["_meta"]["text_sha256"] for x in new] == [state_key(x["state"]) for x in new] and new[0]["_meta"]["text_sha256"] != new[1]["_meta"]["text_sha256"]
    assert {x["_meta"]["text_sha256"] for x in codereviewer_candidates(rows, licences, {}, v1=True)} == {text_digest(hunk)}
    assert state_key({"b": 1, "a": "X  y"}) == text_digest('{"a": "X  y", "b": 1}')


def commits():
    """Eight commitpackft-shaped candidates in one language and eight repositories; one subject holds a U+2028."""
    out = []
    for i in range(8):
        subject = f"Fix bug {i}" + ("\u2028more" if i == 3 else "")
        out.append({"state": f"diff --git a/f{i} b/f{i}\n+x{i}", "questions": {"change_type": q_choice("What kind of change is this?", COMMIT_TYPES, "fix", "commitpackft_type")},
                    "_label": "fix", "_subject": subject, "_lang": "go",
                    "_meta": {"id": f"commitpackft/go/{i}", "source": "commitpackft", "group_id": f"commitpackft/r{i}", "text_sha256": text_digest(f"d{i}")}})
    return out


def test_a_new_build_admits_commits_on_their_final_record():
    """The message question exists only after selection; new builds check the final record and redo the selection without
    the records that fail, keeping the message labels exactly balanced. devtools-v1 checked before the question was added."""
    everyone = lambda counts: (lambda x: True)
    v1, _, v1_dropped = choose("commitpackft", commits(), 8, everyone, "seed")
    assert v1_dropped == 0 and not all(line_safe(x) for x in v1)          # this seed shows the U+2028 message under v1
    pool = commits()
    new, _, dropped = choose("commitpackft", pool, 8, everyone, "seed", final_ok=line_safe)
    assert dropped >= 1 and all(line_safe(x) for x in new)
    labels = [x["questions"]["message_match"]["label"] for x in new if "message_match" in x["questions"]]
    assert len(labels) >= 6 and sum(labels) == len(labels) // 2
    assert all("message_match" not in x["questions"] and "shown_message" not in x["_meta"] for x in pool)   # a redo starts from clean candidates
    assert choose("commitpackft", commits(), 8, everyone, "seed", final_ok=line_safe)[0] == new


POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="the download-dir checks are POSIX lstat/mode rules")


@POSIX_ONLY
def test_private_dir_accepts_owned_dirs_and_creates_missing(tmp_path):
    """A missing directory is created 0700; an existing user-owned directory (even 0755, as older caches were made)
    passes; a symlink to one does not."""
    import stat
    from scripts.build_devtools_v1 import private_dir
    new = private_dir(tmp_path / "fresh" / "nested")
    assert new.is_dir() and stat.S_IMODE(new.lstat().st_mode) == 0o700
    owned = tmp_path / "owned"; owned.mkdir(mode=0o755)
    assert private_dir(owned) == owned
    link = tmp_path / "link"; link.symlink_to(owned, target_is_directory=True)
    with pytest.raises(ValueError, match="not a private download directory"):
        private_dir(link)


@POSIX_ONLY
def test_fetch_downloads_through_a_private_temp_file(tmp_path):
    """fetch writes to a securely created .<name>.XXXXXX.part inside the private directory and os.replace()s it over the
    target: a planted <name>.part symlink is left alone where the old code's curl -o wrote through it onto the victim."""
    import hashlib
    from scripts.build_devtools_v1 import fetch
    src = tmp_path / "src"; src.write_bytes(b"payload")
    raw = tmp_path / "raw"; raw.mkdir(mode=0o700)
    victim = tmp_path / "victim"; victim.write_text("keep me", encoding="utf-8")
    (raw / "data.bin.part").symlink_to(victim)
    target = fetch(src.as_uri(), raw / "data.bin", sha256=hashlib.sha256(b"payload").hexdigest())
    assert target.read_bytes() == b"payload"
    assert victim.read_text(encoding="utf-8") == "keep me" and (raw / "data.bin.part").is_symlink()
    assert sorted(p.name for p in raw.iterdir()) == ["data.bin", "data.bin.part"]   # no temp files remain


@POSIX_ONLY
def test_fetch_refuses_a_writable_directory(tmp_path):
    """A group/world-writable download directory is refused before anything is fetched (that is where the planted
    .part symlink attack lived)."""
    from scripts.build_devtools_v1 import fetch
    src = tmp_path / "src"; src.write_bytes(b"payload")
    shared = tmp_path / "shared"; shared.mkdir(); shared.chmod(0o777)
    with pytest.raises(ValueError, match="not a private download directory"):
        fetch(src.as_uri(), shared / "data.bin")
    assert not list(shared.iterdir())


@POSIX_ONLY
def test_fetch_a_failed_download_leaves_nothing(tmp_path):
    """curl failing leaves neither the temp file nor the target behind."""
    import subprocess
    from scripts.build_devtools_v1 import fetch
    raw = tmp_path / "raw"; raw.mkdir(mode=0o700)
    with pytest.raises(subprocess.CalledProcessError):
        fetch((tmp_path / "missing").as_uri(), raw / "gone.bin")
    assert not (raw / "gone.bin").exists() and not list(raw.iterdir())
