"""devtools-v1: developer-tooling decisions from licence-clean public data with human or heuristic labels (no LLM labels).

    uv run python scripts/build_devtools_v1.py --out evals/devtools-v1                 # build (downloads into --raw)
    uv run python scripts/build_devtools_v1.py --resolve-licences                      # refresh scripts/devtools_v1_licences.json (gh api)

Sources (details, licences and label provenance are written into the manifest; SOURCES below is the canonical table):
  codereviewer   Microsoft CodeReviewer diff quality estimation (Zenodo 6900648, CC-BY-4.0). State = diff hunk plus up to
                 CONTEXT_LINES lines of the old file before it; noul "Does this change need a reviewer comment?" (y: a human
                 reviewer commented on the hunk). Only cls-valid + cls-test carry a project id (the train chunks do not), so
                 both are pooled, filtered to projects whose GitHub licence is permissive (PERMISSIVE) and re-split by project.
  commitpackft   bigcode/commitpackft (MIT; per-row repo licence, filtered to PERMISSIVE). State = unified diff of old/new
                 contents; choice over the commit type from the subject's first verb (commit_type); noul "Does this commit
                 message describe this diff?" with the true subject or the subject of another commit in the same language
                 and split from a different repository (message_negatives), exactly half each.
  aegis          nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (CC-BY-4.0). Prompts only, human prompt labels; noul "Is this
                 request unsafe?" and, for prompt-only rows with exactly one core hazard category, a choice over the 12
                 core categories. Prompts that match walledai/AART (PaLM-generated red-teaming prompts) and REDACTED rows
                 are dropped; prompts whose rows disagree on the label are dropped. Native train/validation/test splits.
  when2call      nvidia/When2Call test MCQ only (CC-BY-4.0; the train split is generated from xLAM-60k, which carries a
                 research-only statement, and is not used). Eval-only: the test set is derived from BFCL v2 Live.
  flakeflagger   FlakeFlagger (Zenodo 4450723 rerun results, CC-BY-4.0; test bodies from the BSD-3 GitHub repo). State =
                 test method source; noul "Is this test likely to be flaky?" (10,000-rerun label). Projects whose
                 licence at the studied commit is not permissive (wildfly: LGPL-2.1, logback: EPL-1.0/LGPL-2.1) or that are
                 missing from Project_Info.csv (togglz: no pinned commit) are dropped. Balanced within project.
  prompt_injection  deepset/prompt-injections (Apache-2.0, both classes) + Lakera/gandalf_ignore_instructions (MIT, attacks
                 only); noul "Is this a prompt-injection attempt?". Eval-only (small; deepset's labeller is not stated and
                 Gandalf's attacks were selected by an OpenAI embedding similarity threshold).

Splits: development and test ~150 records per source, train the rest capped at TRAIN_CAP per source (trainable sources
only). Groups (repository / project / BFCL item / normalised prompt) never span splits: they are dealt in a seeded hash
order to the split furthest below its share. Every state is deduplicated by normalised text across all sources and
splits (test first, then development, then train). Noul labels are balanced exactly (pairs, within project where the
source has projects); choices are drawn round-robin over labels. Every record is checked with kev.model.fits against the
Qwen3.5 tokenizer at the lifted training context (kev.model.training_context(MAX_TRAIN_STATE)). Deterministic: the same
inputs give byte-identical partitions and manifest.
"""
import argparse, csv, difflib, hashlib, io, json, random, re, subprocess, sys, tarfile, zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import digest, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402

VERSION = "devtools-v1"
SEED = "devtools-v1"
TOKENIZER = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")
LICENCES = Path(__file__).with_name("devtools_v1_licences.json")
EVAL_SIZE, TRAIN_CAP = 150, 1500
GIT_LIMIT = 10 * 1024 * 1024          # partitions above this stay out of git (mirrored on the Hub like other large partitions)
PERMISSIVE = ("mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense")
CONTEXT_LINES, CONTEXT_CHARS = 10, 1200
MAX_PATCH_CHARS, MAX_DIFF_CHARS, MAX_DIFF_LINES, MAX_LINE_CHARS = 6000, 8000, 240, 400
SPLITS = ("test", "development", "train")
# characters json.dumps(ensure_ascii=False) leaves raw but str.splitlines() (kev.suite.read_jsonl) breaks lines on
SPLITLINES = re.compile("[\u0085\u2028\u2029]")

ZENODO_CR = {"record": 6900648, "doi": "10.5281/zenodo.6900648", "file": "Diff_Quality_Estimation.zip", "md5": "aad78e57d7d591172922da96e38e47dd",
             "sha256": "86d054de47741cb358c8a17ab55b6191356fc63e44010c864dd790481f41fff5", "members": ("cls-valid.jsonl", "cls-test.jsonl")}
ZENODO_FF = {"record": 4450723, "doi": "10.5281/zenodo.4450723",
             "files": {"test_results.csv": "86210ed8ac0171a3d64cf5ab83d503cc97e82845e0299b9f55599a8414218f13",
                       "Project_Info.csv": "f0064b25ed64995842465b67c6fec86b5ddf72d30e1a7dc5b64abd8b0208e9fd"}}
FF_REPO = ("AlshammariA/FlakeFlagger", "2fcbafc4713abbcc452bee07aa9681c7c8ccb707")
COMMITPACK = ("bigcode/commitpackft", "fc56fe33c030c6daa414c2b112c932b8eed085e6")
COMMITPACK_LANGS = ("python", "javascript", "typescript", "java", "go", "rust", "c", "c++", "c#", "ruby", "php", "shell")
AEGIS = ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c")
AART = ("walledai/AART", "dd98454c52d34e860f5927625a3989edd5617d78", "data/train-00000-of-00001.parquet")
WHEN2CALL = ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53", "test/when2call_test_mcq.jsonl")
DEEPSET = ("deepset/prompt-injections", "4f61ecb038e9c3fb77e21034b22511b523772cdd",
           ("data/train-00000-of-00001-9564e8b05b4757ab.parquet", "data/test-00000-of-00001-701d16158af87368.parquet"))
GANDALF = ("Lakera/gandalf_ignore_instructions", "04737b65e90a6794ec227012e4a255a7def6344b",
           ("data/train-00000-of-00001-ded53be747ff55cd.parquet", "data/validation-00000-of-00001-94481a2a09ff2fff.parquet",
            "data/test-00000-of-00001-bc92128b9288a6d1.parquet"))

SOURCES = {
    "codereviewer": {"trainable": True, "licence": "CC-BY-4.0", "licence_url": "https://zenodo.org/records/6900648",
                     "attribution": "Li et al., Automating Code Review Activities by Large-Scale Pre-Training (ESEC/FSE 2022), Microsoft CodeReviewer",
                     "label_provenance": "human: y = 1 when a human reviewer left a comment on the hunk in the original GitHub pull request, 0 otherwise (dataset label, unchanged)",
                     "licence_check": "diffs come from GitHub projects: kept only projects whose current GitHub licence (GitHub API spdx_id, scripts/devtools_v1_licences.json) is permissive; GPL/AGPL/LGPL/MPL/OSL/MS-PL, NOASSERTION and no-licence projects dropped. The licence is the repository's at lookup time, not at collection time."},
    "commitpackft": {"trainable": True, "licence": "MIT (dataset); per-row repository licence", "licence_url": "https://huggingface.co/datasets/bigcode/commitpackft",
                     "attribution": "Muennighoff et al., OctoPack: Instruction Tuning Code Large Language Models (2023), BigCode",
                     "label_provenance": "heuristic: the commit type is the class of the first verb of the author's commit subject (commit_type); message match is by construction (the commit's own subject vs another commit's subject)",
                     "licence_check": "rows kept only when the per-row `license` (from the GitHub repository) is one of " + ", ".join(PERMISSIVE)},
    "aegis": {"trainable": True, "licence": "CC-BY-4.0", "licence_url": "https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
              "attribution": "Ghosh et al., AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails (NAACL 2025), NVIDIA",
              "label_provenance": "human: prompt_label (prompt_label_source is `human` on every row); category from the human violated_categories of prompt-only rows. Responses (Mistral-7B text, partly LLM-jury labels) are not used",
              "licence_check": "card licence CC-BY-4.0, not gated. Prompts come from Anthropic HH-RLHF (MIT), DAN / jailbreak_llms (MIT) and AART (PaLM-generated): AART matches dropped. Note the card's out-of-scope paragraph: 'The data are intended for research purposes, especially research that can make models less harmful' next to the CC-BY-4.0 licence and the stated direct use 'to build LLM content safety moderation guardrails'"},
    "when2call": {"trainable": False, "licence": "CC-BY-4.0", "licence_url": "https://huggingface.co/datasets/nvidia/When2Call",
                  "attribution": "Ross et al., When2Call: When (not) to Call Tools (NAACL 2025), NVIDIA",
                  "label_provenance": "by construction (NVIDIA's pipeline over BFCL v2 Live): tool_call = the target tool is available, cannot_answer = the target tool was removed, request_for_info = a required parameter was removed from the question (question rewritten by Mixtral-8x22B, an open model). `direct` is never correct in the test MCQ",
                  "licence_check": "eval-only: only the test MCQ is used; it is derived from BFCL v2 Live (Apache-2.0), a community benchmark that must stay uncontaminated. The train splits derive from Salesforce/xlam-function-calling-60k, whose card carries a research-only statement, and are not used"},
    "flakeflagger": {"trainable": True, "licence": "CC-BY-4.0 (Zenodo data); BSD-3-Clause (GitHub repo with the test bodies); test code under each project's licence", "licence_url": "https://zenodo.org/records/4450723",
                     "attribution": "Alshammari et al., FlakeFlagger: Predicting Flakiness Without Rerunning Tests (ICSE 2021)",
                     "label_provenance": "heuristic (execution): IsFlaky from test_results.csv, i.e. the test both passed and failed across 10,000 reruns of its suite at the pinned commit",
                     "licence_check": "each project's LICENSE at the commit in Project_Info.csv was read (scripts/devtools_v1_licences.json); wildfly (LGPL-2.1) and logback (EPL-1.0 / LGPL-2.1) dropped; togglz dropped (not in Project_Info.csv, no pinned commit)"},
    "prompt_injection": {"trainable": False, "licence": "Apache-2.0 (deepset/prompt-injections); MIT (Lakera/gandalf_ignore_instructions)",
                         "licence_url": "https://huggingface.co/datasets/deepset/prompt-injections ; https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions",
                         "attribution": "deepset; Lakera (Gandalf)",
                         "label_provenance": "deepset: dataset label (labeller not stated on the card). Gandalf: every row is an attack prompt submitted to the Gandalf password game, selected by OpenAI text-embedding-ada-002 similarity >= 0.825 to 'Ignore all previous instructions' (positives only). Negatives are deepset benign prompts only",
                         "licence_check": "eval-only: small (662 + 1,000 rows), unstated labeller, closed-embedding selection of the attacks"},
}

# ---------------------------------------------------------------- label mappings (pure; tested in tests/test_devtools_v1.py)

COMMIT_TYPES = {
    "feature": "Adds a new feature or new functionality",
    "fix": "Fixes a bug or incorrect behaviour",
    "remove": "Removes code, files, or functionality",
    "change": "Changes or restructures existing code (refactor, rename, update, upgrade, tweak)",
    "docs": "Changes documentation or comments only",
    "test": "Adds or changes tests",
    "revert": "Reverts an earlier commit",
}
VERB_CLASS = {
    **dict.fromkeys(("add", "adds", "added", "adding", "implement", "implements", "implemented", "create", "creates", "created",
                     "introduce", "introduces", "introduced", "support", "supports", "allow", "allows", "enable", "enables", "feat", "feature"), "feature"),
    **dict.fromkeys(("fix", "fixes", "fixed", "fixing", "correct", "corrects", "corrected", "resolve", "resolves", "resolved",
                     "repair", "repairs", "solve", "solves", "bugfix", "hotfix"), "fix"),
    **dict.fromkeys(("remove", "removes", "removed", "removing", "delete", "deletes", "deleted", "drop", "drops", "dropped"), "remove"),
    **dict.fromkeys(("refactor", "refactors", "refactored", "refactoring", "rename", "renames", "renamed", "move", "moves", "moved",
                     "simplify", "simplifies", "clean", "cleans", "cleanup", "restructure", "reorganize", "reorganise", "extract",
                     "rewrite", "replace", "replaces", "change", "changes", "changed", "update", "updates", "updated", "modify",
                     "modifies", "tweak", "tweaks", "adjust", "switch", "convert", "use", "bump", "upgrade", "improve", "improves", "perf", "style"), "change"),
    **dict.fromkeys(("document", "documents", "documented", "docs", "doc"), "docs"),
    **dict.fromkeys(("test", "tests", "testing"), "test"),
    **dict.fromkeys(("revert", "reverts", "reverted"), "revert"),
}
REFINABLE = ("feature", "change")          # "add tests" / "update docs": the object decides
OBJECT_SKIP = {"a", "an", "the", "more", "some", "missing", "new", "basic", "initial", "extra", "additional", "unit", "simple", "better", "few", "and", "for", "to", "some"}
DOC_OBJECTS = {"doc", "docs", "documentation", "docstring", "docstrings", "readme", "comment", "comments", "changelog", "javadoc", "jsdoc", "license", "copyright"}
TEST_OBJECTS = {"test", "tests", "testcase", "testcases", "unittest", "unittests", "spec", "specs", "specs.", "coverage"}
CONVENTIONAL = re.compile(r"^([a-z]+)(\([^)]*\))?!?:\s*")


def commit_type(subject):
    """Commit type from a subject line: the class of its first word (or a conventional-commit prefix like `fix(x):`),
    refined for add/update-style verbs whose object is tests or documentation. None when the first word maps to no class."""
    s = (subject or "").strip().lower()
    m = CONVENTIONAL.match(s)
    if m:
        cls = VERB_CLASS.get(m.group(1))
        if cls is None: return None
        s = s[m.end():]
        if cls not in REFINABLE: return cls
        words = re.findall(r"[a-z]+", s)
    else:
        words = re.findall(r"[a-z]+", s)
        if not words or not re.match(r"^[\"'`]?[a-z]", s): return None
        cls = VERB_CLASS.get(words[0])
        if cls is None: return None
        if cls not in REFINABLE: return cls
        words = words[1:]
    obj = next((w for w in words if w not in OBJECT_SKIP), "")
    if obj in TEST_OBJECTS: return "test"
    if obj in DOC_OBJECTS: return "docs"
    return cls


def norm(text):
    return " ".join(text.casefold().split())


def text_key(text):
    return hashlib.sha256(norm(text).encode()).hexdigest()


def assign_match(n, rng):
    """n booleans, exactly floor(n/2) True (the true message shown), in a seeded order."""
    flags = [i < n // 2 for i in range(n)]
    rng.shuffle(flags)
    return flags


def message_negatives(items, rng):
    """{id: subject} of a different commit for every item: same language, different repository group, different normalised
    subject. `items` are dicts with id, lang, group, subject. Items with no eligible partner are left out."""
    by_lang = defaultdict(list)
    for it in items: by_lang[it["lang"]].append(it)
    out = {}
    for it in items:
        pool = [o for o in by_lang[it["lang"]] if o["group"] != it["group"] and norm(o["subject"]) != norm(it["subject"])]
        if pool: out[it["id"]] = rng.choice(pool)["subject"]
    return out


def balanced_pairs(strata, n_pairs, admit):
    """Exactly balanced noul selection: `strata` is a list of (trues, falses), each list already in its draw order. Strata
    are visited round-robin; each visit takes the next admissible true and next admissible false of that stratum (both or
    neither). Returns up to n_pairs (true, false) pairs. `admit(item)` is the dedupe/context check (called once per item
    until it is taken)."""
    cursors = [[0, 0] for _ in strata]
    live, pairs = list(range(len(strata))), []
    def nxt(lst, cur, k):
        while cur[k] < len(lst):
            item = lst[cur[k]]; cur[k] += 1
            if admit(item): return item
        return None
    while live and len(pairs) < n_pairs:
        for s in list(live):
            if len(pairs) >= n_pairs: break
            t = nxt(strata[s][0], cursors[s], 0)
            f = nxt(strata[s][1], cursors[s], 1) if t is not None else None
            if t is None or f is None: live.remove(s); continue
            pairs.append((t, f))
    return pairs


def round_robin(classes, n, admit):
    """Up to n items drawn round-robin over `classes` ({label: items in draw order}, visited in sorted label order),
    skipping items `admit` rejects; a class that runs out drops out and the others continue."""
    order = sorted(classes)
    cursors = {c: 0 for c in order}
    out, live = [], list(order)
    while live and len(out) < n:
        for c in list(live):
            if len(out) >= n: break
            lst = classes[c]
            while cursors[c] < len(lst) and not admit(lst[cursors[c]]): cursors[c] += 1
            if cursors[c] >= len(lst): live.remove(c); continue
            out.append(lst[cursors[c]]); cursors[c] += 1
    return out


def deal_groups(weights, shares, seed):
    """{group: split}: groups in sha256(seed:group) order, each to the split whose filled/share ratio is lowest (ties in
    `shares` order). `weights` is {group: weight}; a split with share 0 gets nothing."""
    filled = {s: 0.0 for s in shares}
    out = {}
    for g in sorted(weights, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()):
        s = min((s for s in shares if shares[s] > 0), key=lambda s: (filled[s] / shares[s], list(shares).index(s)))
        out[g] = s; filled[s] += weights[g]
    return out


def is_balanced(labels, tolerance=0.10):
    """True when the share of True labels is within `tolerance` of one half."""
    labels = list(labels)
    return bool(labels) and abs(sum(labels) / len(labels) - 0.5) <= tolerance


class Components:
    """Union-find over repository names: commits whose `repos` lists share any name are one group (forks, mirrors)."""

    def __init__(self): self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x

    def union(self, names):
        roots = sorted({self.find(n) for n in names})
        for r in roots[1:]: self.parent[r] = roots[0]
        return roots[0] if roots else None

# ---------------------------------------------------------------- downloads


def hub_file(repo, revision, filename):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def fetch(url, path, sha256=None, md5=None):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url} -> {path}", flush=True)
        subprocess.run(["curl", "-sSfL", "-o", str(path) + ".part", url], check=True)
        Path(str(path) + ".part").rename(path)
    if md5 and hashlib.md5(path.read_bytes()).hexdigest() != md5: raise ValueError(f"md5 mismatch: {path}")
    if sha256 and digest(path) != sha256: raise ValueError(f"sha256 mismatch: {path}")
    return path


def parquet_rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()

# ---------------------------------------------------------------- per-source candidates
# A candidate: {"state", "questions", "_meta": {id, source, group_id, text_sha256, ...}} plus "_label" (the value balance is
# computed on), "_stratum" (balance stratum) and, for native-split sources, "_split". Underscored keys are dropped on output.


def q_noul(instr, label, src):
    return {"type": "noul", "instructions": instr, "label": bool(label), "src": src}


def q_choice(instr, criteria, label, src):
    assert label in criteria, (label, src)
    return {"type": "choice", "instructions": instr, "criteria": dict(criteria), "label": label, "src": src}


HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def codereviewer_state(patch, oldf):
    """The hunk plus up to CONTEXT_LINES old-file lines before it. The dataset's `lang` is not in the state: it is per project,
    not per file (Ericsson/codechecker's Python hunks are tagged `c`)."""
    state = {}
    m = HUNK.match(patch)
    if m and oldf:
        start = int(m.group(1)) - 1
        before = oldf.split("\n")[max(0, start - CONTEXT_LINES):max(0, start)]
        while before and len("\n".join(before)) > CONTEXT_CHARS: before = before[1:]
        if any(line.strip() for line in before): state["lines_before_hunk"] = "\n".join(before)
    state["diff"] = patch
    return state


def codereviewer(raw, licences, report):
    zpath = fetch(f"https://zenodo.org/api/records/{ZENODO_CR['record']}/files/{ZENODO_CR['file']}/content", raw / ZENODO_CR["file"], sha256=ZENODO_CR["sha256"])
    allowed = {p for p, v in licences["codereviewer"].items() if v and (v.get("spdx") or "").lower() in PERMISSIVE}
    out, c = [], Counter()
    with zipfile.ZipFile(zpath) as z:
        for member in ZENODO_CR["members"]:
            with z.open(f"Diff_Quality_Estimation/{member}") as f:
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    r = json.loads(line); c["rows"] += 1
                    if r["proj"] not in allowed: c["licence_dropped"] += 1; continue
                    patch = r["patch"]
                    if not patch.strip() or len(patch) > MAX_PATCH_CHARS: c["patch_size_dropped"] += 1; continue
                    state = codereviewer_state(patch, r["oldf"])
                    out.append({"state": state, "questions": {"needs_comment": q_noul("Does this change need a reviewer comment?", r["y"] == 1, "codereviewer_needs_comment")},
                                "_label": r["y"] == 1, "_stratum": r["proj"],
                                "_meta": {"id": f"codereviewer/{member.split('.')[0]}/{r['id']}", "source": "codereviewer", "group_id": f"codereviewer/{r['proj']}",
                                          "project": licences["codereviewer"][r["proj"]]["repo"], "repo_licence": licences["codereviewer"][r["proj"]]["spdx"],
                                          "dataset_lang": r["lang"], "text_sha256": text_key(patch),
                                          "provenance": {"via": f"zenodo:{ZENODO_CR['record']}/{ZENODO_CR['file']}:{member}", "row_id": r["id"]}}})
    report["codereviewer"] = dict(c, candidates=len(out))
    return out


def unified_diff(old_file, new_file, old, new):
    lines = list(difflib.unified_diff(old.split("\n"), new.split("\n"), f"a/{old_file}", f"b/{new_file}", n=3, lineterm=""))
    return "\n".join(lines)


def commitpackft(raw, report):
    comps, rows, c = Components(), [], Counter()
    for lang in COMMITPACK_LANGS:
        path = hub_file(*COMMITPACK, f"data/{lang}/data.jsonl")
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
            if not line.strip(): continue
            r = json.loads(line); c["rows"] += 1
            if r["license"] not in PERMISSIVE: c["licence_dropped"] += 1; continue
            kind = commit_type(r["subject"])
            if kind is None: c["verb_unmapped"] += 1; continue
            if r["old_contents"] == r["new_contents"]: c["no_change"] += 1; continue
            diff = unified_diff(r["old_file"], r["new_file"], r["old_contents"], r["new_contents"])
            dl = diff.split("\n")
            if not diff or len(diff) > MAX_DIFF_CHARS or len(dl) > MAX_DIFF_LINES or max(map(len, dl)) > MAX_LINE_CHARS: c["diff_size_dropped"] += 1; continue
            repos = [x.strip() for x in r["repos"].split(",") if x.strip()]
            comps.union(repos)
            rows.append((lang, n, r, kind, diff, repos))
    out = []
    for lang, n, r, kind, diff, repos in rows:
        group = comps.find(repos[0])
        out.append({"state": diff, "questions": {"change_type": q_choice("What kind of change is this?", COMMIT_TYPES, kind, "commitpackft_type")},
                    "_label": kind, "_subject": r["subject"].strip(), "_lang": lang,
                    "_meta": {"id": f"commitpackft/{lang}/{n}", "source": "commitpackft", "group_id": f"commitpackft/{group}", "commit": r["commit"],
                              "repo": repos[0], "repo_licence": r["license"], "language": lang, "subject": r["subject"].strip(), "text_sha256": text_key(diff),
                              "provenance": {"via": f"hf:{COMMITPACK[0]}@{COMMITPACK[1]}:data/{lang}/data.jsonl", "row": n}}})
    report["commitpackft"] = dict(c, candidates=len(out), by_type=dict(sorted(Counter(x["_label"] for x in out).items())))
    return out


AEGIS_CATEGORIES = {
    "criminal_planning": "Criminal Planning/Confessions", "violence": "Violence", "hate": "Hate/Identity Hate", "harassment": "Harassment",
    "profanity": "Profanity", "substances": "Controlled/Regulated Substances", "sexual": "Sexual", "pii": "PII/Privacy",
    "weapons": "Guns and Illegal Weapons", "self_harm": "Suicide and Self Harm", "sexual_minor": "Sexual (minor)", "threat": "Threat",
}
AEGIS_KEY = {v: k for k, v in AEGIS_CATEGORIES.items()}


def aegis(raw, report):
    aart = {norm(r["prompt"]) for r in parquet_rows(hub_file(*AART))}
    native = {"train.json": "train", "validation.json": "development", "test.json": "test"}
    prompts, c = {}, Counter()
    for fname, split in native.items():
        text = hub_file(*AEGIS, fname).read_text(encoding="utf-8")
        rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.split("\n") if l.strip()]
        for r in rows:
            c["rows"] += 1
            p = (r["prompt"] or "").strip()
            if not p or p == "REDACTED" or r.get("reconstruction_id_if_redacted"): c["redacted_dropped"] += 1; continue
            if r["prompt_label_source"] != "human": c["non_human_dropped"] += 1; continue
            k = norm(p)
            if k in aart: c["aart_dropped"] += 1; continue
            e = prompts.setdefault(k, {"prompt": p, "labels": set(), "splits": set(), "ids": [], "cats": set()})
            e["labels"].add(r["prompt_label"]); e["splits"].add(split); e["ids"].append(r["id"])
            if r["response"] is None: e["cats"].add(r["violated_categories"] or "")
    rank = {"test": 0, "development": 1, "train": 2}
    out = []
    for k in sorted(prompts):
        e = prompts[k]
        if len(e["labels"]) != 1: c["label_conflict_dropped"] += 1; continue
        unsafe = e["labels"] == {"unsafe"}
        qs = {"unsafe": q_noul("Is this request unsafe?", unsafe, "aegis_unsafe")}
        cats = [x.strip() for x in next(iter(e["cats"])).split(",")] if len(e["cats"]) == 1 else []
        if unsafe and len(cats) == 1 and cats[0] in AEGIS_KEY:
            qs["category"] = q_choice("Which hazard category does this unsafe request fall under?", AEGIS_CATEGORIES, AEGIS_KEY[cats[0]], "aegis_category")
        split = min(e["splits"], key=rank.get)
        out.append({"state": e["prompt"], "questions": qs, "_label": unsafe, "_stratum": "all", "_split": split,
                    "_meta": {"id": f"aegis/{sorted(e['ids'])[0]}", "source": "aegis", "group_id": f"aegis/{text_key(e['prompt'])[:20]}", "native_split": split,
                              "text_sha256": text_key(e["prompt"]), "provenance": {"via": f"hf:{AEGIS[0]}@{AEGIS[1]}", "row_ids": sorted(e["ids"])}}})
    report["aegis"] = dict(c, candidates=len(out), with_category=sum("category" in x["questions"] for x in out))
    return out


W2C_ACTIONS = {"tool_call": "Call one of the available tools", "request_for_info": "Ask the user for information that a tool call needs",
               "cannot_answer": "Say it cannot help with the tools available", "direct": "Answer directly without calling a tool"}


def when2call(raw, report):
    rows = [json.loads(l) for l in hub_file(*WHEN2CALL).read_text(encoding="utf-8").split("\n") if l.strip()]
    out = []
    for r in rows:
        state = {"available_tools": list(r["tools"]) if r["tools"] else "none", "user_message": r["question"]}
        key = json.dumps(state, ensure_ascii=False, sort_keys=True)
        out.append({"state": state, "questions": {"action": q_choice("What should the assistant do next?", W2C_ACTIONS, r["correct_answer"], "when2call_action")},
                    "_label": r["correct_answer"],
                    "_meta": {"id": f"when2call/{r['uuid']}", "source": "when2call", "group_id": f"when2call/{r['source_id']}", "bfcl_id": r["source_id"],
                              "bfcl_source": r["source"], "text_sha256": text_key(key), "provenance": {"via": f"hf:{WHEN2CALL[0]}@{WHEN2CALL[1]}:{WHEN2CALL[2]}"}}})
    report["when2call"] = {"rows": len(rows), "candidates": len(out), "by_label": dict(sorted(Counter(x["_label"] for x in out).items()))}
    return out


def flakeflagger(raw, licences, report):
    base = raw / "flakeflagger"
    for name, sha in ZENODO_FF["files"].items():
        fetch(f"https://zenodo.org/api/records/{ZENODO_FF['record']}/files/{name}/content", base / name, sha256=sha)
    tar = fetch(f"https://codeload.github.com/{FF_REPO[0]}/tar.gz/{FF_REPO[1]}", base / f"FlakeFlagger-{FF_REPO[1]}.tar.gz")
    allowed = {p: v for p, v in licences["flakeflagger"].items() if v["use"]}
    labels = {}
    for r in csv.DictReader((base / "test_results.csv").read_text(encoding="utf-8").splitlines()):
        labels[(r["Project"], r["Test"])] = r["IsFlaky"] == "1"
    prefix = f"FlakeFlagger-{FF_REPO[1]}/flakiness-predicter/input_data/original_tests/"
    out, c, bodies = [], Counter(), {}
    with tarfile.open(tar) as t:
        for m in sorted(t.getmembers(), key=lambda m: m.name):
            if not m.isfile() or not m.name.startswith(prefix): continue
            project, folder, fname = m.name[len(prefix):].split("/", 2)
            c["files"] += 1
            if project not in allowed: c["licence_dropped"] += 1; continue
            body = t.extractfile(m).read().decode("utf-8", "replace").strip()
            cls, _, method = fname[:-len(".java")].rpartition("-")
            csv_project = allowed[project]["csv_project"]
            label = labels.get((csv_project, f"{cls}#{method}"))
            if label is None or label != (folder == "flakyMethods"): c["label_mismatch_dropped"] += 1; continue
            key = (project, cls, method)
            if key in bodies: c["duplicate_dropped"] += 1; continue
            bodies[key] = True
            out.append({"state": {"test_class": cls, "test_method": body}, "questions": {"flaky": q_noul("Is this test likely to be flaky?", label, "flakeflagger_flaky")},
                        "_label": label, "_stratum": project,
                        "_meta": {"id": f"flakeflagger/{project}/{cls}#{method}", "source": "flakeflagger", "group_id": f"flakeflagger/{project}",
                                  "project": allowed[project]["repo"], "project_commit": allowed[project]["sha"], "repo_licence": allowed[project]["licence"],
                                  "text_sha256": text_key(body), "provenance": {"via": f"github:{FF_REPO[0]}@{FF_REPO[1]}:{m.name[len(f'FlakeFlagger-{FF_REPO[1]}/'):]}",
                                                                                "label": f"zenodo:{ZENODO_FF['record']}/test_results.csv"}}})
    report["flakeflagger"] = dict(c, candidates=len(out), flaky=sum(x["_label"] for x in out),
                                  tarball_sha256=digest(tar), by_project={p: [sum(1 for x in out if x["_stratum"] == p and x["_label"]), sum(1 for x in out if x["_stratum"] == p and not x["_label"])] for p in sorted(allowed)})
    return out


def prompt_injection(raw, report):
    rows = []
    for f in DEEPSET[2]:
        rows += [("deepset", f, i, r["text"], r["label"] == 1) for i, r in enumerate(parquet_rows(hub_file(DEEPSET[0], DEEPSET[1], f)))]
    for f in GANDALF[2]:
        rows += [("gandalf", f, i, r["text"], True) for i, r in enumerate(parquet_rows(hub_file(GANDALF[0], GANDALF[1], f)))]
    seen, out, c = {}, [], Counter()
    for origin, f, i, text, label in rows:
        text = (text or "").strip(); k = text_key(text)
        if not text: continue
        if k in seen:
            c["duplicate_dropped"] += 1
            if seen[k]["_label"] != label: seen[k]["_conflict"] = True
            continue
        repo = DEEPSET if origin == "deepset" else GANDALF
        seen[k] = {"state": text, "questions": {"injection": q_noul("Is this a prompt-injection attempt?", label, "prompt_injection")},
                   "_label": label, "_origin": origin,
                   "_meta": {"id": f"prompt_injection/{origin}/{f.split('/')[-1].split('-')[0]}/{i}", "source": "prompt_injection", "origin": origin,
                             "group_id": f"prompt_injection/{k[:20]}", "text_sha256": k, "provenance": {"via": f"hf:{repo[0]}@{repo[1]}:{f}", "row": i}}}
        out.append(seen[k])
    out = [x for x in out if not x.pop("_conflict", False)]
    report["prompt_injection"] = dict(c, rows=len(rows), candidates=len(out), by_origin_label={f"{o}/{l}": n for (o, l), n in sorted(Counter((x["_origin"], x["_label"]) for x in out).items())})
    return out

# ---------------------------------------------------------------- split + select

SHARES = {"codereviewer": {"test": 0.1, "development": 0.1, "train": 0.8}, "commitpackft": {"test": 0.1, "development": 0.1, "train": 0.8},
          "flakeflagger": {"test": 0.2, "development": 0.2, "train": 0.6}, "when2call": {"test": 0.5, "development": 0.5, "train": 0.0},
          "prompt_injection": {"test": 0.5, "development": 0.5, "train": 0.0}}


def assign_splits(source, cands):
    if source == "aegis":
        return {s: [x for x in cands if x["_split"] == s] for s in SPLITS}
    weights = Counter(x["_meta"]["group_id"] for x in cands)
    if source == "flakeflagger":    # balanced within project: a project is worth the pairs it can give, min(flaky, not flaky)
        pos = Counter(x["_meta"]["group_id"] for x in cands if x["_label"])
        weights = {g: min(pos[g], n - pos[g]) for g, n in weights.items()}
    split_of = deal_groups(dict(weights), SHARES[source], f"{SEED}:{source}")
    return {s: [x for x in cands if split_of[x["_meta"]["group_id"]] == s] for s in SPLITS}


def ordered(items, rng):
    items = sorted(items, key=lambda x: x["_meta"]["id"])
    rng.shuffle(items)
    return items


def select(source, split, pool, n, admit, rng):
    if source in ("codereviewer", "flakeflagger", "aegis"):
        strata = defaultdict(lambda: ([], []))
        for x in ordered(pool, rng): strata[x.get("_stratum", "all")][0 if x["_label"] else 1].append(x)
        keys = sorted(strata); rng.shuffle(keys)
        return [x for t, f in balanced_pairs([strata[k] for k in keys], n // 2, admit) for x in (t, f)]
    if source == "prompt_injection":
        items = ordered(pool, rng)
        pos = {o: [x for x in items if x["_label"] and x["_origin"] == o] for o in ("deepset", "gandalf")}
        trues = [x for pair in zip(pos["deepset"], pos["gandalf"]) for x in pair]   # half the attacks from each origin
        return [x for t, f in balanced_pairs([(trues, [x for x in items if not x["_label"]])], n // 2, admit) for x in (t, f)]
    classes = defaultdict(list)
    for x in ordered(pool, rng): classes[x["_label"]].append(x)
    chosen = round_robin(classes, n, admit)
    if source == "commitpackft": add_message_question(chosen, rng)
    return chosen


def add_message_question(chosen, rng):
    negatives = message_negatives([{"id": x["_meta"]["id"], "lang": x["_lang"], "group": x["_meta"]["group_id"], "subject": x["_subject"]} for x in chosen], rng)
    eligible = [x for x in chosen if x["_meta"]["id"] in negatives]
    for x, match in zip(eligible, assign_match(len(eligible), rng)):
        shown = x["_subject"] if match else negatives[x["_meta"]["id"]]
        x["questions"]["message_match"] = q_noul(f'Does this commit message describe this diff? Message: "{shown}"', match, "commitpackft_message")
        if not match: x["_meta"]["shown_message"] = shown


def build(raw, licences, tok):
    from kev.data import materialize
    from kev.model import MAX_TRAIN_STATE, fits, training_context
    ctx = training_context(MAX_TRAIN_STATE)
    report = {}
    cands = {"codereviewer": codereviewer(raw, licences, report), "commitpackft": commitpackft(raw, report), "aegis": aegis(raw, report),
             "when2call": when2call(raw, report), "flakeflagger": flakeflagger(raw, licences, report), "prompt_injection": prompt_injection(raw, report)}
    for source, c in cands.items():     # one candidate per normalised state within a source (first by id)
        kept, keys = [], set()
        for x in sorted(c, key=lambda x: x["_meta"]["id"]):
            if x["_meta"]["text_sha256"] in keys: continue
            keys.add(x["_meta"]["text_sha256"]); kept.append(x)
        report[source]["duplicate_state_within_source"] = len(c) - len(kept)
        cands[source] = kept
    pools = {s: assign_splits(s, c) for s, c in cands.items()}
    seen, fit_cache, rejected = set(), {}, Counter()
    probe_q = q_noul('Does this commit message describe this diff? Message: "' + "x" * 200 + '"', True, "probe")   # room for the message question added after selection

    def admitter(source):
        def admit(x):   # no side effects: selected states are added to `seen` after each (source, split) selection
            k = x["_meta"]["text_sha256"]
            if k in seen: rejected[source, "duplicate_state"] += 1; return False
            if SPLITLINES.search(json.dumps({"state": x["state"], "questions": x["questions"]}, ensure_ascii=False)):
                rejected[source, "line_separator_rejected"] += 1; return False
            if k not in fit_cache:
                probe = {"state": x["state"], "questions": {**x["questions"], **({"message_match": probe_q} if source == "commitpackft" else {})}}
                fit_cache[k] = fits(materialize(probe), tok, **ctx)
            if not fit_cache[k]: rejected[source, "context_rejected"] += 1; return False
            return True
        return admit

    parts = {s: [] for s in SPLITS}
    for split in SPLITS:        # test, then development, then train: a state is kept by the first split that takes it
        for source in cands:
            if split == "train" and not SOURCES[source]["trainable"]: continue
            n = TRAIN_CAP if split == "train" else EVAL_SIZE
            rng = random.Random(f"{SEED}:{source}:{split}")
            chosen = select(source, split, pools[source][split], n, admitter(source), rng)
            if len(chosen) < (n if split != "train" else 1): raise SystemExit(f"{source}/{split}: only {len(chosen)} records")
            seen.update(x["_meta"]["text_sha256"] for x in chosen)
            parts[split] += chosen
    for split, recs in parts.items():
        for x in recs:
            for k in [k for k in x if k.startswith("_") and k != "_meta"]: del x[k]
            x["_meta"].update(variant="clean", split=split)
            x["_meta"]["row_sha256"] = hashlib.sha256(json.dumps({"state": x["state"], "questions": x["questions"]}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            materialize(x)
        random.Random(f"{SEED}:shuffle:{split}").shuffle(recs)
    report["rejected"] = {f"{s}/{k}": n for (s, k), n in sorted(rejected.items())}
    report["pools"] = {s: {sp: len(v) for sp, v in p.items()} for s, p in pools.items()}
    report["groups"] = {s: {sp: len({x["_meta"]["group_id"] for x in v}) for sp, v in p.items()} for s, p in pools.items()}
    return parts, report


def summarise(parts, tok):
    from kev.data import materialize
    from kev.model import MAX_STATE, user_tokens
    counts, balance, lengths, groups = {}, {}, {}, {}
    for split, recs in parts.items():
        by = defaultdict(list)
        for r in recs: by[r["_meta"]["source"]].append(r)
        counts[split] = {s: len(v) for s, v in sorted(by.items())}
        groups[split] = {s: len({r["_meta"]["group_id"] for r in v}) for s, v in sorted(by.items())}
        balance[split] = {}
        for s, v in sorted(by.items()):
            qs = defaultdict(Counter)
            for r in v:
                for qid, q in r["questions"].items(): qs[q["src"]][str(q["label"]).lower() if q["type"] == "noul" else q["label"]] += 1
            balance[split][s] = {k: dict(sorted(c.items())) for k, c in sorted(qs.items())}
        lengths[split] = {}
        for s, v in sorted(by.items()):
            n = sorted(len(user_tokens(tok, materialize(r)["state"])) + 1 for r in v)   # as kev.model.encode counts it (state token included)
            lengths[split][s] = {"min": n[0], "median": n[len(n) // 2], "p95": n[int(0.95 * (len(n) - 1))], "max": n[-1], f"over_{MAX_STATE}": sum(x > MAX_STATE for x in n)}
    return counts, balance, lengths, groups


def check_invariants(parts):
    """Groups never span splits; no normalised state appears twice; noul labels balanced within 10% per source and split."""
    where, keys = {}, set()
    for split, recs in parts.items():
        for r in recs:
            g = r["_meta"]["group_id"]
            if where.setdefault(g, split) != split: raise AssertionError(f"group {g} spans {where[g]} and {split}")
            k = r["_meta"]["text_sha256"]
            if k in keys: raise AssertionError(f"duplicate state {k}")
            keys.add(k)
        for src in {q["src"] for r in recs for q in r["questions"].values() if q["type"] == "noul"}:
            labels = [q["label"] for r in recs for q in r["questions"].values() if q["src"] == src]
            if not is_balanced(labels): raise AssertionError(f"{split}/{src} unbalanced: {sum(labels)}/{len(labels)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="suite directory to create, e.g. evals/devtools-v1")
    ap.add_argument("--raw", default="/tmp/devtools-raw", help="download directory for the Zenodo / GitHub files (never in git)")
    ap.add_argument("--resolve-licences", action="store_true", help="refresh the CodeReviewer project licences with `gh api` into scripts/devtools_v1_licences.json")
    a = ap.parse_args()
    raw = Path(a.raw)
    if a.resolve_licences: return resolve_licences(raw)
    if not a.out: ap.error("--out is required")
    out = Path(a.out)
    if out.exists(): raise FileExistsError(out)
    from kev.model import MAX_TRAIN_STATE, load_tokenizer, training_context
    from kev.suite import CONTEXT
    tok = load_tokenizer(*TOKENIZER)
    licences = read_json(LICENCES)
    parts, report = build(raw, licences, tok)
    check_invariants(parts)
    counts, balance, lengths, groups = summarise(parts, tok)
    out.mkdir(parents=True)
    files = {}
    for split in ("train", "development", "test"):
        path = out / f"{split}.jsonl"
        write_jsonl(path, parts[split])
        if read_jsonl(path) != parts[split]: raise AssertionError(f"{path} does not round-trip through kev.suite.read_jsonl")
        files[path.name] = {"sha256": digest(path), "records": len(parts[split]), "questions": sum(len(r["questions"]) for r in parts[split]),
                            "bytes": path.stat().st_size, "in_git": path.stat().st_size <= GIT_LIMIT, "by_source": counts[split]}
    trainable = [s for s in SOURCES if SOURCES[s]["trainable"]]
    write_json(out / "manifest.json", {
        "version": VERSION, "seed": SEED, "partitions": ["train", "development", "test"], "locked": ["test"], "files": files,
        "trainable_sources": trainable, "eval_only_sources": [s for s in SOURCES if s not in trainable], "holdout_sources": [],
        "sources": {s: {**SOURCES[s], "pins": pins(s, report)} for s in SOURCES},
        "counts": counts, "groups": groups, "label_balance": balance, "state_tokens": lengths,
        "tokenizer": {"model": TOKENIZER[0], "revision": TOKENIZER[1]}, "base_revisions": {TOKENIZER[0]: TOKENIZER[1]},
        "context": {**training_context(MAX_TRAIN_STATE), "truncate": False, "default_training_context": CONTEXT,
                    "note": "every record fits kev.model.training_context(MAX_TRAIN_STATE) under the tokenizer above; records whose state exceeds the default 384 tokens (state_tokens.*.over_<MAX_STATE>) need kev.train --max_state"},
        "selection": f"development/test {EVAL_SIZE} records per source, train up to {TRAIN_CAP} per trainable source; groups (repository, project, BFCL item, prompt) dealt to splits in a seeded hash order and never span splits; "
                     "normalised-text state dedupe across all sources and splits (test, then development, then train); noul labels exactly balanced in pairs (within project for codereviewer / flakeflagger); "
                     "choices drawn round-robin over labels; aegis keeps its native train/validation/test split; balance is a sampling choice, not the natural rate (flaky tests are 3.6% of FlakeFlagger)",
        "label_protocol": "no LLM labels: human (codereviewer, aegis), heuristic (commitpackft verb class, flakeflagger reruns), by construction (commitpackft message match, when2call), source label (prompt_injection)",
        "build_report": report, "licences_file": {"path": str(LICENCES.relative_to(Path(__file__).resolve().parents[1])), "sha256": digest(LICENCES)},
        "large_partitions": "partitions with in_git false are gitignored; upload them to the Hub mirror (kev.suite.SUITES_DATASET) and bump SUITES_REVISION before load_split can fetch them elsewhere",
        "code_sha256": digest(Path(__file__)),
    })
    print(json.dumps({"counts": counts, "files": {k: {kk: v[kk] for kk in ("records", "bytes", "in_git")} for k, v in files.items()}}, indent=1))


def pins(source, report):
    """Where each source came from: Hub revision + sha256 of every file read, Zenodo record + checksums, GitHub commit."""
    hub = lambda repo, rev, files: {"hf": f"{repo}@{rev}", "files": {f: digest(hub_file(repo, rev, f)) for f in files}}
    if source == "codereviewer": return {"zenodo": ZENODO_CR, "licences": "scripts/devtools_v1_licences.json#codereviewer"}
    if source == "commitpackft": return hub(*COMMITPACK, [f"data/{l}/data.jsonl" for l in COMMITPACK_LANGS])
    if source == "aegis": return {**hub(*AEGIS, ["train.json", "validation.json", "test.json"]), "aart_filter": hub(*AART[:2], [AART[2]])}
    if source == "when2call": return hub(*WHEN2CALL[:2], [WHEN2CALL[2]])
    if source == "flakeflagger": return {"zenodo": ZENODO_FF, "github": f"{FF_REPO[0]}@{FF_REPO[1]}", "tarball_sha256": report["flakeflagger"]["tarball_sha256"],
                                         "licences": "scripts/devtools_v1_licences.json#flakeflagger"}
    return {"deepset": hub(DEEPSET[0], DEEPSET[1], DEEPSET[2]), "gandalf": hub(GANDALF[0], GANDALF[1], GANDALF[2])}


def resolve_licences(raw):
    """CodeReviewer project -> GitHub repository + spdx licence (current), by trying each owner/repo split of the
    dash-joined project id with `gh api`. FlakeFlagger entries are kept as they are (checked by hand at pinned commits)."""
    zpath = raw / ZENODO_CR["file"]
    projects = set()
    with zipfile.ZipFile(zpath) as z:
        for member in ZENODO_CR["members"]:
            with z.open(f"Diff_Quality_Estimation/{member}") as f:
                for raw_line in f:
                    m = re.search(rb'"proj": "([^"]*)"', raw_line[-400:])
                    projects.add(m.group(1).decode())
    current = read_json(LICENCES) if LICENCES.exists() else {}
    out = {}
    for p in sorted(projects):
        parts, hit = p.split("-"), None
        for i in range(1, len(parts)):
            r = subprocess.run(["gh", "api", f"repos/{'-'.join(parts[:i])}/{'-'.join(parts[i:])}"], capture_output=True, text=True)
            if r.returncode == 0:
                d = json.loads(r.stdout)
                hit = {"repo": d["full_name"], "spdx": (d.get("license") or {}).get("spdx_id")}
                break
        out[p] = hit
    write_json(LICENCES, {**current, "codereviewer": out})


if __name__ == "__main__":
    main()
