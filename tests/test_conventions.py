"""Source conventions: facts that have one canonical home must not be re-derived elsewhere.

Each rule is (what it guards, regex, files allowed to match). A failure means a second copy of a rule that already has
a home; call the canonical helper instead (the table in .agents/skills/thermonuclear-code-review/SKILL.md lists them).
Run: uv run python -m pytest tests/test_conventions.py -q
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("kev", "scripts", "space", "tests", "modal_app.py")

RULES = [
    ("head.pt is read and written through kev.checkpoint (Meta, read_meta, write_meta)",
     r"torch\.(load|save)\([^\n]*head\.pt", {"kev/checkpoint.py"}),
    ("KEV_DTYPE/KEV_MERGE/KEV_ATTN/KEV_LORA_SCALE/KEV_TEMPERATURE/KEV_BACKEND/KEV_CUDA_GRAPHS/KEV_PROGRAM are read only by LoadOptions.from_env",
     r"environ(\.get)?\(?\[?\s*\"KEV_(DTYPE|MERGE|ATTN|LORA_SCALE|TEMPERATURE|BACKEND|CUDA_GRAPHS|PROGRAM)\"", {"kev/checkpoint.py"}),
    ("a checkpoint becomes a model only through kev.checkpoint (Checkpoint.load picks the torch, MLX or ExecuTorch implementation)",
     r"MLXDecisionModel\(|merge_lora\(|ExecuTorchDecisionModel\(|load_program\(", {"kev/checkpoint.py", "kev/mlx_model.py", "kev/executorch_model.py", "tests/test_mlx.py", "tests/test_executorch.py"}),
    ("option keys come from kev.api.question_keys",
     r"\[\s*\"false\"\s*,\s*\"true\"\s*\]|\[str\(i\) for i in range\(len\(", {"kev/api.py", "tests/test_unit.py"}),   # the unit test pins the contract
    ("the training context is kev.model.MAX_STATE/MAX_BRANCH/MAX_PACKED, lifted only through kev.model.training_context (kev.suite.CONTEXT in manifests), and kev.model.fits",
     r"(?<![\w.])(>|<=|>=|<)\s*2048\b|\b2048\s*(<|>)|max_(branch|state|packed)\"?\s*[=:]\s*\d{3,}", {"kev/model.py"}),
    ("text files are read and written as UTF-8 (kev.suite.read_json/read_jsonl/write_json/write_jsonl, or an explicit encoding=); "
     "the platform locale must never decide how a frozen partition is decoded (issue #12)",
     r"\.read_text\(\)|\.write_text\((?![^\n]*encoding=)|json\.loads?\(open\(|encoding=None|(?<![\w.])open\((?![^\n]*encoding=)(?![^\n]*\"[rwax]b\")", {"kev/suite.py"}),
    ("suite manifests are read through kev.suite.read_manifest",
     r"manifest\.json\"\)\.read_text\(\)", {"kev/suite.py"}),
    ("device selection, synchronize and empty_cache go through kev.device (the Space is a CUDA-only one-off)",
     r"is_available\(\) else|torch\.(mps|cuda)\.(synchronize|empty_cache|current_allocated_memory|max_memory_allocated)\(", {"kev/device.py", "space/app.py"}),
    ("the isolation sibling probe is kev.experiment.ISOLATION_PROBE (fp32 mechanism check and served isolation read the same question)",
     r"CRANE-9274", {"kev/experiment.py"}),
    ("which partitions stay out of git is kev.suite.GIT_LIMIT",
     r"10 \* 1024 \* 1024", {"kev/suite.py"}),
    ("the pinned Qwen3.5 tokenizer suite builders admit records under is kev.suite.ADMISSION_TOKENIZER",
     r"1001bb4d826a52d1f399e183466143f4da7b741b", {"kev/suite.py", "kev/transfer_v9.py"}),   # transfer_v9 pins every Qwen3.5 base it scores
    ("a state's normalised-text hash (text_sha256) is kev.suite.text_digest",
     r"\.casefold\(\)\.split\(\)\)\.encode\(\)", {"kev/suite.py", "kev/data.py"}),   # kev.suite imports kev.data, so kev.data keeps its inline copy
]


def sources():
    for entry in SCANNED:
        path = ROOT / entry
        yield from (p for p in ([path] if path.is_file() else sorted(path.rglob("*.py"))) if "__pycache__" not in p.parts and p != Path(__file__))


def test_private_suites_keep_their_partitions_out_of_git():
    """A manifest that names its own mirror (kev.suite: a held-out suite) publishes hashes only: no partition in git."""
    import json, subprocess
    tracked = set(subprocess.run(["git", "ls-files", "evals"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split())
    for manifest in sorted((ROOT / "evals").rglob("manifest.json")):
        if "mirror" not in json.loads(manifest.read_text(encoding="utf-8")): continue
        leaked = [f for f in tracked if f.startswith(str(manifest.parent.relative_to(ROOT)) + "/") and f.endswith(".jsonl")]
        assert not leaked, f"{manifest.parent} names a private mirror but tracks partitions: {leaked}"


def test_published_claims_trace_to_committed_evidence():
    from scripts.verify_claims import verify

    assert verify(ROOT) == []


@pytest.mark.parametrize("what,pattern,allowed", RULES, ids=[r[0][:60] for r in RULES])
def test_single_home(what, pattern, allowed):
    regex = re.compile(pattern)
    offenders = []
    for path in sources():
        rel = str(path.relative_to(ROOT))
        if rel in allowed:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if regex.search(code):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, f"{what}\n" + "\n".join(offenders)
