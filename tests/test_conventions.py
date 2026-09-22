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
    ("KEV_DTYPE/KEV_MERGE/KEV_ATTN/KEV_LORA_SCALE/KEV_TEMPERATURE/KEV_BACKEND are read only by LoadOptions.from_env",
     r"environ(\.get)?\(?\[?\s*\"KEV_(DTYPE|MERGE|ATTN|LORA_SCALE|TEMPERATURE|BACKEND)\"", {"kev/checkpoint.py"}),
    ("a checkpoint becomes a model only through kev.checkpoint (Checkpoint.load picks the torch or MLX implementation)",
     r"MLXDecisionModel\(|merge_lora\(", {"kev/checkpoint.py", "kev/mlx_model.py", "tests/test_mlx.py"}),
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
]


def sources():
    for entry in SCANNED:
        path = ROOT / entry
        yield from (p for p in ([path] if path.is_file() else sorted(path.rglob("*.py"))) if "__pycache__" not in p.parts and p != Path(__file__))


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
