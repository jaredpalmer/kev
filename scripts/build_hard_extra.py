"""More hard-v1 training records from fresh seeds: the same generators and training templates (0-3), a different RNG seed,
every state deduplicated against all three frozen hard-v1 partitions; ids get their own prefix. Development and test are
untouched, so they still measure unseen templates. Round-14 data (PLAN.md).

    uv run python scripts/build_hard_extra.py --n 12000 --seed hard-v1x-20260924 --out evals/round14/hard-extra
"""
import argparse, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hard_v1 as H  # noqa: E402
from kev.suite import digest, load_split, write_json, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--seed", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seen = {r["_meta"]["text_sha256"] for split in ("train", "development", "test") for r in load_split("evals/hard-v1", split, allow_test=True)}
    frozen = len(seen)
    H.SEED = a.seed   # build_split reads the module seed for every family RNG and the final shuffle
    report = {}
    recs = H.build_split("train", a.n, seen, H.Checker(), report)
    for r in recs:
        for k in ("id", "group_id"): r["_meta"][k] = r["_meta"][k].replace("hard-v1/", "hard-v1x/", 1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", recs)
    write_json(out / "manifest.json", {"version": "hard-v1x", "seed": a.seed, "records": len(recs), "questions": sum(len(r["questions"]) for r in recs),
                                       "by_family": dict(Counter(r["_meta"]["family"] for r in recs)), "deduplicated_against_frozen_states": frozen,
                                       "templates": list(H.TEMPLATE_SPLITS["train"]), "draw_report": report, "sha256": digest(out / "train.jsonl"),
                                       "code_sha256": {"build_hard_extra.py": digest(__file__), "build_hard_v1.py": digest(H.__file__)}})
    print(len(recs), "records", digest(out / "train.jsonl")[:12], dict(Counter(r["_meta"]["family"] for r in recs)))


if __name__ == "__main__":
    main()
