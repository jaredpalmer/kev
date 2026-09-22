"""Diagnostic: how much knowledge does the *base* model show on the frozen MMLU/SciQ transfer items with a plain
letter-token readout, versus what our trained pointer head gets on the same items?

    uv run python scripts/base_mmlu_probe.py --base Qwen/Qwen3-4B-Base --suite evals/v4/transfer-v4 --tasks mmlu,sciq

Prints accuracy per task for the base model (zero-shot, next-token logits over the option letters). Compare with the
per-task numbers in runs/<study>/<trial>/result.json -> transfer.tasks. Read-only; touches no suite files.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kev.device import default_device
from kev.suite import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--suite", default="evals/v4/transfer-v4")
    ap.add_argument("--tasks", default="mmlu,sciq")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--out", help="write benchmark-compatible rows.json/report.json here (comparable with paired bootstraps)")
    ap.add_argument("--prompt", choices=["plain", "semif"], default="plain",
                    help="semif: SemIf's readout (github.com/TheoLeeCJ/SemIf core.direct_messages): chat template, system instruction, JSON {evidence, criterion, options} payload, letter logits; for instruct models")
    ap.add_argument("--split", default="development")
    ap.add_argument("--all_questions", action="store_true", help="score every question of a record, not only the first (teacher passes over training records)")
    ap.add_argument("--revision", default=None, help="pin the Hub revision (recorded in the report)")
    ap.add_argument("--adapter", default=None, help="diagnostic: merge this Kev LoRA checkpoint into the base, then read letter logits through the base lm_head. "
                                                   "Separates 'the adapted backbone forgot X' from 'the pointer readout cannot express X'.")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.revision)
    dtype = torch.bfloat16 if a.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, revision=a.revision).to(a.device).eval()
    if a.adapter:
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, a.adapter).merge_and_unload()   # Kev adapters are trained on the bare backbone (.model)
        model.eval()
    letters = "ABCDEFGHIJKLMNOP"
    letter_ids = [tok.encode((" " if a.prompt == "plain" else "") + L, add_special_tokens=False)[0] for L in letters]
    if len(set(letter_ids)) != len(letter_ids): raise ValueError("answer-slot tokens collide")
    SEMIF_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
                    "Respond with only its uppercase letter, with no explanation or reasoning.")
    records = [r for r in load_split(a.suite, a.split, allow_test=a.split == "test") if r["_meta"]["variant"] == "clean" and (a.tasks == "all" or r["_meta"]["source"] in a.tasks.split(","))]
    hits, n = {}, {}
    rows = []   # benchmark-compatible rows so the run can be compared with paired bootstraps
    with torch.no_grad():
        for r in records:
            for qid, q in (r["questions"].items() if a.all_questions else [next(iter(r["questions"].items()))]):
                if q["type"] == "noul": q = {**q, "criteria": {"false": "No", "true": "Yes"}, "label": str(bool(q["label"])).lower()}
                elif q["type"] == "score": q = {**q, "criteria": {str(i): c for i, c in enumerate(q["criteria"])}, "label": str(q["label"])}
                keys = list(q["criteria"])
                state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]) if not isinstance(r["state"], dict) else " ".join(f"{k}: {v}" for k, v in r["state"].items())
                if a.prompt == "semif":
                    payload = {"evidence": r["state"], "criterion": q["instructions"], "options": [{"letter": letters[i], "description": q["criteria"][k] or k} for i, k in enumerate(keys)]}
                    prompt = tok.apply_chat_template([{"role": "system", "content": SEMIF_SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(a.device)
                else:
                    prompt = f"{state}\n{q['instructions']}\n" + "\n".join(f"{letters[i]}. {q['criteria'][k]}" for i, k in enumerate(keys)) + "\nAnswer:"
                    ids = tok(prompt, return_tensors="pt").to(a.device)
                logits = model(**ids).logits[0, -1].float()
                probs = torch.softmax(logits[letter_ids[: len(keys)]], -1).tolist()
                pred = int(max(range(len(keys)), key=probs.__getitem__))
                src = r["_meta"]["source"]
                hits[src] = hits.get(src, 0) + int(keys[pred] == q["label"]); n[src] = n.get(src, 0) + 1
                m = r["_meta"]
                rows.append({"id": m["id"], "group": m["group_id"], "question": qid, "source": src, "task": r["questions"][qid]["src"], "type": r["questions"][qid]["type"],
                             "variant": m["variant"], "keys": keys, "label": keys.index(q["label"]), "pair_id": m.get("pair_id"), "sibling": m.get("sibling"), "parent": m["id"],
                             "p": probs, "raw_probability_sum": 1.0, "zero_count": 0})
    summary = {"base": a.base, "readout": "zero-shot next-token letter logits" + (" (SemIf prompt, chat template)" if a.prompt == "semif" else ""), "suite": a.suite, "split": a.split, "revision": a.revision, "adapter": a.adapter,
               "sources": {k: {"n": n[k], "acc": round(hits[k] / n[k], 3)} for k in n}}   # namespaced: a source called "unknowable" must not shadow the report's unknowable block
    if a.out:
        from kev.benchmark import summarize
        from kev.suite import write_json
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        write_json(out / "rows.json", rows)
        rep = summarize(rows, heldout_sources=tuple(n)); rep.update(summary); write_json(out / "report.json", rep)
        summary["clean_acc"] = rep["clean"]["acc"]; summary["brier"] = rep["clean"]["brier"]; summary["paired_flip"] = rep["paired_flip"]
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
