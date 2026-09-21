#!/usr/bin/env python3
"""Generate labelled Kev training records for a workload with any OpenAI-compatible chat model.

    export KEV_GEN_API_KEY=...                       # or OPENAI_API_KEY / AI_GATEWAY_API_KEY
    python3 scripts/generate_data.py workload.json --n 600 --out data/support.jsonl --model gpt-4.1-mini
    python3 scripts/generate_data.py workload.json --dry-run          # print one batch prompt and exit

The workload spec describes the state text and the exact System One questions the deployed model will be asked
(see assets/workload.example.json). Every batch asks the LLM for records whose labels fill the least-represented
options first, so the output is balanced per question. Records are validated (split_data.check_question), deduplicated
by state, and appended to --out as they arrive, so an interrupted run resumes where it stopped.

Endpoints: KEV_GEN_BASE_URL (default https://api.openai.com/v1). Vercel AI Gateway: https://ai-gateway.vercel.sh/v1 with
model ids like openai/gpt-4.1-mini or anthropic/claude-sonnet-4.5. Ollama: http://localhost:11434/v1. Standard library only.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_data import check_question, label_key, normalized_state  # noqa: E402

DEFAULT_BASE_URL = "https://api.openai.com/v1"
SYSTEM = ("You write labelled training examples for a small decision model. Every example is a realistic state text plus the "
          "correct answer to each question. Reply with a single JSON object and nothing else.")


def option_keys(q):
    if q["type"] == "noul": return ["true", "false"]
    if q["type"] == "choice": return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


def describe_question(qid, q):
    if q["type"] == "noul":
        crit = q.get("criteria") or {}
        detail = "".join(f' ({k} = {v})' for k, v in crit.items() if v)
        return f'- {qid} (noul): "{q["instructions"]}"{detail} Label: true or false.'
    if q["type"] == "choice":
        opts = "; ".join(f"{k}" + (f" = {v}" if v not in (None, "") else "") for k, v in q["criteria"].items())
        return f'- {qid} (choice): "{q["instructions"]}" Options: {opts}. Label: the option name exactly as written.'
    levels = "; ".join(f"{i} = {lvl}" for i, lvl in enumerate(q["criteria"]))
    return f'- {qid} (score): "{q["instructions"]}" Levels: {levels}. Label: the level index (an integer).'


def batch_targets(counts, keys, done, batch):
    """How many records of this batch should carry each label so the file ends up balanced: fill the deficit against an
    even split of (done + batch) first, then spread the rest evenly."""
    goal = math.ceil((done + batch) / len(keys))
    deficit = {k: max(0, goal - counts.get(k, 0)) for k in keys}
    total = sum(deficit.values())
    if total == 0: deficit = {k: 1 for k in keys}; total = len(keys)
    raw = {k: batch * v / total for k, v in deficit.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    for k in sorted(keys, key=lambda k: raw[k] - alloc[k], reverse=True)[: batch - sum(alloc.values())]: alloc[k] += 1
    return {k: v for k, v in alloc.items() if v}


def build_prompt(spec, counts, done, batch, examples, rng):
    lines = [f"Domain: {spec['domain']}", f"State: {spec['state']}"]
    if spec.get("state_example") is not None:
        lines.append("State shape (produce states of exactly this shape): " + json.dumps(spec["state_example"], ensure_ascii=False))
    lines.append("Questions the model will be asked about each state. Give the correct label for every one:")
    lines += [describe_question(qid, q) for qid, q in spec["questions"].items()]
    if spec.get("guidance"): lines.append(f"Labelling rules: {spec['guidance']}")
    variety = list(spec.get("variety", []))
    if variety:
        rng.shuffle(variety)
        lines.append("Vary the examples along these axes (cover several per batch): " + "; ".join(variety))
    lines.append(f"Label targets for this batch of {batch} records, per question:")
    for qid, q in spec["questions"].items():
        alloc = batch_targets(counts[qid], option_keys(q), done, batch)
        lines.append(f"- {qid}: " + ", ".join(f"{v} x {k}" for k, v in alloc.items()))
    if examples:
        lines.append("Reference examples in the target style (write new ones, do not copy):")
        for ex in examples: lines.append(json.dumps({"state": ex["state"], "labels": {qid: q["label"] for qid, q in ex["questions"].items()}}, ensure_ascii=False))
    lines.append(f"Make every state different from the others (different people, wording, details, and difficulty; include some hard or ambiguous-looking "
                 f"cases whose label still follows the rules). Random seed for variety: {rng.randrange(10**6)}.")
    lines.append('Return {"records": [{"state": <state>, "labels": {<question id>: <label>, ...}}, ...]} with exactly ' + f"{batch} records.")
    return "\n".join(lines)


def chat(base_url, api_key, model, prompt, json_mode=True, timeout=180, retries=5):
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], "temperature": 1.0}
    if json_mode: body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", "authorization": f"Bearer {api_key}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            if error.code == 400 and json_mode and "response_format" in detail:
                return chat(base_url, api_key, model, prompt, json_mode=False, timeout=timeout, retries=retries)
            if error.code in (401, 403): raise SystemExit(f"authentication failed at {base_url}: {detail}")
            if error.code == 404: raise SystemExit(f"model or endpoint not found ({model} at {base_url}): {detail}")
            if error.code not in (408, 409, 429) and error.code < 500: raise SystemExit(f"HTTP {error.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt == retries - 1: raise SystemExit(f"cannot reach {base_url}: {error}")
        time.sleep(min(2 ** attempt, 30))
    raise SystemExit("the endpoint kept failing; try again later or lower --concurrency")


def parse_records(text):
    text = text.strip()
    if text.startswith("```"): text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0: return []
        try: data = json.loads(text[start:end + 1])
        except json.JSONDecodeError: return []
    records = data.get("records") if isinstance(data, dict) else data
    return records if isinstance(records, list) else []


def coerce_label(q, value):
    if q["type"] == "noul":
        if isinstance(value, bool): return value
        if isinstance(value, str) and value.lower() in ("true", "false"): return value.lower() == "true"
        return value
    if q["type"] == "score":
        if isinstance(value, str) and value.strip().lstrip("-").isdigit(): return int(value)
        if isinstance(value, float) and value.is_integer(): return int(value)
        return value
    if isinstance(value, str) and value not in q["criteria"]:
        matches = [k for k in q["criteria"] if k.casefold() == value.strip().casefold()]
        return matches[0] if len(matches) == 1 else value
    return value


def to_record(spec, item):
    """A generated {"state", "labels"} item -> a labelled request, or None with the reason when it is invalid."""
    if not isinstance(item, dict) or "state" not in item or not isinstance(item.get("labels"), dict): return None, "missing state or labels"
    if item["state"] in (None, ""): return None, "empty state"
    questions = {}
    for qid, q in spec["questions"].items():
        if qid not in item["labels"]: return None, f"no label for {qid}"
        labelled = {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        labelled["label"] = coerce_label(q, item["labels"][qid])
        problems = check_question(qid, labelled)
        if problems: return None, problems[0]
        questions[qid] = labelled
    return {"state": item["state"], "questions": questions}, None


def load_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("domain", "state", "questions"):
        if not spec.get(field): raise SystemExit(f"{path}: the spec needs a non-empty {field!r} field (see assets/workload.example.json)")
    for qid, q in spec["questions"].items():
        probe = {**q, "label": {"noul": True, "choice": next(iter(q.get("criteria") or {}), None), "score": 0}.get(q.get("type"))}
        problems = check_question(qid, probe)
        if problems: raise SystemExit(f"{path}: {problems[0]}")
    return spec


def existing_records(path):
    if not Path(path).exists(): return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Environment: KEV_GEN_API_KEY (or OPENAI_API_KEY / AI_GATEWAY_API_KEY), KEV_GEN_BASE_URL.")
    ap.add_argument("spec", help="workload spec JSON (assets/workload.example.json shows every field)")
    ap.add_argument("--n", type=int, default=600, help="total records wanted in --out (existing valid records count); default 600")
    ap.add_argument("--out", help="JSONL to append to (required unless --dry-run)")
    ap.add_argument("--model", default=os.environ.get("KEV_GEN_MODEL", "gpt-4.1-mini"), help="chat model id at the endpoint; default gpt-4.1-mini")
    ap.add_argument("--base-url", default=os.environ.get("KEV_GEN_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--batch", type=int, default=20, help="records requested per call; default 20")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel calls; default 4")
    ap.add_argument("--examples", help="JSONL of real labelled records to show as style references (up to --n-examples per batch)")
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print one batch prompt (use it yourself or in another tool) and exit")
    a = ap.parse_args()
    spec = load_spec(a.spec)
    rng = random.Random(a.seed)
    examples = [json.loads(l) for l in Path(a.examples).read_text(encoding="utf-8").splitlines() if l.strip()] if a.examples else []
    counts = {qid: Counter() for qid in spec["questions"]}
    have = existing_records(a.out) if a.out else []
    seen = set()
    for r in have:
        seen.add(normalized_state(r["state"]))
        for qid, q in r["questions"].items():
            if qid in counts: counts[qid][label_key(q)] += 1
    if a.dry_run:
        print(build_prompt(spec, counts, len(have), a.batch, rng.sample(examples, min(a.n_examples, len(examples))), rng)); return 0
    if not a.out: ap.error("--out is required (or use --dry-run)")
    key = os.environ.get("KEV_GEN_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("AI_GATEWAY_API_KEY")
    if not key: ap.error("set KEV_GEN_API_KEY (or OPENAI_API_KEY / AI_GATEWAY_API_KEY) for the chat endpoint")
    if a.base_url == DEFAULT_BASE_URL and os.environ.get("AI_GATEWAY_API_KEY") and not os.environ.get("KEV_GEN_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        a.base_url = "https://ai-gateway.vercel.sh/v1"

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rejected, calls, started = Counter(), 0, time.time()
    print(f"{len(have)} records already in {out}; generating up to {a.n} with {a.model} at {a.base_url}", flush=True)
    with out.open("a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(a.concurrency) as pool:
        while len(have) < a.n and calls < 3 * math.ceil(a.n / a.batch) + a.concurrency:
            wanted = min(a.concurrency, math.ceil((a.n - len(have)) / a.batch))
            futures = []
            for _ in range(wanted):
                prompt = build_prompt(spec, counts, len(have), a.batch, rng.sample(examples, min(a.n_examples, len(examples))), random.Random(rng.random()))
                futures.append(pool.submit(chat, a.base_url, key, a.model, prompt)); calls += 1
            for future in as_completed(futures):
                added = 0
                for item in parse_records(future.result()):
                    record, why = to_record(spec, item)
                    if record is None: rejected[why] += 1; continue
                    norm = normalized_state(record["state"])
                    if norm in seen: rejected["duplicate state"] += 1; continue
                    if len(have) >= a.n: break
                    seen.add(norm); have.append(record); added += 1
                    for qid, q in record["questions"].items(): counts[qid][label_key(q)] += 1
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  +{added} -> {len(have)}/{a.n} ({time.time() - started:.0f}s)", flush=True)
    print(f"{len(have)} records in {out} after {calls} calls; rejected: {dict(rejected) or 'none'}")
    for qid, c in counts.items(): print(f"  {qid}: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    if len(have) < a.n: print(f"stopped short of {a.n}: the model kept returning invalid or duplicate records; check the spec's guidance", file=sys.stderr); return 1
    print(f"next: python3 scripts/split_data.py {out} --out {out.with_suffix('')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
