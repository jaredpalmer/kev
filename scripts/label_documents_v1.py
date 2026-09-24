"""Label documents-v1 candidates with LLMs through the Vercel AI Gateway (PLAN_27b B2, revised protocol), blind to the
native labels. One call per (model, document) answers all of that document's questions; answers are cached per model and
split, so a rerun resumes; a call that failed (an `error` result: a non-retryable HTTP status or retries exhausted) is
kept in the answers file for the record but is not cached, so the next run retries it. A shared spend ledger (labels/spend.json, written after every labelled result) enforces a
hard cap across runs. The cap holds for sequential runs of this script (two concurrent runs each read the ledger once and
overwrite each other's total); within a run, the calls already in flight when it trips (at most --workers - 1) still
finish and are recorded, so the ledger can end slightly above the cap. A response without `usage.cost` is not free, it is
unknown: those are counted in the ledger ("uncosted") and the run stops once more than --max-uncosted have been seen.

    export AI_GATEWAY_API_KEY=$(cat ~/.config/kev/ai_gateway_key)
    uv run python scripts/label_documents_v1.py --split train --models deepseek/deepseek-v3.2,alibaba/qwen3-235b-a22b-thinking
    uv run python scripts/label_documents_v1.py --split test --models anthropic/claude-opus-4.5,openai/gpt-5,google/gemini-3-flash

Roles (the protocol): training labels come from two open-weight teachers and are kept only where both agree with the native
label; development and test labels are the native label checked by a three-judge panel from other model families. No Jev.
"""
import argparse, json, os, re, sys, threading, time, urllib.error, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import read_json, read_jsonl  # noqa: E402

WORK = Path("runs/documents-v1-work")
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
SYSTEM = ("You label consumer financial complaints. For each question choose exactly one option key, using only what the "
          "complaint says. If several options fit, choose the one that best describes the consumer's main problem.")
lock = threading.Lock()


def prompt(rec):
    lines = [f"Complaint:\n<<<\n{rec['state']}\n>>>\n", "Questions:"]
    for qid, q in rec["questions"].items():
        lines.append(f"[{qid}] {q['instructions']}")
        lines += [f"  - {k}: {d}" for k, d in q["criteria"].items()]
    shape = ", ".join(f'"{qid}": {{"label": "<option key>", "reason": "<one short sentence>"}}' for qid in rec["questions"])
    lines.append(f"\nAnswer with only a JSON object: {{{shape}}}")
    return "\n".join(lines)


def parse(text, rec):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: obj = json.loads(m.group())
    except json.JSONDecodeError: return None
    out = {}
    for qid, q in rec["questions"].items():   # well-formed JSON of the wrong shape is an unlabelled question, never an exception
        a = obj.get(qid)
        label, reason = (a.get("label"), a.get("reason")) if isinstance(a, dict) else (a, None)
        out[qid] = {"label": label if isinstance(label, str) and label in q["criteria"] else None, "reason": reason[:300] if isinstance(reason, str) else ""}
    return out


def answered(path):
    """Ids with a cached answer: every result except failed calls, which the next run retries."""
    return {r["id"] for r in read_jsonl(path) if "error" not in r} if path.exists() else set()


def save_ledger(path, ledger):
    """Write the ledger atomically, so a crash mid-write cannot leave a truncated spend.json behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def call(model, rec, key, retries=6):
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(rec)}],
            "max_tokens": 6000 if "thinking" in model or model.startswith(("openai/gpt-5", "google/gemini")) else 800}
    if model.startswith("openai/gpt-5"): body["reasoning_effort"] = "low"
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data, {"authorization": f"Bearer {key}", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r: resp = json.load(r)
            text = resp["choices"][0]["message"].get("content") or ""
            cost = (resp.get("usage") or {}).get("cost")   # None: the gateway did not price this response
            return {"answers": parse(text, rec), "cost": None if cost is None else float(cost), "raw": text[:2000]}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1: time.sleep(2 ** attempt + 1); continue
            return {"answers": None, "cost": 0.0, "error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            if attempt < retries - 1: time.sleep(2 ** attempt + 1); continue
            return {"answers": None, "cost": 0.0, "error": f"{type(e).__name__}: {e}"}


def label(model, todo, out, key, ledger, ledger_path, *, cap, max_uncosted, workers):
    """Label `todo` with one model, appending each answer to `out` and saving the ledger after every result. The spend
    checks run on the worker that recorded the result, so a call that starts after the cap trips sees `stop` and is never
    made. Returns (counts, whether the run stopped early)."""
    stop, tally = threading.Event(), Counter()

    def label_one(rec):
        if stop.is_set(): return
        try: res = call(model, rec, key)
        except BaseException: stop.set(); raise
        with lock:
            cost = res["cost"] or 0.0
            ledger["total"] += cost; ledger["by_model"][model] = ledger["by_model"].get(model, 0.0) + cost
            if res["cost"] is None: tally["uncosted"] += 1; ledger["uncosted"] = ledger.get("uncosted", 0) + 1
            f.write(json.dumps({"id": rec["_meta"]["id"], "model": model, **res}) + "\n"); f.flush()
            save_ledger(ledger_path, ledger)
            if "error" in res: tally["failed"] += 1
            else: tally["labelled"] += 1; tally["unparsed"] += res["answers"] is None
            if (tally["labelled"] + tally["failed"]) % 100 == 0: print(f"  {tally['labelled']}/{len(todo)} (failed {tally['failed']}, unparsed {tally['unparsed']}, uncosted {tally['uncosted']}) ${ledger['total']:.2f}", flush=True)
            if ledger["total"] >= cap and not stop.is_set():
                stop.set(); print(f"  spend cap ${cap:.2f} reached; stopping", flush=True)
            if ledger.get("uncosted", 0) > max_uncosted and not stop.is_set():
                stop.set(); print(f"  {ledger['uncosted']} responses without usage.cost (more than {max_uncosted}); spend unknown, stopping", flush=True)

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(label_one, r) for r in todo]): fut.result()
    save_ledger(ledger_path, ledger)
    return tally, stop.is_set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "development", "test"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--cap", type=float, default=80.0, help="hard cap in dollars across every run (the ledger at runs/documents-v1-work/labels/spend.json)")
    ap.add_argument("--limit", type=int, default=0, help="label only the first N documents (a smoke run)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-uncosted", type=int, default=5, help="stop once more than this many responses (across runs, in the ledger) carried no usage.cost")
    ap.add_argument("--work", default=str(WORK), help="work directory (runs/documents-v2-work for the held-out set)")
    a = ap.parse_args()
    work = Path(a.work)
    key = os.environ["AI_GATEWAY_API_KEY"]
    if "jev" in a.models.lower(): raise SystemExit("no Jev in any role")
    recs = read_jsonl(work / "candidates" / f"{a.split}.jsonl")
    if a.limit: recs = recs[:a.limit]
    ledger_path = work / "labels" / "spend.json"; ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = read_json(ledger_path) if ledger_path.exists() else {"total": 0.0, "by_model": {}}
    if ledger.get("uncosted", 0) > a.max_uncosted:
        raise SystemExit(f"{ledger['uncosted']} earlier responses carried no usage.cost (limit {a.max_uncosted}); reconcile {ledger_path} with the gateway's billing first")
    for model in a.models.split(","):
        out = work / "labels" / a.split / (model.replace("/", "__") + ".jsonl"); out.parent.mkdir(parents=True, exist_ok=True)
        done = answered(out)
        todo = [r for r in recs if r["_meta"]["id"] not in done]
        print(f"{model} on {a.split}: {len(done)} cached, {len(todo)} to label; spend so far ${ledger['total']:.2f} of ${a.cap:.2f}", flush=True)
        tally, stopped = label(model, todo, out, key, ledger, ledger_path, cap=a.cap, max_uncosted=a.max_uncosted, workers=a.workers)
        print(f"  done: {tally['labelled']} labelled, {tally['failed']} failed (retried next run), {tally['unparsed']} unparsed, {tally['uncosted']} uncosted; ledger ${ledger['total']:.2f}", flush=True)
        if stopped: break


if __name__ == "__main__":
    main()
