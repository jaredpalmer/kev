"""Where the full-weight arms of rounds 19-20 lose scienthoon, from committed rows only (report only, no new reads).

scienthoon-v1 is 291 synthetic support tickets with three questions each: `queue` (Choice), `priority` (Score whose label
follows an org rule absent from the text) and `angry` (Noul, "The customer sounds angry."). Every ticket body is an issue
sentence plus an optional closing phrase, either neutral ("Appreciate any help.", "답변 기다리겠습니다.") or angry ("This is
unacceptable and I want it fixed today.", "정말 화가 납니다. 당장 처리해 주세요."), so the text class of each `angry` label can be
read off the body. The script compares every 27B checkpoint with committed scienthoon rows against Kev-27B (paired
record-clustered bootstrap, kev.rounds.paired: 2,000 resamples, seed 0, micro), splits the `angry` errors by text class
and ticket kind, and decomposes the pooled-externals and short-state deltas of the round-19/20 arms by suite and task.

    uv run python scripts/scienthoon_drift.py --out runs/r20-scienthoon

Rows come from this checkout, or, for reads that live only on the research archive, from `git show <tag>:<path>`.
Argmax accuracy does not depend on temperature, so rows are compared as saved (served_at 1.0 restores raw logits).
"""
import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from kev.metrics import served_at
from kev.rounds import paired
from kev.suite import read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "research-archive-2026-09-24"
SUITE = "evals/external/scienthoon-v1/development.jsonl"
REFERENCE = "kev-27b"

# (name, rows directory, kind, note). kind: lora (LoRA from the base), lora-delta (LoRA delta from a LoRA checkpoint),
# full (full-weight SFT from the base), wise (a round-20 interpolation), external.
SIDES = [
    ("kev-27b", "runs/r6-27bv2-s2-scienthoon", "lora", "Kev-27B, B1 v2 seed 2 (released)"),
    ("b1v2-s1", "runs/r6-27bv2-s1-scienthoon", "lora", "B1 v2 seed 1: Kev-27B's recipe, other seed"),
    ("b1-A", "runs/r6-27b-A-scienthoon", "lora", "B1 trial A: v7 recipe, 1 epoch"),
    ("b1-B", "runs/r6-27b-B-scienthoon", "lora", "B1 trial B"),
    ("2ep-s1", "runs/r6-27b2ep-s1-scienthoon", "lora", "round-6 follow-up: v7 recipe, 2 epochs, seed 1"),
    ("2ep-s2", "runs/r6-27b2ep-s2-scienthoon", "lora", "round-6 follow-up: 2 epochs, seed 2"),
    ("r7-docs", "runs/r7-27b-s1-scienthoon", "lora-delta", "round 7: documents delta from trial A"),
    ("r10-skills", "runs/r10-27b-skills-scienthoon", "lora-delta", "round 10: skills delta from Kev-27B"),
    ("r17-a", "runs/r17-27b-r10k-lr2e5-scienthoon", "lora-delta", "round 17 arm (a): skills delta from Kev-27B, replay 10,000"),
    ("r19-a", "runs/r19-27b-lr2e6-scienthoon", "full", "round 19 (a): full weights on sft-v1, lr 2e-6"),
    ("r19-b", "runs/r19-27b-lr5e6-scienthoon", "full", "round 19 (b): full weights on sft-v1, lr 5e-6"),
    ("r19-c", "runs/r19-27b-olddata-scienthoon", "full", "round 19 (c): full weights on Kev-27B's own data, lr 5e-6, 2 epochs"),
    *[(f"{arm}-w{w}", f"runs/r20-27b-{arm}-w{w}-scienthoon", "wise", f"round 20: {w / 100:.2f} x round-19 ({arm}) + {1 - w / 100:.2f} x base")
      for arm in ("a", "b") for w in (85, 70, 50)],
    ("autojev", "runs/autojev-scienthoon", "external", "AutoJev-27B (full weights, 73k synthetic decisions, same base)"),
    ("jev", "runs/jev-scienthoon-v1", "external", "Jev, converted from scienthoon's results/jev_synth.jsonl"),
]
ANGRY_PHRASES = ("This is unacceptable", "정말 화가 납니다", "진짜 실망입니다", "disputing the charge", "Ridiculous")
NEUTRAL_PHRASES = ("Let me know what you need from me.", "답변 기다리겠습니다.", "Appreciate any help.", "Thanks in advance.", "확인 부탁드립니다.")
INQUIRIES = ("Question about your return policy", "Feature suggestion", "Do you ship to Canada?", "영업시간 문의", "제품 사양 질문")
HANGUL = re.compile("[\uac00-\ud7a3]")


def rows_of(directory):
    """Scored rows of one read: this checkout's rows.json, else the archive tag's, else None."""
    path = ROOT / directory / "rows.json"
    if path.exists(): return served_at(read_json(path), 1.0)
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{ARCHIVE}:{directory}/rows.json"], capture_output=True, text=True)
    return served_at(json.loads(shown.stdout), 1.0) if shown.returncode == 0 else None


def ticket_facts(records):
    """{record id: facts} read off each ticket's text: the closing phrase's class, inquiry or problem, language."""
    facts = {}
    for r in records:
        s, label = r["state"], r["questions"]["angry"]["label"]
        body = s["body"]
        text = "angry" if any(a in body for a in ANGRY_PHRASES) else "neutral"
        facts[r["_meta"]["id"]] = {"text": text, "angry_label": label, "label_matches_text": label == (text == "angry"),
                                   "kind": "inquiry" if s["subject"] in INQUIRIES else "problem",
                                   "issue_language": "ko" if HANGUL.search(s["subject"]) else "en",
                                   "closer": next((p for p in (*ANGRY_PHRASES, *NEUTRAL_PHRASES) if p in body), None),
                                   "tier": s["customer_tier"], "channel": s["channel"], "subject": s["subject"], "body": body}
    return facts


def correct(row):
    return int(np.argmax(row["p"])) == row["label"]


def angry_confusion(rows, facts):
    """`angry` predictions split by what the text says: false positives on neutral text (by ticket kind and language),
    false negatives on angry text, and errors on the 15 labels that contradict the text."""
    out = collections.Counter()
    for r in rows:
        if r["task"] != "scienthoon_angry": continue
        f, said = facts[r["id"]], bool(np.argmax(r["p"]) == 1)
        out["predicted_angry"] += said
        if not f["label_matches_text"]: out["errors_on_contradicting_labels"] += not correct(r); continue
        if f["text"] == "neutral" and said:
            out["false_positive_on_neutral_text"] += 1; out[f"false_positive_{f['kind']}"] += 1; out[f"false_positive_{f['issue_language']}"] += 1
        if f["text"] == "angry" and not said: out["false_negative_on_angry_text"] += 1
    neutral = [r["p"][1] for r in rows if r["task"] == "scienthoon_angry" and facts[r["id"]]["text"] == "neutral" and facts[r["id"]]["label_matches_text"]]
    out = dict(out); out["mean_p_angry_on_neutral_text"] = float(np.mean(neutral))
    return out


def priority_profile(rows):
    got = [(int(np.argmax(r["p"])), r["label"]) for r in rows if r["task"] == "scienthoon_priority"]
    return {"predicted": dict(sorted(collections.Counter(p for p, _ in got).items())), "mean_predicted_minus_label": float(np.mean([p - y for p, y in got]))}


def delta(candidate, reference, keep=lambda r: True):
    c, ref = [r for r in candidate if keep(r)], [r for r in reference if keep(r)]
    x = paired(c, ref, "acc")
    return {"n": len(c), "delta_pp": 100 * x["delta"], "ci95_pp": [100 * v for v in x["ci95"]]}


def panel_rows(directories):
    rows = []
    for d in directories:
        got = rows_of(d)
        if got is None: raise FileNotFoundError(d)
        rows += got
    return rows


def by_task(candidate, reference):
    """Net questions gained (+) or lost (-) per task, candidate minus reference, on shared questions."""
    ref = {(r["id"], r["question"]): r for r in reference}
    net, n = collections.Counter(), collections.Counter()
    for r in candidate:
        other = ref.get((r["id"], r["question"]))
        if other is None: continue
        net[r["task"]] += correct(r) - correct(other); n[r["task"]] += 1
    return {t: {"net": net[t], "n": n[t]} for t in sorted(n, key=lambda t: net[t])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/r20-scienthoon")
    a = ap.parse_args()
    facts = ticket_facts(read_jsonl(ROOT / SUITE))
    sides = {name: (rows_of(d), d, kind, note) for name, d, kind, note in SIDES}
    missing = [n for n, (rows, *_) in sides.items() if rows is None]
    sides = {n: s for n, s in sides.items() if s[0] is not None}
    reference = sides[REFERENCE][0]
    tasks = ("scienthoon_queue", "scienthoon_angry", "scienthoon_priority")
    report = {"suite": SUITE, "reference": REFERENCE, "archive": ARCHIVE, "missing_reads": missing,
              "untrained_base": "no scienthoon read of Qwen/Qwen3.8-27B exists (runs/probes/qwen38-27b-* covers semif-v1, transfer-v4/v9 and wanli-v1 only)",
              "tickets": {"n": len(facts), "angry_text": sum(f["text"] == "angry" for f in facts.values()),
                          "angry_labels": sum(f["angry_label"] for f in facts.values()),
                          "labels_contradicting_text": sorted(i for i, f in facts.items() if not f["label_matches_text"])},
              "sides": {}}
    for name, (rows, d, kind, note) in sides.items():
        side = {"rows": d, "kind": kind, "note": note, "acc": float(np.mean([correct(r) for r in rows])),
                "tasks": {t: float(np.mean([correct(r) for r in rows if r["task"] == t])) for t in tasks},
                "angry": angry_confusion(rows, facts), "priority": priority_profile(rows)}
        if name != REFERENCE and all(r["label"] == o["label"] for r, o in zip(sorted(rows, key=lambda r: (r["id"], r["question"])), sorted(reference, key=lambda r: (r["id"], r["question"])))):
            side["vs_reference"] = {"all": delta(rows, reference),
                                    **{t.removeprefix("scienthoon_"): delta(rows, reference, lambda r, t=t: r["task"] == t) for t in tasks},
                                    "without_angry": delta(rows, reference, lambda r: r["task"] != "scienthoon_angry")}
        report["sides"][name] = side
    lora = [s["acc"] for n, s in report["sides"].items() if s["kind"] == "lora"]
    report["lora_from_base"] = {"n": len(lora), "mean": float(np.mean(lora)), "sd": float(np.std(lora, ddof=1)), "min": min(lora), "max": max(lora),
                                "reference_rank": 1 + sorted(lora, reverse=True).index(report["sides"][REFERENCE]["acc"]),
                                "mean_false_positives": float(np.mean([s["angry"].get("false_positive_on_neutral_text", 0) for s in report["sides"].values() if s["kind"] == "lora"]))}
    # the angry questions Kev-27B gets right and a round-19 full-weight arm gets wrong
    ref_by = {(r["id"], r["question"]): r for r in reference}
    flips = collections.Counter()
    probs = collections.defaultdict(dict)
    for name in ("r19-a", "r19-b", "r19-c"):
        for r in sides[name][0]:
            if r["task"] == "scienthoon_angry" and correct(ref_by[(r["id"], r["question"])]) and not correct(r):
                flips[r["id"]] += 1
            if r["task"] == "scienthoon_angry": probs[r["id"]][name] = r["p"][1]
    for r in reference:
        if r["task"] == "scienthoon_angry": probs[r["id"]][REFERENCE] = r["p"][1]
    flipped = sorted(flips)
    report["flips"] = {"questions": len(flipped), "in_all_three_arms": sum(v == 3 for v in flips.values()),
                       "neutral_text_gold_false": sum(facts[i]["text"] == "neutral" and not facts[i]["angry_label"] for i in flipped),
                       "problem_tickets": sum(facts[i]["kind"] == "problem" for i in flipped),
                       "sample": [{"id": i, **{k: facts[i][k] for k in ("angry_label", "text", "kind", "tier", "subject", "body")},
                                   "p_angry": {k: round(v, 3) for k, v in probs[i].items()}, "arms_wrong": flips[i]} for i in flipped[:20]]}
    # the pooled externals (SemIf, scienthoon, WANLI-v2, TypeSafe) and the short-state panel, decomposed
    parent = {"semif": "runs/r6-27bv2-s2-semif", "scienthoon": "runs/r6-27bv2-s2-scienthoon", "wanli2": "runs/r6-27bv2-s2-wanli2", "typesafe": "runs/r6-27bv2-s2-typesafe",
              "short": ["runs/r6-27b-v2/01-trial-1/transfer", "runs/r19-P27-r3test"]}
    arms = {"r19-a": "runs/r19-27b-lr2e6", "r19-b": "runs/r19-27b-lr5e6", "r19-c": "runs/r19-27b-olddata", "b-w50": "runs/r20-27b-b-w50"}
    report["pooled_externals"], report["short"] = {}, {}
    for name, prefix in arms.items():
        suites = {s: panel_rows([f"{prefix}-{s}"]) for s in ("semif", "scienthoon", "wanli2", "typesafe")}
        refs = {s: panel_rows([parent[s]]) for s in suites}
        pooled, pooled_ref = [r for s in suites for r in suites[s]], [r for s in refs for r in refs[s]]
        report["pooled_externals"][name] = {
            "all": delta(pooled, pooled_ref),
            "without_scienthoon_angry": delta(pooled, pooled_ref, lambda r: r["task"] != "scienthoon_angry"),
            "net_questions": {s: sum(v["net"] for v in by_task(suites[s], refs[s]).values()) for s in suites},
            "scienthoon_angry_net": by_task(suites["scienthoon"], refs["scienthoon"])["scienthoon_angry"]["net"]}
        transfer = f"{prefix}-transfer4" if name.startswith(("a-", "b-")) else f"{prefix}/00-trial-0/transfer"
        short, short_ref = panel_rows([transfer, f"{prefix}-r3test"]), panel_rows(parent["short"])
        tasks_net = by_task(short, short_ref)
        report["short"][name] = {"all": delta(short, short_ref), "largest_losses": dict(list(tasks_net.items())[:6]),
                                 "net": sum(v["net"] for v in tasks_net.values())}
    # the same two guards applied to Kev-27B's LoRA siblings (every external read on this checkout or the archive)
    report["lora_siblings_under_the_guards"] = {}
    for name, d, kind, _ in SIDES:
        if kind != "lora" or name == REFERENCE: continue
        prefix = d.removesuffix("-scienthoon")
        try:
            pooled = panel_rows([f"{prefix}-{s}" for s in ("semif", "scienthoon", "wanli2", "typesafe")])
        except FileNotFoundError:
            continue
        pooled_ref = panel_rows([parent[s] for s in ("semif", "scienthoon", "wanli2", "typesafe")])
        s = report["sides"][name]["vs_reference"]["all"]
        report["lora_siblings_under_the_guards"][name] = {"scienthoon": s, "scienthoon_guard_passes": s["ci95_pp"][0] >= -2,
                                                          "pooled_externals": (p := delta(pooled, pooled_ref)), "pooled_guard_passes": p["ci95_pp"][0] >= -1.5}
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    write_json(out / "drift.json", report)
    for name, s in report["sides"].items():
        v = s.get("vs_reference", {})
        print(f"{name:11} {s['kind']:10} acc {s['acc']:.3f}  queue {s['tasks']['scienthoon_queue']:.3f} angry {s['tasks']['scienthoon_angry']:.3f} "
              f"priority {s['tasks']['scienthoon_priority']:.3f}  angry FP(neutral) {s['angry'].get('false_positive_on_neutral_text', 0):2} "
              f"FN(angry) {s['angry'].get('false_negative_on_angry_text', 0):2}" + (f"  vs Kev-27B {v['all']['delta_pp']:+.1f} [{v['all']['ci95_pp'][0]:+.1f}, {v['all']['ci95_pp'][1]:+.1f}]"
              f" without angry {v['without_angry']['delta_pp']:+.1f} [{v['without_angry']['ci95_pp'][0]:+.1f}, {v['without_angry']['ci95_pp'][1]:+.1f}]" if v else ""))
    print(json.dumps({k: report[k] for k in ("lora_from_base", "pooled_externals", "short")}, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
