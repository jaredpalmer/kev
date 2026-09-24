"""README figure: the current Kev family against Jev, per out-of-domain source.

    uv run python scripts/plot_family.py            # -> docs/kev-family.png, docs/kev-family-summary.json

Reads saved result.json files only (development partitions; the locked test is not plotted). Style: scripts/chartstyle.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from kev.suite import read_json, write_json
from chartstyle import GRID, JEV, KEV, TEXT, TEXT2, body, display, heading, rule, strip, use_style

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("sciq", "SciQ"), ("qnli", "QNLI"), ("contrastive_authorization", "Policy: authorization"), ("composition_held_and_or", "Rule: (A or B) and C"),
         ("composition_held_or_not", "Rule: (A and B) or not C"), ("tweet_offensive", "TweetEval offensive"), ("paws", "PAWS"),
         ("composition_held_conditional", "Rule: if A then not B else C"), ("mmlu", "MMLU, 4-way"), ("contrastive_deadline", "Policy: deadline (3-level Score)"), ("emotion", "Emotion, 6-way")]
MODELS = [("kev-0.8b", "runs/r15-08b/00-trial-0/result.json"), ("kev-4b", "runs/r10-skills/00-trial-0/result.json"),
          ("kev-9b", "runs/night2-9b-du/00-trial-0/result.json"), ("kev-27b", "runs/release/kev-27b-v2/result.json")]
JEV_PATH = "runs/jev-transfer-v4/report.json"


def transfer(path):
    r = read_json(ROOT / path)
    t = r["transfer"] if "transfer" in r else r
    return {k: v["acc"] for k, v in t["tasks"].items()}, t["clean"]["acc"], t["clean"]["brier"]


def main():
    use_style()
    data = {n: transfer(p) for n, p in MODELS}
    jev_tasks, jev_acc, jev_brier = transfer(JEV_PATH)
    fig = plt.figure(figsize=(14, 9.5))
    heading(fig, .05, .93, "Where Kev matches Jev and where it does not, on data Kev never trained on", size=22)
    body(fig, .05, .89, "Accuracy per source on the frozen out-of-domain suite (transfer-v4, 764 records, development partition). Same items for every model. Jev via Vercel AI Gateway.", size=11.5)
    rule(fig, .05, .96, .865)

    ax = fig.add_axes([.25, .2, .68, .59])
    y = np.arange(len(TASKS))
    for yi, (key, _) in zip(y, TASKS):
        vals = {n: 100 * data[n][0][key] for n, _ in MODELS}; vals["Jev"] = 100 * jev_tasks[key]
        lo, hi = min(vals.values()), max(vals.values())
        ax.plot([lo, hi], [yi, yi], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        for n, v in vals.items():
            ax.scatter([v], [yi], s=95, color=JEV if n == "Jev" else KEV[n], zorder=3, edgecolor="white", linewidth=1.2)
        # direct labels: the best Kev and Jev, offset vertically so they never overlap
        best = max((n for n, _ in MODELS), key=lambda n: vals[n])
        ax.annotate(f"{vals[best]:.0f}", (vals[best], yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5, color=KEV[best], weight="medium")
        ax.annotate(f"{vals['Jev']:.0f}", (vals["Jev"], yi), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9.5, color=JEV, weight="medium")
    ax.set_yticks(y); ax.set_yticklabels([label for _, label in TASKS], fontsize=12); ax.set_ylim(len(TASKS) - .5, -.5)
    ax.set_xlim(20, 104); ax.set_xticks(range(20, 101, 20)); ax.set_xticklabels([f"{v}%" for v in range(20, 101, 20)], fontsize=10.5, color=TEXT2)
    ax.grid(axis="x", color=GRID, lw=.8, zorder=0); ax.tick_params(length=0); strip(ax)
    # key as direct text, once, with each model's overall accuracy
    key = [("Jev", JEV, jev_acc)] + [(n, KEV[n], data[n][1]) for n, _ in reversed(MODELS)]
    for i, (n, c, acc) in enumerate(key):
        fig.text(.05 + .135 * i, .82, "●", fontsize=13, color=c, ha="left", va="baseline")
        fig.text(.05 + .135 * i + .016, .82, f"{display(n)}  {100 * acc:.1f}%", fontsize=11, color=TEXT, ha="left", va="baseline")
    close = sum(data["kev-27b"][0][k] >= jev_tasks[k] - 0.03 for k, _ in TASKS)
    body(fig, .05, .135, f"Notice: Kev-27B is within three points of Jev, or ahead, on {close} of the {len(TASKS)} sources; the 4B and 9B are close on the classification-shaped ones.\n"
         "The gap is concentrated in knowledge (MMLU) and, below 27B, day-precision date arithmetic (the deadline policy).\n"
         "Sources: QNLI, SciQ, TweetEval, PAWS, MMLU, Emotion; the policy families and rule structures shown were never trained.", size=10.5, va="top")
    body(fig, .05, .022, "LoRA r=16 + pointer head. Kev-0.8B, 4B, 9B on Qwen3.5 base models; Kev-27B on the post-trained Qwen3.8-27B.  Regenerate: uv run python scripts/plot_family.py", size=9.5)

    out = ROOT / "docs/kev-family.png"
    fig.savefig(out, dpi=170, metadata={"Title": "Kev family vs Jev, out of domain"}); plt.close(fig)
    write_json(ROOT / "docs/kev-family-summary.json", {"models": {n: {"path": p, "transfer_acc": data[n][1], "transfer_brier": data[n][2], "tasks": data[n][0]} for n, p in MODELS},
                                                       "jev": {"path": JEV_PATH, "transfer_acc": jev_acc, "transfer_brier": jev_brier, "tasks": jev_tasks}})
    print(out, {n: round(data[n][1], 3) for n, _ in MODELS}, "jev", round(jev_acc, 3))


if __name__ == "__main__":
    main()
