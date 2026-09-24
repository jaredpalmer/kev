"""Benchmark card: docs/kev-benchmark.png (1600x900, 16:9). Style: scripts/chartstyle.py.

    uv run python scripts/plot_tweet.py            # light, docs/kev-benchmark.png
    uv run python scripts/plot_tweet.py --dark     # black background, docs/kev-benchmark-dark.png

Reads saved result files only; nothing is typed in by hand except labels.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import chartstyle as cs
from kev.suite import read_json
from chartstyle import JEV, KEV, body, display, heading, hbars, rule, stat, use_style

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("kev-27b", "Qwen3.8-27B", "runs/release/kev-27b-v2/result.json"),
          ("kev-9b", "Qwen3.5-9B", "runs/night2-9b-du/00-trial-0/result.json"),
          ("kev-4b", "Qwen3.5-4B", "runs/r10-skills/00-trial-0/result.json"),
          ("kev-0.8b", "Qwen3.5-0.8B", "runs/r15-08b/00-trial-0/result.json")]
BASES = [("Qwen3.5-9B", "untrained base", "runs/probes/qwen35-9b-base-base-transfer-v4/report.json"),
         ("Qwen3.5-4B instruct", "untrained, SemIf prompt", "runs/probes/qwen35-4b-semif-transfer-v4/report.json")]
LOCKED = [("kev-27b", "runs/locked/kev-27b-v2-ungated/summary.json"),
          ("kev-9b", "runs/locked/kev-9b-night2-du-ungated/summary.json"),
          ("kev-4b", "runs/locked/kev-4b-r10-ungated/summary.json"),
          ("kev-0.8b", "runs/locked/kev-08b-r15-ungated/summary.json")]
JEV_T = "runs/jev-transfer-v4/report.json"


def clean(path):
    r = read_json(ROOT / path)
    if "transfer" in r: r = r["transfer"]
    return r["clean"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dark", action="store_true", help="black background, white/gray text")
    dark = ap.parse_args().dark
    use_style(dark=dark)
    fig = plt.figure(figsize=(16, 9))
    heading(fig, .05, .905, "Kev: open decision models, measured against Jev", size=30)
    body(fig, .05, .855, "Typed questions in, probabilities out, one forward pass, no text generation. LoRA + pointer head on Qwen bases.", size=14)
    rule(fig, .05, .95, .82)

    rows = [(n, sub, 100 * clean(p)["acc"], KEV[n]) for n, sub, p in MODELS]
    rows += [(n, sub, 100 * clean(p)["acc"], cs.NEUTRAL) for n, sub, p in BASES]
    rows.sort(key=lambda r: -r[2])
    jev = clean(JEV_T)
    rows.insert(0, ("Jev", "TypeSafe, hosted", 100 * jev["acc"], JEV))

    ax = fig.add_axes([.24, .235, .39, .52])
    heading(fig, .05, .77, "Accuracy on data Kev never trained on", size=17)
    hbars(ax, [display(r[0]) for r in rows], [r[2] for r in rows], [r[3] for r in rows], xlim=(0, 100), fmt="{:.1f}%",
          emphasize=[i for i, r in enumerate(rows) if r[0].startswith("kev")], sublabels=[r[1] for r in rows], ticks=range(0, 101, 25), label_size=13, value_size=14)
    n_records = read_json(ROOT / MODELS[0][2])["transfer"]["coverage"]["evaluated_records"]
    body(fig, .05, .115, f"Bars: {n_records} frozen out-of-domain records from QNLI, SciQ, PAWS, MMLU, Emotion, TweetEval and held-out policy rules; the same development items for every row.\n"
         "Gray rows: zero-shot letter logits, no Kev fine-tuning. Right: locked tests, read once per release; Jev has not been evaluated on these partitions.", size=11)

    for i, (name, path) in enumerate(LOCKED):
        locked = read_json(ROOT / path)["suites"]["transfer"]["clean"]
        stat(fig, .70, .77 - .16 * i, f"{display(name)}, out of domain, locked test", f"{100 * locked['acc']:.1f}%",
             f"Brier {locked['brier']:.3f}  ·  ECE {100 * locked['ece']:.1f}%", color=KEV[name], value_size=28)

    body(fig, .05, .05, "github.com/jaredpalmer/kev  ·  huggingface.co/jaredpalmer  ·  Apache-2.0  ·  every number reproducible from frozen, checksummed suites", size=11)
    out = ROOT / ("docs/kev-benchmark-dark.png" if dark else "docs/kev-benchmark.png")
    fig.savefig(out, dpi=100, metadata={"Title": "Kev benchmarks"}); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
