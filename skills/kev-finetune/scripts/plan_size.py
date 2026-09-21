#!/usr/bin/env python3
"""How many records to generate so the fine-tune vs baseline comparison can be statistically significant.

    python3 scripts/plan_size.py workload.json                              # default: detect +5 accuracy points at 80% power
    python3 scripts/plan_size.py workload.json --min-gain 0.03 --baseline-acc 0.85
    python3 scripts/plan_size.py --from-result runs/support-v1/result.json  # post hoc: how much more data would make the observed gain significant

The skill compares the fine-tuned model with the baseline on the same development questions (paired). The number of
development questions needed to detect an accuracy gain of `min_gain` follows the McNemar (paired) approximation:
discordant pairs = questions the two models answer differently, assumed to be the gain plus 2 x `regressions` (the share the
baseline gets right and the fine-tune gets wrong). Records = questions / questions-per-record, and the development
partition is `--development` of the file, so total records = development records / that fraction. The unpaired
two-proportion number is printed as the conservative upper bound. Standard library only.
"""
import argparse
import json
import math
import sys
from pathlib import Path

Z = {0.8: 0.8416, 0.9: 1.2816, 0.95: 1.6449}
Z_ALPHA = 1.96   # two-sided 95%


def paired_questions(gain, regressions, power):
    """McNemar sample size: n = (z_a sqrt(p_d) + z_b sqrt(p_d - gain^2))^2 / gain^2, p_d = share of discordant pairs."""
    p_d = gain + 2 * regressions
    return math.ceil((Z_ALPHA * math.sqrt(p_d) + Z[power] * math.sqrt(max(p_d - gain ** 2, 1e-9))) ** 2 / gain ** 2)


def unpaired_questions(p1, gain, power):
    p2 = min(p1 + gain, 0.999)
    return math.ceil((Z_ALPHA + Z[power]) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / gain ** 2)


def plan(questions_per_record, baseline_acc, gain, regressions, power, development, calibration):
    q_paired = paired_questions(gain, regressions, power)
    q_unpaired = unpaired_questions(baseline_acc, gain, power)
    dev_records = math.ceil(q_paired / questions_per_record)
    cal_records = math.ceil(max(100 / questions_per_record, dev_records * calibration / development))   # >= 100 questions for a stable temperature fit
    total = math.ceil(max(dev_records / development, cal_records / calibration))
    return {"development_questions": q_paired, "development_questions_unpaired_bound": q_unpaired, "development_records": dev_records,
            "calibration_records": cal_records, "train_records": total - dev_records - cal_records, "total_records": total,
            "assumptions": {"baseline_acc": baseline_acc, "min_gain": gain, "regressions": regressions, "power": power, "alpha": 0.05,
                            "questions_per_record": questions_per_record, "development_fraction": development, "calibration_fraction": calibration}}


def from_result(path, power):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    boot = r.get("bootstrap", {}).get("acc")
    if not boot: raise SystemExit("result.json has no baseline bootstrap; run train with the baseline enabled")
    delta, (lo, hi) = boot["micro_acc_delta"], boot["ci95"]
    n_q = r["development"]["n_questions"]; dev_records = r["data"]["development"]["records"]; frac = dev_records / sum(r["data"][p]["records"] for p in ("train", "calibration", "development"))
    half = (hi - lo) / 2
    print(f"observed accuracy delta {delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] on {n_q} development questions ({dev_records} records)")
    if lo > 0 or hi < 0: print("already significant at 95%"); return 0
    if abs(delta) < 0.005: print("the observed gain is about zero: more data will not make it significant; change the data (see errors.jsonl), not the amount"); return 0
    se = half / Z_ALPHA                                   # standard error of the delta at the current n
    need_q = math.ceil(n_q * (se * (Z_ALPHA + Z[power]) / delta) ** 2)   # se shrinks with sqrt(n); need |delta| / se_new >= z_a + z_b
    need_records = math.ceil(need_q / (n_q / dev_records) / frac)
    have = round(dev_records / frac)
    if need_records > 10 * have:
        print(f"the observed gain would need ~{need_records} records (>{10}x what you have) to reach significance: it is too small to chase with volume; improve the data (errors.jsonl, guidance) or accept the model on its calibration/coverage merits")
    else:
        print(f"to detect {delta:+.3f} at {int(power * 100)}% power: ~{need_q} development questions, i.e. ~{need_records} total records at the same split (you have {have})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?", help="workload spec (questions per record is read from it)")
    ap.add_argument("--questions", type=int, help="questions per record (overrides the spec)")
    ap.add_argument("--baseline-acc", type=float, default=0.75, help="expected baseline accuracy on your questions (0.75 if unknown; run evaluate on the released checkpoint to measure it)")
    ap.add_argument("--min-gain", type=float, default=0.05, help="smallest accuracy gain worth detecting; default 0.05")
    ap.add_argument("--regressions", type=float, default=0.05, help="share of questions the fine-tune is expected to newly get wrong; default 0.05")
    ap.add_argument("--power", type=float, default=0.8, choices=[0.8, 0.9, 0.95])
    ap.add_argument("--development", type=float, default=0.15, help="development fraction passed to split_data.py; default 0.15")
    ap.add_argument("--calibration", type=float, default=0.15)
    ap.add_argument("--from-result", help="a run's result.json: post-hoc estimate instead of a plan")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    a = ap.parse_args()
    if a.from_result: return from_result(a.from_result, a.power)
    if not 0 < a.min_gain < 0.5 or not 0 < a.baseline_acc < 1 or not 0 <= a.regressions < 0.5: ap.error("min-gain in (0, 0.5), baseline-acc in (0, 1), regressions in [0, 0.5)")
    q = a.questions or (len(json.loads(Path(a.spec).read_text(encoding="utf-8"))["questions"]) if a.spec else None)
    if not q: ap.error("give a workload spec or --questions")
    p = plan(q, a.baseline_acc, a.min_gain, a.regressions, a.power, a.development, a.calibration)
    if a.json: print(json.dumps(p, indent=1)); return 0
    print(f"To detect a {a.min_gain:+.0%} accuracy gain over a {a.baseline_acc:.0%} baseline at {int(a.power * 100)}% power (paired, 95% two-sided):")
    print(f"  development questions: {p['development_questions']} paired (unpaired bound {p['development_questions_unpaired_bound']})")
    print(f"  with {q} question(s) per record and a {a.development:.0%} development split: {p['development_records']} development records")
    print(f"  calibration: {p['calibration_records']} records (>= 100 questions for the temperature fit)")
    print(f"  generate at least {p['total_records']} records  ->  python3 scripts/generate_data.py {a.spec or 'workload.json'} --n {p['total_records']} --out data/<name>.jsonl")
    if p["total_records"] > 3000: print("  that is a lot: consider a larger --min-gain for the first round, or measure the baseline first (evaluate) so the target is realistic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
