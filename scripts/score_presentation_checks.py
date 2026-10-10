"""Score-head order sensitivity, report-only (jaredpalmer/kev#161).

Kev's order tooling targets Choice (`/v1/systemone/permute`, `--perm_kl`, kev.evaluate's Permutation / IIA,
`kev.benchmark --rotations`); nothing measures order sensitivity on Score. This script ports the two checks
jaredpalmer/kev#161 ran by hand, in the style of laya's research/eval/presentation_checks.py
(NandhaKishorM/laya#259), against the same predictor interface kev.benchmark uses (a request record in,
probabilities out), so a checkpoint (LocalPredictor) and an endpoint (RemotePredictor) are measured the same way:

- first-slot rate: the same Score question is asked over every order of its K levels (all K! orders when that is
  at most --orders-max, else that many sampled orders) on --states states; the share of decisions landing on
  slot 0. No positional preference means 1/K (0.333 at K=3).
- identical-option control: all K levels share one text, so no answer is more right than another and any spread
  is presentation noise; slot 0's value minus the mean over slots, in logits when the predictor reports them
  (LocalPredictor does, at the temperature it loads — pass --raw-logits for temperature 1.0, what #161 calls
  "raw"; the issue's numbers are logit-space), else in probabilities.

One caveat the numbers need: Score levels are an ordered scale, so reordering them changes how the question reads as
well as where the answer sits — that is why --rotations leaves Score alone. A high first-slot rate therefore mixes
position bias with order-as-meaning, and only the identical-option control, where the texts are interchangeable,
isolates position. Read the two together, and not the rate as pure bias. The report repeats this next to the numbers.

Report-only on purpose: #98 declined to grow kev.benchmark's modes, so this never touches it. It writes one
JSON report (--out) and prints the same numbers. It is not a gate: it names what a checkpoint or endpoint does,
not what to ship.
"""
import argparse
import itertools
import json
import math
import random
import sys

QID = "score_q"
INSTRUCTIONS = "Answer the question using the scale defined by the levels."
ORDERED_SCALE_CAVEAT = (
    "Score levels are an ordered scale: reordering them changes how the question reads, not only where the answer "
    "sits, so the first-slot rate mixes position bias with order-as-meaning (why --rotations leaves Score alone). "
    "Only the identical-option control, whose texts are interchangeable, isolates position. Read the two together "
    "and not the rate as pure bias.")
DEFAULT_LEVEL_TEXTS = "low,medium,high"
DEFAULT_STATES = (
    "Ticket {i}: the customer cannot sign in and a deadline is tomorrow morning.",
    "Ticket {i}: a scheduled report arrived an hour late; nothing else is affected.",
    "Ticket {i}: the payment webhook dropped events for the last ten minutes in production.",
    "Ticket {i}: a user asks whether dark mode can be enabled on their account.",
    "Ticket {i}: the nightly backup finished with two files unreadable; the archive passed checksum.",
)


def level_orders(k, orders_max):
    """-> the orders the levels are asked in: every permutation when k! <= orders_max, else a deterministic
    sample of orders_max of them (seeded, so two runs of the same command measure the same orders)."""
    if k < 1:
        raise ValueError(f"need at least one level, got {k}")
    every = list(itertools.permutations(range(k)))
    if len(every) <= orders_max:
        return every
    rng = random.Random(0)
    return rng.sample(every, orders_max)


def score_request(state, criteria, label=0):
    """A labelled one-question Score request in the serving format kev.benchmark feeds predictors. The label and
    src are required by the internal record (materialize) and ignored here: report-only, no accuracy is claimed."""
    return {"state": state, "questions": {QID: {"type": "score", "instructions": INSTRUCTIONS,
                                                "criteria": list(criteria), "label": label,
                                                "src": "presentation"}}}


def build_states(n, data=None):
    """-> the states to ask over: the rows of --data (their "state" field) when given, else deterministic
    templates cycled to n. Two runs over the same flags ask the same states."""
    if data is not None:
        states = [row.get("state") for row in data]
        if not all(isinstance(s, str) and s for s in states):
            raise ValueError("--data rows must each carry a non-empty string \"state\"")
    else:
        states = [DEFAULT_STATES[i % len(DEFAULT_STATES)].format(i=i) for i in range(max(n, 1))]
    return states[:n] if data is None else states


def _read(answer, qid, space):
    values = answer.get(space)
    if values is None:
        return None
    return [float(values[qid][str(i)]) for i in range(len(values[qid]))]


def _decide(predictor, state, criteria, spaces):
    answer = predictor(score_request(state, criteria))
    for space in spaces:
        values = _read(answer, QID, space)
        if values is not None:
            return values, space
    raise ValueError(f"predictor returned neither {' nor '.join(spaces)} for {QID}")


def first_slot_rate(predictor, criteria, states, orders):
    """The share of decisions landing on slot 0 over states x orders, plus the per-slot shares (each should be
    1/K) and the expected rate. The decision is the argmax of the probabilities; a predictor that reports logits
    but not probabilities is read in logits (the argmax is the same)."""
    k = len(criteria)
    wins = [0] * k
    n = 0
    for state in states:
        for order in orders:
            criteria_order = [criteria[i] for i in order]
            values, _ = _decide(predictor, state, criteria_order, ("probabilities", "logits"))
            wins[int(max(range(k), key=lambda slot: values[slot]))] += 1
            n += 1
    if n == 0:
        raise ValueError("no states and no orders to ask")
    return {"first_slot_rate": wins[0] / n, "expected": 1 / k, "n_decisions": n,
            "per_slot_rates": [w / n for w in wins], "n_orders": len(orders), "n_levels": k}


def identical_option_control(predictor, level_text, k, states, orders):
    """Slot 0's value minus the mean over slots when every level carries the same text, averaged over states x
    orders, in logits when the predictor reports them and probabilities otherwise. 0 is a flat head; the larger
    the magnitude, the more the first-listed slot is favoured on an unanswerable question."""
    criteria = [level_text] * k
    total, slots, n, space_used = 0.0, [0.0] * k, 0, None
    for state in states:
        for order in orders:
            values, space_used = _decide(predictor, state, criteria, ("logits", "probabilities"))
            total += values[0] - sum(values) / k
            slots = [s + v for s, v in zip(slots, values)]
            n += 1
    if n == 0:
        raise ValueError("no states and no orders to ask")
    return {"control": total / n, "space": space_used, "n_decisions": n,
            "per_slot_means": [s / n for s in slots], "n_orders": len(orders), "n_levels": k}


def measure(predictor, level_texts, states=None, data=None, orders_max=120):
    """-> the report: both checks over the level set. The identical-option control always enumerates every
    order (with one text the orders differ only in position, so sampling buys nothing)."""
    k = len(level_texts)
    states = build_states(len(states) if states else 20, data)
    orders = level_orders(k, orders_max)
    identical = level_orders(k, math.factorial(k))
    return {"levels": level_texts, "n_states": len(states),
            "first_slot": first_slot_rate(predictor, level_texts, states, orders),
            "identical_option": identical_option_control(predictor, level_texts[0], k, states, identical),
            "caveat": ORDERED_SCALE_CAVEAT}


def build_predictor(args):
    """The predictor the flags name: LocalPredictor(--run) or RemotePredictor(--remote), exactly what
    kev.benchmark scores with. Imported lazily so the checks themselves stay importable without a checkpoint."""
    if bool(args.run) == bool(args.remote):
        raise SystemExit("pass exactly one of --run <dir|hub-id> or --remote <url>")
    from kev.device import default_device
    if args.run:
        from kev.predictors import LoadOptions, LocalPredictor
        temperature = 1.0 if args.raw_logits else None
        return LocalPredictor(args.run, args.device or default_device(), opts=LoadOptions(temperature=temperature))
    from kev.predictors import RemotePredictor
    return RemotePredictor(args.remote, model=args.model, api_key=args.api_key)


def main(argv=None):
    parse = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parse.add_argument("--run", help="a run directory or Hub id[@rev] to score in-process")
    parse.add_argument("--remote", help="a TypeSafe System One-compatible endpoint's base URL")
    parse.add_argument("--model", default="kev-latest", help="the model id a --remote request names")
    parse.add_argument("--api-key", default="local", help="the bearer token a --remote request sends")
    parse.add_argument("--device", help="what LocalPredictor loads onto (default: kev.device.default_device())")
    parse.add_argument("--raw-logits", action="store_true",
                       help="score --run at head temperature 1.0 (the issue's \"raw\" column) instead of the checkpoint's")
    parse.add_argument("--levels", type=int, default=3, help="how many levels the Score question has")
    parse.add_argument("--level-texts", default=DEFAULT_LEVEL_TEXTS,
                       help=f"comma-separated level texts (default: {DEFAULT_LEVEL_TEXTS}); --levels is ignored then")
    parse.add_argument("--states", type=int, default=20, help="how many states to ask over")
    parse.add_argument("--data", help="a JSONL of records whose \"state\" fields replace the built-in templates")
    parse.add_argument("--orders-max", type=int, default=120,
                       help="ask at most this many of the K! orders (all of them when K! fits)")
    parse.add_argument("--out", help="write the JSON report here (kev.suite.write_json)")
    args = parse.parse_args(argv)

    level_texts = [t.strip() for t in args.level_texts.split(",") if t.strip()] if args.level_texts else []
    if args.levels < 1:
        raise SystemExit("--levels must be >= 1")
    if not level_texts:
        level_texts = [f"level {i + 1}" for i in range(args.levels)]
    if len(set(level_texts)) != len(level_texts):
        raise SystemExit("--level-texts must be distinct (the identical-option control needs one text to repeat)")
    data = None
    if args.data:
        with open(args.data, encoding="utf-8") as handle:
            data = [json.loads(line) for line in handle if line.strip()]

    predictor = build_predictor(args)
    report = measure(predictor, level_texts, states=[None] * args.states if args.data is None else None,
                     data=data, orders_max=args.orders_max)
    report["target"] = args.run or args.remote
    report["raw_logits"] = bool(args.raw_logits and args.run)
    print(json.dumps(report, indent=2))
    if args.out:
        from kev.suite import write_json
        write_json(args.out, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
