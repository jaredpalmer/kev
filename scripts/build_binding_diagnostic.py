"""Diagnostic for two training-data hypotheses raised by the calibration failure audit (PLAN.md, round 3):

  H2 (role binding): the model decides "X is the same person as Y" by whether a name recurs anywhere in the case, not by
      comparing the two named roles. Strata: match_true, mismatch_clean, mismatch_decoy (a compared name reappears in an
      unrelated role), plus mismatch_decoy_pair (two match atoms sharing one name).
  H1 (elapsed): the model does not compute date differences; a stated day count is what it can use. Strata: elapsed atoms
      rendered plainly vs with an explicit day-count sentence appended to the case.

Eval only. Generated from kev.composition with a fresh seed; states are checked against every frozen suite and the round-3
training corpus. Labels come from evaluate_rule (code), never from a model.

    uv run python scripts/build_binding_diagnostic.py --out evals/diagnostics/binding-v1.jsonl
"""
import argparse, hashlib, json, random, sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.composition import POLICY_WRAPPERS, evaluate_rule, render_rule
from kev.data import materialize
from kev.model import fits, load_tokenizer
from kev.suite import ADMISSION_TOKENIZER, digest, load_split, write_json, write_jsonl

NAMES = ["Mira", "Noah", "Aiko", "Ravi", "Sana", "Elin", "Tomas", "Kofi"]
NOUNS = ["request", "account", "package", "review", "member", "shipment", "entry", "case"]
TREES = {"and_match_num": ("and", 0, 1), "or_match_num": ("or", 0, 1), "if_match": ("if", 0, 1, 2)}


def state_hashes():
    seen = set()
    for path in (ROOT / "evals").rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); s = r.get("state")
                if isinstance(s, dict) and "policy" in s and "case" in s:
                    seen.add(hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest())
    return seen


def record(tree, atoms, facts, order, style, rng, stratum, tag, extra_sentences=()):
    label = evaluate_rule(tree, atoms, facts)
    assert label is not None
    policy = POLICY_WRAPPERS[style].format(rule=render_rule(tree, atoms, style))
    facts_text = " ".join(f"The {k} is {'yes' if facts[k] is True else 'no' if facts[k] is False else facts[k]}." for k in order)
    state = {"policy": policy, "case": " ".join([facts_text, *extra_sentences]).strip()}
    keys = ["accept", "reject"]; rng.shuffle(keys)
    return {"state": state, "questions": {"decision": {"type": "choice", "instructions": "Apply the policy to this case.",
            "criteria": {k: "The policy permits this case" if k == "accept" else "The policy does not permit this case" for k in keys},
            "label": "accept" if label else "reject", "src": f"binding_{stratum}"}},
            "_meta": {"source": "binding_diagnostic", "stratum": stratum, "tag": tag, "tree": tree, "atoms": atoms, "facts": facts}}


def numeric_atom(prefix, rng):
    kind = rng.choice(["lt", "le", "gt", "ge"]); t = rng.randint(5, 60)
    return {"kind": kind, "fields": [f"{prefix} value"], "threshold": t}, [t - 1, t + 1, t + 10]


def binding_records(n_per, seed):
    rng = random.Random(f"{seed}:binding"); out = []
    for i in range(n_per):
        for tname, tree in TREES.items():
            nouns = rng.sample(NOUNS, 4); people = rng.sample(NAMES, 4)
            m = {"kind": "match", "fields": [f"{nouns[0]} signer", f"{nouns[0]} designated approver"], "threshold": 0}
            num, dom = numeric_atom(nouns[1], rng)
            atoms = [m, num] + ([numeric_atom(nouns[2], rng)[0]] if tname == "if_match" else [])
            style = rng.choice((0, 1, 3, 4))
            for stratum in ("match_true", "mismatch_clean", "mismatch_decoy_role", "mismatch_decoy_pair"):
                facts = {a["fields"][0]: rng.choice(dom) for a in atoms[1:]}
                facts["routing reference"] = rng.randint(100, 999)
                a_tree, a_atoms = tree, list(atoms)
                if stratum == "match_true":
                    facts[m["fields"][0]] = facts[m["fields"][1]] = people[0]
                else:
                    facts[m["fields"][0]], facts[m["fields"][1]] = people[0], people[1]
                if stratum == "mismatch_decoy_role":
                    facts[f"{nouns[3]} reviewer"] = rng.choice(people[:2])          # a compared name recurs in an unrelated role
                if stratum == "mismatch_decoy_pair":
                    m2 = {"kind": "match", "fields": [f"{nouns[2]} signer", f"{nouns[2]} designated approver"], "threshold": 0}
                    a_atoms = [m, num, m2] if tname != "if_match" else [m, num, m2]
                    a_tree = ("and", ("and", 0, 1), 2) if tname == "and_match_num" else ("or", ("and", 0, 1), 2) if tname == "or_match_num" else ("if", 0, 1, 2)
                    facts[m2["fields"][0]] = facts[m2["fields"][1]] = people[1]  # the second pair matches, using a name from the first pair
                order = list(facts); rng.shuffle(order)
                out.append(record(a_tree, a_atoms, facts, order, style, rng, stratum, f"{tname}/{i}"))
    return out


def elapsed_records(n_per, seed):
    rng = random.Random(f"{seed}:elapsed"); out = []
    for i in range(n_per):
        nouns = rng.sample(NOUNS, 3); t = rng.randint(5, 60)
        e = {"kind": "elapsed", "fields": [f"{nouns[0]} start date", f"{nouns[0]} end date"], "threshold": t}
        num, dom = numeric_atom(nouns[1], rng)
        tree = rng.choice([("and", 0, 1), ("or", 0, 1)])
        start = date(2027, rng.randint(1, 8), rng.randint(1, 28)); gap = t + rng.choice([-7, -1, 0, 1, 10])
        facts = {e["fields"][0]: start.isoformat(), e["fields"][1]: (start + timedelta(days=gap)).isoformat(), num["fields"][0]: rng.choice(dom), "routing reference": rng.randint(100, 999)}
        order = list(facts); rng.shuffle(order); style = rng.choice((0, 1, 3, 4))
        out.append(record(tree, [e, num], facts, order, style, rng, "elapsed_plain", f"{i}/gap{gap - t:+d}"))
        out.append(record(tree, [e, num], facts, order, style, rng, "elapsed_daycount", f"{i}/gap{gap - t:+d}",
                          extra_sentences=(f"The {e['fields'][1]} is {gap} days after the {e['fields'][0]}.",)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", default="binding-diag-2026-09-21"); ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(); out = ROOT / a.out
    if out.exists(): raise FileExistsError(out)
    recs = binding_records(a.n, a.seed) + elapsed_records(a.n, a.seed)
    seen = state_hashes(); toks = [load_tokenizer(*ADMISSION_TOKENIZER)]
    kept = []
    for r in recs:
        h = hashlib.sha256(json.dumps(r["state"], sort_keys=True).encode()).hexdigest()
        if h in seen: continue
        if not fits(materialize(r), *toks): continue
        r["_meta"]["text_sha256"] = h; seen.add(h); kept.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, kept)
    strata = Counter(r["_meta"]["stratum"] for r in kept); labels = Counter((r["_meta"]["stratum"], r["questions"]["decision"]["label"]) for r in kept)
    manifest = {"records": len(kept), "dropped": len(recs) - len(kept), "seed": a.seed, "sha256": digest(out), "strata": dict(strata),
                "labels_by_stratum": {f"{s}/{l}": n for (s, l), n in sorted(labels.items())}, "source": "binding_diagnostic (eval only; kev.composition generator, code labels)",
                "code_sha256": digest(Path(__file__))}
    write_json(out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
