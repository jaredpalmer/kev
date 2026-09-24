"""hard-v1 families other than long_policy (scripts/hard_v1_policy.py): tradeoff, probability, multi_hop,
temporal_numeric, judge and ambiguous. Each family is `generate(ctx, t) -> item | None` plus `solve(facts) -> {qid:
canonical answer}`; the builder (scripts/build_hard_v1.py) turns an item into a record and takes every label from
`solve`, never from the generator. Six surface templates per family; templates 4 and 5 are held out of training.
"""
import math
import re
from datetime import date, datetime, timedelta
from fractions import Fraction

from scripts import hard_v1_policy as policy
from scripts.hard_v1_common import (CITIES, COMPANIES, business_days_after, choice_q, day, dollars, is_business_day, money,
                                    noul_q, people, roman, score_q, value_q, words)

N_TEMPLATES = 6


def frame(ctx, t, title, lines, speaker="Ops lead"):
    """Render a list of fact sentences in template t's layout: email prose, ticket bullets, JSON object, chat, memo or a
    numbered form. The facts are identical across layouts; only the surface changes."""
    rng = ctx.rng
    if t == 0:
        return f"Subject: {title}\n\nHi all,\n\n" + " ".join(lines) + "\n\nThanks."
    if t == 1:
        return f"TICKET {rng.randint(1000, 99999)}: {title}\n" + "\n".join(f"- {x}" for x in lines)
    if t == 2:
        return {"topic": title, "facts": list(lines)}
    if t == 3:
        out = []
        for i, x in enumerate(lines):
            out.append(f"{speaker}: {x}")
            if i % 2 == 1 and i < len(lines) - 1: out.append(rng.choice(["Analyst: ok, go on.", "Analyst: got it.", "Analyst: noted."]))
        return "\n".join(out)
    if t == 4:
        return f"MEMO\nRe: {title}\n\n" + "\n\n".join(" ".join(lines[i:i + 3]) for i in range(0, len(lines), 3))
    return f"CASE FILE: {title}\n" + "\n".join(f"{i + 1}) {x}" for i, x in enumerate(lines))


def slug(name):
    return re.sub(r"_+", "_", "".join(c if c.isalnum() else "_" for c in name.lower())).strip("_")


def per_mille(p):
    """A probability (Fraction) as tenths of a percent, rounded half up: the canonical value of a percentage option."""
    return int(math.floor(p * 1000 + Fraction(1, 2)))


def fmt_pm(v):
    return f"{v / 10:.1f}%"


def prob_phrases(p, style):
    """Surface forms of a whole-percent probability p (Fraction) in a template's number style: a chance ("a 35% chance"),
    a rate ("35% of the time") and a share of a population ("35% of {unit}")."""
    n = int(p * 100)
    f = Fraction(n, 100)
    if style == "dec":
        d = f"{float(p):.2f}".rstrip("0").rstrip(".")
        return {"chance": f"a probability of {d}", "rate": f"with probability {d}", "share": f"a proportion {d} of all {{unit}}"}
    if style == "ratio":
        return {"chance": f"a {f.numerator}-in-{f.denominator} chance", "rate": f"{f.numerator} times in {f.denominator}", "share": f"{f.numerator} in {f.denominator} {{unit}}"}
    if style == "words":
        w = f"{words(n)} percent"
        return {"chance": f"a {w} chance", "rate": f"{w} of the time", "share": f"{w} of {{unit}}"}
    if style == "outof":
        return {"chance": f"a {n}-in-100 chance", "rate": f"in {n} out of every 100 cases", "share": f"{n} out of every 100 {{unit}}"}
    return {"chance": f"a {n}% chance", "rate": f"{n}% of the time", "share": f"{n}% of {{unit}}"}


# ================================================================================================================ tradeoff
UNITS = {"usd": lambda v: dollars(v), "ms": lambda v: f"{v} ms", "pct": lambda v: f"{v}%", "days": lambda v: f"{v} days",
         "kg": lambda v: f"{v / 10:.1f} kg", "hours": lambda v: f"{v} hours", "weeks": lambda v: f"{v} weeks", "min": lambda v: f"{v} minutes",
         "score": lambda v: f"{v}/10", "people": lambda v: f"{v} desks", "jobs": lambda v: f"{v} jobs", "uptime": lambda v: f"{v / 100:.2f}%", "bps": lambda v: f"{v / 100:.2f}%"}
# numeric attribute: (key, label, unit, (lo, hi, step), better)
TRADE_CONTEXTS = [
    {"what": "a managed cloud provider for the analytics platform", "noun": "provider", "tco": True,
     "names": ["Nimbus", "Stratus Cloud", "Arcadia Hosting", "Blueshift", "Corelink", "Driftnet", "Halcyon Compute"],
     "num": [("cost", "monthly cost", "usd", (800, 5000, 50), "lower"), ("latency", "p95 latency", "ms", (40, 260, 5), "lower"),
             ("uptime", "uptime commitment", "uptime", (9950, 9999, 1), "higher")],
     "bools": [("soc2", "SOC 2 Type II report"), ("eu", "EU data residency")]},
    {"what": "a freight carrier for the Rotterdam to Milan lane", "noun": "carrier", "tco": False,
     "names": ["Rhenus Line", "Alpfreight", "Kestrel Haulage", "Meridian Cargo", "Portway", "Vireo Transport"],
     "num": [("cost", "price per shipment", "usd", (900, 4000, 25), "lower"), ("transit", "transit time", "days", (2, 12, 1), "lower"),
             ("damage", "damage rate", "bps", (20, 300, 5), "lower")],
     "bools": [("reefer", "temperature-controlled trailers"), ("tracking", "live GPS tracking")]},
    {"what": "a laptop model for the staff hardware refresh", "noun": "model", "tco": False,
     "names": ["Aero 14", "Slate Pro", "Vector X1", "Orbit 13", "Keystone 15", "Nova Air"],
     "num": [("cost", "unit price", "usd", (700, 2200, 10), "lower"), ("battery", "battery life", "hours", (6, 20, 1), "higher"),
             ("weight", "weight", "kg", (10, 24, 1), "lower")],
     "bools": [("warranty", "three-year on-site warranty"), ("lte", "built-in LTE")]},
    {"what": "a managed database service for the orders system", "noun": "service", "tco": True,
     "names": ["TideDB", "Quarry SQL", "Lumen Data", "Polar Store", "Granite DB", "Wren Cloud SQL"],
     "num": [("cost", "monthly cost", "usd", (600, 4000, 50), "lower"), ("latency", "write latency", "ms", (2, 40, 1), "lower"),
             ("rpo", "recovery point objective", "min", (1, 60, 1), "lower")],
     "bools": [("pitr", "point-in-time recovery"), ("multiregion", "multi-region replicas")]},
    {"what": "a marketing agency for the spring campaign", "noun": "agency", "tco": False,
     "names": ["Brightside Creative", "Northlight Studio", "Paper Kite", "Signal & Co", "Tallgrass Media", "Wildfern Agency"],
     "num": [("cost", "fee", "usd", (5000, 30000, 500), "lower"), ("lead", "lead time", "weeks", (2, 12, 1), "lower"),
             ("portfolio", "portfolio rating", "score", (3, 10, 1), "higher")],
     "bools": [("b2b", "B2B campaign experience"), ("inhouse", "in-house video production")]},
    {"what": "an office lease for the new Leeds team", "noun": "office", "tco": True,
     "names": ["Canal Wharf", "Park Row", "Mill Yard", "Station House", "Albion Court", "Wellington Place"],
     "num": [("cost", "monthly rent", "usd", (4000, 15000, 100), "lower"), ("commute", "average commute", "min", (15, 70, 1), "lower"),
             ("desks", "capacity", "people", (20, 80, 1), "higher")],
     "bools": [("parking", "on-site parking"), ("access", "step-free access")]},
    {"what": "a CI/CD provider for the platform team", "noun": "provider", "tco": True,
     "names": ["Buildkite Lite", "Pipewright", "Relay CI", "Forge Runner", "Tessel Build", "Loop Deploy"],
     "num": [("cost", "monthly cost", "usd", (300, 3000, 25), "lower"), ("build", "median build time", "min", (4, 30, 1), "lower"),
             ("concurrency", "parallel jobs", "jobs", (4, 64, 2), "higher")],
     "bools": [("selfhosted", "self-hosted runners"), ("sso", "SSO login")]},
    {"what": "a payment processor for the online store", "noun": "processor", "tco": False,
     "names": ["Paylane", "Tillpoint", "Coinbridge", "Settle Pay", "Ledgerly", "Quickcheck"],
     "num": [("cost", "fee per transaction", "bps", (150, 350, 5), "lower"), ("payout", "payout delay", "days", (1, 7, 1), "lower"),
             ("uptime", "uptime commitment", "uptime", (9950, 9999, 1), "higher")],
     "bools": [("multicurrency", "multi-currency settlement"), ("disputes", "a chargeback protection service")]},
]
SCORES = ["reliability", "support quality", "ease of use", "security", "scalability", "vendor stability", "ease of integration", "documentation"]
WEIGHTS = [(50, 30, 20), (40, 40, 20), (60, 25, 15), (45, 35, 20), (40, 35, 25), (70, 20, 10)]
TRADE_Q = ["Which {noun} should be chosen?", "Which {noun} does the decision rule select?", "Which option should the team pick?",
           "Following the stated requirements and priorities, which {noun} wins?", "Which {noun} best satisfies the brief?",
           "Given everything above, which {noun} is the right choice?"]
NONE_DESC = ["None of the options meets every requirement", "No option qualifies", "Reject all options: none meets the requirements"]


def trade_attr(ctx, spec):
    key, label, unit, (lo, hi, step), better = spec
    return {"key": key, "label": label, "unit": unit, "better": better, "lo": lo, "hi": hi, "step": step}


def meets(option, cons):
    for c in cons:
        v = option[c["attr"]]
        if c["op"] == "<=" and not v <= c["value"]: return False
        if c["op"] == ">=" and not v >= c["value"]: return False
        if c["op"] == "is" and v is not True: return False
    return True


def trade_rank(f, name):
    o = f["options"][name]
    if f["rule"] == "lex":
        return tuple((o[a] if b == "lower" else -o[a]) for a, b in f["priority"])
    if f["rule"] == "weighted":
        return (-sum(w * o[s] for s, w in f["weights"].items()),)
    return (o["setup"] + 12 * f["years"] * o["cost"],)


def solve_tradeoff(f):
    feasible = [n for n in f["options"] if meets(f["options"][n], f["constraints"])]
    out = {}
    if not feasible:
        out["choice"] = "none_qualifies"
    else:
        ranks = sorted((trade_rank(f, n), n) for n in feasible)
        if len(ranks) > 1 and ranks[0][0] == ranks[1][0]: raise ValueError("tie")
        out["choice"] = slug(ranks[0][1])
    if f.get("check"): out["meets"] = meets(f["options"][f["check"]], f["constraints"])
    return out


def gen_tradeoff(ctx, t):
    rng = ctx.rng
    c = rng.choice(TRADE_CONTEXTS)
    rule = ctx.pick(f"trade_rule_{c['tco']}", ["lex", "lex", "weighted", "tco"] if c["tco"] else ["lex", "lex", "weighted", "weighted"])
    if rule == "tco" and not c["tco"]: raise AssertionError("total cost rule needs a recurring cost")
    none_case = rng.random() < 0.07
    n = rng.randint(3, 5)
    names = rng.sample(c["names"], n)
    attrs = [trade_attr(ctx, s) for s in c["num"]]
    opts = {}
    for name in names:
        o = {a["key"]: rng.randrange(a["lo"], a["hi"] + 1, a["step"]) if a["step"] > 1 else rng.randint(a["lo"], a["hi"]) for a in attrs}
        for b, _ in c["bools"]: o[b] = rng.random() < 0.65
        opts[name] = o
    scores = rng.sample(SCORES, 3) if rule == "weighted" else []
    for name in names:
        for s in scores: opts[name][s] = rng.randint(3, 10)
        if rule == "tco": opts[name]["setup"] = rng.randrange(0, 24001, 500)
    # priorities
    if rule == "lex":
        prim, tie = rng.sample(attrs, 2)
        priority = [(prim["key"], prim["better"]), (tie["key"], tie["better"])]
        cand = [a for a in attrs if a is not prim]
    else:
        priority = []
        cand = [a for a in attrs if a["key"] != "cost"] if rule == "tco" else list(attrs)
    # constraints: one or two, made to bind
    cons = []
    kinds = rng.sample(["num", "bool"], rng.randint(1, 2)) if cand else ["bool"]
    for kind in kinds:
        if kind == "bool":
            b, label = rng.choice(c["bools"])
            cons.append({"attr": b, "op": "is", "value": True, "label": label})
        else:
            a = rng.choice(cand)
            vals = sorted({opts[nm][a["key"]] for nm in names})
            if len(vals) < 2: continue
            i = rng.randrange(len(vals) - 1)
            cut = (vals[i] + vals[i + 1]) // 2 if vals[i + 1] - vals[i] > 1 else vals[i]
            if a["better"] == "lower":
                cons.append({"attr": a["key"], "op": "<=", "value": max(vals[i], cut - cut % max(1, a["step"])), "label": a["label"], "unit": a["unit"]})
            else:
                cons.append({"attr": a["key"], "op": ">=", "value": vals[i + 1], "label": a["label"], "unit": a["unit"]})
    if not cons: return None
    if rule == "lex" and rng.random() < 0.35:   # a tie on the first priority among feasible options, broken by the second
        feas = [nm for nm in names if meets(opts[nm], cons)]
        if len(feas) >= 2:
            best = min(feas, key=lambda nm: opts[nm][priority[0][0]] * (1 if priority[0][1] == "lower" else -1))
            other = rng.choice([nm for nm in feas if nm != best])
            opts[other][priority[0][0]] = opts[best][priority[0][0]]
    if none_case:
        for nm in names:
            if meets(opts[nm], cons):
                cc = rng.choice(cons)
                if cc["op"] == "is": opts[nm][cc["attr"]] = False
                elif cc["op"] == "<=": opts[nm][cc["attr"]] = cc["value"] + rng.randint(1, 5) * max(1, next(a["step"] for a in attrs if a["key"] == cc["attr"]))
                else: opts[nm][cc["attr"]] = cc["value"] - rng.randint(1, 3) * max(1, next(a["step"] for a in attrs if a["key"] == cc["attr"]))
    weights = dict(zip(scores, rng.choice(WEIGHTS))) if rule == "weighted" else {}
    years = rng.choice([2, 3, 4, 5]) if rule == "tco" else None
    want = ctx.coin("trade_meets")   # the option the noul asks about meets the requirements half the time
    pool = [nm for nm in names if meets(opts[nm], cons) == want]
    check = rng.choice(pool or names)
    f = {"rule": rule, "options": opts, "constraints": [{k: v for k, v in x.items() if k in ("attr", "op", "value")} for x in cons],
         "priority": priority, "weights": weights, "years": years, "check": check}
    try:
        truth = solve_tradeoff(f)
    except ValueError:
        return None
    if rule == "weighted" and truth["choice"] != "none_qualifies":   # a clear winner: at least 0.2 points on the 10-point scale
        feas = sorted((sum(w * opts[nm][s] for s, w in weights.items()) for nm in names if meets(opts[nm], cons)), reverse=True)
        if len(feas) > 1 and feas[0] - feas[1] < 20: return None
    # naive answer: best on the first criterion ignoring the requirements; keep most records where it is wrong
    naive = min(names, key=lambda nm: trade_rank({**f, "constraints": []}, nm))
    if slug(naive) == truth["choice"] and rng.random() < 0.7: return None
    # requirement text
    def req_text(x):
        if x["op"] == "is": return f"it must offer {x['label']}"
        v = UNITS[x["unit"]](x["value"])
        return f"its {x['label']} must be at most {v}" if x["op"] == "<=" else f"its {x['label']} must be at least {v}"
    reqs = [req_text(x) for x in cons]
    if rule == "lex":
        (pa, pb), (ta, tb) = priority
        lab = {a["key"]: a["label"] for a in attrs}
        word = lambda b: "lowest" if b == "lower" else "highest"
        prio = (f"Among the options that meet every requirement, choose the one with the {word(pb)} {lab[pa]}; "
                f"if two or more are tied on {lab[pa]}, choose the one with the {word(tb)} {lab[ta]}.")
    elif rule == "weighted":
        prio = (f"Each option has been scored from 1 to 10 on {', '.join(scores[:-1])} and {scores[-1]} (higher is better). Among the options "
                f"that meet every requirement, choose the highest weighted score using the weights "
                + ", ".join(f"{w}% {s}" for s, w in weights.items()) + ".")
    else:
        prio = (f"Among the options that meet every requirement, choose the lowest total cost over {years} years: the one-time setup fee "
                f"plus {years} years of the monthly cost.")
    shown = [a for a in attrs] + [{"key": s, "label": f"{s} score", "unit": "score"} for s in scores]
    if rule == "tco": shown.append({"key": "setup", "label": "one-time setup fee", "unit": "usd"})
    def val(o, a):
        return UNITS[a["unit"]](o[a["key"]])
    def yn(v): return "yes" if v else "no"
    blabels = dict(c["bools"])
    rows = {nm: {**{a["label"]: val(opts[nm], a) for a in shown}, **{blabels[b]: yn(opts[nm][b]) for b in blabels}} for nm in names}
    intro = f"We need to choose {c['what']}."
    reqline = "Hard requirements: " + "; ".join(reqs) + "."
    if t == 0:
        cols = list(next(iter(rows.values())))
        table = "| option | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n" + "\n".join(
            f"| {nm} | " + " | ".join(rows[nm][k] for k in cols) + " |" for nm in names)
        state = f"{intro}\n\n{reqline}\n{prio}\n\n{table}"
    elif t == 1:
        state = intro + "\n" + reqline + "\n" + prio + "\n" + "\n".join(f"- {nm}: " + "; ".join(f"{k} {v}" for k, v in rows[nm].items()) for nm in names)
    elif t == 2:
        paras = [f"{nm} quoted " + ", ".join(f"{k} of {v}" if v not in ("yes", "no") else (f"{'with' if v == 'yes' else 'without'} {k}") for k, v in rows[nm].items()) + "." for nm in names]
        state = f"Hi team,\n\n{intro} After the calls this week: " + " ".join(paras) + f"\n\nReminder of the brief: {'; '.join(reqs)}. {prio}\n\nCheers"
    elif t == 3:
        state = {"decision": intro, "hard_requirements": reqs, "selection_rule": prio, "options": rows}
    elif t == 4:
        state = (f"SPEC SHEET: {c['what']}\n" + "\n".join(f"{nm} | " + " | ".join(f"{k}={v}" for k, v in rows[nm].items()) for nm in names)
                 + "\n" + "\n".join(f"R{i + 1}: {r}" for i, r in enumerate(reqs)) + f"\nSelection: {prio}")
    else:
        lines = [f"Procurement meeting notes: {intro}"]
        who = people(rng, 3)
        lines.append(f"{who[0]} (finance) and {who[1]} (security) set the hard requirements: " + "; ".join(reqs) + ".")
        lines.append(f"{who[2]} summarised the quotes. " + " ".join(f"{nm}: " + ", ".join(f"{k} {v}" for k, v in rows[nm].items()) + "." for nm in names))
        lines.append(f"Agreed rule: {prio}")
        state = "\n".join(lines)
    options = {slug(nm): None for nm in names}
    if none_case or rng.random() < 0.25:
        options["none_qualifies"] = rng.choice(NONE_DESC)
    if truth["choice"] not in options: return None
    src = "hard_tradeoff"
    q1, o1 = choice_q(ctx, TRADE_Q[t].format(noun=c["noun"]), options, truth["choice"], src)
    q2 = noul_q(rng.choice([f"Does {check} meet every hard requirement?", f"Is {check} compliant with all the hard requirements?",
                            f"Does {check} pass the hard requirements?"]), src)
    f["options_q"] = {"choice": o1}
    return {"state": state, "questions": {"choice": q1, "meets": q2}, "facts": f, "meta": {"subtype": rule, "context": c["noun"], "none_case": none_case}}


# ============================================================================================================= probability
PROB_STYLE = ["pct", "dec", "ratio", "pct", "words", "outof"]
BUCKETS = ["Under 10%", "10% to 30%", "30% to 50%", "50% to 70%", "70% to 90%", "Over 90%"]
EDGES = [Fraction(1, 10), Fraction(3, 10), Fraction(1, 2), Fraction(7, 10), Fraction(9, 10)]
# (context, unit, class, the system's action in the present tense, in the past participle)
BAYES = [("A fraud model flags card transactions", "transactions", "fraudulent", "flags", "flagged"),
         ("An inspection camera flags units on the production line", "units", "defective", "flags", "flagged"),
         ("A security system raises alerts on login attempts", "login attempts", "malicious", "raises an alert on", "raised an alert on"),
         ("A churn model marks customers as at risk", "customers", "going to cancel within 90 days", "marks", "marked"),
         ("A spam filter quarantines inbound emails", "emails", "spam", "quarantines", "quarantined"),
         ("A credit model flags loan applications", "applications", "going to default", "flags", "flagged")]


def bucket(p):
    return sum(p >= e for e in EDGES)


def near_edge(p, tol=Fraction(15, 1000)):
    return any(abs(p - e) < tol for e in EDGES)


def solve_probability(f):
    k, P = f["kind"], {x: Fraction(v) for x, v in f["p"].items()}
    out = {}
    if k == "ev":
        evs = {n: sum(Fraction(p) * v for p, v in o["outcomes"]) - o["cost"] for n, o in f["projects"].items()}
        best = sorted(evs.items(), key=lambda kv: -kv[1])
        if best[0][1] == best[1][1]: raise ValueError("tie")
        out["choice"] = slug(best[0][0])
        out["value"] = int(evs[f["asked"]])
    elif k == "bayes":
        b, s, fp = P["base"], P["sens"], P["fpr"]
        ppv = b * s / (b * s + (1 - b) * fp)
        out["value"] = per_mille(ppv)
        out["more_likely"] = ppv > Fraction(1, 2)
        out["bucket"] = bucket(ppv)
    elif k == "independent":
        ps = [Fraction(x) for x in f["ps"]]
        allp = math.prod(ps)
        val = {"all": allp, "any_fail": 1 - allp, "any": 1 - math.prod(1 - x for x in ps)}[f["ask"]]
        out["value"] = per_mille(val)
        out["bucket"] = bucket(val)
    elif k == "compare":
        a = Fraction(f["a"]["p1"]) * Fraction(f["a"]["p2"]) if f["a"]["op"] == "and" else 1 - (1 - Fraction(f["a"]["p1"])) * (1 - Fraction(f["a"]["p2"]))
        b = Fraction(f["b"])
        out["choice"] = "first_event" if a > b else "second_event" if b > a else "equally_likely"
    elif k == "table":
        n = f["counts"]   # {row: {col: count}}
        r, c = f["row"], f["col"]
        val = {"col_given_row": Fraction(n[r][c], sum(n[r].values())), "row_given_col": Fraction(n[r][c], sum(n[x][c] for x in n))}[f["ask"]]
        out["value"] = per_mille(val)
    return out


def gen_probability(ctx, t):
    rng = ctx.rng
    kind = ctx.pick("prob_kind", ["ev", "bayes", "independent", "compare", "table"])
    style = PROB_STYLE[t]
    ph = lambda p, kind: prob_phrases(Fraction(p), style)[kind]
    src = "hard_probability"
    qwords = ["", "Using the figures above: ", "Based only on these numbers: ", "", "Per the figures given, ", "Question: "][t]
    if kind == "ev":
        n = rng.randint(3, 4)
        names = rng.sample(["Project Atlas", "Project Beacon", "Project Cedar", "Project Delta", "Project Ember", "Project Fjord", "Project Garnet"], n)
        projects = {}
        for nm in names:
            k = rng.randint(2, 3)
            cuts = sorted(rng.sample(range(5, 100, 5), k - 1))
            probs = [a - b for a, b in zip(cuts + [100], [0] + cuts)]
            payoffs = sorted([rng.randrange(-200, 900, 10) * 1000 for _ in range(k)], reverse=True)
            projects[nm] = {"outcomes": [(f"{p}/100", v) for p, v in zip(probs, payoffs)], "cost": rng.randrange(0, 150, 10) * 1000}
        asked = rng.choice(names)
        f = {"kind": "ev", "p": {}, "projects": projects, "asked": asked}
        try: truth = solve_probability(f)
        except ValueError: return None
        evs = {nm: sum(Fraction(p) * v for p, v in o["outcomes"]) - o["cost"] for nm, o in projects.items()}
        ranked = sorted(evs.values(), reverse=True)
        if ranked[0] - ranked[1] < 5000: return None
        naive = max(names, key=lambda nm: max(v for _, v in projects[nm]["outcomes"]))
        if slug(naive) == truth["choice"] and rng.random() < 0.7: return None
        lines = [f"The planning team must fund exactly one of {n} projects and wants the highest expected net value (expected payoff minus up-front cost)."]
        for nm in names:
            o = projects[nm]
            outs = "; ".join(f"{ph(p, 'chance')} of {'a gain' if v >= 0 else 'a loss'} of {dollars(abs(v))}" for p, v in o["outcomes"])
            lines.append(f"{nm} costs {dollars(o['cost'])} up front and has {outs}.")
        q1, o1 = choice_q(ctx, qwords + "Which project has the highest expected net value?", {slug(nm): None for nm in names}, truth["choice"], src)
        ev_a = evs[asked]; o = projects[asked]
        distract = [int(ev_a + o["cost"]), int(max(v for _, v in o["outcomes"]) - o["cost"]), int(sum(v for _, v in o["outcomes"]) / len(o["outcomes"]) - o["cost"]),
                    int(ev_a - o["cost"]), int(-ev_a)]
        q2, o2 = value_q(ctx, qwords + f"What is the expected net value of {asked}?", int(ev_a), distract, lambda v: ("-" if v < 0 else "") + dollars(abs(v)), src)
        f["options_q"] = {"choice": o1, "value": o2}
        return {"state": frame(ctx, t, "Project funding decision", lines, "Planner"), "questions": {"choice": q1, "value": q2}, "facts": f, "meta": {"subtype": kind}}
    if kind == "bayes":
        ctxt, unit, cls, verb, done = rng.choice(BAYES)
        want = ctx.coin("bayes")
        for _ in range(200):
            base = rng.choice([1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60])
            sens = rng.choice([70, 75, 80, 85, 90, 95, 98, 99])
            fpr = rng.choice([1, 2, 3, 5, 8, 10, 15, 20])
            ppv = Fraction(base * sens, base * sens + (100 - base) * fpr)
            if (ppv > Fraction(1, 2)) == want and abs(ppv - Fraction(1, 2)) > Fraction(3, 100) and not near_edge(ppv): break
        else:
            return None
        p = {"base": f"{base}/100", "sens": f"{sens}/100", "fpr": f"{fpr}/100"}
        lines = [f"{ctxt}.", "Historically, " + ph(Fraction(base, 100), "share").format(unit=unit) + f" are {cls}.",
                 f"When one of the {unit} is {cls}, the system {verb} it {ph(Fraction(sens, 100), 'rate')}.",
                 f"When one of the {unit} is not {cls}, the system still {verb} it {ph(Fraction(fpr, 100), 'rate')}."]
        if t in (1, 3):
            tail = lines[1:]; rng.shuffle(tail); lines = lines[:1] + tail
        f = {"kind": "bayes", "p": p}
        truth = solve_probability(f)
        target = f"one of the {unit} that the system has {done}"
        B, S, FP = Fraction(base, 100), Fraction(sens, 100), Fraction(fpr, 100)
        distract = [per_mille(S), per_mille(1 - FP), per_mille(B), per_mille(B * S), per_mille(S - FP), per_mille(B * S / (B * S + FP))]
        q1, o1 = value_q(ctx, qwords + f"What is the probability that {target} is actually {cls}?", truth["value"], distract, fmt_pm, src)
        if rng.random() < 0.5:
            q2 = noul_q(qwords + f"Is {target} more likely than not to be {cls}?", src)
            qid2 = "more_likely"
        else:
            q2 = score_q(qwords + f"How likely is it that {target} is {cls}?", BUCKETS, src)
            qid2 = "bucket"
        f["options_q"] = {"value": o1}
        f["qid2"] = qid2
        return {"state": frame(ctx, t, "Alert precision", lines, "Analyst"), "questions": {"value": q1, qid2: q2}, "facts": f, "meta": {"subtype": kind}}
    if kind == "independent":
        k = rng.randint(2, 4)
        ps = [rng.choice([60, 70, 75, 80, 85, 90, 92, 95, 97, 99]) for _ in range(k)]
        ask = rng.choice(["all", "any_fail", "any"])
        steps = rng.sample(["the payment gateway", "the fraud check", "the warehouse API", "the courier booking", "the address validator", "the tax service"], k)
        if ask == "any":
            lines = [f"An order is confirmed if at least one of {k} redundant suppliers can fill it; each supplier's availability is independent of the others."]
            lines += [f"Supplier {chr(65 + i)} is able to fill an order {ph(Fraction(p, 100), 'rate')}." for i, p in enumerate(ps)]
            ask_text = "What is the probability that at least one supplier can fill the order?"
        else:
            lines = [f"A checkout succeeds only if all {k} independent steps succeed: {', '.join(steps)}."]
            lines += [f"{s[0].upper() + s[1:]} succeeds {ph(Fraction(p, 100), 'rate')}." for s, p in zip(steps, ps)]
            ask_text = "What is the probability that a checkout succeeds?" if ask == "all" else "What is the probability that a checkout fails?"
        f = {"kind": "independent", "p": {}, "ps": [f"{p}/100" for p in ps], "ask": ask}
        truth = solve_probability(f)
        val = Fraction(truth["value"], 1000)
        if near_edge(val): return None
        P = [Fraction(p, 100) for p in ps]
        prod, prodf = math.prod(P), math.prod(1 - x for x in P)
        distract = [per_mille(min(P)), per_mille(1 - prod), per_mille(prod), per_mille(1 - prodf), per_mille(prodf), per_mille(sum(1 - x for x in P)), per_mille(sum(P) / len(P))]
        q1, o1 = value_q(ctx, qwords + ask_text, truth["value"], distract, fmt_pm, src)
        q2 = score_q(qwords + ask_text.replace("What is the probability", "How likely is it"), BUCKETS, src)
        f["options_q"] = {"value": o1}
        f["qid2"] = "bucket"
        return {"state": frame(ctx, t, "Reliability estimate", lines, "Engineer"), "questions": {"value": q1, "bucket": q2}, "facts": f, "meta": {"subtype": kind}}
    if kind == "compare":
        op = rng.choice(["and", "or"])
        p1, p2 = rng.choice(range(20, 95, 5)), rng.choice(range(20, 95, 5))
        a = Fraction(p1 * p2, 10000) if op == "and" else 1 - Fraction((100 - p1) * (100 - p2), 10000)
        want = ctx.pick("compare", ["first", "second", "first", "second", "equal"])
        if want == "equal":
            if (a * 100).denominator != 1: return None
            b = a
        else:
            delta = Fraction(rng.choice([1, 2, 3, 5, 8]), 100)
            b = a - delta if want == "first" else a + delta
            b = Fraction(round(b * 100), 100)
            if not 0 < b < 1 or b == a: return None
        e1, e2, e3 = rng.sample(["a new customer renews after the trial", "the shipment clears customs on the first day", "the server patch installs without errors",
                                 "the supplier delivers on time", "the audit finds no issues", "the ad campaign beats its target", "the candidate accepts the offer"], 3)
        first = f"At least one of these happens: {e1}, or {e2}" if op == "or" else f"Both of these happen: {e1}, and {e2}"
        lines = [f"There is {ph(Fraction(p1, 100), 'chance')} that {e1}.", f"There is {ph(Fraction(p2, 100), 'chance')} that {e2}; these two are independent.",
                 f"There is {ph(b, 'chance')} that {e3}."]
        f = {"kind": "compare", "p": {}, "a": {"op": op, "p1": f"{p1}/100", "p2": f"{p2}/100"}, "b": f"{b.numerator}/{b.denominator}"}
        truth = solve_probability(f)
        q1, o1 = choice_q(ctx, qwords + "Which is more likely?", {"first_event": first, "second_event": f"That {e3}",
                                                                  "equally_likely": "They are equally likely"}, truth["choice"], src)
        f["options_q"] = {"choice": o1}
        return {"state": frame(ctx, t, "Comparing odds", lines, "Analyst"), "questions": {"choice": q1}, "facts": f, "meta": {"subtype": kind}}
    # table
    row_names, col_names, what = rng.choice([(("chat", "email"), ("escalated", "not escalated"), "support tickets last quarter by channel and outcome"),
                                             (("new", "returning"), ("converted", "did not convert"), "store visitors last month by type and outcome"),
                                             (("night shift", "day shift"), ("defective", "passed"), "units inspected last week by shift and result"),
                                             (("mobile", "desktop"), ("abandoned cart", "completed purchase"), "checkout sessions yesterday by device and outcome")])
    counts = {r: {c: rng.randint(15, 400) for c in col_names} for r in row_names}
    r, c = rng.choice(row_names), col_names[0]
    ask = rng.choice(["col_given_row", "row_given_col"])
    lines = [f"Counts of {what}:"] + [f"{rr}: {counts[rr][col_names[0]]} {col_names[0]}, {counts[rr][col_names[1]]} {col_names[1]}." for rr in row_names]
    f = {"kind": "table", "p": {}, "counts": counts, "row": r, "col": c, "ask": ask}
    truth = solve_probability(f)
    tot = sum(sum(v.values()) for v in counts.values())
    distract = [per_mille(Fraction(counts[r][c], sum(counts[r].values()))), per_mille(Fraction(counts[r][c], sum(counts[x][c] for x in counts))),
                per_mille(Fraction(counts[r][c], tot)), per_mille(Fraction(sum(counts[x][c] for x in counts), tot)), per_mille(Fraction(sum(counts[r].values()), tot))]
    ask_text = (f"Among {r} records, what share were {c}?" if ask == "col_given_row" else f"Among records that were {c}, what share came from {r}?")
    q1, o1 = value_q(ctx, qwords + ask_text, truth["value"], distract, fmt_pm, src)
    f["options_q"] = {"value": o1}
    return {"state": frame(ctx, t, "Quarterly breakdown", lines, "Analyst"), "questions": {"value": q1}, "facts": f, "meta": {"subtype": kind}}


# =============================================================================================================== multi_hop
TITLES = ["Analyst", "Engineer", "Manager", "Senior Manager", "Director", "Vice President", "Chief Executive"]
RANK = {x: i for i, x in enumerate(TITLES)}
SERVICES = ["auth-service", "billing-api", "search-indexer", "notification-hub", "inventory-db", "checkout-web", "pricing-engine", "user-profile",
            "reporting-etl", "cdn-edge", "session-cache", "order-queue", "email-relay", "fraud-scorer", "catalog-api", "payments-gateway", "ledger-service", "geo-lookup"]
ENTITIES = ["Aldermoor Holdings", "Birchfield Capital", "Cresta Group", "Dunmore Ventures", "Eastgate Partners", "Fenwick Industries", "Garrow Trust",
            "Hollis Investments", "Ivel Holdings", "Juno Capital", "Kilnworth plc", "Lysander Group"]
TEAMS = ["Payments", "Search", "Identity", "Storage", "Messaging", "Checkout", "Data Platform", "Mobile"]
COMPONENTS = ["card tokenizer", "ranking model", "login service", "blob store", "push gateway", "cart service", "event pipeline", "iOS release train"]


def up_chain(reports, who):
    chain = []
    while who in reports:
        who = reports[who]; chain.append(who)
    return chain


def solve_multi_hop(f):
    k = f["kind"]
    if k == "org":
        chain = up_chain(f["reports"], f["requester"])
        if f["rule"] == "skip": ans = chain[1]
        else: ans = next(p for p in chain if RANK[f["titles"][p]] >= RANK[f["min_title"]])
        return {"approver": slug(ans)}
    if k == "deps":
        hard = {}
        for a, b, soft in f["edges"]:   # a depends on b
            if not soft: hard.setdefault(b, set()).add(a)
        seen, stack = set(), [f["down"]]
        while stack:
            x = stack.pop()
            for y in hard.get(x, ()):
                if y not in seen: seen.add(y); stack.append(y)
        hit = [s for s in f["candidates"] if s in seen]
        if len(hit) > 1: raise ValueError("several affected")
        return {"affected": slug(hit[0]) if hit else "none_affected"}
    if k == "ownership":
        stakes = {e: {h: Fraction(p) for h, p in hs.items()} for e, hs in f["stakes"].items()}   # entity -> holder -> share
        def controls(holder, target, depth=0):
            if depth > 6: return False
            held = sum((p for h, p in stakes.get(target, {}).items() if h == holder or controls(holder, h, depth + 1)), Fraction(0))
            return held > Fraction(1, 2)
        ctrl = [h for h in f["candidates"] if controls(h, f["target"])]
        # the ultimate controller: controls the target and is not itself controlled by another candidate
        top = [h for h in ctrl if not any(controls(o, h) for o in f["candidates"] if o != h)]
        out = {"controller": slug(top[0]) if top else "no_controller"}
        def econ(holder, target, depth=0):
            if depth > 6: return Fraction(0)
            return sum((p if h == holder else p * econ(holder, h, depth + 1)) for h, p in stakes.get(target, {}).items())
        out["interest"] = per_mille(econ(f["top"], f["target"]))
        return out
    if k == "oncall":
        d = date.fromisoformat(f["date"])
        def away(p):
            return any(date.fromisoformat(a) <= d <= date.fromisoformat(b) for a, b in f["leave"].get(p, []))
        team = f["routes"][f["component"]]
        for p in (f["primary"][team], f["secondary"][team], f["manager"][team]):
            if not away(p): return {"paged": slug(p)}
        raise ValueError("nobody available")
    raise ValueError(k)


MH_Q = {"org": ["Who must approve this request?", "Whose approval does the policy require for this request?", "Who is the required approver?",
                "Under the approval rule, who signs off on this request?", "Which person has to approve the request?", "Identify the approver required by the rule."],
        "deps": ["Which of these services will be affected by the outage?", "Which listed service is impacted?", "Which of the following goes down too?",
                 "Which service on this list loses functionality?", "Which of these will the outage reach?", "Which listed service is hit by the outage?"],
        "ownership": ["Which entity ultimately controls {t}?", "Who has ultimate control of {t}?", "Which holder controls {t} at the top of the chain?",
                      "Under the control rule, who ultimately controls {t}?", "Name the ultimate controller of {t}.", "Who is the ultimate controlling entity of {t}?"],
        "oncall": ["Who gets paged for this alert?", "Which person should be paged?", "Who receives the page?",
                   "Following the paging rules, who is paged?", "Who is the right person to page?", "Whom does the alert page?"]}


def gen_multi_hop(ctx, t):
    rng = ctx.rng
    kind = ctx.pick("mh_kind", ["org", "deps", "ownership", "oncall"])
    src = "hard_multi_hop"
    if kind == "org":
        names = people(rng, 14)
        firsts = [n.split()[0] for n in names]
        ceo = firsts[0]
        titles, reports = {ceo: "Chief Executive"}, {}
        levels = [[ceo]]
        pool = firsts[1:]
        ladder = ["Vice President", "Director", "Senior Manager", "Manager", "Engineer"]
        for depth, title in enumerate(ladder):
            nxt = []
            for boss in levels[-1]:
                for _ in range(rng.randint(1, 2) if depth < 4 else 1):
                    if not pool: break
                    p = pool.pop(); nxt.append(p); reports[p] = boss
                    titles[p] = title if not (title == "Senior Manager" and rng.random() < 0.4) else rng.choice(["Manager", "Director"])
            levels.append(nxt)
            if not pool: break
        leaves = [p for p in firsts if p in reports and p not in reports.values() and len(up_chain(reports, p)) >= 3]
        if not leaves: return None
        req = rng.choice(leaves)
        # titles must not increase downward in a way that makes the rule read oddly; keep the ladder monotone along the chain
        chain = up_chain(reports, req)
        for lo, hi in zip([req] + chain, chain):
            if RANK[titles[hi]] < RANK[titles[lo]]: titles[hi] = titles[lo]
        rule = rng.choice(["skip", "title"])
        min_title = rng.choice(["Director", "Senior Manager", "Vice President"])
        amount = rng.randrange(2000, 60000, 250)
        f = {"kind": "org", "reports": reports, "titles": titles, "requester": req, "rule": rule, "min_title": min_title}
        truth = solve_multi_hop(f)
        if rule == "title" and truth["approver"] == slug(chain[0]) and rng.random() < 0.7: return None
        facts = [f"{p} reports to {b}." for p, b in reports.items()]
        rng.shuffle(facts)
        tfacts = [f"{p} is a{'n' if titles[p][0] in 'AEIOU' else ''} {titles[p]}." for p in titles]
        rng.shuffle(tfacts)
        policy_line = (f"Purchase requests of this size must be approved by the requester's manager's manager (the skip-level manager)." if rule == "skip" else
                       f"Purchase requests of this size must be approved by the nearest person above the requester in the reporting line whose title is {min_title} or more senior. "
                       f"Seniority, from junior to senior: {', '.join(TITLES)}.")
        lines = [policy_line] + facts[:len(facts) // 2] + tfacts + facts[len(facts) // 2:] + [f"{req} has submitted a purchase request for {dollars(amount)}."]
        pool = [slug(p) for p in [chain[0], ceo] + chain[1:] + rng.sample([x for x in firsts if x != req], 4)]
        cands = list(dict.fromkeys([truth["approver"]] + pool))[:5]
        q1, o1 = choice_q(ctx, MH_Q["org"][t], {c: None for c in cands}, truth["approver"], src)
        f["options_q"] = {"approver": o1}
        return {"state": frame(ctx, t, "Approval routing", lines, "HR"), "questions": {"approver": q1}, "facts": f, "meta": {"subtype": kind, "rule": rule, "hops": len(chain)}}
    if kind == "deps":
        svc = rng.sample(SERVICES, 11)
        down = svc[0]
        hops = rng.randint(2, 4)
        chain = svc[1:1 + hops]          # chain[0] depends on down, chain[i] depends on chain[i-1]
        others = svc[1 + hops:]
        edges = [(chain[0], down, False)] + [(chain[i], chain[i - 1], False) for i in range(1, hops)]
        use_soft = rng.random() < 0.5
        # distractors: a service down depends on (reverse direction), one depending on an unrelated service, one via a soft edge
        edges.append((down, others[0], False))
        edges.append((others[1], others[2], False))
        edges.append((others[3], rng.choice([others[0], others[2]]), False))
        if use_soft: edges.append((others[4], rng.choice([down, chain[0]]), True))
        edges.append((others[5], others[1], False))
        target_affected = rng.random() > 0.12
        cands = ([chain[-1]] if target_affected else []) + [others[0], others[3], others[4] if use_soft else others[1]]
        f = {"kind": "deps", "edges": [list(e) for e in edges], "down": down, "candidates": cands}
        try: truth = solve_multi_hop(f)
        except ValueError: return None
        facts = [f"{a} depends on {b}" + (" (soft dependency: it degrades gracefully and keeps working if " + b + " is down)." if s else ".") for a, b, s in edges]
        rng.shuffle(facts)
        lines = ["Outages propagate along hard dependencies: if a service goes down, every service that depends on it, directly or through other hard dependencies, is affected. Soft dependencies do not propagate outages."]
        lines += facts + [f"{down} is currently down."]
        options = {slug(c): None for c in cands}
        if truth["affected"] == "none_affected" or rng.random() < 0.3: options["none_affected"] = "None of these services is affected"
        q1, o1 = choice_q(ctx, MH_Q["deps"][t], options, truth["affected"], src)
        f["options_q"] = {"affected": o1}
        return {"state": frame(ctx, t, f"Outage: {down}", lines, "SRE"), "questions": {"affected": q1}, "facts": f, "meta": {"subtype": kind, "hops": hops, "soft": use_soft}}
    if kind == "ownership":
        ents = rng.sample(ENTITIES, 6)
        top, m1, m2, target, rival, other = ents
        shape = ctx.pick("own_shape", ["chain", "split", "none"])
        if shape == "chain":
            s1, s2 = rng.randint(51, 90), rng.randint(51, 80)
            stakes = {m1: {top: s1, other: 100 - s1}, target: {m1: s2, rival: 100 - s2}}
            cands = [top, m1, rival, other]
        elif shape == "split":
            a, b = rng.randint(51, 90), rng.randint(51, 90)
            # the two controlled subsidiaries together hold a majority, but the outside rival is the largest single holder
            x = rng.randint(20, 33); y = rng.randint(51 - x, min((99 - x) // 2, 99 - 2 * x, 45))
            r = 100 - x - y
            if x + y <= 50 or r <= max(x, y): return None
            stakes = {m1: {top: a, other: 100 - a}, m2: {top: b, other: 100 - b}, target: {m1: x, m2: y, rival: r}}
            cands = [top, rival, m1, m2]
        else:
            a = rng.randint(51, 90); x = rng.randint(20, 40); r = rng.randint(20, 45)
            if x + r >= 95: return None
            stakes = {m1: {top: a, other: 100 - a}, target: {m1: x, rival: r, m2: 100 - x - r}}
            if 100 - x - r > 50: return None
            cands = [top, rival, m1, m2]
        f = {"kind": "ownership", "stakes": {e: {h: f"{p}/100" for h, p in hs.items()} for e, hs in stakes.items()}, "candidates": cands, "target": target, "top": top}
        truth = solve_multi_hop(f)
        facts = [f"{h} owns {p}% of {e}." for e, hs in stakes.items() for h, p in hs.items()]
        rng.shuffle(facts)
        lines = ["An entity controls a company if it holds more than 50% of that company's shares, counting shares it holds directly and shares held by companies it controls."] + facts
        options = {slug(c): None for c in cands}
        options["no_controller"] = "No entity controls it"
        q1, o1 = choice_q(ctx, MH_Q["ownership"][t].format(t=target), options, truth["controller"], src)
        E = Fraction(truth["interest"], 1000)
        chain_vals = [Fraction(p, 100) for hs in stakes.values() for p in hs.values()]
        distract = [per_mille(max(chain_vals)), per_mille(Fraction(sum(p for h, p in stakes[target].items() if h in (m1, m2)), 100)),
                    per_mille(E * 2 if E < Fraction(1, 2) else E / 2), per_mille(Fraction(stakes[target].get(m1, 0), 100)), per_mille(Fraction(stakes[m1][top], 100))]
        q2, o2 = value_q(ctx, f"What is {top}'s effective economic interest in {target} (its share of {target}'s profits through all holdings)?", truth["interest"], distract, fmt_pm, src)
        f["options_q"] = {"controller": o1, "interest": o2}
        return {"state": frame(ctx, t, "Group structure", lines, "Legal"), "questions": {"controller": q1, "interest": q2}, "facts": f, "meta": {"subtype": kind, "shape": shape}}
    # oncall
    teams = rng.sample(TEAMS, 3)
    comps = rng.sample(COMPONENTS, 4)
    names = [n.split()[0] for n in people(rng, 9)]
    routes = {comps[0]: teams[0], comps[1]: teams[1], comps[2]: teams[2], comps[3]: teams[0]}
    primary = {tm: names[i] for i, tm in enumerate(teams)}
    secondary = {tm: names[3 + i] for i, tm in enumerate(teams)}
    manager = {tm: names[6 + i] for i, tm in enumerate(teams)}
    comp = rng.choice(comps)
    team = routes[comp]
    d = date(2026, 1, 1) + timedelta(days=rng.randint(0, 600))
    want = ctx.pick("oncall", ["primary", "secondary", "manager"])
    leave = {}
    def span(on):
        """A leave period that covers the alert date (on) or misses it by a few days, before or after."""
        if on:
            a, b = d - timedelta(days=rng.randint(0, 5)), d + timedelta(days=rng.randint(0, 6))
        elif rng.random() < 0.5:
            a = d + timedelta(days=rng.randint(1, 5)); b = a + timedelta(days=rng.randint(0, 7))
        else:
            b = d - timedelta(days=rng.randint(1, 3)); a = b - timedelta(days=rng.randint(0, 6))
        return [a.isoformat(), b.isoformat()]
    leave[primary[team]] = [span(want in ("secondary", "manager"))]
    if want == "manager": leave[secondary[team]] = [span(True)]
    elif rng.random() < 0.5: leave[secondary[team]] = [span(False)]
    for tm in teams:
        if tm != team and rng.random() < 0.6: leave[primary[tm]] = [span(rng.random() < 0.5)]
    f = {"kind": "oncall", "routes": routes, "primary": primary, "secondary": secondary, "manager": manager, "leave": leave, "component": comp, "date": d.isoformat()}
    try: truth = solve_multi_hop(f)
    except ValueError: return None
    ds = "iso" if t in (2, 5) else "us"
    lines = ["Paging rule: an alert pages the owning team's primary on-call; if the primary is on leave that day, it pages the secondary; if the secondary is also on leave, it pages the team's engineering manager."]
    facts = [f"Alerts from the {c} are owned by the {tm} team." for c, tm in routes.items()]
    facts += [f"This week's {tm} primary on-call is {primary[tm]} and the secondary is {secondary[tm]}." for tm in teams]
    facts += [f"The {tm} engineering manager is {manager[tm]}." for tm in teams]
    facts += [f"{p} is on leave from {day(date.fromisoformat(a), ds)} to {day(date.fromisoformat(b), ds)} inclusive." for p, spans in leave.items() for a, b in spans]
    rng.shuffle(facts)
    lines += facts + [f"On {day(d, ds)}, an alert fired from the {comp}."]
    cands = list(dict.fromkeys([primary[team], secondary[team], manager[team], primary[rng.choice([x for x in teams if x != team])]]))
    q1, o1 = choice_q(ctx, MH_Q["oncall"][t], {slug(c): None for c in cands}, truth["paged"], src)
    f["options_q"] = {"paged": o1}
    return {"state": frame(ctx, t, f"Alert from {comp}", lines, "SRE"), "questions": {"paged": q1}, "facts": f, "meta": {"subtype": kind}}


# ======================================================================================================== temporal_numeric
HOLIDAY_NAMES = ["Founders' Day", "Spring Holiday", "Harvest Day", "Company Day", "Remembrance Day", "Midsummer Holiday", "Civic Day", "Heritage Day"]
TZ = [("New York", -4), ("London", 1), ("Tokyo", 9), ("Mumbai", 5.5), ("Sydney", 10), ("Berlin", 2), ("Denver", -6), ("Singapore", 8),
      ("Adelaide", 9.5), ("Kathmandu", 5.75), ("Sao Paulo", -3), ("Dubai", 4), ("Honolulu", -10), ("Auckland", 12)]
TEMPORAL_Q = {
    "deadline": ["By what date is the response due?", "What is the deadline?", "On which date is the response due at the latest?",
                 "When is the last day to respond on time?", "Compute the due date.", "What due date should be recorded?"],
    "diff": ["How many days after the invoice date was the payment received?", "How many days passed between the invoice date and the payment?",
             "Count the days from the invoice date to the payment date.", "What is the gap in days between invoice and payment?",
             "Number of days from invoice to payment:", "How many calendar days elapsed between the invoice and the payment?"],
    "tz": ["What is the local start time in {b}?", "When does the meeting start for the attendee in {b}, in local time?", "Convert the start time to {b} local time.",
           "What local date and time is the meeting in {b}?", "Local start time in {b}:", "At what local time does the {b} attendee join?"],
    "recurring": ["On which date does it happen?", "What is that date?", "Which date is it?", "Give the date.", "Date of the requested occurrence:", "When does it fall?"],
    "prorata": ["What refund is due?", "How much should be refunded?", "What is the correct refund amount?", "Compute the refund.", "Refund due:", "What amount must be returned to the customer?"],
    "shift": ["What is the gross pay for this shift?", "How much is the worker paid for the shift?", "What should the payslip show for this shift?",
              "Compute the shift pay.", "Shift pay due:", "What does the shift earn?"],
}


def fmt_dt(s, style):
    d = datetime.fromisoformat(s)
    hm = d.strftime("%I:%M %p").lstrip("0")
    return f"{d:%A}, {d:%B} {d.day}, {hm}" if style != "24h" else f"{d:%a} {d.day} {d:%b} {d:%H:%M}"


def solve_temporal(f):
    k = f["kind"]
    if k == "deadline":
        hol = {date.fromisoformat(x) for x in f["holidays"]}
        return {"due": business_days_after(date.fromisoformat(f["received"]), f["n"], hol).isoformat()}
    if k == "diff":
        n = (date.fromisoformat(f["paid"]) - date.fromisoformat(f["invoice"])).days
        return {"days": n, "late": n > f["limit"]}
    if k == "tz":
        a = datetime.fromisoformat(f["start"])
        return {"local": (a + timedelta(hours=f["off_b"] - f["off_a"])).isoformat(timespec="minutes")}
    if k == "recurring":
        if f["pattern"] == "every_n_weeks":
            return {"date": (date.fromisoformat(f["first"]) + timedelta(weeks=f["n"] * (f["k"] - 1))).isoformat()}
        if f["pattern"] == "business_days":
            d, k_ = date.fromisoformat(f["first"]), 1
            while k_ < f["k"]:
                d = business_days_after(d, f["n"], set()); k_ += 1
            return {"date": d.isoformat()}
        if f["pattern"] == "monthly_rollback":
            d = date(f["year"], f["month"], f["dom"])
            while d.weekday() >= 5: d -= timedelta(days=1)
            return {"date": d.isoformat()}
        if f["pattern"] == "last_business":
            nxt = date(f["year"] + (f["month"] == 12), f["month"] % 12 + 1, 1)
            d = nxt - timedelta(days=1)
            while d.weekday() >= 5: d -= timedelta(days=1)
            return {"date": d.isoformat()}
    if k == "prorata":
        s, c = date.fromisoformat(f["start"]), date.fromisoformat(f["cancel"])
        end = date.fromisoformat(f["end"])
        total = (end - s).days
        used = (c - s).days + 1
        cents = Fraction(f["price_cents"] * (total - used), total)
        return {"refund": int(math.floor(cents + Fraction(1, 2)))}
    if k == "shift":
        start = datetime.fromisoformat(f["start"]); end = datetime.fromisoformat(f["end"])
        minutes = (end - start).seconds // 60 - f["break_min"]
        base = min(minutes, f["ot_after_h"] * 60)
        ot = max(0, minutes - f["ot_after_h"] * 60)
        cents = Fraction(f["rate_cents"] * base, 60) + Fraction(f["rate_cents"] * 3 * ot, 120)
        return {"pay": int(math.floor(cents + Fraction(1, 2)))}
    raise ValueError(k)


def gen_temporal(ctx, t):
    rng = ctx.rng
    kind = ctx.pick("temp_kind", ["deadline", "diff", "tz", "recurring", "prorata", "shift"])
    src = "hard_temporal_numeric"
    dstyle = ["weekday", "weekday_eu", "weekday", "weekday", "weekday_eu", "weekday"][t]
    ostyle = ["us", "eu", "iso", "us", "eu", "us"][t]   # options carry no weekday: the weekday is what the question tests
    fd = lambda s: day(date.fromisoformat(s), ostyle)
    q = TEMPORAL_Q[kind][t]
    if kind == "deadline":
        rec = date(2025, 1, 1) + timedelta(days=rng.randint(0, 900))
        n = rng.randint(3, 15)
        naive_end = business_days_after(rec, n, set())
        window = [rec + timedelta(days=i) for i in range(1, (naive_end - rec).days + 1) if (rec + timedelta(days=i)).weekday() < 5]
        hol = set(rng.sample(window, min(len(window), rng.randint(1, 3))))
        outside = {naive_end + timedelta(days=rng.randint(8, 40)), rec - timedelta(days=rng.randint(3, 30))}
        allh = sorted(hol | outside)
        f = {"kind": "deadline", "received": rec.isoformat(), "n": n, "holidays": [h.isoformat() for h in allh]}
        truth = solve_temporal(f)
        who = rng.choice(COMPANIES)
        lines = [f"{who} must answer every formal complaint within {n} business days of receiving it.",
                 "Business days are Monday to Friday, excluding the public holidays listed here; the day the complaint is received does not count.",
                 "Public holidays this year: " + "; ".join(f"{name} ({day(h, dstyle)})" for name, h in zip(rng.sample(HOLIDAY_NAMES, len(allh)), allh)) + ".",
                 f"A complaint was received on {day(rec, dstyle)}."]
        due = date.fromisoformat(truth["due"])
        distract = [rec + timedelta(days=n), naive_end, business_days_after(rec, n + 1, set(allh)), business_days_after(rec, n - 1, set(allh)),
                    naive_end + timedelta(days=len(hol)), due + timedelta(days=1)]
        q1, o1 = value_q(ctx, q, truth["due"], [x.isoformat() for x in distract], fd, src)
        f["options_q"] = {"due": o1}
        return {"state": frame(ctx, t, "Complaint response deadline", lines, "Compliance"), "questions": {"due": q1}, "facts": f, "meta": {"subtype": kind}}
    if kind == "diff":
        inv = date(2025, 1, 1) + timedelta(days=rng.randint(0, 900))
        limit = rng.choice([14, 30, 45, 60])
        late = ctx.coin("late")
        n = limit + rng.randint(1, 12) if late else limit - rng.randint(0, 12)
        paid = inv + timedelta(days=n)
        f = {"kind": "diff", "invoice": inv.isoformat(), "paid": paid.isoformat(), "limit": limit}
        truth = solve_temporal(f)
        ds = ["us", "eu", "iso", "us", "eu", "us"][t]
        lines = [f"Invoice {rng.randint(10000, 99999)} was issued on {day(inv, ds)}.", f"Payment reached the account on {day(paid, ds)}.",
                 f"A late fee applies when payment arrives more than {limit} days after the invoice date."]
        rng.shuffle(lines)
        months = (paid.year - inv.year) * 12 + paid.month - inv.month
        distract = [n + 1, n - 1, months * 30 + paid.day - inv.day, n + 2, n - 2]
        q1, o1 = value_q(ctx, q, truth["days"], distract, lambda v: f"{v} days", src)
        q2 = noul_q(rng.choice(["Does the late fee apply?", "Is a late fee due on this invoice?", "Should the late fee be charged?"]), src)
        f["options_q"] = {"days": o1}
        return {"state": frame(ctx, t, "Invoice payment timing", lines, "Finance"), "questions": {"days": q1, "late": q2}, "facts": f, "meta": {"subtype": kind}}
    if kind == "tz":
        (ca, oa), (cb, ob) = rng.sample(TZ, 2)
        d0 = datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 500))
        start = d0.replace(hour=rng.randint(0, 23), minute=rng.choice([0, 15, 30, 45]))
        f = {"kind": "tz", "start": start.isoformat(timespec="minutes"), "off_a": oa, "off_b": ob}
        truth = solve_temporal(f)
        off = lambda o: f"UTC{'+' if o >= 0 else '-'}{int(abs(o))}" + (f":{int(round((abs(o) % 1) * 60)):02d}" if abs(o) % 1 else "")
        lines = [f"A call is scheduled for {fmt_dt(f['start'], '12h')} {ca} time.", f"On that date {ca} is on {off(oa)} and {cb} is on {off(ob)}.",
                 f"One attendee is in {cb}."]
        loc = datetime.fromisoformat(truth["local"])
        distract = [(start + timedelta(hours=oa - ob)), (start + timedelta(hours=ob - oa - 1)), (start + timedelta(hours=ob - oa + 1)),
                    loc - timedelta(days=1) if loc.date() != start.date() else loc + timedelta(days=1), loc + timedelta(hours=12)]
        q1, o1 = value_q(ctx, q.format(b=cb), truth["local"], [x.isoformat(timespec="minutes") for x in distract], lambda s: fmt_dt(s, "12h"), src)
        f["options_q"] = {"local": o1}
        return {"state": frame(ctx, t, "Meeting across time zones", lines, "Coordinator"), "questions": {"local": q1}, "facts": f, "meta": {"subtype": kind}}
    if kind == "recurring":
        pattern = rng.choice(["every_n_weeks", "business_days", "monthly_rollback", "last_business"])
        if pattern == "every_n_weeks":
            first = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300)); n = rng.choice([1, 2, 3]); k = rng.randint(3, 9)
            f = {"kind": "recurring", "pattern": pattern, "first": first.isoformat(), "n": n, "k": k}
            every = {1: "every week", 2: "every other week", 3: "every three weeks"}[n]
            lines = [f"The payroll run happens {every} on the same weekday.", f"The first run of the year was on {day(first, dstyle)}.",
                     f"Question concerns the {k}{'th' if k > 3 else ['', 'st', 'nd', 'rd'][k]} run of the year, counting the first run as run 1."]
            truth = solve_temporal(f)
            dd = date.fromisoformat(truth["date"])
            distract = [dd + timedelta(weeks=n), dd - timedelta(weeks=n), first + timedelta(weeks=n * k), dd + timedelta(days=1), first + timedelta(days=7 * (k - 1))]
        elif pattern == "business_days":
            first = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
            while first.weekday() >= 5: first += timedelta(days=1)
            n, k = rng.choice([2, 3, 4]), rng.randint(3, 6)
            f = {"kind": "recurring", "pattern": pattern, "first": first.isoformat(), "n": n, "k": k}
            lines = [f"A backup verification runs every {n} business days (Monday to Friday; there are no holidays in this period).",
                     f"The first verification was on {day(first, dstyle)}.", f"Question concerns verification number {k}, counting the first as number 1."]
            truth = solve_temporal(f)
            dd = date.fromisoformat(truth["date"])
            distract = [first + timedelta(days=n * (k - 1)), business_days_after(dd, n, set()), business_days_after(first, n * k, set()), dd + timedelta(days=1), dd - timedelta(days=1)]
        else:
            weekend = rng.random() < 0.65   # most draws land on a weekend, so the roll-back rule matters
            for _ in range(100):
                year, month, dom = 2026 + rng.randint(0, 1), rng.randint(1, 12), rng.choice([1, 5, 10, 15, 20, 25, 28])
                nxt = date(year + (month == 12), month % 12 + 1, 1)
                probe = date(year, month, dom) if pattern == "monthly_rollback" else nxt - timedelta(days=1)
                if (probe.weekday() >= 5) == weekend: break
            first_of = date(year, month, 1)
            if pattern == "monthly_rollback":
                f = {"kind": "recurring", "pattern": pattern, "year": year, "month": month, "dom": dom}
                lines = [f"Rent is collected on day {dom} of each month; if that day falls on a Saturday or Sunday, it is collected on the Friday before.",
                         f"{day(first_of, 'us')} is a {first_of:%A}.", f"Question concerns the collection in {first_of:%B %Y}."]
                truth = solve_temporal(f)
                raw = date(year, month, dom)
                distract = [raw, raw + timedelta(days=(7 - raw.weekday()) % 7 or 1), raw - timedelta(days=1), raw + timedelta(days=1), raw - timedelta(days=3)]
            else:
                f = {"kind": "recurring", "pattern": pattern, "year": year, "month": month}
                nxt = date(year + (month == 12), month % 12 + 1, 1)
                last = nxt - timedelta(days=1)
                lines = [f"Invoices go out on the last business day (Monday to Friday) of each month.", f"{day(last, 'us')} is a {last:%A}.",
                         f"Question concerns the invoice run for {first_of:%B %Y}."]
                truth = solve_temporal(f)
                distract = [last, last - timedelta(days=1), last - timedelta(days=2), last - timedelta(days=3), nxt]
        q1, o1 = value_q(ctx, q, truth["date"], [x.isoformat() for x in distract], fd, src)
        f["options_q"] = {"date": o1}
        return {"state": frame(ctx, t, "Recurring schedule", lines, "Ops lead"), "questions": {"date": q1}, "facts": f, "meta": {"subtype": f"recurring_{pattern}"}}
    if kind == "prorata":
        s = date(2025, 1, 1) + timedelta(days=rng.randint(0, 800))
        try: end = s.replace(year=s.year + 1)
        except ValueError: return None
        total = (end - s).days
        c = s + timedelta(days=rng.randint(10, total - 10))
        price = rng.choice([1200, 2190, 899, 4800, 3650, 1499, 600]) * 100 + rng.choice([0, 0, 99])
        f = {"kind": "prorata", "start": s.isoformat(), "end": end.isoformat(), "cancel": c.isoformat(), "price_cents": price}
        truth = solve_temporal(f)
        ds = ["us", "eu", "iso", "us", "eu", "us"][t]
        lines = [f"A customer paid {money(price)} for an annual plan running from {day(s, ds)} up to (but not including) {day(end, ds)}.",
                 f"They cancelled on {day(c, ds)}.",
                 "Refund rule: refund the annual price multiplied by the number of unused days and divided by the number of days in the plan year; the cancellation day counts as used. Round to the nearest cent."]
        used = (c - s).days + 1
        rnd = lambda x: int(math.floor(x + Fraction(1, 2)))
        distract = [rnd(Fraction(price * (total - used + 1), total)), rnd(Fraction(price * (total - used), 365 if total == 366 else 366)),
                    rnd(Fraction(price * (12 - ((c.year - s.year) * 12 + c.month - s.month)), 12)), rnd(Fraction(price * used, total)), rnd(Fraction(price * (total - used - 1), total))]
        q1, o1 = value_q(ctx, q, truth["refund"], distract, money, src)
        f["options_q"] = {"refund": o1}
        return {"state": frame(ctx, t, "Subscription cancellation", lines, "Billing"), "questions": {"refund": q1}, "facts": f, "meta": {"subtype": kind, "leap": total == 366}}
    # shift
    d0 = datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 500))
    start = d0.replace(hour=rng.choice([6, 7, 8, 14, 15, 18, 20, 21, 22, 23]), minute=rng.choice([0, 15, 30, 45]))
    length = rng.randint(6 * 4, 13 * 4) * 15
    end = start + timedelta(minutes=length)
    brk = rng.choice([0, 20, 30, 45, 60])
    rate = rng.choice([1650, 1800, 1925, 2100, 2275, 2400, 2850, 3100])
    ot_after = 8
    f = {"kind": "shift", "start": start.isoformat(timespec="minutes"), "end": end.isoformat(timespec="minutes"), "break_min": brk, "rate_cents": rate, "ot_after_h": ot_after}
    truth = solve_temporal(f)
    hm = lambda x: x.strftime("%H:%M") if t in (1, 2, 4) else x.strftime("%I:%M %p").lstrip("0")
    lines = [f"A warehouse worker clocked in at {hm(start)} on {day(start.date(), 'us')} and clocked out at {hm(end)}" + (" the next day." if end.date() != start.date() else "."),
             f"They took a {brk}-minute unpaid break." if brk else "They took no break.", f"The hourly rate is {money(rate)}.",
             f"Paid time beyond {ot_after} hours in a shift is paid at 1.5 times the hourly rate."]
    minutes = length - brk
    rnd = lambda x: int(math.floor(x + Fraction(1, 2)))
    distract = [rnd(Fraction(rate * length, 60)), rnd(Fraction(rate * minutes, 60)), rnd(Fraction(rate * 3 * minutes, 120)),
                rnd(Fraction(rate * min(length, 480), 60) + Fraction(rate * 3 * max(0, length - 480), 120)), rnd(Fraction(rate * 480, 60) + Fraction(rate * 2 * max(0, minutes - 480), 60)),
                truth["pay"] + rate // 2, truth["pay"] - rate]
    q1, o1 = value_q(ctx, q, truth["pay"], distract, money, src)
    f["options_q"] = {"pay": o1}
    return {"state": frame(ctx, t, "Shift pay check", lines, "Payroll"), "questions": {"pay": q1}, "facts": f, "meta": {"subtype": kind, "overtime": minutes > 480}}


# ==================================================================================================================== judge
JUDGE_Q = ["Is the proposed answer correct?", "Is the colleague's result right?", "Is this answer correct?",
           "Does the assistant's final answer match the correct result?", "Verify: is the stated result correct?", "Is the student's final answer correct?"]
JUDGE_Q2 = ["What is the correct answer?", "What should the result be?", "What is the right value?", "What is the correct final answer?",
            "Correct result:", "What is the correct final answer to the item?"]
LOGIC_ATOMS = [("the customer is a loyalty member", "member"), ("the order total is over $100", "over"), ("it is the customer's birthday month", "birthday"),
               ("the item is on clearance", "clearance"), ("the order ships to a domestic address", "domestic"), ("the customer used a promo code", "promo")]


def fmt_num(v, dp):
    return f"{v:,.{dp}f}"


def judge_value(f):
    """The exact answer of a judge item (a Fraction, or a bool for logic items)."""
    k, a = f["kind"], {x: Fraction(v) for x, v in f["args"].items()} if f["kind"] != "logic" else f["args"]
    if k == "order": return (a["q1"] * a["p1"] + a["q2"] * a["p2"]) * (1 - a["d"]) * (1 + a["tax"])
    if k == "convert": return a["x"] * a["factor"] + a["offset"]
    if k == "pct_change": return (a["new"] - a["old"]) / a["old"] * 100
    if k == "reverse_pct": return a["after"] / (1 - a["p"])
    if k == "wavg": return (a["n1"] * a["p1"] + a["n2"] * a["p2"]) / (a["n1"] + a["n2"])
    if k == "fence": return a["length"] / a["gap"] + 1
    if k == "rate": return a["items"] / (a["r1"] + a["r2"])
    if k == "logic":
        v = a["values"]
        x, y, z = (v[n] for n in a["atoms"])
        return {"and_or": x and (y or z), "or_and": (x and y) or z, "and_not": x and not y, "not_or": not (x or y)}[a["form"]]
    raise ValueError(k)


def rounded(v, dp):
    """Round half up to dp decimals; returns an int count of 10**-dp units (the canonical value of a numeric answer)."""
    return int(math.floor(v * 10 ** dp + Fraction(1, 2)))


def solve_judge(f):
    v = judge_value(f)
    if f["kind"] == "logic":
        return {"correct": f["proposed"] == v}
    r = rounded(v, f["dp"])
    return {"correct": f["proposed"] == r, "value": r}


def gen_judge(ctx, t):
    rng = ctx.rng
    kind = ctx.pick("judge_kind", ["order", "convert", "pct_change", "reverse_pct", "wavg", "fence", "rate", "logic"])
    correct = ctx.coin("judge")
    src = "hard_judge"
    F = Fraction
    if kind == "order":
        q1, q2 = rng.randint(2, 12), rng.randint(1, 6)
        p1, p2 = F(rng.randint(300, 9000), 100), F(rng.randint(500, 20000), 100)
        d, tax = F(rng.choice([5, 10, 15, 20, 25]), 100), F(rng.choice([5, 6, 7, 8, 10, 20]), 100)
        args = {"q1": q1, "p1": p1, "q2": q2, "p2": p2, "d": d, "tax": tax}
        text = (f"An order has {q1} notebooks at {money(int(p1 * 100))} each and {q2} desk lamps at {money(int(p2 * 100))} each. A {int(d * 100)}% discount applies to the "
                f"whole order, and then {int(tax * 100)}% sales tax is added. What is the final total, to the nearest cent?")
        sub = q1 * p1 + q2 * p2
        errors = {"discount_first_line": (q1 * p1 * (1 - d) + q2 * p2) * (1 + tax), "tax_on_gross": sub * (1 - d) + sub * tax,
                  "missing_quantity": (q1 * p1 + p2) * (1 - d) * (1 + tax), "no_tax": sub * (1 - d), "flat_discount": (sub - d * 100) * (1 + tax)}
        dp, unit = 2, "$"
        work = lambda v: f"Subtotal {money(int(sub * 100))}; after discount and tax: {money(rounded(v, 2))}."
    elif kind == "convert":
        conv = rng.choice([("kilometres", "miles", F(1, 1) / F("1.609344"), 0, "1 mile = 1.609344 km"), ("miles", "kilometres", F("1.609344"), 0, "1 mile = 1.609344 km"),
                           ("kilograms", "pounds", 1 / F("0.45359237"), 0, "1 pound = 0.45359237 kg"), ("pounds", "kilograms", F("0.45359237"), 0, "1 pound = 0.45359237 kg"),
                           ("degrees Celsius", "degrees Fahrenheit", F(9, 5), 32, "F = C x 9/5 + 32"), ("litres", "US gallons", 1 / F("3.785411784"), 0, "1 US gallon = 3.785411784 litres")])
        src_u, dst_u, factor, offset, hint = conv
        x = F(rng.randint(5, 900)) if offset == 0 else F(rng.randint(-20, 45))
        args = {"x": x, "factor": factor, "offset": offset}
        text = f"Convert {x} {src_u} to {dst_u} ({hint}). Give the answer to one decimal place."
        errors = {"inverted": x / factor + offset if factor != 1 else x * 2, "missing_offset": x * factor if offset else x * factor * 10, "decimal_slip": (x * factor + offset) * 10,
                  "offset_first": (x + offset) * factor if offset else x * factor + 1}
        dp, unit = 1, dst_u
        work = lambda v: f"{x} x {float(factor):.6g}" + (f" + {offset}" if offset else "") + f" = {fmt_num(rounded(v, 1) / 10, 1)}"
    elif kind == "pct_change":
        old = rng.randint(40, 5000); new = old + rng.choice([-1, 1]) * rng.randint(1, old // 2 + 1)
        if new <= 0: return None
        args = {"old": old, "new": new}
        text = f"Monthly active users went from {old:,} to {new:,}. What is the percentage change, to one decimal place (negative for a decrease)?"
        errors = {"wrong_base": F(new - old, new) * 100, "sign_flip": F(old - new, old) * 100, "absolute": F(new - old), "ratio": F(new, old) * 100}
        dp, unit = 1, "%"
        work = lambda v: f"({new:,} - {old:,}) / {old:,} x 100 = {fmt_num(rounded(v, 1) / 10, 1)}%"
    elif kind == "reverse_pct":
        p = F(rng.choice([10, 15, 20, 25, 30, 40]), 100); orig = F(rng.randint(20, 900))
        after = orig * (1 - p)
        if after.denominator != 1 and (after * 100).denominator != 1: return None
        args = {"after": after, "p": p}
        text = f"After a {int(p * 100)}% discount, a jacket costs {money(int(after * 100))}. What was the price before the discount, to the nearest cent?"
        errors = {"wrong_base": after * (1 + p), "subtracted": after - after * p, "added_points": after + p * 100}
        dp, unit = 2, "$"
        work = lambda v: f"{money(int(after * 100))} / (1 - {int(p * 100)}%) = {money(rounded(v, 2))}"
    elif kind == "wavg":
        n1, n2 = rng.randint(10, 400), rng.randint(10, 400)
        p1, p2 = F(rng.randint(200, 5000), 100), F(rng.randint(200, 5000), 100)
        args = {"n1": n1, "p1": p1, "n2": n2, "p2": p2}
        text = (f"A shop bought {n1} units at {money(int(p1 * 100))} each and later {n2} units at {money(int(p2 * 100))} each. "
                "What is the average cost per unit across all units bought, to the nearest cent?")
        errors = {"simple_mean": (p1 + p2) / 2, "swapped_weights": (n2 * p1 + n1 * p2) / (n1 + n2), "total_over_batches": (n1 * p1 + n2 * p2) / 2}
        dp, unit = 2, "$"
        work = lambda v: f"Average = {money(rounded(v, 2))}"
    elif kind == "fence":
        gap = rng.choice([2, 3, 4, 5, 6, 8, 10]); length = gap * rng.randint(4, 40)
        args = {"length": length, "gap": gap}
        text = f"Posts are placed every {gap} metres along a straight {length}-metre fence, with a post at both ends. How many posts are needed?"
        errors = {"fencepost": F(length, gap), "double_end": F(length, gap) + 2, "area_like": F(length * gap)}
        dp, unit = 0, "posts"
        work = lambda v: f"{length} / {gap} ... = {rounded(v, 0)} posts"
    elif kind == "rate":
        r1, r2 = rng.randint(30, 200), rng.randint(30, 200); items = (r1 + r2) * rng.randint(2, 12) + rng.choice([0, (r1 + r2) // 2])
        args = {"items": items, "r1": r1, "r2": r2}
        text = f"Machine A packs {r1} boxes per hour and machine B packs {r2} boxes per hour. Working together, how many hours do they need to pack {items:,} boxes? Give one decimal place."
        errors = {"average_rate": F(items, (r1 + r2) / F(2)), "sum_of_times": F(items, r1) + F(items, r2), "one_machine": F(items, max(r1, r2))}
        dp, unit = 1, "hours"
        work = lambda v: f"{items:,} / ({r1} + {r2}) = {fmt_num(rounded(v, 1) / 10, 1)} hours"
    else:
        atoms = rng.sample(LOGIC_ATOMS, 3)
        form = rng.choice(["and_or", "or_and", "and_not", "not_or"])
        values = {a[1]: rng.random() < 0.5 for a in atoms}
        args = {"atoms": [a[1] for a in atoms], "form": form, "values": values}
        A, B, C = (a[0] for a in atoms)
        rule = {"and_or": f"a discount applies if {A} and, in addition, either {B} or {C}",
                "or_and": f"a discount applies if both {A} and {B}, or if {C}",
                "and_not": f"a discount applies if {A} but not if {B}",
                "not_or": f"a discount applies only if neither {A} nor {B}"}[form]
        facts = "; ".join(a[0] if values[a[1]] else "it is not true that " + a[0] for a in atoms)
        text = f"Rule: {rule}. Facts: {facts}. Does the discount apply?"
        f = {"kind": "logic", "args": args}
        truth = judge_value(f)
        proposed = truth if correct else not truth
        f["proposed"] = proposed
        f["error"] = None if correct else "logic"
        answer = "Yes, the discount applies." if proposed else "No, the discount does not apply."
        return _judge_item(ctx, t, text, answer, None, f, src, None, None)
    f = {"kind": kind, "args": {k: (str(v) if isinstance(v, Fraction) else v) for k, v in args.items()}, "dp": dp}
    truth = judge_value(f)
    if kind == "fence" and truth.denominator != 1: return None
    r_true = rounded(truth, dp)
    errs = {k: rounded(v, dp) for k, v in errors.items()}
    errs = {k: v for k, v in errs.items() if v != r_true}
    if not errs: return None
    if correct:
        proposed, error = r_true, None
    else:
        error = rng.choice(sorted(errs)); proposed = errs[error]
    f["proposed"], f["error"] = proposed, error
    show = lambda v: (money(v) if unit == "$" else f"{fmt_num(v / 10 ** dp, dp)}{'%' if unit == '%' else ' ' + unit}")
    answer = show(proposed)
    work_txt = None
    if t in (1, 3, 5) and kind in ("pct_change", "convert", "reverse_pct", "rate"):
        work_txt = work(Fraction(proposed, 10 ** dp))
    distract = list(errs.values()) + [proposed + 1, r_true + 10 ** dp]
    return _judge_item(ctx, t, text, answer, work_txt, f, src, (r_true, distract), show)


def _judge_item(ctx, t, text, answer, work, f, src, value, show):
    rng = ctx.rng
    hedge = rng.choice(["", "", "I think ", "Pretty sure it's ", "Final answer: ", "Answer: "])
    if t == 0: state = f"Question: {text}\nProposed answer: {hedge}{answer}"
    elif t == 1: state = f"A colleague worked this out.\n{text}\nTheir result: {hedge}{answer}" + (f"\nWorking: {work}" if work else "")
    elif t == 2: state = {"question": text, "proposed_answer": f"{hedge}{answer}"}
    elif t == 3: state = f"User: {text}\nAssistant: {hedge}{answer}" + (f" ({work})" if work else "")
    elif t == 4: state = f"REVIEW REQUEST\nItem: {text}\nSubmitted result: {hedge}{answer}"
    else: state = f"Exam item: {text}\nStudent response: {hedge}{answer}" + (f"\nStudent working: {work}" if work else "")
    qs = {"correct": noul_q(JUDGE_Q[t], src)}
    if value is not None:
        q2, o2 = value_q(ctx, JUDGE_Q2[t], value[0], value[1], show, src)
        qs["value"] = q2
        f["options_q"] = {"value": o2}
    return {"state": state, "questions": qs, "facts": f, "meta": {"subtype": f["kind"], "error": f["error"]}}


# ================================================================================================================ ambiguous
INSUFFICIENT = [("insufficient_information", "Cannot be determined from the information given"), ("cannot_determine", "Not enough information to decide"),
                ("insufficient_information", "The facts provided do not settle this"), ("undetermined", "Undetermined: a fact the rule needs is missing"),
                ("not_enough_information", "The information given is not enough to decide"), ("cannot_be_determined", "It cannot be determined from what is stated")]


def months_between(a, b):
    """Whole months from a to b (b >= a), counting a month only once its day of month is reached."""
    m = (b.year - a.year) * 12 + b.month - a.month
    return m - (b.day < a.day)


def amb_decide(f):
    """The outcome a scenario's rule gives, or None when the stated facts do not settle it. A missing fact is None in
    `params`; the rule is evaluated over every value the missing fact could take (its domain is stated in the scenario)."""
    s, p = f["scenario"], f["params"]
    D = lambda k: date.fromisoformat(p[k]) if p.get(k) else None
    if s == "return_window":
        order, delivered, req = D("order"), D("delivered"), D("request")
        if delivered is None:   # delivery happens on or after the order date, so a request within the window of the order date is within the window of delivery too
            return "accept_return" if (req - order).days <= p["window"] else None
        return "accept_return" if (req - delivered).days <= p["window"] else "reject_return"
    if s == "approval":
        if p["amount"] < p["no_approval_below"]: return "approval_valid"
        if p["title"] is None: return None
        need = "Director" if p["amount"] > p["director_above"] else "Manager"
        return "approval_valid" if RANK[p["title"]] >= RANK[need] else "approval_invalid"
    if s == "tenure":
        apply_, hired, moved = D("apply"), D("hired"), D("moved")
        if hired is None:   # employment began on or before the team move, so a team move long enough ago settles it
            return "eligible" if months_between(moved, apply_) >= 12 * p["years"] else None
        return "eligible" if months_between(hired, apply_) >= 12 * p["years"] else "not_eligible"
    if s == "shipping":
        if p["discount_cents"] is None:   # a discount can only lower the total
            return "standard_shipping" if p["subtotal_cents"] < p["threshold_cents"] else None
        return "free_shipping" if p["subtotal_cents"] - p["discount_cents"] >= p["threshold_cents"] else "standard_shipping"
    if s == "overtime":
        known = sum(p["hours"][:4])
        if p["hours"][4] is None:
            if known > 40: return "overtime_due"
            if known + p["max_shift"] <= 40: return "no_overtime"
            return None
        return "overtime_due" if known + p["hours"][4] > 40 else "no_overtime"
    if s == "warranty":
        purchase, claim = D("purchase"), D("claim")
        if months_between(purchase, claim) >= p["months"]: return "not_covered"
        if p["cause"] is None: return None
        return "repair_under_warranty" if p["cause"] == "defect" else "not_covered"
    if s == "price_match":
        ours, theirs = p["ours_cents"], p["theirs_cents"]
        if theirs >= ours: return "no_adjustment"
        if theirs < ours * (100 - p["max_gap_pct"]) / 100: return "decline_match"
        if p["in_stock"] is None: return None
        return "match_price" if p["in_stock"] else "decline_match"
    raise ValueError(s)


def solve_ambiguous(f):
    out = amb_decide(f)
    return {"decision": out if out is not None else f["insufficient_key"]}


AMB_Q = ["What is the correct decision?", "How should this case be decided under the rule?", "What does the rule say should happen?",
         "Which outcome does the policy require?", "Decide the case.", "Select the outcome the rule requires."]


def amb_scenario(ctx, name, want_nondeciding):
    """(policy sentence, fact sentences with the deciding fact present, deciding sentence index, params, missing key,
    outcome options, explicit-unknown sentence) for one scenario; the caller removes the deciding sentence for the absent
    twin. Returns None when a draw does not give the requested kind of case."""
    rng = ctx.rng
    who = people(rng, 3)
    first = who[0].split()[0]
    ds = rng.choice(["us", "eu", "iso"])
    base = date(2025, 1, 1) + timedelta(days=rng.randint(0, 800))
    if name == "return_window":
        window = rng.choice([14, 30, 45])
        order = base
        if want_nondeciding: req = order + timedelta(days=rng.randint(3, window - 1))
        else: req = order + timedelta(days=window + rng.randint(1, 20))
        delivered = order + timedelta(days=rng.randint(1, min(12, (req - order).days - 1)))
        params = {"window": window, "order": order.isoformat(), "delivered": delivered.isoformat(), "request": req.isoformat()}
        facts = [f"{who[0]} ordered a standing desk on {day(order, ds)}.", f"The desk was delivered on {day(delivered, ds)}.",
                 f"{first} asked to return it on {day(req, ds)}.", "The desk is unused and in its original packaging."]
        pol = f"Returns are accepted if the return request is made within {window} days of delivery."
        opts = {"accept_return": "Accept the return", "reject_return": "Reject the return as outside the window"}
        return pol, facts, 1, params, "delivered", opts, "The delivery date is not recorded on the order."
    if name == "approval":
        small, director = rng.choice([(250, 5000), (500, 10000), (1000, 25000)])
        band = rng.choice(["mid", "big"]) if not want_nondeciding else "small"
        amount = {"small": rng.randint(20, small - 1), "mid": rng.randint(small, director), "big": rng.randint(director + 1, director * 3)}[band]
        title = rng.choice(["Analyst", "Engineer", "Manager", "Senior Manager", "Director", "Vice President"])
        params = {"amount": amount, "no_approval_below": small, "director_above": director, "title": title}
        other_title = rng.choice(["Manager", "Director", "Vice President"])
        facts = [f"{who[0]} submitted a purchase of {dollars(amount)} for conference travel.", f"The purchase was approved by {who[1]}.",
                 f"{who[1]} is a{'n' if title[0] in 'AEIOU' else ''} {title}.", f"{who[2]}, who leads the travel desk, is a{'n' if other_title[0] in 'AEIOU' else ''} {other_title}."]
        pol = (f"Purchases under {dollars(small)} need no approval. Purchases from {dollars(small)} up to {dollars(director)} must be approved by a Manager or more senior; "
               f"purchases above {dollars(director)} must be approved by a Director or more senior. Seniority: {', '.join(TITLES)}.")
        opts = {"approval_valid": "The purchase is properly approved", "approval_invalid": "The approval is not valid under the rule"}
        return pol, facts, 2, params, "title", opts, f"{who[1]}'s job title is not listed in the directory."
    if name == "tenure":
        years = rng.choice([2, 3, 5])
        apply_ = base
        if want_nondeciding:
            moved = apply_ - timedelta(days=365 * years + rng.randint(40, 400))
            hired = moved - timedelta(days=rng.randint(30, 900))
        else:
            moved = apply_ - timedelta(days=rng.randint(30, 365 * years - 40))
            hired = moved - timedelta(days=rng.randint(10, 365 * years + 400))
        params = {"years": years, "apply": apply_.isoformat(), "hired": hired.isoformat(), "moved": moved.isoformat()}
        facts = [f"{who[0]} applied for the sabbatical on {day(apply_, ds)}.", f"{first} joined the company on {day(hired, ds)}.",
                 f"{first} moved to the {rng.choice(TEAMS)} team on {day(moved, ds)}.", f"{first}'s manager supports the application."]
        pol = f"Employees are eligible for a sabbatical once they have completed {years} years of continuous employment with the company on the date they apply."
        opts = {"eligible": "Eligible for the sabbatical", "not_eligible": "Not yet eligible"}
        return pol, facts, 1, params, "hired", opts, f"{first}'s start date with the company is not in the file."
    if name == "shipping":
        thr = rng.choice([50, 75, 100, 150]) * 100
        sub = thr - rng.randint(100, thr // 3) if want_nondeciding else thr + rng.randint(100, thr // 2)
        disc = rng.choice([500, 1000, 1500, 2000, 2500, 3000, 4000])
        params = {"threshold_cents": thr, "subtotal_cents": sub, "discount_cents": disc}
        facts = [f"{who[0]}'s basket totals {money(sub)} before discounts.", f"The coupon {first} applied took {money(disc)} off.",
                 "The basket contains three items from the same warehouse.", f"{first} chose standard delivery."]
        pol = f"Orders ship free when the order total after all discounts is at least {money(thr)}; otherwise the standard shipping fee applies."
        opts = {"free_shipping": "Ship the order free", "standard_shipping": "Charge the standard shipping fee"}
        return pol, facts, 1, params, "discount_cents", opts, "A coupon was applied at checkout, but its value is not shown."
    if name == "overtime":
        mx = rng.choice([10, 12])
        if want_nondeciding:
            if rng.random() < 0.5: hours = [rng.randint(10, mx) for _ in range(4)]            # already over 40
            else: hours = [rng.randint(4, 7) for _ in range(4)]                              # cannot reach 40
        else:
            hours = [rng.randint(6, 10) for _ in range(4)]
            if sum(hours) > 40 or sum(hours) + mx <= 40: return None
        fri = rng.randint(4, mx)
        params = {"hours": hours + [fri], "max_shift": mx}
        days_ = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        facts = [f"{who[0]} worked {h} hours on {d_}." for h, d_ in zip(hours + [fri], days_)] + [f"{first} did not work at the weekend."]
        pol = f"Hours beyond 40 in a Monday-to-Sunday week are overtime. No shift may exceed {mx} hours."
        opts = {"overtime_due": "Overtime is due for this week", "no_overtime": "No overtime is due for this week"}
        return pol, facts, 4, params, "hours4", opts, f"{first}'s Friday timesheet has not been submitted."
    if name == "warranty":
        months = rng.choice([12, 24, 36])
        purchase = base
        if want_nondeciding: claim = purchase + timedelta(days=int(30.5 * months) + rng.randint(20, 300))
        else: claim = purchase + timedelta(days=rng.randint(30, int(30.4 * months) - 20))
        cause = rng.choice(["defect", "accident"])
        params = {"months": months, "purchase": purchase.isoformat(), "claim": claim.isoformat(), "cause": cause}
        facts = [f"{who[0]} bought a coffee machine on {day(purchase, ds)}.", f"{first} made a warranty claim on {day(claim, ds)}.",
                 ("The technician found a faulty heating element from the factory." if cause == "defect" else "The technician found the machine had been dropped."),
                 f"{first} registered the machine online."]
        pol = f"The warranty covers manufacturing defects for {months} months from the purchase date. Accidental damage is never covered."
        opts = {"repair_under_warranty": "Repair under warranty", "not_covered": "Not covered by the warranty"}
        return pol, facts, 2, params, "cause", opts, "The technician's report on the cause has not come back yet."
    # price_match
    ours = rng.randint(80, 900) * 100
    gap = rng.choice([15, 20, 25])
    if want_nondeciding:
        theirs = rng.choice([int(ours * (100 - gap - rng.randint(3, 20)) / 100), ours + rng.randint(1, 50) * 100])
    else:
        theirs = int(ours * (100 - rng.randint(2, gap - 2)) / 100)
    stock = rng.random() < 0.5
    params = {"ours_cents": ours, "theirs_cents": theirs, "max_gap_pct": gap, "in_stock": stock}
    rival = rng.choice(COMPANIES)
    facts = [f"{who[0]} asked us to match {rival}'s price for the same blender model.", f"Our price is {money(ours)}; {rival} lists it at {money(theirs)}.",
             f"{rival} {'had the blender in stock' if stock else 'was out of stock'} when {first} asked.", f"{first} is a returning customer."]
    pol = (f"We match a competitor's lower price for an identical item if the competitor has it in stock at the time of the request, "
           f"but we never match a price more than {gap}% below ours. If the competitor's price is not lower, no adjustment is needed.")
    opts = {"match_price": "Match the competitor's price", "decline_match": "Decline the price match", "no_adjustment": "No adjustment needed"}
    return pol, facts, 2, params, "in_stock", opts, f"Nobody checked whether {rival} had it in stock."


AMB_SCENARIOS = ["return_window", "approval", "tenure", "shipping", "overtime", "warranty", "price_match"]


def gen_ambiguous_pair(ctx, t):
    """Two records sharing a group: the intact case and its twin with the deciding fact removed. In about one pair in
    five the removed fact turns out not to matter (the rule is settled either way), so "a fact is missing" alone never
    predicts the insufficient-information answer."""
    rng = ctx.rng
    name = ctx.pick("amb_scenario", AMB_SCENARIOS)
    nondeciding = rng.random() < 0.2
    got = amb_scenario(ctx, name, nondeciding)
    if got is None: return None
    pol, facts, idx, params, missing, opts, unknown = got
    ikey, idesc = INSUFFICIENT[t]
    base = {"scenario": name, "insufficient_key": ikey}
    intact_f = {**base, "params": params, "deciding_present": True}
    absent_params = dict(params)
    if missing == "hours4": absent_params["hours"] = params["hours"][:4] + [None]
    else: absent_params[missing] = None
    absent_f = {**base, "params": absent_params, "deciding_present": False}
    o_intact, o_absent = amb_decide(intact_f), amb_decide(absent_f)
    if o_intact is None: return None
    if nondeciding != (o_absent is not None): return None
    options = {**opts, ikey: idesc}
    explicit = rng.random() < 0.45
    absent_facts = facts[:idx] + ([unknown] if explicit else []) + facts[idx + 1:]
    def lay(fs):
        return [pol] + fs
    items = []
    for twin, fs, fct in (("intact", facts, intact_f), ("absent", absent_facts, absent_f)):
        truth = solve_ambiguous(fct)
        q, o = choice_q(ctx, AMB_Q[t], options, truth["decision"], "hard_ambiguous")
        fct = {**fct, "options_q": {"decision": o}}
        items.append({"state": frame(ctx, t, name.replace("_", " ").capitalize(), lay(fs), "Case handler"), "questions": {"decision": q}, "facts": fct,
                      "meta": {"subtype": name, "twin": twin, "absent_kind": ("nondeciding" if nondeciding else "deciding") if twin == "absent" else None,
                               "explicit_unknown": explicit if twin == "absent" else None}})
    return items


# ================================================================================================================ registry
def solve_policy(f):
    return policy.solve(f)


FAMILIES = {
    "long_policy": (policy.generate, solve_policy),
    "tradeoff": (gen_tradeoff, solve_tradeoff),
    "probability": (gen_probability, solve_probability),
    "multi_hop": (gen_multi_hop, solve_multi_hop),
    "temporal_numeric": (gen_temporal, solve_temporal),
    "judge": (gen_judge, solve_judge),
    "ambiguous": (gen_ambiguous_pair, solve_ambiguous),
}


def labels(family, facts, questions):
    """Every question's label from the family's solver: a solver answer is mapped to the option key whose canonical value
    it equals (choice), used as is (noul: bool), or as the level index (score). Raises unless exactly one option matches."""
    truth = FAMILIES[family][1](facts)
    out = {}
    opts = facts.get("options_q") or facts.get("options") or {}
    for qid, q in questions.items():
        ans = truth[qid]
        if q["type"] == "choice":
            keys = [k for k, v in opts[qid].items() if v == ans]
            if len(keys) != 1: raise ValueError(f"{family}/{qid}: {len(keys)} options match the solver's answer {ans!r}")
            out[qid] = keys[0]
        elif q["type"] == "noul":
            out[qid] = bool(ans)
        else:
            out[qid] = int(ans)
    return out
