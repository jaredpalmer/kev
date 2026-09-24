"""hard-v1 families other than long_policy (scripts/hard_v1_policy.py): tradeoff, multi_hop and ambiguous here, and the
computed-answer families probability, temporal_numeric and judge in scripts/hard_v1_numeric.py. Each family is
`generate(ctx, t) -> item | None` plus `solve(facts) -> {qid: canonical answer}`; the builder (scripts/build_hard_v1.py)
turns an item into a record and takes every label from `solve`, never from the generator. Six surface templates per
family; templates 4 and 5 are held out of training. FAMILIES below is the registry.
"""
from datetime import date, timedelta
from fractions import Fraction

from scripts import hard_v1_numeric as numeric
from scripts import hard_v1_policy as policy
from scripts.hard_v1_common import COMPANIES, choice_q, day, dollars, fmt_pm, frame, money, noul_q, people, per_mille, slug, value_q


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


def trade_attr(spec):
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
    attrs = [trade_attr(s) for s in c["num"]]
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
        policy_line = ("Purchase requests of this size must be approved by the requester's manager's manager (the skip-level manager)." if rule == "skip" else
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
    items = []
    for twin, fs, fct in (("intact", facts, intact_f), ("absent", absent_facts, absent_f)):
        truth = solve_ambiguous(fct)
        q, o = choice_q(ctx, AMB_Q[t], options, truth["decision"], "hard_ambiguous")
        fct = {**fct, "options_q": {"decision": o}}
        items.append({"state": frame(ctx, t, name.replace("_", " ").capitalize(), [pol] + fs, "Case handler"), "questions": {"decision": q}, "facts": fct,
                      "meta": {"subtype": name, "twin": twin, "absent_kind": ("nondeciding" if nondeciding else "deciding") if twin == "absent" else None,
                               "explicit_unknown": explicit if twin == "absent" else None}})
    return items


# ================================================================================================================ registry
# the answers that decline to pick an option: a missing deciding fact (INSUFFICIENT) or no option qualifying (the tradeoff
# and multi_hop solvers' "none" keys); build_hard_v1.summary counts them per family
ABSTAIN_KEYS = tuple(dict.fromkeys(k for k, _ in INSUFFICIENT)) + ("none_qualifies", "none_affected", "no_controller")

FAMILIES = {
    "long_policy": (policy.generate, policy.solve),
    "tradeoff": (gen_tradeoff, solve_tradeoff),
    "probability": (numeric.gen_probability, numeric.solve_probability),
    "multi_hop": (gen_multi_hop, solve_multi_hop),
    "temporal_numeric": (numeric.gen_temporal, numeric.solve_temporal),
    "judge": (numeric.gen_judge, numeric.solve_judge),
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
