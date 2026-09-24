"""hard-v1's computed-answer families: probability, temporal_numeric and judge (a proposed answer to a computation that
is right or carries one planted error). Same contract as scripts/hard_v1_families.py, which registers them: `gen_*(ctx, t)
-> item | None` plus `solve_*(facts) -> {qid: canonical answer}`.
"""
import math
from datetime import date, datetime, timedelta
from fractions import Fraction

from scripts.hard_v1_common import (COMPANIES, business_days_after, choice_q, day, dollars, fmt_pm, frame, money, noul_q, per_mille,
                                    score_q, slug, value_q, words)


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
                lines = ["Invoices go out on the last business day (Monday to Friday) of each month.", f"{day(last, 'us')} is a {last:%A}.",
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
