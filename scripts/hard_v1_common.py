"""Shared pieces of the hard-v1 generators (scripts/build_hard_v1.py): name pools, number/date/money rendering, the
surface layouts (frame), balanced option placement and the question builders. Every label in hard-v1 is computed by a family's solver from the record's
`_meta.facts`; the builders here only lay options out and never decide a label.
"""
import math
import random
import re
from collections import defaultdict
from datetime import timedelta
from fractions import Fraction

LETTERS = "abcdefgh"

FIRST = ["Mira", "Noah", "Priya", "Tomas", "Aiko", "Lena", "Omar", "Sana", "Jonas", "Ravi", "Elin", "Kofi", "Hana", "Diego",
         "Farah", "Luca", "Ines", "Mateo", "Yusuf", "Chloe", "Arjun", "Greta", "Tariq", "Nadia", "Emeka", "Sofia", "Bram",
         "Leila", "Kenji", "Amara", "Felix", "Rosa", "Idris", "Maya", "Viktor", "Zainab", "Owen", "Petra", "Samir", "Tess",
         "Dmitri", "Ayla", "Colm", "Nia", "Hugo", "Wen", "Rafael", "Esme", "Kwame", "Lotte", "Ashok", "Fiona", "Jae", "Beatriz"]
LAST = ["Okafor", "Lindqvist", "Haddad", "Moreau", "Tanaka", "Kowalski", "Brennan", "Castillo", "Varga", "Nakamura", "Osei",
        "Petrov", "Rahman", "Silva", "Fischer", "Dubois", "Ivanova", "Mensah", "Novak", "Quinn", "Sato", "Abara", "Bauer",
        "Costa", "Delgado", "Eriksen", "Farouk", "Gallagher", "Horvat", "Iyer", "Jensen", "Keller", "Laine", "Mbeki", "Nguyen",
        "Ortiz", "Pajari", "Reyes", "Sandoval", "Thorne", "Ueda", "Vance", "Whitfield", "Yilmaz", "Zeller"]
COMPANIES = ["Northwind Logistics", "Harbor & Pine", "Cobalt Ridge", "Larkspur Health", "Tidewater Foods", "Granite Peak Software",
             "Juniper Analytics", "Blue Heron Retail", "Copperline Energy", "Silverleaf Media", "Redwood Freight", "Aster Labs",
             "Brightwater Clinics", "Kestrel Systems", "Marigold Home", "Oakhaven Bank", "Pinecrest Hotels", "Quarry Street Studio",
             "Riverside Dental", "Summit Parcel", "Thistle Apparel", "Umber Robotics", "Vantage Print", "Willow Creek Farms",
             "Yardley Engineering", "Zephyr Travel", "Alder Finance", "Bramble Games", "Cinder Security", "Driftwood Furniture"]

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
        "eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n):
    """English words for 0 <= n < 1000 ("forty-two")."""
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    rest = n % 100
    return ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + words(rest))


def person(rng, used=None):
    while True:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if used is None or name not in used:
            if used is not None: used.add(name)
            return name


def people(rng, n):
    """n distinct full names whose first names are also distinct (so a first-name reference is unambiguous)."""
    firsts = rng.sample(FIRST, n)
    lasts = rng.sample(LAST, n)
    return [f"{f} {l}" for f, l in zip(firsts, lasts)]


def money(cents, style="plain"):
    """$1,234.50 (plain), $1,234 when whole and style == "whole_ok", or "1,234.50 USD" (code)."""
    whole, frac = divmod(cents, 100)
    body = f"{whole:,}" if (style == "whole_ok" and frac == 0) else f"{whole:,}.{frac:02d}"
    return f"{body} USD" if style == "code" else f"${body}"


def dollars(n):
    return f"${n:,}"


def day(d, style="us"):
    if style == "iso": return d.isoformat()
    if style == "eu": return f"{d.day} {d:%B} {d.year}"
    if style == "weekday": return f"{d:%A}, {d:%B} {d.day}, {d.year}"
    if style == "weekday_eu": return f"{d:%A} {d.day} {d:%B} {d.year}"
    return f"{d:%B} {d.day}, {d.year}"


def roman(n):
    out = ""
    for v, s in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v: out += s; n -= v
    return out


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


class Ctx:
    """Per (family, split) generation context: the seeded RNG every draw comes from, and a shuffled bag of label positions
    per option count so correct answers are spread evenly over positions (exactly balanced within each block of n)."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.bags = defaultdict(list)

    def position(self, n):
        bag = self.bags[n]
        if not bag:
            bag.extend(range(n)); self.rng.shuffle(bag)
        return bag.pop()

    def place(self, label, keys):
        others = [k for k in keys if k != label]
        self.rng.shuffle(others)
        others.insert(self.position(len(keys)), label)
        return others

    def coin(self, key="noul"):
        """Balanced boolean: a shuffled bag of [True, False] per key."""
        bag = self.bags[("coin", key)]
        if not bag:
            bag.extend([True, False]); self.rng.shuffle(bag)
        return bag.pop()

    def pick(self, key, items):
        """Balanced choice among items: each item once per block, in shuffled order (used to spread target outcomes)."""
        bag = self.bags[("pick", key)]
        if not bag:
            bag.extend(items); self.rng.shuffle(bag)
        return bag.pop()


def choice_q(ctx, instructions, options, label, src):
    """Choice question over semantic keys (options: key -> description, canonical order); the label is placed at a
    balanced position and the rest shuffled. Returns (question, option_values) with option_values[key] = key."""
    if label not in options: raise ValueError(f"label {label!r} not among options")
    if len(options) < 2: raise ValueError("a choice needs at least two options")
    keys = ctx.place(label, list(options))
    return ({"type": "choice", "instructions": instructions, "criteria": {k: options[k] for k in keys}, "src": src},
            {k: k for k in keys})


def value_q(ctx, instructions, correct, distractors, fmt, src, n=5, keys=LETTERS):
    """Choice question whose options are computed values (money in cents, day counts, ISO dates, ...): the correct value
    plus up to n-1 distinct distractors (distinct after formatting), keyed a, b, c, ... in display order. Returns
    (question, option_values) with option_values[key] = the canonical value the solver compares against."""
    shown, values = {fmt(correct)}, [correct]
    for v in distractors:
        if len(values) == n: break
        if fmt(v) not in shown and v != correct:
            shown.add(fmt(v)); values.append(v)
    if len(values) < 3: raise ValueError("too few distinct values for a choice")
    order = ctx.place(0, list(range(len(values))))
    opts = {keys[i]: values[j] for i, j in enumerate(order)}
    return ({"type": "choice", "instructions": instructions, "criteria": {k: fmt(v) for k, v in opts.items()}, "src": src}, opts)


def noul_q(instructions, src, criteria=None):
    q = {"type": "noul", "instructions": instructions, "src": src}
    if criteria: q["criteria"] = criteria
    return q


def score_q(instructions, levels, src):
    return {"type": "score", "instructions": instructions, "criteria": list(levels), "src": src}


def business_days_after(start, n, holidays):
    """The n-th business day after `start` (start itself not counted); weekends and `holidays` are skipped."""
    d, count = start, 0
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in holidays: count += 1
    return d

