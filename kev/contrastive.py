"""Programmatic contrastive pairs with certain labels (PLAN.md step 2).

Each item is a policy plus a case described by a few sentences. Every sentence carries zero or more explicit facts;
the label is a pure function `evaluate(facts)` of the facts present, returning UNDETERMINED when a required fact is
missing. A pair is the same item with exactly one sentence changed so that the label flips.

Two code-level checks run on every pair before it is kept (no model anywhere):
  ablation   : removing any evidence sentence (one that carries a required fact) makes the label UNDETERMINED,
               so the answer provably lives in the state, not in the instruction, the options, or the policy alone;
  invariance : removing any filler sentence leaves the label unchanged.
Evidence status is derived from the evaluator, not declared. Both siblings share `pair_id`/`family_id`, and
splits are assigned by family so siblings never straddle a split.
"""
import hashlib
import json
import random
from datetime import date, timedelta

UNDETERMINED = "UNDETERMINED"

NAMES = ["Mira", "Noah", "Priya", "Tomas", "Aiko", "Lena", "Omar", "Sana", "Jonas", "Ravi", "Elin", "Kofi"]
ROLES = ["account owner", "billing manager", "support agent", "warehouse lead"]
ITEMS = ["a pair of running shoes", "a desk lamp", "a wireless keyboard", "a rain jacket", "a coffee grinder", "a backpack"]
PROGRAMS = ["the volunteer driver program", "the apprenticeship", "the rental agreement", "the night-shift roster"]


def _day(d):
    """The day as "March 5, 2026": strftime's "%-d" (no zero padding) is a glibc/BSD extension that Windows rejects."""
    return f"{d:%B} {d.day}, {d:%Y}"


def _need(facts, *keys):
    return all(k in facts for k in keys)


# A family returns (item_a, item_b). An item is {policy, sentences: [(text, facts)], evaluate, question}.
# The two items must differ in exactly one sentence, and evaluate() must give different labels.

def family_return_window(rng):
    window = rng.choice([14, 30, 45, 60]); item = rng.choice(ITEMS); name = rng.choice(NAMES)
    bought = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    def evaluate(f):
        return (f["request"] - f["purchase"]).days <= window if _need(f, "request", "purchase") else UNDETERMINED
    def build(days):
        request = bought + timedelta(days=days)
        return {"policy": f"Returns are accepted only if the return request is submitted within {window} days of the purchase date.",
                "sentences": [(f"{name} bought {item} on {_day(bought)}.", {"purchase": bought}),
                              (f"The return request was submitted on {_day(request)}.", {"request": request}),
                              (f"The order was paid by card and shipped to {name}'s home address.", {})],
                "evaluate": evaluate, "question": {"type": "noul", "instructions": "Is this return request within the policy window?"}}
    return build(rng.randint(1, window - 1)), build(window + rng.randint(1, 30))


def family_spend_threshold(rng):
    limit = rng.choice([250, 500, 1000, 2500]); name = rng.choice(NAMES); role = rng.choice(ROLES)
    def evaluate(f):
        return ("auto_approved" if f["amount"] <= limit else "director_signoff") if _need(f, "amount") else UNDETERMINED
    def build(amount):
        return {"policy": f"Expense claims of ${limit:,} or less are approved automatically. Claims above ${limit:,} require director sign-off.",
                "sentences": [(f"{name}, the {role}, submitted an expense claim.", {}),
                              (f"The claim total is ${amount:,}.", {"amount": amount}),
                              ("Receipts were attached for every line item.", {})],
                "evaluate": evaluate, "question": {"type": "choice", "instructions": "How is this claim handled under the policy?",
                                                   "criteria": {"auto_approved": "Approved without further review", "director_signoff": "Requires director sign-off", "rejected": "Rejected outright"}}}
    return build(limit - rng.randint(1, limit // 2)), build(limit + rng.randint(1, limit))


def family_authorization(rng):
    approver, other = rng.sample(NAMES, 2); account = rng.randint(10, 99); amount = rng.choice([40, 120, 350, 900])
    def evaluate(f):
        return f["signer"] == f["approver"] if _need(f, "signer", "approver") else UNDETERMINED
    def build(signer):
        return {"policy": "A refund is authorized only when its sole authorization was signed by someone who may authorize refunds for that account.",
                "sentences": [(f"Only {approver} may authorize refunds for account {account}.", {"approver": approver}),
                              (f"The sole authorization for this refund on account {account} was signed by {signer}.", {"signer": signer}),
                              (f"The refund amount is ${amount}.", {})],
                "evaluate": evaluate, "question": {"type": "noul", "instructions": "Is the refund authorized?"}}
    return build(approver), build(other)


def family_age_eligibility(rng):
    minimum = rng.choice([16, 18, 21, 25]); name = rng.choice(NAMES); program = rng.choice(PROGRAMS)
    def evaluate(f):
        return f["age"] >= minimum if _need(f, "age") else UNDETERMINED
    def build(age):
        return {"policy": f"Applicants must be at least {minimum} years old to be eligible for {program}.",
                "sentences": [(f"{name} applied to join {program}.", {}),
                              (f"{name} is {age} years old.", {"age": age}),
                              ("The application form was complete and signed.", {})],
                "evaluate": evaluate, "question": {"type": "noul", "instructions": "Is the applicant eligible?"}}
    return build(minimum + rng.randint(0, 20)), build(minimum - rng.randint(1, 5))


def family_quantity_limit(rng):
    limit = rng.choice([2, 3, 5, 10]); item = rng.choice(ITEMS); name = rng.choice(NAMES)
    def evaluate(f):
        if not _need(f, "qty"): return UNDETERMINED
        return "within_limit" if f["qty"] <= limit else "slightly_over" if f["qty"] <= 2 * limit else "far_over"
    def build(qty):
        return {"policy": f"Customers may order at most {limit} units of any single item per order. Orders up to double the limit are held for review; larger orders are cancelled.",
                "sentences": [(f"{name} placed an order for {item}.", {}),
                              (f"The order quantity is {qty}.", {"qty": qty}),
                              ("Delivery was requested to a residential address.", {})],
                "evaluate": evaluate, "question": {"type": "choice", "instructions": "What happens to this order?",
                                                   "criteria": {"within_limit": "Processed normally", "slightly_over": "Held for review", "far_over": "Cancelled"}}}
    return build(rng.randint(1, limit)), build(rng.choice([rng.randint(limit + 1, 2 * limit), rng.randint(2 * limit + 1, 4 * limit)]))


def family_deadline(rng):
    name = rng.choice(NAMES); due = date(2026, rng.randint(2, 11), rng.randint(1, 28)); grace = rng.choice([3, 7, 14])
    def evaluate(f):
        if not _need(f, "received", "due"): return UNDETERMINED
        late = (f["received"] - f["due"]).days
        return 0 if late <= 0 else 1 if late <= grace else 2
    def build(offset):
        received = due + timedelta(days=offset)
        return {"policy": f"Reports received by the deadline are on time. Reports received within {grace} days after the deadline are late but accepted. Later reports are refused.",
                "sentences": [(f"The filing deadline for {name}'s report was {_day(due)}.", {"due": due}),
                              (f"The report was received on {_day(received)}.", {"received": received}),
                              ("The report was submitted through the online portal.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "How late is this report?", "criteria": ["On time", "Late but accepted", "Refused"]}}
    return build(-rng.randint(0, 10)), build(rng.choice([rng.randint(1, grace), grace + rng.randint(1, 20)]))


# --- ordinal threshold families as Score questions (PLAN.md: the deadline failure is "no idea how ordinal thresholds work").
# Each has three ordered levels split by two thresholds; one is date-based like deadline but with a different template.

def family_warranty_claim(rng):
    name = rng.choice(NAMES); item = rng.choice(ITEMS); bought = date(2026, rng.randint(1, 6), rng.randint(1, 28))
    standard, extended = rng.choice([(90, 365), (180, 730), (365, 1095)]); part = rng.choice(['stitching', 'battery', 'housing', 'zipper', 'switch'])
    def evaluate(f):
        if not _need(f, "claim", "purchase"): return UNDETERMINED
        age = (f["claim"] - f["purchase"]).days
        return 0 if age <= standard else 1 if age <= extended else 2
    def build(age):
        claim = bought + timedelta(days=age)
        return {"policy": f"Warranty claims made within {standard} days of purchase are covered in full. Claims made after that but within {extended} days are covered at half cost. Later claims are not covered.",
                "sentences": [(f"{name} purchased {item} on {_day(bought)}.", {"purchase": bought}),
                              (f"A warranty claim for it was filed on {_day(claim)}.", {"claim": claim}),
                              (f"The claim describes a defect in the {part}.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "How is this claim covered?", "criteria": ["Covered in full", "Covered at half cost", "Not covered"]}}
    a = rng.choice([rng.randint(1, standard), rng.randint(standard + 1, extended), extended + rng.randint(1, 200)])
    b = rng.choice([x for x in [rng.randint(1, standard), rng.randint(standard + 1, extended), extended + rng.randint(1, 200)] if evaluate({"claim": bought + timedelta(days=x), "purchase": bought}) != evaluate({"claim": bought + timedelta(days=a), "purchase": bought})] or [a])
    return build(a), build(b)


def family_sla_response(rng):
    name = rng.choice(NAMES); target, breach = rng.choice([(4, 24), (8, 48), (24, 72), (1, 8)])
    unit = "hours"; topic = rng.choice(['a login failure', 'a duplicate charge', 'a missing invoice', 'an export error']); queue = rng.choice(['email', 'chat', 'phone'])
    def evaluate(f):
        if not _need(f, "hours"): return UNDETERMINED
        return 0 if f["hours"] <= target else 1 if f["hours"] <= breach else 2
    def build(h):
        return {"policy": f"Support responses within {target} {unit} meet the service level. Responses after {target} but within {breach} {unit} are a minor breach. Anything slower is a major breach.",
                "sentences": [(f"{name} opened a priority ticket about {topic}.", {}),
                              (f"The first response arrived {h} {unit} after the ticket was opened.", {"hours": h}),
                              (f"The ticket was routed through the {queue} queue.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "How does this response time rate against the service level?", "criteria": ["Met", "Minor breach", "Major breach"]}}
    levels = [rng.randint(1, target), rng.randint(target + 1, breach), breach + rng.randint(1, 100)]
    a, b = rng.sample(levels, 2)
    return build(a), build(b)


def family_late_fee(rng):
    name = rng.choice(NAMES); grace, cap = rng.choice([(5, 30), (10, 60), (15, 45)]); amount = rng.choice([120, 450, 980, 2300]); method = rng.choice(['bank transfer', 'card', 'cheque'])
    def evaluate(f):
        if not _need(f, "days_late"): return UNDETERMINED
        return 0 if f["days_late"] <= grace else 1 if f["days_late"] <= cap else 2
    def build(d):
        return {"policy": f"Invoices paid within {grace} days after the due date incur no fee. Payments between {grace + 1} and {cap} days late incur a 2% fee. Payments later than {cap} days incur a 10% fee and a hold on the account.",
                "sentences": [(f"{name}'s invoice for ${amount:,} fell due last quarter.", {}),
                              (f"Payment was received {d} days after the due date.", {"days_late": d}),
                              (f"The payment was made by {method}.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "Which fee tier applies?", "criteria": ["No fee", "2% fee", "10% fee and account hold"]}}
    levels = [rng.randint(0, grace), rng.randint(grace + 1, cap), cap + rng.randint(1, 60)]
    a, b = rng.sample(levels, 2)
    return build(a), build(b)


def family_volume_discount(rng):
    name = rng.choice(NAMES); item = rng.choice(ITEMS); t1, t2 = rng.choice([(10, 50), (25, 100), (5, 20), (100, 500)]); dest = rng.choice(['warehouse', 'storefront', 'branch office'])
    def evaluate(f):
        if not _need(f, "units"): return UNDETERMINED
        return 0 if f["units"] < t1 else 1 if f["units"] < t2 else 2
    def build(u):
        return {"policy": f"Orders of fewer than {t1} units are charged the list price. Orders of {t1} to {t2 - 1} units receive the volume discount. Orders of {t2} units or more receive the wholesale rate.",
                "sentences": [(f"{name} placed a business order for {item}.", {}),
                              (f"The order is for {u} units.", {"units": u}),
                              (f"Delivery is to a {dest}.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "Which pricing tier applies?", "criteria": ["List price", "Volume discount", "Wholesale rate"]}}
    levels = [rng.randint(1, t1 - 1), rng.randint(t1, t2 - 1), t2 + rng.randint(0, t2)]
    a, b = rng.sample(levels, 2)
    return build(a), build(b)


def family_shipping_delay(rng):
    """Day-precision date arithmetic with small thresholds (the skill `deadline` tests): promised vs delivered dates."""
    name = rng.choice(NAMES); item = rng.choice(ITEMS); promised = date(2026, rng.randint(1, 11), rng.randint(1, 28))
    minor, major = rng.choice([(1, 4), (2, 7), (3, 10)]); carrier = rng.choice(["the courier", "the postal service", "a freight partner"])
    def evaluate(f):
        if not _need(f, "delivered", "promised"): return UNDETERMINED
        late = (f["delivered"] - f["promised"]).days
        return 0 if late <= minor else 1 if late <= major else 2
    def build(offset):
        delivered = promised + timedelta(days=offset)
        return {"policy": f"Deliveries up to {minor} day{'s' if minor > 1 else ''} after the promised date count as on time. Deliveries {minor + 1} to {major} days after it are a minor delay and earn a shipping refund. Later deliveries are a major delay and earn a full refund.",
                "sentences": [(f"{name} ordered {item} with delivery promised for {_day(promised)}.", {"promised": promised}),
                              (f"The parcel was delivered on {_day(delivered)}.", {"delivered": delivered}),
                              (f"It was shipped by {carrier}.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "How is this delivery classified?", "criteria": ["On time", "Minor delay: shipping refund", "Major delay: full refund"]}}
    levels = [rng.randint(-3, minor), rng.randint(minor + 1, major), major + rng.randint(1, 30)]
    a, b = rng.sample(levels, 2)
    return build(a), build(b)


FAMILIES = {"return_window": family_return_window, "spend_threshold": family_spend_threshold, "authorization": family_authorization,
            "age_eligibility": family_age_eligibility, "quantity_limit": family_quantity_limit, "deadline": family_deadline,
            "warranty_claim": family_warranty_claim, "sla_response": family_sla_response, "late_fee": family_late_fee, "volume_discount": family_volume_discount,
            "shipping_delay": family_shipping_delay}
ORDINAL_FAMILIES = ("warranty_claim", "sla_response", "late_fee", "volume_discount", "shipping_delay")   # trainable Score-threshold families (deadline stays held out)


def label_of(item, drop=None):
    facts = {}
    for i, (_, f) in enumerate(item["sentences"]):
        if i != drop: facts.update(f)
    return item["evaluate"](facts)


def check_pair(a, b):
    """None if the pair is valid, else the reason it is rejected."""
    la, lb = label_of(a), label_of(b)
    if UNDETERMINED in (la, lb): return "label_undetermined"
    if la == lb: return "labels_equal"
    if sum(x[0] != y[0] for x, y in zip(a["sentences"], b["sentences"])) != 1 or len(a["sentences"]) != len(b["sentences"]):
        return "not_exactly_one_sentence_differs"
    for item, label in ((a, la), (b, lb)):
        evidence = 0
        for i, (_, facts) in enumerate(item["sentences"]):
            got = label_of(item, drop=i)
            if facts and got != UNDETERMINED: return "ablation_failed"      # evidence removed must make it undeterminable
            if not facts and got != label: return "invariance_failed"          # filler removed must not change the label
            evidence += bool(facts)
        if evidence < 1: return "no_evidence_sentence"
    return None


def to_request(item, family, pair_id, sibling, rng):
    order = list(range(len(item["sentences"]))); rng.shuffle(order)
    sentences = [item["sentences"][i][0] for i in order]
    q = {**item["question"], "label": label_of(item), "src": f"contrastive_{family}"}
    return {"state": {"policy": item["policy"], "case": " ".join(sentences)}, "questions": {"decision": q},
            "_meta": {"source": "contrastive", "family": family, "family_id": f"{family}/{pair_id}", "pair_id": pair_id, "sibling": sibling,
                      "repo": None, "revision": None, "split": "generated", "row": pair_id, "id": f"contrastive/{family}/{pair_id}/{sibling}",
                      "text_sha256": hashlib.sha256(" ".join(sentences).casefold().encode()).hexdigest(),
                      "row_sha256": hashlib.sha256(json.dumps([t for t, _ in item["sentences"]]).encode()).hexdigest()}}


def generate(n_pairs_per_family, seed, families=None):
    """Sibling-adjacent records plus a per-family report of rejections."""
    records, report = [], {}
    for family in (families or FAMILIES):
        rng = random.Random(f"{seed}:{family}")
        kept, rejected, reasons, attempts = 0, 0, {}, 0
        while kept < n_pairs_per_family and attempts < 50 * n_pairs_per_family:
            attempts += 1
            a, b = FAMILIES[family](rng)
            why = check_pair(a, b)
            if why:
                rejected += 1; reasons[why] = reasons.get(why, 0) + 1; continue
            pair_id = f"{seed}-{family}-{kept:04d}"
            order_seed = rng.getrandbits(64)
            records += [to_request(a, family, pair_id, "a", random.Random(order_seed)),
                        to_request(b, family, pair_id, "b", random.Random(order_seed))]
            kept += 1
        if kept < n_pairs_per_family:
            raise ValueError(f"{family}: only {kept}/{n_pairs_per_family} pairs passed checks ({reasons})")
        report[family] = {"pairs": kept, "rejected": rejected, "reasons": reasons}
    return records, report


def paired_flip(rows):
    """Pair-level metrics from benchmark rows carrying pair_id/sibling. A model that ignores the state cannot flip."""
    by_pair = {}
    for r in rows:
        if r.get("pair_id"):
            key = (r["pair_id"], r.get("question", "decision"))
            pair = by_pair.setdefault(key, {})
            if r["sibling"] in pair:
                raise ValueError("duplicate contrastive sibling")
            pair[r["sibling"]] = r
    if not by_pair:
        return None
    if any(set(p) != {"a", "b"} for p in by_pair.values()):
        raise ValueError("incomplete contrastive pair")
    prediction = lambda r: r["keys"][max(range(len(r["p"])), key=r["p"].__getitem__)]
    truth = lambda r: r["keys"][r["label"]]
    relevant = [p for p in by_pair.values() if truth(p["a"]) != truth(p["b"])]
    invariant = [p for p in by_pair.values() if truth(p["a"]) == truth(p["b"])]
    both = lambda ps: sum(all(prediction(r) == truth(r) for r in p.values()) for p in ps) / len(ps) if ps else None
    result = {"pairs": len(relevant),
              "flip_rate": sum(prediction(p["a"]) != prediction(p["b"]) for p in relevant) / len(relevant) if relevant else None,
              "both_correct_rate": both(relevant)}
    if invariant:
        result.update(invariant_pairs=len(invariant), invariance_rate=sum(prediction(p["a"]) == prediction(p["b"]) for p in invariant) / len(invariant),
                      invariant_both_correct_rate=both(invariant))
    return result
