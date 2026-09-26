"""longdoc-v1, programmatic part: bundles of generated service agreements at a target length in tokens, where one
agreement decides each question. Written for this suite (no text, pool or template from hard-v1).

A bundle holds N agreements between a named provider and a named customer. Every agreement draws its own terms (payment
days, notice periods, uptime commitment, retention, breach notification, insurance, ...) from shared pools, so the values
one agreement states are the distractors of every other; each names one primary hosting city (unique in the bundle) and
lists its covered sites and their service tiers in a closing schedule. Four questions per bundle, each about one agreement:

  locate   which agreement names <city> as its primary hosting location (options: up to five agreements of the bundle)
  detail   one term of a named agreement, its clause placed at 10 %, 50 % or 90 % of the state (DEPTHS; cycled per record)
  multihop the service credit for an outage at a named site: the site's tier is in the closing schedule, the credit per
           tier in the service-levels section near the start of the same agreement
  absent   one term of a named agreement that states it (half) or omits it entirely while other agreements state it
           (half); the right answer is then "not stated"

detail, multihop and absent always offer the "not stated" option. Labels are computed by `solve` from `_meta.facts` (the
builder re-solves them after a JSON round trip); the generator never writes a label directly.
"""
import random

DEPTHS = (0.1, 0.5, 0.9)
NOT_STATED = "The agreement does not state this"
LETTERS = "abcde"

# ------------------------------------------------------------------------------------------------------------- pools
PREFIX = ["Alder", "Basalt", "Beacon", "Birchwood", "Calder", "Canopy", "Cardinal", "Cedarline", "Clearwater", "Corvid",
          "Crescent", "Delta Row", "Eastgate", "Ember", "Fairlight", "Fernhill", "Flint", "Foxglove", "Galena", "Glasshouse",
          "Halcyon", "Hawthorn", "Heron", "Ironbark", "Jadeport", "Kelp", "Keystone", "Lanternfish", "Larch", "Lodestar",
          "Maple Court", "Meridian", "Millbrook", "Moraine", "Nettle", "Northgate", "Obsidian", "Orchard", "Osprey", "Palisade",
          "Pewter", "Quillon", "Quince", "Rainier", "Rookery", "Saltmarsh", "Sandpiper", "Sequoia", "Shale", "Sorrel",
          "Stonecrop", "Sycamore", "Tamarack", "Teal", "Thornbury", "Tolland", "Umberfield", "Verdant", "Wexford", "Wren"]
NOUN = ["Analytics", "Cloud", "Data", "Digital", "Freight", "Grid", "Health", "Hosting", "Infrastructure", "Insights",
        "Labs", "Logistics", "Media", "Networks", "Payments", "Platforms", "Retail", "Robotics", "Security", "Software",
        "Storage", "Systems", "Telecom", "Therapeutics", "Utilities", "Ventures", "Works", "Foods", "Mobility", "Energy"]
SUFFIX = ["Inc.", "LLC", "Ltd", "plc", "GmbH", "S.A.", "Pty Ltd", "Corp."]
CITIES = ["Aberdeen", "Adelaide", "Albuquerque", "Antwerp", "Asheville", "Aarhus", "Bergen", "Bilbao", "Boise", "Bratislava",
          "Brno", "Calgary", "Cardiff", "Charleston", "Cork", "Curitiba", "Dayton", "Dresden", "Dunedin", "Durban", "Eindhoven",
          "Fresno", "Galway", "Gdansk", "Geelong", "Genoa", "Ghent", "Graz", "Halifax", "Hobart", "Irvine", "Izmir", "Kaunas",
          "Knoxville", "Kosice", "Leeds", "Leipzig", "Linz", "Lyon", "Malmo", "Mannheim", "Memphis", "Montpellier", "Nantes",
          "Newcastle", "Omaha", "Oulu", "Porto", "Poznan", "Quebec City", "Reno", "Rotterdam", "Salzburg", "Santander",
          "Savannah", "Spokane", "Tampere", "Toulouse", "Tucson", "Turku", "Utrecht", "Valencia", "Vilnius", "Winnipeg",
          "Wroclaw", "Zaragoza", "Tallinn", "Riga", "Plzen", "Bologna", "Trieste", "Lausanne", "Basel", "Innsbruck"]
SITES = ["Harlow depot", "Kessler Street warehouse", "Bayview clinic", "Northfield plant", "Juniper Road office", "Ridgeway store",
         "Canal Street hub", "Marston yard", "Westbrook lab", "Old Mill campus", "Fenwick terminal", "Sutton Row call centre",
         "Pinehurst branch", "Quayside office", "Linden Park depot", "Grange Road plant", "Orchard Lane store", "Hilltop data room",
         "Riverside kitchen", "Tanner Street studio", "Beacon Hill branch", "Southport warehouse", "Elm Court clinic",
         "Kingsway office", "Foundry Lane workshop", "Meadowbank depot", "Crown Street store", "Ashby distribution centre"]
TIERS = ("Tier 1", "Tier 2", "Tier 3")
CREDITS = ["5%", "7.5%", "10%", "12.5%", "15%", "20%", "25%", "30%"]

# term key -> section, values, clause wordings ({v} = value, {P}/{C} = provider / customer), question
TERMS = {
    "payment_days": ("Fees and Payment", [15, 20, 30, 45, 60, 75, 90], "{v} days",
                     ["{C} shall pay each undisputed invoice within {v} of the invoice date.",
                      "Undisputed invoices are payable {v} after receipt by {C}."],
                     "Within how many days of an invoice must the customer pay it?"),
    "late_interest": ("Fees and Payment", ["0.5%", "0.75%", "1%", "1.25%", "1.5%", "2%"], "{v} per month",
                      ["Overdue amounts accrue interest at {v} until paid.", "{P} may charge interest on late payments at {v}."],
                      "What interest rate applies to late payments?"),
    "price_cap": ("Fees and Payment", ["2%", "3%", "4%", "5%", "6%", "8%"], "{v}",
                  ["{P} may increase its fees once a year by no more than {v}.",
                   "Any annual fee increase is capped at {v} of the fees for the preceding year."],
                  "By how much at most may the provider raise its fees each year?"),
    "termination_notice": ("Termination", [30, 45, 60, 90, 120, 180], "{v} days",
                           ["Either party may terminate this Agreement for convenience on {v}' written notice.",
                            "This Agreement may be ended by either party without cause by giving {v}' notice in writing."],
                           "How much written notice does termination for convenience require?"),
    "liability_cap": ("Limitation of Liability", [3, 6, 9, 12, 18, 24], "the fees paid in the {v} months before the claim",
                      ["Each party's total liability under this Agreement is limited to {v}.",
                       "The aggregate liability of either party shall not exceed {v}."],
                      "What is each party's total liability capped at?"),
    "uptime": ("Service Levels", ["99.0%", "99.5%", "99.8%", "99.9%", "99.95%", "99.99%"], "{v}",
               ["{P} shall make the Services available at least {v} of each calendar month (the Uptime Commitment).",
                "The Uptime Commitment is availability of {v} in each calendar month."],
               "What monthly availability does the provider commit to?"),
    "retention_days": ("Data Protection", [7, 14, 30, 60, 90, 180, 365], "{v} days",
                       ["Within {v} of termination, {P} shall delete all Customer Data in its possession.",
                        "{P} shall retain Customer Data for no more than {v} after this Agreement ends and then destroy it."],
                       "Within how long after the agreement ends must the provider delete customer data?"),
    "breach_hours": ("Data Protection", [12, 24, 36, 48, 72, 96], "{v} hours",
                     ["{P} shall notify {C} of any personal data breach within {v} of becoming aware of it.",
                      "Notice of a security incident affecting Customer Data must reach {C} within {v}."],
                     "How quickly must the provider notify the customer of a data breach?"),
    "insurance": ("Insurance", ["$1,000,000", "$2,000,000", "$3,000,000", "$5,000,000", "$10,000,000"], "{v}",
                  ["{P} shall maintain professional liability insurance of at least {v} per claim.",
                   "Throughout the term {P} will carry errors and omissions cover with a limit of not less than {v}."],
                  "What minimum professional liability insurance must the provider carry?"),
    "audit_notice": ("Audit", [5, 10, 15, 20, 30, 45], "{v} business days",
                     ["{C} may audit {P}'s compliance once a year on {v}' written notice.",
                      "An audit under this section requires {v}' prior notice to {P}."],
                     "How much notice must the customer give before an audit?"),
    "renewal_months": ("Term and Renewal", [6, 12, 18, 24, 36], "{v} months",
                       ["After the initial term this Agreement renews automatically for successive periods of {v}.",
                        "Unless either party gives notice of non-renewal, the term extends by {v} at a time."],
                       "For how long does the agreement renew each time?"),
    "response_hours": ("Service Levels", [1, 2, 4, 8, 12], "{v} hours",
                       ["{P} shall respond to Severity 1 incidents within {v}.",
                        "The response time for a Severity 1 incident is {v} from the time it is reported."],
                       "What is the response time for a Severity 1 incident?"),
}
SECTION_ORDER = ["Definitions", "Services", "Service Levels", "Fees and Payment", "Term and Renewal", "Termination",
                 "Limitation of Liability", "Data Protection", "Insurance", "Audit"]
HOSTING = ["{P} shall host Customer Data primarily in its {city} facility.",
           "The primary hosting location for Customer Data is {P}'s data centre in {city}."]

# boilerplate sections: (title, sentence pool); a section takes 2-5 of its sentences in order. Nothing in them decides a question.
FILLER = [
    ("Confidentiality", ["Each party shall keep the other's Confidential Information secret and use it only to perform this Agreement.",
                         "Confidential Information does not include information that is or becomes public through no fault of the receiving party.",
                         "A party may disclose Confidential Information where required by law, after giving the other party notice where lawful.",
                         "The obligations in this section survive termination for five years.",
                         "On request, the receiving party shall return or destroy the disclosing party's Confidential Information."]),
    ("Force Majeure", ["Neither party is liable for delay caused by events beyond its reasonable control.",
                       "The affected party shall notify the other promptly and use reasonable efforts to resume performance.",
                       "If a force majeure event lasts more than sixty days, either party may terminate the affected Services.",
                       "Payment obligations are not suspended by a force majeure event."]),
    ("Notices", ["Notices under this Agreement must be in writing and delivered by hand, courier or email.",
                 "A notice sent by email is received when sent, unless the sender receives an automated message that it was not delivered.",
                 "Each party shall keep its notice address up to date.", "Notices to {P} must be copied to its legal department."]),
    ("Assignment", ["Neither party may assign this Agreement without the other's prior written consent, which shall not be unreasonably withheld.",
                    "Either party may assign this Agreement to a successor to all or substantially all of its business.",
                    "Any attempted assignment in breach of this section is void."]),
    ("Subcontracting", ["{P} may use subcontractors to perform the Services and remains responsible for their acts and omissions.",
                        "{P} shall maintain a list of subcontractors that process Customer Data and provide it on request.",
                        "{P} shall give {C} notice before engaging a new subcontractor that will process Customer Data."]),
    ("Intellectual Property", ["Each party retains all rights in its pre-existing intellectual property.",
                               "{C} owns all Customer Data and any reports generated from it for {C}.",
                               "{P} grants {C} a non-exclusive licence to use the Services during the term.",
                               "Feedback given by {C} may be used by {P} without restriction."]),
    ("Warranties", ["{P} warrants that the Services will be performed with reasonable skill and care.",
                    "{P} warrants that the Services will materially conform to the Documentation.",
                    "Except as expressly stated, all other warranties are excluded to the extent permitted by law.",
                    "{C} warrants that it has the right to provide Customer Data to {P}."]),
    ("Security", ["{P} shall maintain appropriate technical and organisational security measures.",
                  "{P} shall encrypt Customer Data in transit and at rest using industry-standard methods.",
                  "Access to Customer Data is limited to personnel who need it to provide the Services.",
                  "{P} shall carry out penetration testing of the Services at least annually."]),
    ("Business Continuity", ["{P} shall maintain a business continuity and disaster recovery plan for the Services.",
                             "{P} shall test the plan at least once a year and share a summary of the results on request.",
                             "Backups of Customer Data are stored in a location separate from the primary hosting location."]),
    ("Change Control", ["Either party may request a change to the Services by written change request.",
                        "No change is binding until both parties sign a change order.",
                        "{P} shall assess each change request and state its effect on fees and timelines within ten business days."]),
    ("Compliance", ["Each party shall comply with all laws that apply to its performance of this Agreement.",
                    "{P} shall comply with {C}'s reasonable site and security policies when on {C}'s premises.",
                    "Neither party shall offer or accept any bribe in connection with this Agreement."]),
    ("Publicity", ["Neither party may use the other's name or logo in publicity without prior written approval.",
                   "{P} may list {C} as a customer after {C} has approved the listing in writing."]),
    ("Non-Solicitation", ["During the term and for twelve months after, neither party shall solicit the other's employees who worked on the Services.",
                          "General advertisements not aimed at the other party's employees are not solicitation."]),
    ("Dispute Resolution", ["The parties shall first try to resolve any dispute through their relationship managers.",
                            "A dispute not resolved within twenty business days shall be escalated to senior executives.",
                            "Nothing in this section prevents a party from seeking urgent injunctive relief."]),
    ("Governing Law", ["This Agreement is governed by the laws of the jurisdiction in which {P} has its registered office.",
                       "The courts of that jurisdiction have exclusive jurisdiction over any dispute arising from this Agreement."]),
    ("Entire Agreement", ["This Agreement is the entire agreement between the parties about its subject matter.",
                          "It supersedes all earlier proposals, understandings and agreements about that subject matter.",
                          "Each party confirms that it has not relied on any statement not set out in this Agreement."]),
    ("Severability", ["If any provision is held invalid, the remaining provisions continue in full force.",
                      "The parties shall negotiate in good faith a valid provision that comes closest to the invalid one."]),
    ("Waiver", ["A failure to exercise a right is not a waiver of that right.", "A waiver is effective only if it is in writing and signed."]),
    ("Relationship of the Parties", ["The parties are independent contractors.", "Nothing in this Agreement creates a partnership, agency or joint venture.",
                                     "Neither party may bind the other to any obligation."]),
    ("Customer Responsibilities", ["{C} shall provide {P} with timely access to the information and personnel it reasonably needs.",
                                   "{C} is responsible for the accuracy of the data it submits to the Services.",
                                   "{C} shall keep its user credentials secure and notify {P} of any unauthorised use."]),
    ("Acceptable Use", ["{C} shall not use the Services to store or transmit unlawful material.",
                        "{C} shall not attempt to probe, scan or test the vulnerability of the Services without permission.",
                        "{P} may suspend access that breaches this section after notice where practicable."]),
    ("Reporting", ["{P} shall provide a monthly service report covering availability, incidents and open requests.",
                   "The parties shall hold a quarterly review meeting to discuss the service reports.",
                   "Reports are Confidential Information of both parties."]),
    ("Transition Assistance", ["On termination {P} shall provide reasonable assistance to migrate the Services to {C} or a new provider.",
                               "Transition assistance beyond thirty days is charged at {P}'s then-current rates.",
                               "{P} shall provide Customer Data in a commonly used machine-readable format."]),
    ("Anti-Slavery", ["Each party shall take reasonable steps to ensure there is no forced labour in its supply chain.",
                      "Each party shall provide the other with information about its compliance on reasonable request."]),
    ("Counterparts", ["This Agreement may be signed in any number of counterparts, each of which is an original.",
                      "Electronic signatures are as valid as handwritten ones."]),
    ("Third-Party Rights", ["A person who is not a party to this Agreement has no right to enforce any of its terms."]),
    ("Accessibility", ["{P} shall make reasonable efforts to keep the user interface of the Services accessible to people with disabilities.",
                       "{P} shall publish an accessibility statement for the Services and keep it current."]),
    ("Environmental Commitments", ["{P} shall report the energy use of the facilities that host the Services once a year.",
                                   "{P} shall prefer suppliers that commit to reducing their emissions."]),
]
DEFINITIONS = [("Affiliate", "an entity that controls, is controlled by or is under common control with a party"),
               ("Business Day", "a day other than a Saturday, Sunday or public holiday"),
               ("Customer Data", "all data submitted to the Services by or for {C}"),
               ("Documentation", "the user guides and specifications for the Services published by {P}"),
               ("Effective Date", "the date on which the last party signs this Agreement"),
               ("Services", "the hosted services described in Schedule 1"),
               ("Confidential Information", "information disclosed by one party to the other that is marked or would reasonably be understood as confidential"),
               ("Incident", "an unplanned interruption to, or reduction in the quality of, the Services"),
               ("Severity 1", "an Incident that makes the Services unavailable to all users"),
               ("Personal Data", "any information relating to an identified or identifiable natural person")]

# four new render styles for headings and clause numbers (none of hard-v1's)
STYLES = [
    {"heading": lambda n, t: f"ARTICLE {n} \u2014 {t.upper()}", "clause": lambda n, m, x: f"{n}.{m}  {x}"},
    {"heading": lambda n, t: f"Section {n}: {t}", "clause": lambda n, m, x: f"  ({'abcdefghijklmnopqrstuvwxyz'[m - 1]}) {x}"},
    {"heading": lambda n, t: f"### {n}. {t}", "clause": lambda n, m, x: f"{n}.{m} {x}"},
    {"heading": lambda n, t: f"{t} [\u00a7{n}]", "clause": lambda n, m, x: f"\u00b6{n}.{m} {x}"},
]


# ------------------------------------------------------------------------------------------------------------- solver
def solve(facts, q):
    """The label of question q (its spec in `_meta.questions`) from the bundle's facts: an option key."""
    if q["kind"] == "locate":
        winners = [k for k, i in q["option_docs"].items() if facts["docs"][i]["city"] == q["city"]]
        if len(winners) != 1: raise ValueError("locate needs exactly one offered agreement naming the city")
        return winners[0]
    doc = facts["docs"][q["doc"]]
    if q["kind"] == "multihop":
        tier = doc["sites"].get(q["site"])
        value = doc["credits"][tier] if tier is not None else NOT_STATED
    else:
        value = doc["terms"].get(q["term"], NOT_STATED)
    keys = [k for k, v in q["option_values"].items() if v == value]
    if len(keys) != 1: raise ValueError(f"{q['kind']}: {len(keys)} options match {value!r}")
    return keys[0]


# ------------------------------------------------------------------------------------------------------------- facts
def company(rng, used):
    """A company name whose first word no other party in the bundle uses."""
    while True:
        name = f"{rng.choice(PREFIX)} {rng.choice(NOUN)} {rng.choice(SUFFIX)}"
        if name.split()[0] not in used: used.add(name.split()[0]); return name


def value_text(term, v):
    return TERMS[term][2].format(v=v)


def make_doc(rng, used, city, absent=None, present=None):
    """Facts of one agreement: parties, hosting city, term values (a term is dropped with probability 0.12, `absent` always,
    `present` never), covered sites with their tiers (every tier used), credit per tier (three distinct values)."""
    doc = {"P": company(rng, used), "C": company(rng, used), "city": city, "style": rng.randrange(len(STYLES)), "seed": rng.getrandbits(32)}
    doc["terms"] = {k: rng.choice(spec[1]) for k, spec in TERMS.items() if k != absent and (k == present or rng.random() >= 0.12)}
    doc["credits"] = dict(zip(TIERS, sorted(rng.sample(CREDITS, 3), key=lambda c: -float(c.rstrip("%")))))
    names = rng.sample(SITES, rng.randint(4, 7))
    doc["sites"] = {s: TIERS[i] if i < 3 else rng.choice(TIERS) for i, s in enumerate(names)}
    return doc


# ------------------------------------------------------------------------------------------------------------- rendering
def render_doc(doc, target_chars, key=None, key_position=None):
    """(text, {tag: character offset}) of one agreement of about target_chars characters. Tags: each stated term, "city",
    "credit" (the tier -> credit clause) and "schedule" (the site -> tier list). The document's own RNG (doc["seed"]) makes
    every call produce the same content; `key` (a tag) moves the section holding it to body index key_position (depth control)."""
    rng = random.Random(doc["seed"])
    P, C = doc["P"], doc["C"]
    sub = lambda s: s.replace("{P}", P).replace("{C}", C)
    style = STYLES[doc["style"]]
    sections = {n: [] for n in SECTION_ORDER}
    sections["Definitions"] = [(None, f"\u201c{t}\u201d means {sub(d)}.") for t, d in rng.sample(DEFINITIONS, rng.randint(4, len(DEFINITIONS)))]
    sections["Services"] = [(None, sub(rng.choice(["{P} shall provide the Services described in Schedule 1 to {C} from the Effective Date.",
                                                    "From the Effective Date {P} will supply the Services set out in Schedule 1."]))),
                            ("city", sub(rng.choice(HOSTING)).replace("{city}", doc["city"]))]
    cr = doc["credits"]
    sections["Service Levels"] = [("credit", f"If availability in a calendar month falls below the Uptime Commitment, {P} shall credit {C} a "
                                             f"percentage of that month's fees for each affected site according to its service tier: {TIERS[0]} sites "
                                             f"{cr[TIERS[0]]}, {TIERS[1]} sites {cr[TIERS[1]]} and {TIERS[2]} sites {cr[TIERS[2]]}. "
                                             "Each site's tier is shown in Schedule 2.")]
    for term, spec in TERMS.items():
        wording = rng.choice(spec[3])   # drawn for every term, stated or not, so a dropped term leaves the others' wording unchanged
        if term in doc["terms"]:
            sections[spec[0]].append((term, sub(wording.replace("{v}", value_text(term, doc["terms"][term])))))
    for name in SECTION_ORDER[3:]:
        if not sections[name]:
            sections[name] = [(None, sub(rng.choice(["The parties shall act in good faith in matters covered by this section.",
                                                      "This section is subject to the other terms of this Agreement."])))]
    rng.shuffle(sections["Service Levels"]); rng.shuffle(sections["Fees and Payment"]); rng.shuffle(sections["Data Protection"])
    body = [(n, sections[n]) for n in SECTION_ORDER]
    filler = rng.sample(FILLER, len(FILLER))
    slots = [rng.random() for _ in FILLER]
    size = lambda: sum(len(x) + 12 for _, cl in body for _, x in cl) + 40 * len(body)
    for (title, pool), u in zip(filler, slots):
        if size() >= target_chars: break
        k = 2 + int(u * 10) % max(1, len(pool) - 1)
        body.insert(3 + int(u * 1000) % (len(body) - 2), (title, [(None, sub(x)) for x in pool[:min(k, len(pool))]]))
    if key is not None:
        i = next(j for j, (_, cl) in enumerate(body) if any(t == key for t, _ in cl))
        item = body.pop(i)
        body.insert(min(max(key_position, 0), len(body)), item)
    title = ["Master Services Agreement", "Hosted Services Agreement", "Cloud Services Agreement", "Managed Services Agreement"][doc["seed"] % 4]
    lines = [title.upper(), f"between {P} (the Provider) and {C} (the Customer)", ""]
    offsets, pos = {}, 0
    for line in lines: pos += len(line) + 1
    for n, (name, clauses) in enumerate(body, start=1):
        head = style["heading"](n, name); lines.append(head); pos += len(head) + 1
        for m, (tag, x) in enumerate(clauses, start=1):
            line = style["clause"](n, m, x)
            if tag: offsets[tag] = pos + len(line) - len(x)
            lines.append(line); pos += len(line) + 1
        lines.append(""); pos += 1
    tail = ["SCHEDULE 1 \u2014 SERVICES", "Hosted services, support and maintenance as described in the Documentation.", "",
            "SCHEDULE 2 \u2014 COVERED SITES AND SERVICE TIERS"]
    offsets["schedule"] = pos + sum(len(x) + 1 for x in tail)
    lines += tail + [f"{site}: {tier}" for site, tier in doc["sites"].items()]
    return "\n".join(lines), offsets, len(body)


# ------------------------------------------------------------------------------------------------------------- bundles
DOC_TOKENS = {4096: 760, 8192: 1600, 16384: 3000, 32768: 3500, 65536: 4000}   # typical agreement length per bucket
CHARS_PER_TOKEN = 4.1   # this text under the Qwen3.8 tokenizer (measured 3.9-4.2); the builder measures exact counts


def assemble(docs, texts, order):
    """The state (a string) and each agreement's character start, agreements in `order`."""
    intro = f"Procurement file: {len(docs)} service agreements on record, in no particular order.\n\n"
    parts, starts, pos = [intro], {}, len(intro)
    for k, i in enumerate(order, start=1):
        head = f"===== Agreement {k} of {len(docs)} =====\n"
        starts[i] = pos + len(head)
        block = head + texts[i] + f"\n===== End of agreement {k} =====\n\n"
        parts.append(block); pos += len(block)
    return "".join(parts).rstrip() + "\n", starts


def options(rng, correct, pool, preferred, n=4):
    """correct plus n-1 distinct distractors (values stated elsewhere in the bundle first), shuffled, then NOT_STATED last.
    `correct` may itself be NOT_STATED (then n distractors are drawn and NOT_STATED is the answer)."""
    values = [] if correct == NOT_STATED else [correct]
    for v in rng.sample(sorted(set(preferred), key=str), len(set(preferred))) + rng.sample(pool, len(pool)):
        if len(values) == n: break
        if v != correct and v not in values: values.append(v)
    rng.shuffle(values)
    return dict(zip(LETTERS, values + [NOT_STATED]))


def generate(rng, T, depth, count_tokens):
    """One bundle at bucket T (state tokens, nominal) with its detail clause near `depth`, or None when the draw misses
    the length window or the depth tolerance (the caller draws again). count_tokens(text) -> state tokens."""
    lo, hi = int(0.84 * T), int(0.93 * T)
    base = DOC_TOKENS[T]
    used, cities = set(), rng.sample(CITIES, len(CITIES))
    t_detail, t_absent = rng.sample(sorted(TERMS), 2)
    absent_now = rng.random() < 0.5
    docs = []
    def add(**kw):
        docs.append(make_doc(rng, used, cities[len(docs)], **kw)); return len(docs) - 1
    D = add(present=t_detail); M = add(); A = add(absent=t_absent if absent_now else None, present=None if absent_now else t_absent); L = add()
    lengths = {}
    def text_of(i, key=None, pos=None):
        return render_doc(docs[i], lengths.setdefault(i, rng.uniform(0.7, 1.3) * base * CHARS_PER_TOKEN), key, pos)
    while sum(len(text_of(i)[0]) for i in range(len(docs))) / CHARS_PER_TOKEN < 0.885 * T:
        add()
    others = [i for i in range(len(docs)) if i not in (D, M, A, L)]
    # length: measure once, then drop or add filler agreements to land in [lo, hi]
    texts = {i: text_of(i) for i in range(len(docs))}
    order = [i for i in range(len(docs))]; rng.shuffle(order)
    for _ in range(6):
        n = count_tokens(assemble(docs, {i: t[0] for i, t in texts.items()}, order)[0])
        if lo <= n <= hi: break
        if n > hi and others:
            drop = others.pop(); order.remove(drop); del texts[drop]
        elif n > hi:
            return None
        else:
            j = add(); others.append(j); texts[j] = text_of(j); order.insert(rng.randrange(len(order) + 1), j)
    else:
        return None
    if absent_now and not any(t_absent in docs[i]["terms"] for i in order if i != A):
        return None                                              # an absent term must be stated by some other agreement in the file
    # depth: the detail agreement's slot and the position of its clause's section, chosen to bring the clause nearest `depth`
    n_sections = texts[D][2]
    best = None
    rest = [i for i in order if i != D]
    for slot in range(len(order)):
        for sec in range(n_sections):
            t = text_of(D, t_detail, sec)
            cand = rest[:slot] + [D] + rest[slot:]
            state, starts = assemble(docs, {**{i: texts[i][0] for i in rest}, D: t[0]}, cand)
            frac = (starts[D] + t[1][t_detail]) / len(state)
            if best is None or abs(frac - depth) < abs(best[0] - depth): best = (frac, cand, t)
    _, order, tD = best
    texts[D] = tD
    state, starts = assemble(docs, {i: texts[i][0] for i in order}, order)
    total = count_tokens(state)
    if not lo <= total <= hi: return None
    token_depth = lambda i, tag: round(count_tokens(state[:starts[i] + texts[i][1][tag]]) / total, 4) if tag in texts[i][1] else None
    if abs(token_depth(D, t_detail) - depth) > 0.08: return None
    # facts: only the agreements in the file, re-indexed in file order
    index = {i: k for k, i in enumerate(order)}
    facts = {"docs": [{k: docs[i][k] for k in ("P", "C", "city", "terms", "credits", "sites")} for i in order]}
    name = lambda i: f"the agreement between {docs[i]['P']} and {docs[i]['C']}"
    title = lambda i: f"The agreement between {docs[i]['P']} and {docs[i]['C']}"
    specs, questions = {}, {}
    # locate
    offered = [L] + rng.sample([i for i in order if i != L], min(4, len(order) - 1))
    rng.shuffle(offered)
    specs["locate"] = {"kind": "locate", "city": docs[L]["city"], "option_docs": {LETTERS[j]: index[i] for j, i in enumerate(offered)},
                       "doc": index[L], "depth": token_depth(L, "city")}
    questions["locate"] = {"type": "choice", "instructions": f"Which agreement in this file names {docs[L]['city']} as the primary hosting location for customer data?",
                           "criteria": {LETTERS[j]: title(i) for j, i in enumerate(offered)}}
    # detail and absent
    for qid, i, term in (("detail", D, t_detail), ("absent", A, t_absent)):
        correct = docs[i]["terms"].get(term, NOT_STATED)
        vals = options(rng, correct, TERMS[term][1], [docs[j]["terms"][term] for j in order if j != i and term in docs[j]["terms"]])
        specs[qid] = {"kind": qid, "term": term, "doc": index[i], "option_values": vals, "stated": term in docs[i]["terms"],
                      "depth": token_depth(i, term), "target_depth": depth if qid == "detail" else None}
        questions[qid] = {"type": "choice", "instructions": f"Under {name(i)}: {TERMS[term][4]}",
                          "criteria": {k: (v if v == NOT_STATED else value_text(term, v)) for k, v in vals.items()}}
    # multihop
    site = rng.choice(sorted(docs[M]["sites"]))
    cr = docs[M]["credits"]
    vals = options(rng, cr[docs[M]["sites"][site]], CREDITS, [c for c in cr.values()])
    specs["multihop"] = {"kind": "multihop", "site": site, "doc": index[M], "option_values": vals,
                         "depth": token_depth(M, "credit"), "depth_schedule": token_depth(M, "schedule")}
    questions["multihop"] = {"type": "choice", "instructions": f"Under {name(M)}, what service credit applies for a month in which availability at the {site} falls below the Uptime Commitment?",
                             "criteria": dict(vals)}
    return {"state": state, "questions": questions, "facts": facts, "specs": specs, "state_tokens": total, "agreements": len(order)}
