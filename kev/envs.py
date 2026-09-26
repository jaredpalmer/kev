"""Multi-step decision environments with programmatic rewards, for the agentic RL pass after SFT (kev.rl).

Kev never generates text, so an agent step is one typed question: the state is what the episode has revealed so far and
the options are the actions available now. An episode is several such steps, and only its end is rewarded, which is what
a supervised log loss over single decisions cannot express: which record to open next, when enough is known to answer,
and when to escalate instead of guessing.

Investigation: a small staff directory whose records are hidden until opened. The task asks for an attribute reached by a
chain of manager references ("the city of the manager of the manager of Ada"). Actions: open a record, answer with a
value, or escalate. Some episodes redact a record on the chain, so no answer is knowable and escalating is right. Escalating is
rewarded only when the question is unknowable, so "always escalate" is not a safe harbour. Rewards are exact (the world is generated, the solver is the chain walk); nothing comes from a teacher model.

Sources: where is someone based this week? Several sources of different (unstated) reliability can each be asked once, at
a cost; their reports are noisy and can conflict. Actions: ask a source, answer with a candidate, or escalate. The best
action depends on the belief the reports so far support, so no fixed demonstration is optimal: the warm-up clones a
cheap heuristic (ask the first source, repeat its report), and the Bayes-optimal policy (`oracle`, exact by recursion
over the remaining sources) is the ceiling RL is read against, never a training target.

Every episode exposes `request()`, `actions()`, `step(action)`, `done`, `reward` (terminal), `spent` (step costs),
`outcome`, `demo()` (the warm-up's action) and `info()` (per-episode eval statistics, None where not applicable).

Splits draw from disjoint seed namespaces; phrasing templates 0-1 are for training and 2 is held out for evaluation, so an
evaluation reads a generalisation to an unseen surface form as well as unseen worlds.
"""
import random
from dataclasses import dataclass, field

NAMES = ("Ada", "Bruno", "Chen", "Dara", "Emeka", "Farah", "Goran", "Hana", "Ivo", "Jun", "Kira", "Luis", "Mona", "Nils", "Oren", "Priya")
CITIES = ("Austin", "Berlin", "Cairo", "Dublin", "Lagos", "Lima", "Osaka", "Oslo")
TEAMS = ("billing", "search", "payments", "infra", "mobile", "security")
ATTRS = {"city": CITIES, "team": TEAMS}
TEMPLATES = (
    "Find the {attr} of {chain}.",
    "Which {attr} belongs to {chain}? Look it up in the directory.",
    "Report the {attr} for {chain}, using only records you have opened.",
)
TRAIN_TEMPLATES, EVAL_TEMPLATES = (0, 1), (2,)
REWARD = {"correct": 1.0, "wrong": -1.0, "out_of_budget": -0.5}
ESCALATE = {True: -0.5, False: 0.5}   # by whether the answer was knowable: escalating everything must not be a safe harbour
STEP_COST = 0.02   # per record opened: prefer short investigations, but never so much that guessing beats looking


def chain_text(start, hops):
    return "the manager of " * hops + start


@dataclass
class Investigation:
    """One episode. `people` maps name -> {"manager", "city", "team"}; the task asks for `attr` of the person `hops`
    manager-steps up from `start`. `redacted` names a record that opens as unavailable (None: every record opens)."""
    people: dict
    start: str
    hops: int
    attr: str
    template: int
    redacted: str | None = None
    budget: int = 0
    opened: list = field(default_factory=list)
    done: bool = False
    reward: float = 0.0
    spent: float = 0.0
    outcome: str = ""

    def __post_init__(self):
        self.budget = self.budget or self.hops + 3

    def chain(self):
        """The names the solver opens, in order: start, its manager, ..., the target."""
        names = [self.start]
        for _ in range(self.hops): names.append(self.people[names[-1]]["manager"])
        return names

    def answer(self):
        """The true value, or None when a record the chain needs is redacted (the question is unknowable)."""
        return None if self.redacted in self.chain() else self.people[self.chain()[-1]][self.attr]

    def record_text(self, name):
        if name == self.redacted: return f"{name}: record unavailable"
        p = self.people[name]
        return f"{name}: manager {p['manager'] or 'none'}; city {p['city']}; team {p['team']}"

    def actions(self):
        """Choice keys, in a fixed order: open each unopened record, answer each value of the asked attribute, escalate."""
        return [f"open {n}" for n in self.people if n not in self.opened] + [f"answer {v}" for v in ATTRS[self.attr]] + ["escalate"]

    def request(self):
        """The current step as a SystemOneRequest body (one choice question over the available actions)."""
        state = {"task": TEMPLATES[self.template].format(attr=self.attr, chain=chain_text(self.start, self.hops)),
                 "directory": ", ".join(self.people),
                 "opened records": [self.record_text(n) for n in self.opened] or "none yet",
                 "records you may still open": self.budget - len(self.opened)}
        criteria = {k: ("look up this person's record" if k.startswith("open") else
                        "stop and hand the task to a human: the answer cannot be determined" if k == "escalate" else
                        "final answer") for k in self.actions()}
        return {"state": state, "questions": {"action": {"type": "choice", "instructions": "Choose the next action.", "criteria": criteria}}}

    def step(self, action):
        """Apply one action key; returns the step reward (the terminal reward on the last step, the step cost before)."""
        if self.done or action not in self.actions(): raise ValueError(f"invalid action {action!r}")
        if action.startswith("open "):
            self.opened.append(action[5:]); self.spent += STEP_COST
            if len(self.opened) > self.budget: return self._end("out_of_budget") - STEP_COST
            return -STEP_COST
        if action == "escalate": return self._end("escalate")
        truth = self.answer()
        return self._end("correct" if truth is not None and action == f"answer {truth}" else "wrong")

    def _end(self, outcome):
        self.done, self.outcome = True, outcome
        self.reward = ESCALATE[self.answer() is not None] if outcome == "escalate" else REWARD[outcome]
        return self.reward

    def oracle(self):
        """The solver's next action: walk the chain, escalate at a redacted record, answer once the target is open."""
        for name in self.chain():
            if name not in self.opened: return f"open {name}"
            if name == self.redacted: return "escalate"
        return f"answer {self.answer()}"

    demo = oracle

    def info(self):
        knowable = self.answer() is not None
        return {"knowable_correct": self.outcome == "correct" if knowable else None,
                "knowable_escalated": self.outcome == "escalate" if knowable else None,
                "unknowable_escalated": self.outcome == "escalate" if not knowable else None,
                "out_of_budget": self.outcome == "out_of_budget"}


def investigation(seed, split="train", n_people=6, max_hops=2, p_redact=0.25):
    """A reproducible episode: `split` ("train" or "eval") selects the seed namespace and the template set."""
    if split not in ("train", "eval"): raise ValueError(f"unknown split {split!r}")
    rng = random.Random(f"investigation:{split}:{seed}")
    names = rng.sample(NAMES, n_people)
    people = {}
    for i, n in enumerate(names):   # managers point earlier in the list, so every chain ends at the root
        people[n] = {"manager": rng.choice(names[:i]) if i else None, "city": rng.choice(CITIES), "team": rng.choice(TEAMS)}
    hops = min(rng.randint(1, max_hops), max(len(_ancestors(people, n)) for n in names))
    start = rng.choice([n for n in names if len(_ancestors(people, n)) >= hops])
    ep = Investigation(people=dict(rng.sample(sorted(people.items()), n_people)), start=start, hops=hops,
                       attr=rng.choice(sorted(ATTRS)), template=rng.choice(TRAIN_TEMPLATES if split == "train" else EVAL_TEMPLATES))
    if rng.random() < p_redact: ep.redacted = rng.choice(ep.chain())
    return ep


def _ancestors(people, name):
    out = []
    while people[name]["manager"]: name = people[name]["manager"]; out.append(name)
    return out


SOURCE_KINDS = {"badge log": 0.9, "travel booking": 0.8, "HR record": 0.7, "team calendar": 0.6, "colleague's guess": 0.4}
SOURCE_TEMPLATES = (
    "Where is {name} based this week? Answer only if the reports support it.",
    "Find out which office {name} is working from this week.",
    "Which city is {name} in this week? Escalate rather than guess.",
)
SOURCE_REWARD = {"correct": 1.0, "wrong": -1.0, "escalate": 0.0}
ASK_COST = 0.05


@dataclass
class Sources:
    """One episode: `truth` is among `candidates`; `sources` maps each askable source to the report it gives when asked
    (drawn at generation: the truth with the kind's reliability, else a uniformly wrong candidate)."""
    name: str
    candidates: list
    truth: str
    sources: dict
    template: int
    asked: list = field(default_factory=list)
    answered: str = ""
    done: bool = False
    reward: float = 0.0
    spent: float = 0.0
    outcome: str = ""

    def actions(self):
        return [f"ask {k}" for k in self.sources if k not in self.asked] + [f"answer {c}" for c in self.candidates] + ["escalate"]

    def request(self):
        state = {"task": SOURCE_TEMPLATES[self.template].format(name=self.name), "candidates": ", ".join(self.candidates),
                 "sources": ", ".join(self.sources),
                 "reports so far": [f"the {k} says {self.sources[k]}" for k in self.asked] or "none yet"}
        criteria = {k: ("ask this source (each question has a small cost)" if k.startswith("ask ") else
                        "stop and hand the task to a human: the reports do not settle it" if k == "escalate" else "final answer")
                    for k in self.actions()}
        return {"state": state, "questions": {"action": {"type": "choice", "instructions": "Choose the next action.", "criteria": criteria}}}

    def step(self, action):
        if self.done or action not in self.actions(): raise ValueError(f"invalid action {action!r}")
        if action.startswith("ask "):
            self.asked.append(action[4:]); self.spent += ASK_COST
            return -ASK_COST
        self.done, self.answered = True, action[7:] if action.startswith("answer ") else ""
        self.outcome = "escalate" if action == "escalate" else "correct" if action == f"answer {self.truth}" else "wrong"
        self.reward = SOURCE_REWARD[self.outcome]
        return self.reward

    def posterior(self, asked=None):
        """P(truth = c | the reports of `asked` (default: those asked so far)), uniform prior."""
        w = {c: 1.0 for c in self.candidates}
        for k in self.asked if asked is None else asked:
            r = SOURCE_KINDS[k]
            for c in w: w[c] *= r if c == self.sources[k] else (1 - r) / (len(self.candidates) - 1)
        z = sum(w.values())
        return {c: v / z for c, v in w.items()}

    def _value(self, reports):
        """(Bayes-optimal expected return from here, best action) given `reports` {source: report}."""
        w = {c: 1.0 for c in self.candidates}
        for k, rep in reports.items():
            r = SOURCE_KINDS[k]
            for c in w: w[c] *= r if c == rep else (1 - r) / (len(self.candidates) - 1)
        z = sum(w.values()); post = {c: v / z for c, v in w.items()}
        best_c = max(post, key=post.get)
        best = max((0.0, "escalate"), (2 * post[best_c] - 1, f"answer {best_c}"))
        for k in self.sources:
            if k in reports: continue
            r = SOURCE_KINDS[k]
            v = -ASK_COST
            for rep in self.candidates:
                p_rep = sum(post[c] * (r if c == rep else (1 - r) / (len(self.candidates) - 1)) for c in self.candidates)
                v += p_rep * self._value({**reports, k: rep})[0]
            if v > best[0] + 1e-12: best = (v, f"ask {k}")
        return best

    def oracle(self):
        """The Bayes-optimal next action (the ceiling, not a training target)."""
        return self._value({k: self.sources[k] for k in self.asked})[1]

    def optimal_value(self):
        return self._value({})[0]

    def demo(self):
        """The warm-up heuristic: ask the first source, then answer what it said."""
        return f"ask {next(iter(self.sources))}" if not self.asked else f"answer {self.sources[self.asked[0]]}"

    def info(self):
        return {"escalated": self.outcome == "escalate", "asks": len(self.asked), "optimal_value": self.optimal_value(),
                "answer_posterior": self.posterior()[self.answered] if self.answered else None}


def sources(seed, split="train", n_sources=3, n_candidates=4):
    if split not in ("train", "eval"): raise ValueError(f"unknown split {split!r}")
    rng = random.Random(f"sources:{split}:{seed}")
    candidates = rng.sample(CITIES, n_candidates); truth = rng.choice(candidates)
    reports = {}
    for k in rng.sample(sorted(SOURCE_KINDS), n_sources):
        reports[k] = truth if rng.random() < SOURCE_KINDS[k] else rng.choice([c for c in candidates if c != truth])
    return Sources(name=rng.choice(NAMES), candidates=candidates, truth=truth, sources=reports,
                   template=rng.choice(TRAIN_TEMPLATES if split == "train" else EVAL_TEMPLATES))


ENVS = {"investigation": investigation, "sources": sources}
