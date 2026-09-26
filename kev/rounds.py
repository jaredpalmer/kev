"""Registered research rounds as data: one spec per round, one engine for every stage.

A round (PLAN.md, rounds 5-20) always has the same shape. Arms are config-only trials (or checkpoints made from them,
round 20's interpolations) compared with a parent checkpoint;
each arm and its parent are read on a list of suites; a rule compares every arm with its parent on paired,
record-clustered bootstraps (primaries with a lower bound, guards with thresholds, pooled panels); the passing arm with
the best score is the size's candidate; confirmation stages (test partitions, the locked read) are read once. The spec
(`experiments/rounds/r<N>.json`, schema below) is committed before any training or read, like the PLAN registration it
encodes.

    uv run python -m kev.rounds validate      experiments/rounds/r18.json   # well-formed, every parent read exists, plans validate
    uv run python -m kev.rounds launch        experiments/rounds/r18.json   # modal_app.py::study for each study (after `modal deploy`)
    uv run python -m kev.rounds watch         experiments/rounds/r18.json   # pull + read each finished trial once, then the read-out
    uv run python -m kev.rounds launch-reads  experiments/rounds/r18.json [--arms a,b] [--stage tests --arm a]
    uv run python -m kev.rounds readout       experiments/rounds/r18.json   # -> runs/r18-readout/round18.json + a table
    uv run python -m kev.rounds confirm       experiments/rounds/r18.json --stage tests [--arm a]   # -> runs/r18-verdict/<size>-<stage>.json

Every comparison is served-vs-served: each side's rows are served at the temperature fitted on its own trial's
development rows (`kev.metrics.served`), or, for the arms of a round that registers a `temperature` pool, on that pool;
unknowable records are scored only by `unknowable_report`, and every delta is `kev.metrics.paired_bootstrap` (2,000
resamples, seed 0, micro), the registered read since round 5.

Spec (paths are relative to the repo root; templates take {round}, {arm}, {size}, {tag}):
    round, registered                 the round number and where its registration lives
    archive                           a recorded round: the git tag holding the plans, runs and data it names (validate lists
                                      what this checkout lacks instead of failing; launch refuses a record)
    gpu, app                          Modal GPU and KEV_APP_NAME for launches (optional)
    studies   {name: {plan, suite, transfer, gpu, timeout, budget}}   one modal_app.py::study call each (budget >= its admission bound)
    reads     {tag: {suite, flags?} | {entrypoint: "locked_test", decision}}  what each read tag scores (entrypoint: ENTRYPOINTS)
    read_timeout {size: seconds}      overrides modal_app.READ_TIMEOUTS for one size (a 27B's fp32 reads)
    locked_args  {size: [args]}       extra modal_app.py::locked_test switches for one size (a 27B's GPU memory)
    parents   {name: {trial, checkpoint?, reads: {tag: dir}}}   checkpoint (Hub id[@rev]) only when /<trial>/checkpoint is not on the volume
    arms      {name: {trial?, checkpoint?, parent, reads?, select?, transfer_read?, trained_on?}}   name = "<size>-<label>"; reads
                                      default to arm_reads; select false = reported, never the candidate (attribution arms); an
                                      arm without a trial (an interpolated checkpoint) names its /runs/... checkpoint, needs the
                                      round's `temperature` pool (it has no development rows) and, if the rule reads
                                      "transfer", a transfer_read; trained_on [suite dirs] names what such a checkpoint was
                                      trained on (default: covered by the round's trial arms, e.g. the trials it interpolates)
    arm_reads template of an arm's read directory (default "runs/r{round}-{arm}-{tag}")
    transfer_read  a read tag (round level, or per arm, which wins; null = the trial's own) whose rows are an arm's "transfer"
                                      rows instead of its trial's in-trial transfer read (a checkpoint without a trial has none)
    temperature {reads: [tags], sources?: {tag: [source]}, exclude_reads?: [tags]}   every ARM (parents keep their trial's
                                      development rows) is served at the temperature fitted, as everywhere (kev.metrics.served),
                                      on the knowable rows of `reads` pooled (a read listed in `sources` gives only those
                                      sources' rows), minus every record whose id is in the `exclude_reads` rows; the arm's own
                                      rule-stage reads are used at every stage; no pooled tag may sit in a panel that a
                                      temperature-dependent criterion reads (see pool_problems), and no pooled read may share
                                      data with any arm's training data (pool_training_problems: round 19's failure mode)
    drop_ids  record ids dropped on both sides of every comparison (devtools-v1's duplicated ids)
    rule      {panels, unknowable?, criteria, rank}
    confirm   {stage: {candidate_reads, parent_reads?, panels, criteria}}
A panel is {reads: [tags], metrics: [bootstrapped], report: [value only], source?: filter, versus?: {name: dir}, by_length?};
the tag "transfer" is the trial's own in-trial transfer read (or the arm's transfer_read). by_length (true, or {edges?:
[tokens], tokenizer?: [name, revision]}) adds acc / ece / brier / confident_error_rate per state-token bucket
(kev.metrics.calibration_by_length: "<metric>_<bucket>" entries, e.g. ece_16k_plus, value only) with state tokens counted
from the reads' suite records. A criterion is {left, op, right, plus?} where left is a path
("<panel>.<metric>.<candidate|parent|delta|lower|upper>", "<panel>.<metric>_<bucket>.<candidate|parent>",
"unknowable.<candidate|parent>") or a list [a, b] meaning a - b, and right is a number or a path (plus is added to it).
rank is a list of {by: path | [paths summed], order}.

Every arm's read-out records where its temperature came from (`temperature_source`: the pool with its reads and question
count, or its trial's development rows with their suite) and its parent's (`parent_temperature_source`: its trial's
development rows, plus `shipped`, the head.pt temperature when it is here), and the printed table warns when an arm's rows
are a training corpus's (in distribution: round 19's failure mode). From round 21 (POOL_REQUIRED_FROM) a rule or
confirmation with a temperature-dependent criterion must register a `temperature` pool: validate and launch refuse it
otherwise; earlier rounds get a warning, so their recorded specs keep validating.
"""
import argparse
import functools
import operator
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

from kev.metrics import (LENGTH_EDGES, LENGTH_METRICS, calibration_by_length, length_buckets, metrics, paired_bootstrap, raw_row, recorded,
                         scored_rows, served, served_at, tempered_row, unknowable_report)
from kev.suite import ADMISSION_TOKENIZER, digest, file_lock, load_split, read_json, read_manifest, write_json

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = 2000   # the registered resample count since round 5
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
FIELDS = ("candidate", "parent", "delta", "lower", "upper")
TRANSFER = "transfer"   # the in-trial transfer read (runs/<study>/<trial>/transfer)
STAGGER = 60            # seconds between `modal run` launches: Modal refuses more than ~3 app creations a minute
ENTRYPOINTS = ("benchmarks", "locked_test")   # the modal_app.py entrypoints a read can go through
IN_FLIGHT = 7200        # seconds an arm's launched reads are presumed running (the longest suite timeout in modal_app.READ_TIMEOUTS)


# --- rows ------------------------------------------------------------------------------------------------------------

def paired(candidate, reference, metric):
    """The registered paired read: record-clustered bootstrap of candidate - reference, micro, 2,000 resamples, seed 0."""
    x = paired_bootstrap(candidate, reference, samples=SAMPLES, seed=0, metric=metric, aggregation="micro")
    return {"delta": x[f"micro_{metric}_delta"], "ci95": x["ci95"]}


def temperature(trial, root=ROOT):
    """The temperature fitted on a trial's own development rows (decision-v7's, or its suite's; how every checkpoint here is
    served unless the round registers a `temperature` pool for its arms)."""
    return served(read_json(Path(root) / trial / "development/rows.json"), [])[0]


def served_clean(rows, t):
    """Every clean row served at t, unknowable records included (what unknowable_report scores; served_at drops them)."""
    return [tempered_row(raw_row(recorded(r)), t) for r in rows if r["variant"] == "clean"]


class Pool(NamedTuple):
    """Where a pooled temperature is fitted (spec `temperature`, resolved to one arm's read directories)."""
    reads: list     # directories whose rows are pooled
    sources: dict   # {directory: [source, ...]}: the only sources pooled from that read (absent = every source)
    exclude: list   # directories whose record ids are removed from the pool


def select_rows(reads, exclude=()):
    """(pooled rows, rows before the exclusion): every rows.json of reads, as (path, sources or None), concatenated with
    each read limited to its sources, minus every record whose id appears in an exclude rows.json. The one row selection
    of a pooled temperature, here and in scripts/calibrate_checkpoint.py (which writes it into a checkpoint)."""
    excluded = {r["id"] for path in exclude for r in read_json(path)}
    rows = [r for path, sources in reads for r in read_json(path) if sources is None or r["source"] in sources]
    return [r for r in rows if r["id"] not in excluded], rows


def pooled_temperature(pool, root=ROOT):
    """(temperature, fit report): the temperature() objective (kev.metrics.served: knowable clean rows, TEMPERATURE_FIT)
    on select_rows of the pool."""
    kept, rows = select_rows([(Path(root) / d / "rows.json", pool.sources.get(d)) for d in pool.reads], [Path(root) / d / "rows.json" for d in pool.exclude])
    return served(kept, [])[0], {"reads": pool.reads, **({"sources": pool.sources} if pool.sources else {}), "exclude_reads": pool.exclude,
                                 "questions": len(scored_rows(kept)), "excluded_questions": len(scored_rows(rows)) - len(scored_rows(kept))}


class Side:
    """One checkpoint of a comparison: its trial, where each read tag's rows live and its served temperature (fitted on
    the trial's development rows, or on `pool` when the round registers one for its arms)."""

    def __init__(self, trial, dirs, root=ROOT, drop=(), pool=None, checkpoint=None, suite=None, shipped=None):
        self.trial, self.dirs, self.root, self.drop, self.pool, self.checkpoint = trial, dirs, Path(root), set(drop), pool, checkpoint
        self.suite = suite       # the suite the trial trained and was scored on (its development rows'), when known
        self.shipped = shipped   # a parent's: () -> the temperature its checkpoint ships (head.pt), or None when not here
        self._t, self._served, self.fit = None, {}, None

    def rows_path(self, tag):
        return self.root / self.dirs[tag] / "rows.json"

    def has(self, tag):
        return tag in self.dirs and self.rows_path(tag).exists()

    def absent_fit(self):
        """The rows the temperature needs that are not here (pool directories, or the trial's development rows)."""
        needed = [*self.pool.reads, *self.pool.exclude] if self.pool else [f"{self.trial}/development"]
        return [d for d in needed if not (self.root / d / "rows.json").exists()]

    @property
    def t(self):
        if self._t is None:
            if self.pool: self._t, self.fit = pooled_temperature(self.pool, self.root)
            else: self._t = temperature(self.trial, self.root)
        return self._t

    def temperature_source(self):
        """Where the served temperature was fitted (read-out provenance): the pool (its reads and knowable question count),
        or the trial's development rows, their suite and whether that suite is a training corpus (then the temperature is
        in distribution: round 19's failure mode)."""
        if self.pool: return {"kind": "pool", "reads": self.pool.reads, "questions": self.fit["questions"] if self.fit else None}
        m = suite_manifest(self.suite) if self.suite else None
        return {"kind": "trial development rows", "rows": f"{self.trial}/development", "suite": self.suite,
                "training_corpus": bool(trainable_sources(m)) if m is not None else None, **({"shipped": self.shipped()} if self.shipped else {})}

    def served(self, tag):
        if tag not in self._served: self._served[tag] = served_at(read_json(self.rows_path(tag)), self.t)
        return self._served[tag]

    def panel(self, spec):
        """A panel's served, knowable rows, reads concatenated in the order listed."""
        rows = [r for tag in spec["reads"] for r in self.served(tag)]
        return [r for r in rows if r["id"] not in self.drop and ("source" not in spec or r["source"] == spec["source"])]

    def unknowable_share(self, tag):
        return unknowable_report(served_clean(read_json(self.rows_path(tag)), self.t))["share_at_0_9"] if self.has(tag) else None


# --- spec ------------------------------------------------------------------------------------------------------------

def load(path):
    spec = read_json(path)
    spec.setdefault("arm_reads", "runs/r{round}-{arm}-{tag}")
    return spec


def size_of(arm):
    return arm.split("-")[0]


def locations(where, spec, **values):
    """{tag: dir} from a template string (one for every tag of the spec) or an explicit {tag: template} mapping."""
    tags = [*spec.get("reads", {}), TRANSFER]
    if isinstance(where, str):
        return {tag: where.format(round=spec["round"], tag=tag, **values) for tag in tags}
    return {tag: d.format(round=spec["round"], tag=tag, **values) for tag, d in where.items()}


def transfer_read(spec, arm):
    """The read tag whose rows are the arm's "transfer" rows, or None for its trial's in-trial transfer read."""
    a = spec["arms"][arm]
    return a["transfer_read"] if "transfer_read" in a else spec.get("transfer_read")


def arm_side(spec, arm, root=ROOT, stage=None):
    """The arm's side. Its "transfer" rows and its temperature pool come from its rule-stage reads at every stage (the
    development reads it was selected on); a stage's candidate_reads locate the stage's own panels."""
    a = spec["arms"][arm]
    development = locations(a.get("reads", spec["arm_reads"]), spec, arm=arm, size=size_of(arm))
    where = spec["confirm"][stage]["candidate_reads"] if stage else a.get("reads", spec["arm_reads"])
    tag = transfer_read(spec, arm)
    transfer = development.get(tag) if tag else f"{a['trial']}/transfer" if a.get("trial") else None
    dirs = {**locations(where, spec, arm=arm, size=size_of(arm)), **({TRANSFER: transfer} if transfer else {})}
    pool, registered = None, spec.get("temperature")
    if registered:
        fit = {**development, **({TRANSFER: transfer} if transfer else {})}
        pool = Pool([fit[r] for r in registered["reads"]], {fit[r]: s for r, s in registered.get("sources", {}).items()}, [fit[r] for r in registered.get("exclude_reads", [])])
    suite = trial_suite(spec, a["trial"], root) if a.get("trial") and not pool else None
    return Side(a.get("trial"), dirs, root, spec.get("drop_ids", ()), pool, a.get("checkpoint"), suite)


def parent_side(spec, arm, root=ROOT, stage=None):
    p = spec["parents"][spec["arms"][arm]["parent"]]
    where = (spec["confirm"][stage].get("parent_reads") if stage else None) or p["reads"]
    dirs = {**locations(where, spec, arm=arm, size=size_of(arm)), TRANSFER: f"{p['trial']}/transfer"}
    return Side(p["trial"], dirs, root, spec.get("drop_ids", ()), suite=trial_suite(spec, p["trial"], root), shipped=lambda: shipped_temperature(p, root))


def rule_tags(rule):
    return {tag for panel in rule["panels"].values() for tag in panel["reads"]} | ({rule["unknowable"]} if rule.get("unknowable") else set())


def _paths(criterion):
    left = criterion["left"] if isinstance(criterion["left"], list) else [criterion["left"]]
    return [*left, *([criterion["right"]] if isinstance(criterion["right"], str) else [])]


class Validation(NamedTuple):
    problems: list   # the spec is malformed, or something a launch or a read-out needs is missing
    archived: list   # files a recorded round names that this checkout does not carry (they live on spec["archive"])


def validate(spec, root=ROOT, rows=True, plans=True, partitions=False):
    """Validation(problems, archived). rows: every parent's development rows and every read the rule needs from it exist
    under root (what a read-out, and so a launch, needs); plans: every read suite exists, every plan trial passes
    kev.experiment.validated_trial against its suite manifest and fits its budget; partitions: plans go through load_plan
    instead (verifies the partitions, may fetch them from the Hub). A recorded round (`archive`: the git tag holding its
    evidence) lists absent files under `archived` instead of failing; a round without it must have them all."""
    problems, archived, root = [], [], Path(root)
    absent = (archived if spec.get("archive") else problems).append
    for key in ("round", "registered", "parents", "arms", "reads", "rule"):
        if key not in spec: problems.append(f"missing key {key!r}")
    if problems: return Validation(problems, archived)
    tags = {*spec["reads"], TRANSFER}
    for name, r in spec["reads"].items():
        if r.get("entrypoint", "benchmarks") not in ENTRYPOINTS: problems.append(f"read {name}: entrypoint {r['entrypoint']!r} is not one of {ENTRYPOINTS}"); continue
        if r.get("entrypoint", "benchmarks") == "benchmarks" and not r.get("suite"): problems.append(f"read {name}: no suite")
        if r.get("entrypoint") == "locked_test" and not r.get("decision"): problems.append(f"read {name}: locked_test needs a decision suite")
        if plans and r.get("suite") and not (root / r["suite"]).exists(): absent(f"read {name}: {r['suite']} not in this checkout")
    stages = {None: spec["rule"], **spec.get("confirm", {})}
    reads_transfer = any(TRANSFER in panel["reads"] for rule in stages.values() for panel in rule["panels"].values())
    for name, a in spec["arms"].items():
        if a.get("parent") not in spec["parents"]: problems.append(f"arm {name}: unknown parent {a.get('parent')!r}")
        if "-" not in name: problems.append(f"arm {name}: name must be <size>-<label>")
        tag = transfer_read(spec, name)
        if tag is not None and (tag not in spec["reads"] or spec["reads"][tag].get("entrypoint", "benchmarks") != "benchmarks"):
            problems.append(f"arm {name}: transfer_read {tag!r} is not a benchmarks read of this spec")
        if tag is not None and isinstance(a.get("reads"), dict) and tag not in a["reads"]: problems.append(f"arm {name}: its reads do not locate its transfer_read {tag!r}")
        if not a.get("trial"):
            if not str(a.get("checkpoint", "")).startswith("/runs/"): problems.append(f"arm {name}: an arm without a trial needs a checkpoint on the runs volume (/runs/...)")
            if not spec.get("temperature"): problems.append(f"arm {name}: no trial, so no development rows to fit its temperature on; register a `temperature` pool")
            if reads_transfer and tag is None: problems.append(f"arm {name}: no trial, so no in-trial transfer read; give it a transfer_read")
    problems += pool_problems(spec, stages)
    conflicts, unknown_training = pool_training_problems(spec, root)
    problems += conflicts + pool_required_problems(spec)
    for line in unknown_training: absent(line)
    for stage, rule in stages.items():
        where = f"confirm.{stage}" if stage else "rule"
        for pname, panel in rule["panels"].items():
            unknown = set(panel["reads"]) - tags
            if unknown: problems.append(f"{where} panel {pname}: unknown read tags {sorted(unknown)}")
            if not panel.get("metrics") and not panel.get("report"): problems.append(f"{where} panel {pname}: nothing to compute")
            problems += [f"{where} panel {pname}: {p}" for p in by_length_problems(panel, spec)]
        for cname, c in rule["criteria"].items():
            if c.get("op") not in OPS: problems.append(f"{where} criterion {cname}: op {c.get('op')!r}")
            for path in _paths(c):
                if not _known_path(path, rule): problems.append(f"{where} criterion {cname}: unknown path {path!r}")
        for key in rule.get("rank", []):
            for path in key["by"] if isinstance(key["by"], list) else [key["by"]]:
                if not _known_path(path, rule): problems.append(f"{where} rank: unknown path {path!r}")
        if stage and "candidate_reads" not in rule: problems.append(f"{where}: no candidate_reads")
    if rule_tags(spec["rule"]) - tags: problems.append(f"rule.unknowable: unknown read tag {spec['rule']['unknowable']!r}")
    for study, s in spec.get("studies", {}).items() if plans else ():
        if not (root / s["plan"]).exists(): absent(f"study {study}: plan {s['plan']} not in this checkout")
        elif (root / s["suite"] / "manifest.json").exists(): problems += _validate_plan(study, s, root, partitions)
        else: absent(f"study {study}: suite {s['suite']} not in this checkout")
    if rows:
        for pname, p in spec["parents"].items():
            if not (root / p["trial"] / "development/rows.json").exists(): absent(f"parent {pname}: {p['trial']}/development/rows.json not in this checkout")
        for arm in spec["arms"]:
            side = parent_side(spec, arm, root)
            for tag in sorted(rule_tags(spec["rule"]) - {spec["rule"].get("unknowable")}):   # the parent's unknowable read is reported, not required
                if not side.has(tag): absent(f"arm {arm}: parent read {tag} not in this checkout ({side.dirs.get(tag, 'no location')})")
    return Validation(problems, archived)


def _known_path(path, rule):
    parts = path.split(".")
    if parts[0] == "unknowable": return len(parts) == 2 and parts[1] in ("candidate", "parent") and bool(rule.get("unknowable"))
    panel = rule["panels"].get(parts[0])
    if panel is None or len(parts) != 3 or parts[2] not in FIELDS: return False
    value_only = parts[1] in panel.get("report", ()) or parts[1] in by_length_names(panel)   # report and by-length metrics have no interval
    return parts[1] in panel.get("metrics", ()) or (value_only and parts[2] in ("candidate", "parent"))


# --- calibration by state length -------------------------------------------------------------------------------------

def by_length_options(panel):
    """(edges, tokenizer) of a panel's by_length option: true = kev.metrics.LENGTH_EDGES and kev.suite.ADMISSION_TOKENIZER."""
    option = panel.get("by_length")
    option = {} if option is True else option
    return tuple(option.get("edges", LENGTH_EDGES)), tuple(option.get("tokenizer", ADMISSION_TOKENIZER))


def by_length_names(panel):
    """{"<metric>_<bucket>": metric} for the entries a by_length panel reports (empty for any other panel, or invalid edges)."""
    if not panel.get("by_length"): return {}
    try:
        return {f"{m}_{name}": m for m in LENGTH_METRICS for name, _, _ in length_buckets(by_length_options(panel)[0])}
    except (ValueError, AttributeError, TypeError):
        return {}


def by_length_problems(panel, spec):
    option = panel.get("by_length")
    if not option: return []
    if option is not True and (not isinstance(option, dict) or set(option) - {"edges", "tokenizer"}):
        return ["by_length is true or {edges?, tokenizer?}"]
    problems = []
    try:
        edges, tokenizer = by_length_options(panel)   # tuple() of a non-iterable raises: a malformed option is a problem, not a crash
        length_buckets(edges)
        if len(tokenizer) != 2 or not all(isinstance(x, str) for x in tokenizer): problems.append("by_length: tokenizer is [name, revision]")
    except (ValueError, TypeError) as error:
        problems.append(f"by_length: {error}" if isinstance(error, ValueError) else f"by_length: edges are a list of token counts and tokenizer is [name, revision] ({error})")
    if TRANSFER in panel["reads"]: problems.append("by_length counts state tokens from a read's suite; 'transfer' names no single suite")
    problems += [f"by_length: read {tag!r} has no suite" for tag in panel["reads"] if tag in spec["reads"] and not spec["reads"][tag].get("suite")]
    return problems


@functools.cache
def state_lengths(suite, split, tokenizer=ADMISSION_TOKENIZER):
    """{record id: state tokens} of a suite partition with the tokenizer (name, revision): the counts a by_length panel
    buckets both sides by. A state's token count is its encoded state segment, as kev.model.encode builds it: the <state>
    token plus user_tokens of the materialised state (len + 1, what encode checks against max_state and kev.serve reports
    as `state_tokens`, seg.count(0)); the one definition behind a row's `state_tokens` (kev.metrics.calibration_by_length)."""
    from kev.data import materialize
    from kev.model import load_tokenizer, user_tokens
    tok = load_tokenizer(*tokenizer)
    return {r["_meta"]["id"]: 1 + len(user_tokens(tok, materialize(r)["state"])) for r in load_split(ROOT / suite, split, allow_test=split == "test")}


def panel_lengths(spec, panel):
    """{record id: state tokens} over a by_length panel's reads (each read's suite and partition)."""
    tokenizer, out = by_length_options(panel)[1], {}
    for tag in panel["reads"]:
        read = spec["reads"][tag]
        for rid, n in state_lengths(read["suite"], read_split(read), tokenizer).items():
            if out.setdefault(rid, n) != n: raise ValueError(f"record {rid!r} has different states in the reads of one by_length panel")
    return out


# --- temperature pools against training data (round 19's failure mode) ------------------------------------------------

ROUND_19 = ("round 19's failure mode: a temperature fitted on held-out items of a checkpoint's own training corpus is in distribution "
            "(round 19 served its SFT arms at T 0.955 fitted on sft-v1 development rows: breadth-v1 ECE 0.059); fit on a pool of "
            "held-out datasets instead (round 20: breadth-v1 ECE 0.0085)")
FIT_SPLITS = ("calibration", "development")   # a training corpus's held-out partitions: never a temperature pool


def read_split(read):
    """The partition a read scores: kev.benchmark's development, `--split X`, or the test under --allow-test / locked_test."""
    flags = read.get("flags", "").split()
    if read.get("entrypoint") == "locked_test" or "--allow-test" in flags: return "test"
    for i, flag in enumerate(flags):
        if flag.startswith("--split="): return flag.split("=", 1)[1]
        if flag == "--split" and i + 1 < len(flags): return flags[i + 1]
    return "development"


def suite_dir(path):
    """'evals/...' for a suite path given relative, absolute or as a container saw it (/root/kev/evals/sft-v1); None if none."""
    parts = Path(str(path)).parts
    return str(Path(*parts[parts.index("evals"):])) if "evals" in parts else None


@functools.cache
def _suites():
    """({manifest sha256: suite dir}, {directory name: [suite dirs]}) of every suite in this checkout."""
    by_digest, by_name = {}, {}
    for m in sorted((ROOT / "evals").rglob("manifest.json")):
        d = str(m.parent.relative_to(ROOT))
        by_digest[digest(m)] = d
        by_name.setdefault(m.parent.name, []).append(d)
    return by_digest, by_name


def suite_by_digest(sha256):
    """The suite dir whose manifest.json hashes to sha256 (what provenance, head.pt and report.json record), or None."""
    return _suites()[0].get(sha256)


def suite_manifest(suite):
    return read_manifest(ROOT / suite) if (ROOT / suite / "manifest.json").exists() else None


def listed_sources(manifest):
    """Every source a manifest names: trainable, held-out, eval-only and its `sources` table (a dict or a list)."""
    return {*manifest.get("trainable_sources", []), *manifest.get("holdout_sources", []), *manifest.get("eval_only_sources", []), *manifest.get("sources", [])}


def trainable_sources(manifest):
    """The sources a suite offers for training (`trainable_sources`, or `sources` entries marked trainable): a suite with
    any is a training corpus, whose calibration and development partitions are held-out items of training sources."""
    table = manifest.get("sources", {})
    return {*manifest.get("trainable_sources", []), *(k for k, v in table.items() if isinstance(v, dict) and v.get("trainable"))} if isinstance(table, dict) else set(manifest.get("trainable_sources", []))


class Training(NamedTuple):
    """What a checkpoint was trained on, as far as this checkout can tell."""
    suites: frozenset     # suite dirs: its training suite, the components that suite names, a plan's `data` directory
    sources: frozenset    # the sources those suites train
    unlisted: tuple       # training suites whose sources this checkout cannot list


def training_data(suites, data=None):
    """Training over suite dirs plus a `data` file: each suite's trainable sources (every listed source for a data-only
    manifest without `trainable_sources`, e.g. round6/b1v2) plus the suites its manifest names as components (sft-v1's
    `inputs.components`: documents-v1-train -> evals/documents-v1, hard-v1-extra-train -> evals/hard-v1). A `data` file
    counts through its directory's suite; one outside evals/ (a `kev.train --data` run) cannot be checked, so it is
    `unlisted` like any training this checkout cannot place, never silently dropped."""
    data_dir = suite_dir(Path(data).parent) if data else None
    seen, sources, unlisted, todo = set(), set(), [] if data_dir or not data else [f"data {data} (outside evals/)"], [d for d in (*suites, data_dir) if d]
    while todo:
        d = todo.pop(0)
        if d in seen: continue
        seen.add(d)
        m = suite_manifest(d)
        own = (trainable_sources(m) if "trainable_sources" in m else listed_sources(m)) if m is not None else set()
        if not own: unlisted.append(d)
        sources |= own
        inputs = (m or {}).get("inputs")
        for component in inputs.get("components", {}) if isinstance(inputs, dict) else ():
            found = _suites()[1].get(component.removesuffix("-train").removesuffix("-extra"))
            if found: todo += found
            else: unlisted.append(f"{d} component {component}")
    return Training(frozenset(seen), frozenset(sources), tuple(unlisted))


def recorded_training(suite_sha256=None, suite=None, data=None):
    """Training of a checkpoint from what its trainer recorded (provenance.json, head.pt): the suite whose manifest hashes to
    suite_sha256 (else the suite path it was given) and its `data` file's directory; None when no suite of this checkout matches."""
    trained = suite_by_digest(suite_sha256)
    if trained is None and suite and suite_dir(suite) and suite_manifest(suite_dir(suite)) is not None: trained = suite_dir(suite)
    return training_data([trained], data) if trained else None


def _study(spec, trial):
    return next((s for name, s in spec.get("studies", {}).items() if trial.startswith(f"runs/{name}/")), None)


def trial_suite(spec, trial, root=ROOT):
    """The suite a trial trained on and was scored on (its development rows'): its study's in this spec, else the manifest
    its provenance.json hashes; None when neither is here."""
    if study := _study(spec, trial): return suite_dir(study["suite"])
    provenance = Path(root) / trial / "provenance.json"
    return suite_by_digest(read_json(provenance).get("suite_sha256")) if provenance.exists() else None


def trial_training(spec, trial, root=ROOT):
    """Training of a trial: its study's suite plus its plan entry's `data` (trial <NN>-<label> is entry NN), else what its
    provenance.json records; None when neither is here."""
    study, provenance = _study(spec, trial), Path(root) / trial / "provenance.json"
    if study:
        plan = next((p / study["plan"] for p in (Path(root), ROOT) if (p / study["plan"]).exists()), None)
        entries, index = (read_json(plan) if plan else []), Path(trial).name.split("-")[0]
        data = entries[int(index)].get("data") if isinstance(entries, list) and index.isdigit() and int(index) < len(entries) else None
        if plan or not provenance.exists(): return training_data([suite_dir(study["suite"])], data)
    if not provenance.exists(): return None
    p = read_json(provenance)
    return recorded_training(p.get("suite_sha256"), data=p.get("config", {}).get("data"))


def pool_conflicts(pooled, training):
    """[(label, why)] for every way a pooled read shares data with a checkpoint's Training. pooled = [(label, suite, split,
    sources or None)]: (a) the read's suite is training data of the checkpoint (its training suite, or a component or `data`
    suite of it); (b) the sources it pools (its allowlist, else every source its manifest lists) are sources the checkpoint
    trained on; (c) it scores the calibration or development partition of a training corpus (any suite with trainable
    sources). A read whose suite or sources cannot be established is refused too. The one check, for kev.rounds pools and
    scripts/calibrate_checkpoint.py alike."""
    out = []
    for label, suite, split, sources in pooled:
        d = suite_dir(suite) if suite else None
        m = suite_manifest(d) if d else None
        if m is None:
            out.append((label, f"its suite ({suite!r}) has no manifest in this checkout, so what it pools cannot be checked")); continue
        if d in training.suites: out.append((label, f"{d} is training data of the checkpoint"))
        pooled_sources = set(sources) if sources else listed_sources(m)
        if not pooled_sources: out.append((label, f"{d} lists no sources; give the read a `sources` allowlist"))
        if shared := sorted(pooled_sources & training.sources):
            out.append((label, f"it pools {len(shared)} training source(s) {shared[:8]}" + (" ..." if len(shared) > 8 else "")))
        if split in FIT_SPLITS and trainable_sources(m): out.append((label, f"it reads the {split} partition of {d}, a training corpus"))
    return out


def pool_training_problems(spec, root=ROOT):
    """(problems, unknown) of the round's `temperature` pool against every arm's training data (pool_conflicts; message
    names round 19's failure mode). An arm's training comes from `trained_on`, else from its trial (trial_training); an arm
    without either (a checkpoint made from the round's trials) is covered by the trial arms. unknown: arms whose training
    this checkout cannot establish or list (a trial without provenance, a manifest without sources, a `data` file outside
    evals/): a problem for a new round, reported under archived for a recorded one."""
    pool = spec.get("temperature")
    if not pool or not pool.get("reads"): return [], []
    pooled = [(tag, spec["reads"][tag].get("suite"), read_split(spec["reads"][tag]), pool.get("sources", {}).get(tag)) for tag in pool["reads"] if tag in spec["reads"]]
    found, problems, unknown = {}, [], []
    for arm, a in spec["arms"].items():
        if "trained_on" in a:
            if not isinstance(a["trained_on"], list) or not a["trained_on"] or not all(isinstance(s, str) for s in a["trained_on"]):
                problems.append(f"arm {arm}: trained_on is a non-empty list of suite directories"); continue
            training = training_data([suite_dir(s) or s for s in a["trained_on"]])
        elif a.get("trial"):
            training = trial_training(spec, a["trial"], root)
            if training is None:
                unknown.append(f"arm {arm}: what {a['trial']} trained on is unknown (no study of this spec, no provenance.json), so the temperature pool cannot be checked against it"); continue
        else:
            continue
        unknown += [f"arm {arm}: cannot list the sources of {d}, part of its training, so the temperature pool cannot be checked against it" for d in training.unlisted]
        by_tag = {}
        for tag, why in pool_conflicts(pooled, training): by_tag.setdefault(tag, []).append(why)
        for tag, whys in by_tag.items(): found.setdefault((tag, "; ".join(whys)), []).append(arm)
    for (tag, why), arms in found.items():
        problems.append(f"temperature: pooled read {tag!r} shares data with the training of arm(s) {', '.join(arms)}: {why}. This is {ROUND_19}")
    if not any("trained_on" in a or a.get("trial") for a in spec["arms"].values()):
        problems.append("temperature: no arm names its training (a trial or `trained_on`), so the pool cannot be checked against it")
    return problems, unknown


POOL_REQUIRED_FROM = 21   # from this round on, a temperature-dependent criterion without a `temperature` pool is refused


def temperature_criteria(spec):
    """["rule <name>" | "confirm.<stage> <name>"] of every criterion that reads a number the temperature moves."""
    stages = {None: spec["rule"], **spec.get("confirm", {})}
    return sorted({f"{'confirm.' + s if s else 'rule'} {name}" for s, rule in stages.items() for name, c in rule["criteria"].items()
                   if any(_moves_with_temperature(path, rule) for path in _paths(c))})


def _listed(names):
    return f"{len(names)} ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})"


def pool_required_problems(spec):
    """Round >= POOL_REQUIRED_FROM with a temperature-dependent criterion and no `temperature` pool: refused (its arms would
    be served at their trials' development rows, round 19's failure mode). Earlier rounds only warn (calibration_warnings),
    so their recorded specs keep validating."""
    number, moved = spec.get("round"), temperature_criteria(spec)
    if spec.get("temperature") or not moved or not isinstance(number, int) or number < POOL_REQUIRED_FROM: return []
    return [f"temperature: round {number} registers no `temperature` pool, but {_listed(moved)} criteria depend on the temperature, "
            f"so its arms would be served at their trials' development rows. This is {ROUND_19}; register a pool of held-out datasets (copy r20)"]


def shipped_temperature(entry, root=ROOT):
    """The temperature a parent's checkpoint ships (head.pt), read locally only: <trial>/checkpoint/head.pt, else its Hub
    checkpoint's head.pt in the local Hugging Face cache (no network); None when neither is here."""
    path = Path(root) / entry["trial"] / "checkpoint" / "head.pt"
    if not path.exists() and entry.get("checkpoint") and not str(entry["checkpoint"]).startswith("/"):
        from huggingface_hub import try_to_load_from_cache
        repo, _, revision = entry["checkpoint"].partition("@")
        cached = try_to_load_from_cache(repo, "head.pt", revision=revision or None)
        path = Path(cached) if isinstance(cached, str) else path
    if not path.exists(): return None
    from kev.checkpoint import read_meta
    return read_meta(path.parent).temperature


SHIPPED_TOLERANCE = 0.05   # a parent served this far from its shipped temperature, fitted in distribution, is warned about


def calibration_warnings(spec, root=ROOT):
    """Report-only warnings that `validate` and `launch` print. (1) For a round before POOL_REQUIRED_FROM without a
    `temperature` pool whose criteria depend on the temperature (from then on it is a problem: pool_required_problems): each
    trial arm whose development rows are a training corpus's is served in distribution, round 19's failure mode. (2) Every
    parent served at a temperature fitted on a training corpus's development rows that differs from its shipped head.pt
    temperature by more than SHIPPED_TOLERANCE (when both are here)."""
    out, moved = [], temperature_criteria(spec)
    early = isinstance(spec.get("round"), int) and spec["round"] < POOL_REQUIRED_FROM
    for arm, a in spec["arms"].items() if moved and early and not spec.get("temperature") else ():
        suite = trial_suite(spec, a["trial"], root) if a.get("trial") else None
        if suite and trainable_sources(suite_manifest(suite) or {}):
            out.append(f"arm {arm} will be served at a temperature fitted on its trial's {suite} development rows, a training corpus, and "
                       f"{_listed(moved)} criteria depend on it: {ROUND_19}; register a `temperature` pool")
    for name, p in spec["parents"].items():
        suite = trial_suite(spec, p["trial"], root)
        if not (suite and trainable_sources(suite_manifest(suite) or {}) and (Path(root) / p["trial"] / "development/rows.json").exists()): continue
        shipped = shipped_temperature(p, root)
        served_t = temperature(p["trial"], root) if shipped is not None else None
        if shipped is not None and abs(served_t - shipped) > SHIPPED_TOLERANCE:
            out.append(f"parent {name} is served at T {served_t:.3f} fitted on {p['trial']}/development ({suite}, a training corpus), "
                       f"{abs(served_t - shipped):.3f} from the T {shipped:.3f} its checkpoint ships: its reads are not what it serves (report only)")
    return out


TEMPERATURE_FREE = ("acc",)   # the one bootstrapped metric a temperature cannot move (the argmax is invariant)


def _moves_with_temperature(path, rule):
    """Whether a criterion path reads a number the temperature moves: any panel metric but accuracy (a by-length bucket's
    included: acc_16k_plus is accuracy); the unknowable share is scored on records a pool never fits on."""
    panel, metric = (path.split(".") + [""])[:2]
    return panel != "unknowable" and by_length_names(rule["panels"].get(panel, {})).get(metric, metric) not in TEMPERATURE_FREE


def allowlist_problems(tag, suite, sources):
    """A pool read's `sources` allowlist against the sources its suite's manifest lists: a name the suite does not contain
    (a typo) would silently shrink the pool, so it is a problem, as is a suite whose sources cannot be listed."""
    m = suite_manifest(suite_dir(suite)) if suite and suite_dir(suite) else None
    listed = listed_sources(m) if m is not None else set()
    if not listed: return [f"temperature: cannot check the sources allowlist of {tag!r}: {suite!r} lists no sources in this checkout"]
    return [f"temperature: sources for {tag!r} name {unknown} which {suite} does not contain" for unknown in [sorted(set(sources) - listed)] if unknown]


def pool_problems(spec, stages):
    """What is wrong with the round's `temperature` pool: tags that are not reads of the spec, tags an arm cannot locate,
    and pooled tags inside a panel that a temperature-dependent criterion reads (any metric but accuracy, at any stage):
    the temperature would be fitted on the rows it is judged on. The unknowable share is scored on unknowable records,
    which a pool never fits on (knowable rows only), so the unknowable read may be pooled."""
    pool = spec.get("temperature")
    if not pool: return []
    reads, exclude = pool.get("reads") or [], pool.get("exclude_reads", [])
    problems = ["temperature: no reads to pool"] if not reads else []
    problems += [f"temperature: {tag!r} is not a read tag of this spec" for tag in reads if tag not in spec["reads"]]
    problems += [f"temperature: exclude_reads {tag!r} is not a read tag of this spec" for tag in exclude if tag not in spec["reads"] and tag != TRANSFER]
    for tag, sources in pool.get("sources", {}).items():
        if tag not in reads: problems.append(f"temperature: sources for {tag!r}, which is not pooled")
        if not sources or not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            problems.append(f"temperature: sources for {tag!r} must be a non-empty list of source names"); continue
        problems += allowlist_problems(tag, spec["reads"].get(tag, {}).get("suite"), sources)
    for stage, rule in stages.items():
        for cname, c in rule["criteria"].items():
            for path in _paths(c):
                shared = sorted(set(reads) & set(rule["panels"].get(path.split(".")[0], {}).get("reads", [])))
                if shared and _moves_with_temperature(path, rule):
                    problems.append(f"temperature: {shared} pooled, but {'confirm.' + stage if stage else 'rule'} criterion {cname} reads {path}, which the temperature moves")
    for arm, a in spec["arms"].items():
        where = a.get("reads", spec["arm_reads"])
        unplaced = [t for t in [*reads, *exclude] if (t == TRANSFER and not a.get("trial") and transfer_read(spec, arm) is None)
                    or (t != TRANSFER and not isinstance(where, str) and t not in where)]
        if unplaced: problems.append(f"arm {arm}: its reads do not locate the temperature pool's {unplaced}")
    return problems


def _validate_plan(study, s, root, partitions):
    from kev.experiment import load_plan, validated_trial
    from kev.suite import read_manifest
    from kev.budget import compute_bound   # the admission bound modal_app.admit_study refuses a study over
    problems = []
    try:
        trials = load_plan(root / s["suite"], root / s["plan"]) if partitions else [validated_trial(t, read_manifest(root / s["suite"])) for t in read_json(root / s["plan"])]
    except (ValueError, KeyError) as error:
        return [f"study {study}: plan {s['plan']} does not validate ({type(error).__name__}: {error})"]
    bound = compute_bound(s["gpu"], s["timeout"], len(trials), any(t.get("full_ft") for t in trials))
    if bound > s["budget"]: problems.append(f"study {study}: admission bound ${bound:.2f} exceeds budget ${s['budget']:.2f} (modal_app.admit_study would refuse it)")
    return problems


# --- rule ------------------------------------------------------------------------------------------------------------

def compare(candidate, parent, rule, lengths=None):
    """Every panel, the unknowable share and every criterion of one candidate against its parent. Panels whose reads are
    missing on either side are listed under "missing" and their criteria are None; `passed` needs every criterion. A
    candidate served at a pooled temperature records the fit (`temperature_fit`); every candidate records where its
    temperature came from (`temperature_source`); one whose pool rows are missing is reported incomplete with nothing
    computed. lengths(panel) -> {record id: state tokens} serves by_length panels whose rows record no state_tokens."""
    out = {"trial": candidate.trial, "parent": parent.trial, **({"checkpoint": candidate.checkpoint} if candidate.checkpoint else {})}
    if absent := [f"candidate:{d}" for d in candidate.absent_fit()]:
        return {**out, "temperature": None, "missing": absent, "complete": False, "passed": None}
    out.update(temperature=candidate.t, parent_temperature=parent.t, panels={}, missing=[])
    if candidate.fit: out["temperature_fit"] = candidate.fit
    out["temperature_source"] = candidate.temperature_source()
    out["parent_temperature_source"] = parent.temperature_source()   # parents are served at their trial's development rows; `shipped` is head.pt's T
    for name, spec in rule["panels"].items():
        absent = [f"{who}:{side.dirs[t] if t in side.dirs else t}" for who, side in (("candidate", candidate), ("parent", parent)) for t in spec["reads"] if not side.has(t)]
        if absent: out["missing"] += absent; continue
        c, p = candidate.panel(spec), parent.panel(spec)
        mc, mp = metrics(c), metrics(p)
        panel = {"n": len(c)}
        for m in spec.get("metrics", ()):
            panel[m] = {"candidate": mc[m], "parent": mp[m], **paired(c, p, m)}
        for m in spec.get("report", ()):
            panel.setdefault(m, {"candidate": mc[m], "parent": mp[m]})
        if spec.get("by_length"):
            panel.update(_by_length(spec, c, p, lengths))
        for ref, d in spec.get("versus", {}).items():
            rows = [r for r in read_json(candidate.root / d / "rows.json") if r["variant"] == "clean"]   # a reference as it served itself
            panel.setdefault("versus", {})[ref] = {m: {"reference": metrics(rows)[m], **paired(c, rows, m)} for m in spec.get("metrics", ())}
        out["panels"][name] = panel
    if rule.get("unknowable"):
        tag = rule["unknowable"]
        if not candidate.has(tag): out["missing"].append(f"candidate:{candidate.dirs[tag]}")
        out["unknowable"] = {"candidate": candidate.unknowable_share(tag), "parent": parent.unknowable_share(tag)}
    out["criteria"] = {name: _criterion(out, c) for name, c in rule["criteria"].items()}
    out["complete"] = not out["missing"]
    out["passed"] = (out["complete"] and all(out["criteria"].values())) if out["criteria"] else None
    return out


def _by_length(spec, candidate, parent, lengths):
    """A by_length panel's entries: "<metric>_<bucket>" -> {candidate, parent, n} (kev.metrics.calibration_by_length of
    each side's served rows, bucketed by the same counts) and "by_length" -> how the tokens were counted."""
    edges, tokenizer = by_length_options(spec)
    counted = all(r.get("state_tokens") is not None for r in (*candidate, *parent))
    if not counted and lengths is None: raise ValueError("a by_length panel needs state-token counts: rows with state_tokens, or lengths")
    counts = None if counted else lengths(spec)
    c, p = calibration_by_length(candidate, counts, edges), calibration_by_length(parent, counts, edges)
    out = {"by_length": {"edges": list(edges), "tokens": "rows" if counted else {"records": "suite", "tokenizer": list(tokenizer)}}}
    for bucket in c:
        for m in LENGTH_METRICS: out[f"{m}_{bucket}"] = {"candidate": c[bucket][m], "parent": p[bucket][m], "n": c[bucket]["n"]}
    return out


def value(report, path):
    """A number from a comparison by path; None if its panel was not read."""
    parts = path.split(".")
    if parts[0] == "unknowable": return (report.get("unknowable") or {}).get(parts[1])
    entry = report["panels"].get(parts[0], {}).get(parts[1])
    if entry is None: return None
    if parts[2] in ("lower", "upper"): return entry["ci95"][0 if parts[2] == "lower" else 1]
    return entry[parts[2]]


def _criterion(report, c):
    left = [value(report, p) for p in (c["left"] if isinstance(c["left"], list) else [c["left"]])]
    right = value(report, c["right"]) if isinstance(c["right"], str) else c["right"]
    if right is None or any(v is None for v in left): return None
    lhs = left[0] - left[1] if len(left) == 2 else left[0]
    return OPS[c["op"]](lhs, right + c["plus"] if "plus" in c else right)


def rank(arms, keys, selectable):
    """Passing selectable arms per size, best first by the rank keys (a key summing several paths adds them left to right)."""
    def score(report):
        out = []
        for key in keys:
            paths = key["by"] if isinstance(key["by"], list) else [key["by"]]
            total = value(report, paths[0])
            for p in paths[1:]: total = total + value(report, p)
            out.append(-total if key.get("order", "desc") == "desc" else total)
        return tuple(out)
    by_size = {}
    for arm, report in arms.items():
        by_size.setdefault(size_of(arm), [])
        if report.get("passed") and arm in selectable: by_size[size_of(arm)].append(arm)
    return {size: sorted(names, key=lambda a: score(arms[a])) for size, names in by_size.items()}


def readout(spec, root=ROOT):
    """The registered rule applied to every arm that has finished training (development rows present); an arm without a
    trial (a checkpoint made from others) is compared once its pool and reads are there (compare lists what is missing)."""
    arms = {}
    for arm, a in spec["arms"].items():
        if a.get("trial") and not (Path(root) / a["trial"] / "development/rows.json").exists():
            arms[arm] = {"trial": a["trial"], "parent": spec["parents"][a["parent"]]["trial"], "missing": [f"candidate:{a['trial']}/development/rows.json"], "complete": False, "passed": None}
            continue
        arms[arm] = compare(arm_side(spec, arm, root), parent_side(spec, arm, root), spec["rule"], lambda panel: panel_lengths(spec, panel))
    ranking = rank(arms, spec["rule"].get("rank", []), {a for a, x in spec["arms"].items() if x.get("select", True)})
    return {"round": spec["round"], "registered": spec["registered"], "drop_ids": sorted(spec.get("drop_ids", [])), "arms": arms,
            "ranking": ranking, "candidates": {size: names[0] if names else None for size, names in ranking.items()}}


def confirm(spec, stage, arm, root=ROOT):
    """One confirmation stage for the chosen arm against its parent, on the stage's reads."""
    out = compare(arm_side(spec, arm, root, stage), parent_side(spec, arm, root, stage), spec["confirm"][stage], lambda panel: panel_lengths(spec, panel))
    return {"round": spec["round"], "stage": stage, "arm": arm, **out}


# --- printing --------------------------------------------------------------------------------------------------------

RATES = ("acc", "confident_error_rate", "coverage_at_5pct_error", "coverage_at_1pct_error")   # printed in percentage points


def _interval(metric, entry):
    scale, fmt = (100, "+.1f") if metric in RATES else (1, "+.3f")
    return f"{scale * entry['delta']:{fmt}} [{scale * entry['ci95'][0]:{fmt}}, {scale * entry['ci95'][1]:{fmt}}]"


def table(report):
    """One line per comparison: every bootstrapped panel metric (rates in pp), the unknowable share, the verdict, what failed;
    under it, each by_length panel's buckets, and a warning when the arm is served at a temperature fitted on its trial's
    development rows of a training corpus (in distribution: round 19's failure mode)."""
    lines = []
    for arm, r in report.get("arms", {report.get("arm"): report}).items():
        cells = [f"{p}.{m} {_interval(m, e)}" for p, panel in r.get("panels", {}).items() for m, e in panel.items() if isinstance(e, dict) and "ci95" in e]
        if r.get("unknowable"): cells.append(f"unk {r['unknowable']['candidate']}")
        failed = [k for k, v in r.get("criteria", {}).items() if v is False]
        verdict = "incomplete" if not r.get("complete") else {True: "PASS", None: "reported", False: "fail: " + ", ".join(failed)}[r["passed"]]
        lines.append(f"{arm:16} {' | '.join(cells)} -> {verdict}" + (f" (missing {len(r['missing'])})" if r.get("missing") else ""))
        for p, panel in r.get("panels", {}).items():
            if "by_length" not in panel: continue
            buckets = [name for name, _, _ in length_buckets(tuple(panel["by_length"]["edges"])) if panel[f"ece_{name}"]["n"]]
            lines.append(f"{'':16} {p} by state length (candidate/parent): " + " | ".join(
                f"{b} n={panel[f'ece_{b}']['n']} acc {panel[f'acc_{b}']['candidate']:.3f}/{panel[f'acc_{b}']['parent']:.3f} ece {panel[f'ece_{b}']['candidate']:.4f}/{panel[f'ece_{b}']['parent']:.4f}" for b in buckets))
        source = r.get("temperature_source") or {}
        if source.get("kind") == "trial development rows" and source.get("training_corpus"):
            lines.append(f"{'':16} !!! {arm} is served at T {r['temperature']:.3f} fitted on {source['rows']}, held-out items of its training corpus {source['suite']}: "
                         f"in distribution, {ROUND_19.split(': ', 1)[0]}; not a temperature to ship (register a held-out-datasets `temperature` pool)")
    if "candidates" in report: lines.append(f"candidates {report['candidates']}")
    return "\n".join(lines)


# --- Modal orchestration ---------------------------------------------------------------------------------------------

def bench_job(run, suite, name, flags=""):
    """One modal_app.py::benchmarks entry. modal_app.parse_jobs reads the run as everything before the suite, so a pinned
    Hub revision (repo@sha) is safe; the suite and the name must not contain '@' or ',' and flags must start with '--'."""
    if any(c in s for s in (suite, name) for c in "@,") or "," in run or (flags and not flags.startswith("--")):
        raise ValueError(f"cannot encode benchmark job {(run, suite, name, flags)}")
    return "@".join([run, suite, name, *([flags] if flags else [])])


def checkpoint_of(entry):
    """Where a read runs from: the trial's checkpoint on the runs volume (runs/X -> /runs/X/checkpoint) unless declared."""
    return entry["checkpoint"] if "checkpoint" in entry else f"/{entry['trial']}/checkpoint"


def side_reads(spec, arm, rule, who, side, stage=None):
    """[(read tag, directory)] a side needs for a stage, by tag: the rule's reads; "transfer" as the arm's transfer_read (a
    trial's in-trial transfer read is never launched); and, for an arm at the rule stage, its temperature pool."""
    tags = rule_tags(rule)
    if who == "candidate" and stage is None and spec.get("temperature"):
        tags |= {*spec["temperature"]["reads"], *spec["temperature"].get("exclude_reads", [])}
    reads = {}
    for tag in tags:
        if tag != TRANSFER: reads.setdefault(side.dirs[tag], tag)
        elif who == "candidate" and transfer_read(spec, arm): reads.setdefault(side.dirs[tag], transfer_read(spec, arm))
    return sorted((read, d) for d, read in reads.items())


def read_commands(spec, arm, stage=None, root=ROOT, sides=("candidate",)):
    """The `modal run` commands for the missing reads of one arm (and, with sides, its parent): one benchmarks call per side
    with every suite batched, plus one locked_test call per locked read. Reads that exist locally are skipped."""
    rule = spec["confirm"][stage] if stage else spec["rule"]
    size, commands = size_of(arm), []
    for who in sides:
        side = (arm_side if who == "candidate" else parent_side)(spec, arm, root, stage)
        run = checkpoint_of(spec["arms"][arm] if who == "candidate" else spec["parents"][spec["arms"][arm]["parent"]])
        jobs = []
        for tag, d in side_reads(spec, arm, rule, who, side, stage):
            if (root / d / "rows.json").exists(): continue
            r = spec["reads"][tag]
            if r.get("entrypoint") == "locked_test":
                if not run.startswith("/runs/"): raise ValueError(f"locked_test reads a trial on the runs volume, not {run}")
                commands.append(["modal", "run", "modal_app.py::locked_test", "--trial", run.removeprefix("/runs/").removesuffix("/checkpoint"),
                                 "--name", Path(d).relative_to("runs/locked").parts[0], "--decision", r["decision"], "--gpu", spec.get("gpu", "H100"),
                                 *spec.get("locked_args", {}).get(size, [])])
                continue
            if not d.startswith("runs/") or "/" in d.removeprefix("runs/"): raise ValueError(f"{tag}: benchmarks writes runs/<name>, not {d}")
            jobs.append(bench_job(run, r["suite"], d.removeprefix("runs/"), r.get("flags", "")))
        if jobs:
            timeout = spec.get("read_timeout", {}).get(size)
            commands.append(["modal", "run", "--detach", "modal_app.py::benchmarks", "--jobs", ",".join(jobs), "--gpu", spec.get("gpu", "H100"),
                             *(["--timeout", str(timeout)] if timeout else [])])
    return commands


def modal_env(spec):
    return {**os.environ, **({"KEV_APP_NAME": spec["app"]} if spec.get("app") else {}), **({"KEV_GPU": spec["gpu"]} if spec.get("gpu") else {})}


def launch_commands(commands, spec, log_stem, stagger=STAGGER, run=subprocess.Popen, sleep=time.sleep):
    """Start each command in the background (log under runs/), `stagger` seconds apart; returns the processes."""
    procs = []
    for i, cmd in enumerate(commands):
        if i: sleep(stagger)
        log = ROOT / "runs" / f"{log_stem}-{i}.log"
        print("launch:", " ".join(cmd[:4]), "->", log.relative_to(ROOT), flush=True)
        with log.open("w", encoding="utf-8") as f:
            procs.append(run([sys.executable, "-m", *cmd], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=modal_env(spec)))
    return procs


class InFlight(Exception):
    """An arm's reads were launched recently and may still be running; they are not launched again yet."""


def launch_arm_reads(spec, arm, stage=None, sides=("candidate",), stagger=STAGGER, launch=None, now=time.time, log=print):
    """Launch one arm's missing reads once. Under a per-arm lock (runs/.reads-r<N>-<arm>.lock: a watcher and a
    `launch-reads` cannot both launch the arm), the intent is written first (runs/r<N>-reads-<arm>[-<stage>].json: when and
    what), then the commands start. Detached benchmarks cannot be seen from here, so while an earlier intent is younger
    than the reads' timeout the arm counts as in flight: InFlight is raised (the watcher retries on a later pass) instead of
    a second launch whose job would find /runs/bench/<name> taken. Reads whose rows landed are never relaunched."""
    tag = f"r{spec['round']}-reads-{arm}" + (f"-{stage}" if stage else "")
    grace = max(IN_FLIGHT, spec.get("read_timeout", {}).get(size_of(arm), 0))
    with file_lock(ROOT / "runs" / f".reads-r{spec['round']}-{arm}.lock"):
        commands = read_commands(spec, arm, stage, ROOT, sides)
        if not commands: log(f"{arm}: every read has landed"); return []
        intent = ROOT / "runs" / f"{tag}.json"
        if intent.exists() and now() - read_json(intent)["launched_at"] < grace:
            since = read_json(intent)["launched_at"]
            raise InFlight(f"{arm}: reads launched {int(now() - since)} s ago may still be running; not relaunching for another "
                           f"{int(since + grace - now())} s (if they finished on Modal but were never pulled: modal volume get kev-runs /bench/<name> runs/)")
        write_json(intent, {"launched_at": now(), "commands": commands}, atomic=True)
        return (launch or launch_commands)(commands, spec, tag, stagger)


def pull(study, spec):
    """modal_app.py::pull for one study (modal_app.pull_study holds a per-study lock, so concurrent pulls wait)."""
    subprocess.run([sys.executable, "-m", "modal", "run", "modal_app.py::pull", "--name", study], check=True, cwd=ROOT, env=modal_env(spec))


# --- watch -----------------------------------------------------------------------------------------------------------

NETWORK_MARKERS = ("nodename nor servname", "name or service not known", "temporary failure in name resolution", "unavailable", "connection reset",
                   "connection refused", "network is unreachable", "deadline exceeded")


def transient(error):
    """Whether polling a call failed because of this machine's network (DNS drop, reset connection, gRPC UNAVAILABLE)
    rather than because the trial failed. A trial's own exception is re-raised by FunctionCall.get with its original
    type, so only network types and gRPC transport messages count; anything else is the trial's failure."""
    try:
        import modal.exception as me
        modal_types = (me.ConnectionError, me.ClientClosed)
    except ImportError:
        modal_types = ()
    if isinstance(error, (socket.gaierror, ConnectionError, *modal_types)): return True
    return type(error).__module__.startswith(("grpclib", "modal")) and any(m in str(error).lower() for m in NETWORK_MARKERS)


class TrialFailed(Exception):
    """A full-weight trial that failed with an error returns {"failed": ...} (modal_app.failed_trial) instead of raising,
    so Modal does not retry it; poll_modal raises this for it, which watch_studies marks failed like any trial error."""


def poll_modal(call_id):
    """'running' | 'done' for a spawned trial; the trial's exception (or a network error) propagates, and a returned
    failure is raised as TrialFailed."""
    import modal
    try:
        result = modal.FunctionCall.from_id(call_id).get(timeout=0.5)
    except TimeoutError:            # builtin: no output yet (modal.exception.FunctionTimeoutError is not a builtin TimeoutError)
        return "running"
    except modal.exception.OutputExpiredError:   # finished long ago; the result is on the volume
        return "done"
    if isinstance(result, dict) and "failed" in result: raise TrialFailed(result["failed"])
    return "done"


class Unmapped(Exception):
    """A finished call trained no arm of the spec: its reads cannot be launched."""


def watch_studies(studies, on_done, poll=poll_modal, interval=120, max_transient=60, sleep=time.sleep, log=print, root=ROOT, now=time.time):
    """Poll every spawned trial of the studies until each is settled, calling on_done(study, label) for a finished trial
    until it succeeds; returns the finished calls that map to no arm. State lives in runs/<study>.watch.json, replaced
    atomically after every change, so a restarted watcher resumes: finished trials are not polled again and launched reads
    are not launched again. `launching_at` is written before on_done runs; a restart that finds it logs the interrupted
    launch and calls on_done again, which must check what already happened (kev.rounds.launch_arm_reads does). Network
    errors while polling are retried (max_transient in a row marks the call failed); an on_done that raises (a pull that
    lost the network, reads still in flight) is retried on the next pass; one that raises Unmapped leaves the call
    unlaunched, logged, and settled so the watch can end."""
    runs = Path(root) / "runs"
    calls = {study: read_json(runs / f"{study}.spawn.json")["calls"] for study in studies}
    while True:
        settled = True
        for study, study_calls in calls.items():
            path = runs / f"{study}.watch.json"
            state = read_json(path) if path.exists() else {"calls": {}}
            for label, call_id in study_calls.items():
                s = state["calls"].setdefault(label, {"status": "running", "launched": False, "transient": 0})
                if s["status"] == "running":
                    try:
                        s["status"], s["transient"] = poll(call_id), 0
                    except Exception as error:   # noqa: BLE001 - a trial's own failure or this machine's network, told apart here
                        if transient(error) and s["transient"] + 1 < max_transient:
                            s["transient"] += 1; log(f"{study}/{label}: network error, retrying ({type(error).__name__}: {str(error)[:120]})")
                        else:
                            s["status"], s["error"] = "failed", f"{type(error).__name__}: {str(error)[:300]}"; log(f"{study}/{label}: FAILED {s['error']}")
                if s["status"] == "done" and not s["launched"] and not s.get("unmapped"):
                    if "launching_at" in s: log(f"{study}/{label}: a launch started at {s['launching_at']:.0f} was interrupted; checking the arm's reads before any relaunch")
                    s["launching_at"] = now(); write_json(path, state, atomic=True)
                    try:
                        on_done(study, label); s["launched"] = True; s.pop("launching_at"); log(f"{study}/{label}: done, reads launched")
                    except Unmapped as error:
                        s["unmapped"] = True; s.pop("launching_at"); log(f"!!! {study}/{label}: finished but {error}; its reads were NOT launched")
                    except Exception as error:   # noqa: BLE001 - retried on the next pass; only a dead process leaves launching_at behind
                        s.pop("launching_at"); log(f"{study}/{label}: launching reads failed, retrying next pass ({type(error).__name__}: {str(error)[:300]})")
                write_json(path, state, atomic=True)
                settled = settled and (s["status"] == "failed" or s["launched"] or bool(s.get("unmapped")))
        if settled:
            return [f"{study}/{label}" for study in calls for label, s in read_json(runs / f"{study}.watch.json")["calls"].items() if s.get("unmapped")]
        sleep(interval)


def arm_of(spec, study, label):
    """The arm a spawned call trained (trial directories are <NN>-<label>)."""
    return next((a for a, x in spec["arms"].items() if x.get("trial", "").startswith(f"runs/{study}/") and Path(x["trial"]).name.split("-", 1)[1] == label), None)


def watch(spec, interval=120, stagger=STAGGER, reads_timeout=6 * 3600):
    """The round from spawned trials to read-out: when a trial finishes, pull its study and launch that arm's reads
    (`stagger` apart); once every trial is settled, wait for the reads to land and write the read-out. The wait ends when
    every read is there, when every benchmarks process this watcher started has exited (a failed read never lands), or
    after reads_timeout (a restarted watcher has no processes to follow)."""
    last, procs = [0.0], []
    def on_done(study, label):
        arm = arm_of(spec, study, label)
        if arm is None: raise Unmapped(f"no arm of round {spec['round']} has a trial runs/{study}/<NN>-{label}")
        pull(study, spec)
        time.sleep(max(0.0, last[0] + stagger - time.time()))
        procs.extend(launch_arm_reads(spec, arm, stagger=stagger))
        last[0] = time.time()
    unmapped = watch_studies(list(spec.get("studies", {})), on_done, interval=interval)
    deadline = time.time() + reads_timeout
    finished = [a for a, x in spec["arms"].items() if x.get("trial") and (ROOT / x["trial"] / "result.json").exists()]
    while (waiting := [a for a in finished if not all((ROOT / d / "rows.json").exists() for _, d in side_reads(spec, a, spec["rule"], "candidate", arm_side(spec, a)))]) and time.time() < deadline:
        if procs and all(p.poll() is not None for p in procs): break
        print(f"waiting for the reads of {waiting}", flush=True); time.sleep(interval)
    report = write_readout(spec)
    if unmapped: raise SystemExit(f"finished calls that map to no arm, never read: {unmapped} (fix the spec's arms, then launch-reads)")
    return report


def launchable(spec, rows=True):
    """Problems that stop a launch: a recorded round is never relaunched (register a new one), and a new round needs its
    plans, suites and (for trials) its parents' reads, or its read-out could not be computed."""
    if spec.get("archive"): return [f"round {spec['round']} is a record (evidence on {spec['archive']}); register a new round to train or read again"]
    return validate(spec, rows=rows).problems


def study_commands(spec):
    return {name: ["modal", "run", "modal_app.py::study", "--suite", s["suite"], "--plan", s["plan"], "--name", name, "--transfer", s["transfer"],
                   "--budget", str(s["budget"]), "--timeout", str(s["timeout"]), "--gpu", s["gpu"]] for name, s in spec["studies"].items()}


def launch_studies(spec, stagger=STAGGER, run=subprocess.run, sleep=time.sleep):
    """modal_app.py::study for every study of the round (each spawns its trials on the deployed app and returns), `stagger`
    apart, output in runs/<study>.log (a filter on this output can hide the SystemExit that explains a refusal)."""
    for i, (name, cmd) in enumerate(study_commands(spec).items()):
        if i: sleep(stagger)
        print("launch:", " ".join(cmd), flush=True)
        with (ROOT / "runs" / f"{name}.log").open("w", encoding="utf-8") as f:
            run([sys.executable, "-m", *cmd], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=modal_env(spec), check=True)


def write_readout(spec, root=ROOT, out=None):
    report = readout(spec, root)
    out = Path(out or Path(root) / f"runs/r{spec['round']}-readout"); out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"round{spec['round']}.json", report); print(table(report))
    return report


# --- CLI -------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kev.rounds", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "launch", "launch-reads", "watch", "readout", "confirm"):
        p = sub.add_parser(name); p.add_argument("spec")
        if name in ("validate", "readout", "confirm"): p.add_argument("--root", default=str(ROOT), help="checkout whose runs/ holds the rows (default: this one)")
        if name == "validate": p.add_argument("--partitions", action="store_true", help="verify the suites' partitions through load_plan (may fetch from the Hub)")
        if name in ("launch", "launch-reads", "watch"): p.add_argument("--stagger", type=int, default=STAGGER)
        if name in ("launch", "launch-reads"): p.add_argument("--dry-run", action="store_true", help="print the modal commands only")
        if name == "launch-reads": p.add_argument("--arms", help="comma-separated arms (default: every arm with a pulled result, and every arm that is a checkpoint without a trial)"); p.add_argument("--parents", action="store_true", help="also the parents' missing reads")
        if name in ("launch-reads", "confirm"): p.add_argument("--stage", required=name == "confirm"); p.add_argument("--arm", help="the candidate (default: the one the readout names)")
        if name == "watch": p.add_argument("--interval", type=int, default=120)
        if name in ("readout", "confirm"): p.add_argument("--out")
    a = ap.parse_args(argv)
    spec, root = load(a.spec), Path(getattr(a, "root", ROOT))
    for line in calibration_warnings(spec, root) if a.cmd in ("validate", "launch") else ():
        print(f"!!! warning: {line}")
    if a.cmd == "validate":
        problems, archived = validate(spec, root, partitions=a.partitions)
        if archived: print(f"archived (on {spec['archive']}, not in this checkout):\n  " + "\n  ".join(archived))
        print("\n".join(problems) or f"round {spec['round']}: ok ({len(spec['arms'])} arms, {len(spec.get('studies', {}))} studies)")
        raise SystemExit(1 if problems else 0)
    if a.cmd == "readout":
        write_readout(spec, root, a.out); return
    if a.cmd == "confirm":
        arm = a.arm or _candidate(spec, root)
        report = confirm(spec, a.stage, arm, root)
        out = Path(a.out or root / f"runs/r{spec['round']}-verdict"); out.mkdir(parents=True, exist_ok=True)
        write_json(out / f"{size_of(arm)}-{a.stage}.json", report); print(table(report)); return
    problems = launchable(spec, rows=a.cmd != "launch-reads")   # launch-reads is how a missing parent read gets made
    if problems: raise SystemExit("spec does not validate for a launch:\n" + "\n".join(problems))
    if a.cmd == "launch":
        if a.dry_run: print("\n".join(" ".join(c) for c in study_commands(spec).values()))
        else: launch_studies(spec, a.stagger)
        return
    if a.cmd == "watch":
        watch(spec, a.interval, a.stagger); return
    arms = [a.arm or _candidate(spec, root)] if a.stage else (a.arms.split(",") if a.arms else [x for x, v in spec["arms"].items() if not v.get("trial") or (root / v["trial"] / "result.json").exists()])
    sides = ("candidate", "parent") if a.parents or a.stage else ("candidate",)
    if a.dry_run:
        for arm in arms: print("\n".join(" ".join(c) for c in read_commands(spec, arm, a.stage, root, sides)))
        return
    for i, arm in enumerate(arms):
        if i: time.sleep(a.stagger)
        try: launch_arm_reads(spec, arm, a.stage, sides, a.stagger)
        except InFlight as error: print(f"!!! {error}")


def _candidate(spec, root):
    report = read_json(Path(root) / f"runs/r{spec['round']}-readout/round{spec['round']}.json")
    chosen = [arm for arm in report["candidates"].values() if arm]
    if len(chosen) != 1: raise SystemExit(f"the readout names {len(chosen)} candidates ({report['candidates']}); pass --arm")
    return chosen[0]


if __name__ == "__main__":
    main()
