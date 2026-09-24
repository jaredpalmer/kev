import copy
import pathlib
import random

import pytest
import torch

from kev import evaluate
from kev.data import materialize
from kev.suite import record_digest
from kev.train import question_loss


def choice_request():
    return {"state": "The shoes are the wrong size.", "questions": {"reason": {
        "type": "choice", "instructions": "Why return the shoes?",
        "criteria": {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"},
        "label": "size", "src": "fixture",
    }}}


def test_clean_evaluation_does_not_add_options(monkeypatch):
    observed = []

    def predict(tok, model, req):
        observed.append(copy.deepcopy(req))
        rec = materialize(req)
        return rec, [torch.ones(len(q["options"])) / len(q["options"]) for q in rec["questions"]]

    monkeypatch.setattr(evaluate, "_probs", predict)
    evaluate.test_accuracy(None, None, [choice_request() for _ in range(100)], random.Random(1))
    assert all(set(r["questions"]["reason"]["criteria"]) == {"size", "damage", "color"} for r in observed)


def test_none_removed_relabels_and_counts_complete_pairs(monkeypatch):
    observed = []

    def predict(tok, model, req):
        rec = materialize(req)
        observed.append(copy.deepcopy(req))
        return rec, [torch.nn.functional.one_hot(torch.tensor(q["label"]), len(q["options"])).float() for q in rec["questions"]]

    monkeypatch.setattr(evaluate, "_probs", predict)
    report = evaluate.test_none_of_the_above(None, None, [choice_request()], random.Random(1))
    assert report["n"] == 1
    assert report["true_option_present"]["acc"] == 1
    assert report["true_option_removed"]["picks_none_rate"] == 1
    assert observed[1]["questions"]["reason"]["label"] != "size"


def test_score_loss_is_proper_at_true_distribution():
    logits = torch.tensor([0.2, 0.8]).log().requires_grad_()
    loss = sum(p * question_loss(logits, {"label": y, "qtype": "score"}, "cpu", 0.5) for y, p in enumerate([0.2, 0.8]))
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-6


def frozen_request(i=0):
    r = choice_request()
    r["_meta"] = {"id": f"item-{i}", "group_id": f"item-{i}", "source": "fixture", "variant": "clean"}
    return r


def test_contrast_cases_preserve_groups_and_relabel():
    from kev.suite import contrast_cases
    original = frozen_request()
    present, absent, permuted = contrast_cases(original)
    assert present["questions"]["reason"]["label"] == "size"
    assert absent["questions"]["reason"]["label"] == "none_of_these"
    for record in (present, absent, permuted):
        assert record["_meta"]["group_id"] == original["_meta"]["id"]
        materialize(record)
    assert original == frozen_request()


def test_source_sampling_does_not_depend_on_other_sources(monkeypatch):
    from kev import data

    def convert(split, n, rng):
        value = rng.randrange(1000000)
        rng.origins = [{"row": value, "row_sha256": str(value), "text_sha256": str(value)}]
        return [choice_request()]

    monkeypatch.setattr(data, "SOURCES", {"agnews": (convert, "train", "test"), "mnli": (convert, "train", "test")})
    alone = data.build(1, only=["mnli"])
    together = data.build(1)
    assert alone[0] == next(r for r in together if r["_meta"]["source"] == "mnli")


def test_strict_encoding_rejects_truncation():
    from types import SimpleNamespace
    from kev.model import encode

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=list(range(len(text))))

        def convert_tokens_to_ids(self, text):
            return 1000

    rec = {"state": "abcdefgh", "questions": [{"instr": "q", "options": ["a", "b"], "label": 0}]}
    assert encode(Tokenizer(), rec, max_state=4)["state_truncated"]
    with pytest.raises(ValueError, match="state exceeds"):
        encode(Tokenizer(), rec, max_state=4, strict=True)


def test_locked_split_and_hash_verification(tmp_path):
    import json
    from kev.suite import digest, load_split, write_json, write_jsonl
    for name in ("development", "test"):
        write_jsonl(tmp_path / f"{name}.jsonl", [frozen_request()])
    manifest = {"files": {f"{name}.jsonl": {"sha256": digest(tmp_path / f"{name}.jsonl"), "records": 1} for name in ("development", "test")}}
    write_json(tmp_path / "manifest.json", manifest)
    assert len(load_split(tmp_path, "development")) == 1
    with pytest.raises(ValueError, match="locked test"):
        load_split(tmp_path, "test")
    write_jsonl(tmp_path / "development.jsonl", [{}])
    with pytest.raises(ValueError, match="checksum"):
        load_split(tmp_path, "development")


def test_api_payload_excludes_answers_and_metadata():
    from kev.data import api_request
    clean = api_request(frozen_request())
    assert set(clean) == {"state", "questions"}
    assert set(clean["questions"]["reason"]) == {"type", "instructions", "criteria"}


def test_failed_prediction_cannot_produce_partial_score(tmp_path):
    from kev.suite import read_json
    from kev.benchmark import evaluate_records

    def fail(record):
        raise ValueError("invalid prediction")

    out = tmp_path / "evaluation"
    with pytest.raises(ValueError):
        evaluate_records([frozen_request()], fail, out)
    failure = read_json(out / "failure.json")
    assert failure["coverage"]["requested_records"] == 1
    assert failure["coverage"]["evaluated_records"] == 0
    assert failure["coverage"]["rejected_records"] == 1
    assert not (out / "report.json").exists()


def test_overlong_records_are_rejected_only_for_suites_scored_as_published(tmp_path):
    """ContextOverflow from the encoder (any of its three limits) is a counted rejection under skip_overlong and an abort
    otherwise; every other ValueError still aborts either way."""
    from kev.benchmark import evaluate_records
    from kev.model import ContextOverflow
    from kev.suite import read_json
    records = [frozen_request(0), frozen_request(1), frozen_request(2)]
    keys = list(records[0]["questions"]["reason"]["criteria"])

    def predictor(record):
        if record["_meta"]["id"] == "item-1": raise ContextOverflow("branch too long: 1590 tokens with a 7000-token state (row limit 8192)")
        return {"probabilities": {"reason": {k: 1.0 / len(keys) for k in keys}}, "latency_ms": 1.0}

    report, _ = evaluate_records(records, predictor, tmp_path / "published", skip_overlong=True)
    assert report["coverage"]["evaluated_records"] == 2 and report["coverage"]["rejected_records"] == 1
    assert [r["id"] for r in read_json(tmp_path / "published" / "rejected.json")] == ["item-1"]
    with pytest.raises(ContextOverflow):
        evaluate_records(records, predictor, tmp_path / "admitted")
    assert read_json(tmp_path / "admitted" / "failure.json")["error_type"] == "ContextOverflow"

    def other(record):
        raise ValueError("answer IDs differ from request IDs")
    with pytest.raises(ValueError):
        evaluate_records(records, other, tmp_path / "other", skip_overlong=True)


def test_missing_answers_and_nonfinite_probabilities_fail():
    from kev.benchmark import prediction_rows, validate_distribution
    with pytest.raises(ValueError, match="answer IDs"):
        prediction_rows(frozen_request(), {"probabilities": {}})
    with pytest.raises(ValueError, match="non-finite"):
        validate_distribution({"x": float("nan"), "y": 0.5}, ["x", "y"])
    with pytest.raises(ValueError, match="sum"):
        validate_distribution({"x": 0, "y": 0}, ["x", "y"])


def test_task_macro_and_record_bootstrap():
    from kev.benchmark import prediction_rows, summarize
    from kev.metrics import paired_bootstrap
    pred = {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}}
    rows = prediction_rows(frozen_request(), pred)
    report = summarize(rows)
    assert report["objective"] == pytest.approx(-report["clean"]["nll"])
    assert paired_bootstrap(rows, rows, samples=50)["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(rows, [])


def test_trial_config_cannot_change_evaluator_or_read_test():
    from kev.experiment import validated_trial
    manifest = {"base_revisions": {"model": "pinned"}}
    assert validated_trial({"base": "model"}, manifest)["ord_w"] == 0
    for extra in ({"test": True}, {"command": "echo x"}, {"lr": -1}, {"epochs": 2.5}):
        with pytest.raises(ValueError):
            validated_trial({"base": "model", **extra}, manifest)


def test_batched_mask_matches_single_and_pads_are_invisible():
    from kev.model import branch_mask, branch_mask_batch
    a, b = [0, 0, 1, 1, 2], [0, 1, 1]
    m = branch_mask_batch([a, b], "cpu")
    assert m.shape == (2, 1, 5, 5)
    assert torch.equal(m[0:1], branch_mask(a, "cpu"))
    assert torch.equal(m[1:2, :, :3, :3], branch_mask(b, "cpu"))
    allowed = m[1, 0] == 0
    assert not allowed[:3, 3:].any()          # real tokens never attend to padding
    assert allowed[3, 3] and allowed[4, 4]    # padded rows keep the diagonal, so softmax is finite
    assert not allowed[3, 1:3].any()          # pads belong to no question segment (state stays visible; rows are discarded)


def test_eval_only_sources_cannot_be_trained(tmp_path):
    import json
    from kev.data import EVAL_ONLY, TRAINABLE, ALL_SOURCES
    from kev.suite import digest, write_json, write_jsonl
    from kev.experiment import load_plan
    assert "mmlu" in EVAL_ONLY and not set(TRAINABLE) & set(EVAL_ONLY) and set(TRAINABLE) | set(EVAL_ONLY) == set(ALL_SOURCES)
    r = frozen_request(); r["_meta"]["source"] = "mmlu"
    for name in ("train", "calibration", "development"):
        write_jsonl(tmp_path / f"{name}.jsonl", [r])
    write_json(tmp_path / "manifest.json", {"base_revisions": {"m": "x"}, "files": {f"{n}.jsonl": {"sha256": digest(tmp_path / f"{n}.jsonl"), "records": 1} for n in ("train", "calibration", "development")}})
    write_json(tmp_path / "plan.json", [{"base": "m"}])
    with pytest.raises(ValueError, match="eval-only"):
        load_plan(tmp_path, tmp_path / "plan.json")


def test_contrastive_pairs_are_checked_and_labelled_by_code():
    from kev import contrastive
    from kev.benchmark import labels
    from kev.contrastive import FAMILIES, UNDETERMINED, check_pair, generate, label_of, paired_flip
    records, report = generate(5, seed=7)
    assert len(records) == 2 * 5 * len(FAMILIES) and all(v["pairs"] == 5 for v in report.values())
    for a, b in zip(records[::2], records[1::2]):
        assert a["_meta"]["pair_id"] == b["_meta"]["pair_id"] and a["_meta"]["family_id"] == b["_meta"]["family_id"]
        assert a["questions"]["decision"]["label"] != b["questions"]["decision"]["label"]
        assert a["state"]["policy"] == b["state"]["policy"]
        materialize(a); materialize(b)
    # a family whose label leaks into the policy text (no evidence needed) must be rejected by the ablation check
    def leaky(rng):
        def evaluate(f): return True
        item = {"policy": "Everything is allowed.", "sentences": [("Filler.", {}), ("Age is 30.", {"age": 30})], "evaluate": evaluate,
                "question": {"type": "noul", "instructions": "Allowed?"}}
        other = {**item, "sentences": [("Filler.", {}), ("Age is 10.", {"age": 10})], "evaluate": lambda f: False}
        return item, other
    assert check_pair(*leaky(None)) == "ablation_failed"
    # a pair whose two items do not differ in exactly one sentence is rejected
    a, b = FAMILIES["authorization"](random.Random(1))
    b["sentences"][2] = ("The refund amount is $1.", {})
    assert check_pair(a, b) == "not_exactly_one_sentence_differs"
    assert label_of(a, drop=0) == UNDETERMINED
    # paired_flip: a constant model never flips; a perfect model flips every pair and gets both right
    rows = []
    for rec in records[:8]:
        keys, y = labels(rec["questions"]["decision"])
        rows.append({"pair_id": rec["_meta"]["pair_id"], "sibling": rec["_meta"]["sibling"], "keys": keys, "label": y, "p": [1.0 if i == y else 0.0 for i in range(len(keys))]})
    assert paired_flip(rows) == {"pairs": 4, "flip_rate": 1.0, "both_correct_rate": 1.0}
    constant = [{**r, "p": [1.0] + [0.0] * (len(r["keys"]) - 1)} for r in rows]
    assert paired_flip(constant)["flip_rate"] == 0.0


def test_permuted_variants_pair_with_their_parent_not_their_group():
    from kev.benchmark import prediction_rows, summarize
    from kev.suite import contrast_cases
    rows = []
    for i in range(2):
        r = frozen_request(i); r["_meta"]["group_id"] = "shared-pair"     # siblings share a bootstrap group
        variants = contrast_cases(r)
        for rec in [r] + variants:
            rec["_meta"].setdefault("group_id", "shared-pair")
            keys = list(rec["questions"]["reason"]["criteria"])
            rows += prediction_rows(rec, {"probabilities": {"reason": {k: (0.7 if k == rec["questions"]["reason"]["label"] else 0.3 / (len(keys) - 1)) for k in keys}}})
    report = summarize(rows)
    assert report["permutation"]["n"] == 2 and report["permutation"]["flip_rate"] == 0.0


def test_contrastive_eval_split_is_stratified_by_family():
    from collections import Counter
    from kev.contrastive import generate
    recs, _ = generate(6, seed="t", families=["authorization", "deadline"])
    dev, test = [], []
    for i in range(0, len(recs), 2):
        (dev if (i // 2) % 2 == 0 else test).extend(recs[i : i + 2])
    for part in (dev, test):
        fams = Counter(r["_meta"]["family"] for r in part)
        assert set(fams) == {"authorization", "deadline"} and all(v == 6 for v in fams.values())
        assert all(a["_meta"]["pair_id"] == b["_meta"]["pair_id"] for a, b in zip(part[::2], part[1::2]))


def test_coverage_cannot_split_equal_confidence_ties():
    from kev.metrics import coverage_at_error
    correct = [True] * 90 + [False] * 10
    assert coverage_at_error([0.99] * 100, correct, 0.05) == 0.0
    assert coverage_at_error([0.99] * 100, correct[::-1], 0.05) == 0.0
    assert coverage_at_error([0.99] * 100, correct, 0.1) == 1.0
    assert coverage_at_error([], [], 0.05) == 0.0


def test_risk_curve_thresholds_and_nonmonotone_risk():
    from kev.metrics import coverage_at_error, risk_coverage_curve
    curve = risk_coverage_curve([0.99, 0.99, 0.9, 0.8], [True, False, True, True])
    assert [p["accepted"] for p in curve] == [2, 3, 4]
    assert [p["threshold"] for p in curve] == [0.99, 0.9, 0.8]
    assert [p["risk"] for p in curve] == pytest.approx([0.5, 1 / 3, 0.25])
    assert coverage_at_error([0.99, 0.99, 0.9, 0.8], [True, False, True, True], 0.25) == 1.0


@pytest.mark.parametrize("confidence,correct,budget", [
    ([float("nan")], [True], 0.05), ([1.1], [True], 0.05),
    ([0.5], [], 0.05), ([[0.5]], [True], 0.05), ([0.5], [True], -0.1),
])
def test_selective_metrics_reject_invalid_inputs(confidence, correct, budget):
    from kev.metrics import coverage_at_error
    with pytest.raises(ValueError):
        coverage_at_error(confidence, correct, budget)


def test_fixed_threshold_does_not_reselect_using_evaluation_labels():
    from kev.metrics import select_threshold, evaluate_threshold
    threshold = select_threshold([0.99, 0.98, 0.97, 0.6], [True, True, True, False], 0.05)
    assert threshold == 0.97
    result = evaluate_threshold([0.99, 0.7, 0.6], [False, True, True], threshold)
    assert result["accepted"] == 1 and result["errors"] == 1 and result["risk"] == 1.0
    assert result["coverage"] == pytest.approx(1 / 3)
    empty = evaluate_threshold([0.99], [True], None)
    assert empty["coverage"] == 0 and empty["risk"] is None


def test_global_coverage_bootstrap_recomputes_full_statistic():
    from kev.metrics import metrics, paired_bootstrap
    candidate, reference = [], []
    for i in range(20):
        common = {"id": str(i), "group": "one-cluster", "source": "fixture", "task": "fixture",
                  "question": "q", "variant": "clean", "type": "choice", "keys": ["a", "b"], "label": 0}
        candidate.append({**common, "p": [0.99, 0.01] if i < 18 else [0.4, 0.6]})
        reference.append({**common, "p": [0.6, 0.4] if i < 18 else [0.01, 0.99]})
    assert metrics(candidate)["acc"] == metrics(reference)["acc"]
    result = paired_bootstrap(candidate, reference, samples=50, metric="coverage_at_5pct_error", aggregation="micro")
    assert result["micro_coverage_at_5pct_error_delta"] == pytest.approx(0.9)
    assert result["ci95"] == pytest.approx([0.9, 0.9])
    assert result["groups"] == 1
    assert paired_bootstrap(candidate, candidate, samples=50, metric="aurc", aggregation="micro")["ci95"] == [0, 0]


def test_bootstrap_rejects_duplicate_or_inconsistent_pairs():
    from kev.benchmark import prediction_rows
    from kev.metrics import paired_bootstrap
    rows = prediction_rows(frozen_request(), {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}})
    with pytest.raises(ValueError, match="duplicate"):
        paired_bootstrap(rows + rows, rows)
    with pytest.raises(ValueError, match="group"):
        paired_bootstrap(rows, [{**rows[0], "group": "different"}])


def test_temperature_preserves_argmax_but_not_cross_question_ranking():
    import numpy as np
    from kev.metrics import probabilities_at_temperature
    rows = [{"p": [0.6, 0.2, 0.2]}, {"p": [0.55, 0.449, 0.001]}]
    calibrated = [probabilities_at_temperature(r, 2.0) for r in rows]
    assert max(rows[0]["p"]) > max(rows[1]["p"])
    assert calibrated[0].max() < calibrated[1].max()
    assert all(np.argmax(r["p"]) == p.argmax() for r, p in zip(rows, calibrated))


def test_calibration_uses_logits_without_probability_floor_distortion():
    import numpy as np
    from kev.metrics import probabilities_at_temperature
    p = probabilities_at_temperature({"p": [1.0, 0.0], "logits": [0.0, -100.0]}, 2.0)
    assert p[1] == pytest.approx(np.exp(-50), rel=1e-6, abs=0)
    for temperature in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            probabilities_at_temperature({"p": [0.5, 0.5]}, temperature)


def test_unknowable_not_in_raw_or_calibrated_accuracy_denominator():
    from kev.benchmark import prediction_rows, summarize
    rows = prediction_rows(frozen_request(), {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}})
    rows.append({**rows[0], "id": "unk", "source": "unknowable", "task": "unknowable_test"})
    result = summarize(rows, temperature=2.0)
    assert result["clean"]["n"] == result["calibrated_clean"]["n"] == 1
    assert result["metric_policy"]["selective_ties"] == "whole_confidence_groups"


def test_logits_are_recorded_in_option_order():
    from kev.benchmark import prediction_rows
    prediction = {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}},
                  "logits": {"reason": {"color": -2.0, "damage": -2.0, "size": 0.0}}, "inference_temperature": 1.0}
    row = prediction_rows(frozen_request(), prediction)[0]
    assert row["logits"] == [0.0, -2.0, -2.0] and row["inference_temperature"] == 1.0
    prediction["logits"]["reason"]["size"] = float("inf")
    with pytest.raises(ValueError, match="logit"):
        prediction_rows(frozen_request(), prediction)


def test_risk_curve_area_has_explicit_tie_policy():
    from kev.metrics import area_under_risk_coverage
    assert area_under_risk_coverage([0.99, 0.9, 0.8], [True, True, False]) == pytest.approx(1 / 9)
    assert area_under_risk_coverage([0.99] * 10, [True] * 9 + [False]) == pytest.approx(0.1)
    assert area_under_risk_coverage([0.99] * 10, [False] + [True] * 9) == pytest.approx(0.1)


@pytest.mark.parametrize("options", [{}, {"label_smoothing": 0.05}, {"brier_w": 0.5}, {"focal_gamma": 1.0}])
def test_training_losses_match_definitions(options):
    z = torch.tensor([0.4, -0.7, 1.2], requires_grad=True)
    q = {"label": 1, "qtype": "choice"}
    ce = torch.nn.functional.cross_entropy(z[None], torch.tensor([1]))
    target = torch.nn.functional.one_hot(torch.tensor(1), 3).float()
    expected = ce
    if "label_smoothing" in options:
        epsilon = options["label_smoothing"]
        expected = -(((1 - epsilon) * target + epsilon / 3) * z.log_softmax(-1)).sum()
    elif "brier_w" in options:
        expected = ce + options["brier_w"] * (z.softmax(-1) - target).square().sum()
    elif "focal_gamma" in options:
        expected = (1 - z.softmax(-1)[1]) ** options["focal_gamma"] * ce
    actual = question_loss(z, q, "cpu", 0, **options)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(z.grad).all()


def test_brier_mixture_is_proper_and_soft_targets_unchanged():
    distribution = torch.tensor([0.2, 0.3, 0.5])
    logits = distribution.log().requires_grad_()
    expected_loss = sum(p * question_loss(logits, {"label": i, "qtype": "choice"}, "cpu", 0, brier_w=0.5)
                        for i, p in enumerate(distribution))
    expected_loss.backward()
    assert logits.grad.abs().max() < 1e-6
    soft = {"label": 0, "qtype": "choice", "target": [1 / 3] * 3}
    base = question_loss(logits, soft, "cpu", 0)
    for options in ({"label_smoothing": 0.05}, {"brier_w": 0.5}, {"focal_gamma": 1.0}):
        assert torch.equal(question_loss(logits, soft, "cpu", 0, **options), base)


@pytest.mark.parametrize("bad", [{"label_smoothing": -0.1}, {"label_smoothing": 1.1}, {"brier_w": float("nan")},
                                  {"focal_gamma": -1}, {"brier_w": 0.5, "focal_gamma": 1.0}])
def test_loss_options_fail_before_training(bad):
    with pytest.raises(ValueError):
        question_loss(torch.zeros(3), {"label": 0, "qtype": "choice"}, "cpu", 0, **bad)


def test_trial_accepts_one_registered_loss_change():
    from kev.experiment import validated_trial
    manifest = {"base_revisions": {"model": "pinned"}}
    for flag, value in (("label_smoothing", 0.05), ("brier_w", 0.5), ("focal_gamma", 1.0)):
        assert validated_trial({"base": "model", flag: value}, manifest)[flag] == value
    with pytest.raises(ValueError):
        validated_trial({"base": "model", "brier_w": 0.5, "focal_gamma": 1.0}, manifest)


def test_temperature_fit_requires_raw_rows_and_uses_true_logit_nll():
    from kev.metrics import fit_temperature, nll_at_temperature
    row = {"variant": "clean", "source": "fixture", "task": "fixture", "p": [1.0, 0.0],
           "logits": [0.0, -100.0], "label": 1, "inference_temperature": 1.0}
    assert nll_at_temperature(row, 2.0) == pytest.approx(50.0)
    assert fit_temperature([row], aggregation="micro") == pytest.approx(4.0)
    with pytest.raises(ValueError, match="raw logits"):
        fit_temperature([{**row, "inference_temperature": 2.0}])


def test_cross_validated_temperature_is_group_disjoint_and_reports_intervals():
    import numpy as np
    from kev.metrics import cross_validated_temperature
    weights = np.exp([3.0, 0.0]); p = (weights / weights.sum()).tolist()
    rows = []
    for source in ("a", "b"):
        for group in range(10):
            for _ in range(2):
                i = len(rows)
                rows.append({"id": str(i), "source": source, "group": f"g{group}", "task": "t", "type": "choice",
                             "variant": "clean", "question": "q", "keys": ["x", "y"],
                             "label": 1 if i % 4 == 3 else 0, "logits": [3.0, 0.0], "p": p, "inference_temperature": 1.0})
    from kev.metrics import grouped_folds
    fold_id = grouped_folds(rows, 5, 0)
    fold_of = {(r["source"], r["group"]): f for r, f in zip(rows, fold_id)}
    assert all(fold_of[(r["source"], r["group"])] == f for r, f in zip(rows, fold_id))   # a group never straddles folds
    for source in ("a", "b"):
        assert {fold_of[(source, f"g{g}")] for g in range(10)} == set(range(5))   # every source in every fold
    result = cross_validated_temperature(rows, folds=5, samples=200)
    assert len(result["temperatures"]) == 5 and all(t > 1 for t in result["temperatures"])
    assert result["out_of_fold"]["ece"] < result["raw"]["ece"]
    lo, hi = result["ece_ci95"]["delta"]
    assert lo <= hi and isinstance(result["separated"], bool)
    assert result["raw"]["n"] == result["out_of_fold"]["n"] == 40


def test_raw_row_inverts_the_served_temperature():
    import numpy as np
    from kev.metrics import fit_temperature, raw_row, tempered_row
    raw = {"variant": "clean", "source": "s", "task": "t", "type": "choice", "label": 0, "logits": [2.0, -1.0, 0.5],
           "p": (np.exp([2.0, -1.0, 0.5]) / np.exp([2.0, -1.0, 0.5]).sum()).tolist(), "inference_temperature": 1.0}
    served = tempered_row(raw, 2.3)
    back = raw_row(served)
    assert back["inference_temperature"] == 1.0 and np.allclose(back["p"], raw["p"])
    assert np.allclose(np.asarray(back["logits"]) - max(back["logits"]), np.asarray(raw["logits"]) - max(raw["logits"]))
    assert raw_row(raw) is raw
    fit_temperature([back], aggregation="micro")   # accepted as raw
    with pytest.raises(ValueError, match="raw logits"):
        fit_temperature([served])
    with pytest.raises(ValueError, match="recorded none"):
        raw_row({"p": [0.6, 0.4], "inference_temperature": 2.0})
    twice = tempered_row(served, 1.5)                        # a second temperature composes, and raw_row still restores T=1
    assert twice["inference_temperature"] == pytest.approx(2.3 * 1.5) and np.allclose(raw_row(twice)["p"], raw["p"])


def test_workload_report_refuses_rows_without_logits():
    from kev.calibrate import workload_report
    rows = [{"id": str(i), "source": "jev", "group": f"g{i}", "task": "t", "type": "choice", "variant": "clean",
             "question": "q", "keys": ["x", "y"], "label": 0, "p": [0.7, 0.3]} for i in range(10)]
    with pytest.raises(ValueError, match="logits"):
        workload_report(rows, folds=2, samples=10)


def test_workload_report_recovers_a_shared_temperature_and_keeps_accuracy():
    import numpy as np
    from kev.calibrate import format_report, workload_report
    from kev.metrics import tempered_row
    rng = np.random.default_rng(0)
    rows = []
    for source in ("a", "b"):
        for group in range(12):
            for k in range(2):
                i = len(rows)
                label = int(rng.integers(0, 3))
                z = rng.normal(0, 1, 3); z[label] += 1.5          # informative, over-confident at T=1
                p = np.exp(z - z.max()); p /= p.sum()
                raw = {"id": str(i), "source": source, "group": f"g{group}", "task": "t", "type": "choice", "variant": "clean",
                       "question": "q", "keys": ["x", "y", "z"], "label": label, "logits": z.tolist(), "p": p.tolist(), "inference_temperature": 1.0}
                rows.append(tempered_row(raw, 1.7))                # served at the checkpoint's temperature
    report = workload_report(rows, folds=4, seed=0, samples=50)
    arms = report["arms"]
    assert report["shipped_temperature"] == [1.7] and len(report["fold_temperatures"]) == 4
    assert len({round(arms[a]["acc"], 12) for a in arms}) == 1                      # temperature never moves accuracy
    assert arms["workload"]["nll"] <= arms["raw"]["nll"] and arms["workload"]["nll"] <= arms["shipped"]["nll"]   # in-sample fit is optimal on the grid
    assert set(report["oof_vs_shipped"]) == {"ece", "brier", "coverage_at_5pct_error", "aurc"}
    assert all(b["ci95"][0] <= b["ci95"][1] for b in report["oof_vs_shipped"].values())
    assert "workload_oof" in format_report(report)


def test_cross_validated_temperature_rejects_too_few_groups():
    from kev.metrics import cross_validated_temperature
    rows = [{"id": str(i), "source": "a", "group": f"g{i}", "task": "t", "type": "choice", "variant": "clean",
             "label": 0, "logits": [1.0, 0.0], "p": [0.73, 0.27], "inference_temperature": 1.0} for i in range(3)]
    with pytest.raises(ValueError, match="fewer"):
        cross_validated_temperature(rows, folds=5)


def test_training_plan_matches_registered_screen():
    import json
    from pathlib import Path
    from kev.experiment import load_plan
    from kev.suite import read_json
    root = Path(__file__).resolve().parents[1]
    protocol = read_json(root / "experiments/calibration-audit-protocol.json")
    trials = load_plan(root / protocol["data"]["decision_suite"], root / "experiments/calibration-screen-4b.json")
    loss_keys = {"label_smoothing", "brier_w", "focal_gamma"}
    assert len(trials) == len(protocol["screen"]["arms"]) == 4
    assert all({k: v for k, v in t.items() if k not in loss_keys} == {k: v for k, v in trials[0].items() if k not in loss_keys} for t in trials)
    assert trials[0]["init_from"] == protocol["parents"]["4b"]
    for trial, arm in zip(trials, protocol["screen"]["arms"]):
        assert all(trial[k] == arm[k] for k in loss_keys)


def test_modal_compute_bound_includes_requested_memory_and_cpu():
    from modal_app import TRIAL_CPU, TRIAL_MEMORY, compute_bound
    expected = 3.95 + TRIAL_CPU * 0.04730 + TRIAL_MEMORY[1] / 1024 * 0.008
    assert compute_bound("H100", 3600, 1) == pytest.approx(expected)
    with pytest.raises(ValueError):
        compute_bound("unknown", 3600, 1)


def test_modal_worker_preserves_object_dependency_environment():
    from modal_app import worker_environment
    env = worker_environment("kev-calibration-audit", "H100", "named-secret")
    assert env["KEV_APP_NAME"] == "kev-calibration-audit"
    assert env["KEV_GPU"] == "H100" and env["KEV_HF_SECRET"] == "named-secret"
    assert "HF_TOKEN" not in env
    assert "KEV_HF_SECRET" not in worker_environment("kev-research", "H100")


def test_repeat_pull_refetches_trial_dirs_copied_mid_run(monkeypatch, tmp_path, capsys):
    """A trial dir without result.json was copied while the trial ran: the next pull deletes and re-fetches it,
    keeps finished dirs untouched, adds new ones, and names what is still running."""
    import modal_app
    study = tmp_path / "runs" / "s"
    (study / "00-trial-0").mkdir(parents=True); (study / "00-trial-0/result.json").write_text("{}", encoding="utf-8")
    (study / "00-trial-0/keep").write_text("local", encoding="utf-8")
    (study / "01-trial-1/checkpoint").mkdir(parents=True); (study / "01-trial-1/checkpoint/half.bin").write_text("partial", encoding="utf-8")
    (study / "results.jsonl").write_text("stale", encoding="utf-8")
    volume = {"00-trial-0": True, "01-trial-1": True, "02-trial-2": True, "03-trial-3": False}   # name -> finished on the volume
    fetched, aggregated = [], []

    def pull_volume(remote, local_parent):
        name = remote.rsplit("/", 1)[1]
        fetched.append(name)
        (local_parent / name).mkdir()
        if volume[name]: (local_parent / name / "result.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(modal_app, "ROOT", tmp_path)
    monkeypatch.setattr(modal_app, "pull_volume", pull_volume)
    monkeypatch.setattr(modal_app, "volume_names", lambda path: (set(volume), set()))
    monkeypatch.setattr(modal_app.subprocess, "run", lambda cmd, **kw: aggregated.append(cmd))
    assert modal_app.pull_study("s") == study
    assert fetched == ["01-trial-1", "02-trial-2", "03-trial-3"]
    assert (study / "00-trial-0/keep").exists() and (study / "01-trial-1/result.json").exists()
    assert not (study / "01-trial-1/checkpoint").exists() and not (study / "results.jsonl").exists()
    assert "no result.json): ['03-trial-3']" in capsys.readouterr().out
    assert aggregated and "--aggregate" in aggregated[0]


def test_locked_test_passes_timeout_and_memory_through(monkeypatch, tmp_path):
    import modal_app
    calls = {}

    class Fn:
        def with_options(self, **kw): calls.update(kw); return self
        def remote(self, *args): return {"suites": {"decision": {"clean": {"acc": 0.9, "brier": 0.1}}}}

    monkeypatch.setattr(modal_app, "ROOT", tmp_path)
    monkeypatch.setattr(modal_app.modal.Function, "from_name", lambda app, name: Fn())
    monkeypatch.setattr(modal_app, "local_git_commit", lambda: "0" * 40)
    monkeypatch.setattr(modal_app, "pull_volume", lambda remote, local_parent: None)
    locked_test = modal_app.locked_test.info.raw_f
    locked_test("s/00-trial-0", "cand")
    assert calls == {"gpu": modal_app.GPU, "timeout": 3600, "memory": (32768, 49152)}   # run_locked_test's own defaults
    locked_test("s/00-trial-0", "cand", gpu="H200", timeout=14400, memory_mb=131072)
    assert calls == {"gpu": "H200", "timeout": 14400, "memory": (32768, 131072)}


class FakeServed:
    """encode/probs_and_prefix over materialized records. Alone, a question prefers its last option; `leak(others)`
    (the other questions' instructions in the same request) is added to option 0's logit, i.e. broken isolation."""
    def __init__(self, leak=lambda others: 0.0):
        self.leak = leak

    def encode(self, tok, rec):
        return rec

    def probs_and_prefix(self, rec):
        out = []
        for i, q in enumerate(rec["questions"]):
            logits = torch.arange(len(q["options"]), dtype=torch.float)
            logits[0] += self.leak([o["instr"] for j, o in enumerate(rec["questions"]) if j != i])
            out.append(torch.softmax(logits, 0))
        return out, None


def test_served_isolation_compares_alone_with_packed_and_sibling():
    from scripts.serving_bench import isolation
    noul = {"type": "noul", "instructions": "Is it late?", "label": True, "src": "fixture"}
    raw = [{"state": "s", "questions": {"a": choice_request()["questions"]["reason"], "b": noul}}, {"state": "t", "questions": {"b": noul}}]
    exact = isolation(FakeServed(), None, raw)
    assert exact == {k: {"questions": 3, "max_dp": 0.0, "mean_dp": 0.0, "argmax_flips": 0} for k in ("packed", "sibling")}
    # reads its siblings: the two-question record moves when packed, every question moves next to the probe
    crowded = isolation(FakeServed(lambda others: 5.0 * len(others)), None, raw)
    assert crowded["packed"]["argmax_flips"] == 2 and crowded["sibling"]["argmax_flips"] == 3
    assert crowded["packed"]["max_dp"] > 0.5 and 0 < crowded["packed"]["mean_dp"] < crowded["packed"]["max_dp"]
    # reads only the probe's text: packed stays exact, so the sibling read is scored at the question's index after the probe
    secret = isolation(FakeServed(lambda others: 5.0 * any("CRANE" in o for o in others)), None, raw)
    assert secret["packed"] == exact["packed"] and secret["sibling"]["argmax_flips"] == 3


def test_screen_requires_beating_continuation_control_not_just_parent():
    from scripts.review_calibration_screen import screen_checks
    def result(cov):
        return {"micro": {"coverage_at_5pct_error": cov, "acc": 0.8, "aurc": 0.05}, "sources": {"x": {"acc": 0.8}}}
    rule = {"coverage_delta_vs_ce_control_min": 0.05, "coverage_delta_vs_recalibrated_parent_min": 0.05,
            "accuracy_delta_vs_each_min": -0.01, "aurc_delta_vs_each_max": 0, "per_source_accuracy_delta_min": -0.05}
    checks = screen_checks(result(0.6), {"parent": result(0.5), "ce-control": result(0.59)}, rule)
    assert checks["coverage_vs_parent"] and not checks["coverage_vs_ce-control"]


def test_final_audit_partition_remains_bound_to_registration():
    import json
    from pathlib import Path
    from kev.suite import digest, load_split
    from kev.suite import read_json
    root = Path(__file__).resolve().parents[1]
    protocol = read_json(root / "experiments/calibration-audit-protocol.json")
    suite = root / protocol["data"]["development_suite"]
    assert digest(suite / "test.jsonl") == protocol["data"]["fresh_test_sha256"]
    assert digest(suite / "calibration.jsonl") == protocol["data"]["fresh_threshold_sha256"]
    with pytest.raises(ValueError, match="locked test"):
        load_split(suite, "test")


def test_tempered_replay_records_effective_temperature():
    from scripts.calibration_audit import tempered
    from kev.metrics import fit_temperature
    raw = {"variant": "clean", "source": "fixture", "task": "fixture", "p": [0.9, 0.1],
           "logits": [2.197224577, 0.0], "label": 0, "inference_temperature": 1.0}
    rows = tempered([raw], 2.0)
    assert raw["inference_temperature"] == 1.0 and rows[0]["inference_temperature"] == 2.0
    with pytest.raises(ValueError, match="raw logits"):
        fit_temperature(rows)


def test_selective_metrics_include_confidence_ties():
    from kev.metrics import metrics
    rows = [{"p": [0.99, 0.01], "label": y, "type": "noul"} for y in [0, 1]]
    report = metrics(rows)
    assert report["confident_error_rate"] == .5
    assert report["selective"]["0.5"] == {"coverage": 1.0, "accuracy": .5, "confidence_cutoff": .99}

def test_gate_rejects_confident_transfer_failure():
    from kev.experiment import gate_report
    coverage = {"requested_records": 2, "evaluated_records": 2, "requested_questions": 2,
                "evaluated_questions": 2, "rejected_records": 0, "truncated_records": 0}
    report = {"coverage": coverage, "transfer": {"coverage": coverage,
              "clean": {"confident_error_rate": .5}, "paired_flip": {"pairs": 1, "both_correct_rate": 0}}}
    result = gate_report(report, {"passed": True})
    assert not result["passed"]
    assert not result["checks"]["heldout_pairs_at_least_70pct"]
    assert not result["checks"]["transfer_confident_errors_below_10pct"]

def test_uneven_microbatches_have_equal_record_weight():
    from kev.train import accumulation_records
    x = torch.arange(10, dtype=torch.float32)
    gradients = []
    for batch, accum in ((8, 1), (3, 3), (2, 4)):
        w = torch.tensor(1.0, requires_grad=True)
        for mb, start in enumerate(range(0, len(x), batch)):
            chunk = x[start:start + batch]
            ((w * chunk).sum() / accumulation_records(len(x), batch, accum, mb)).backward()
        gradients.append(w.grad.item())
    assert gradients[0] == gradients[2]
    assert accumulation_records(10, 3, 3, 2) == 9
    assert accumulation_records(10, 3, 3, 3) == 1

def test_v3_training_refuses_heldout_structure():
    from kev.suite import validate_training
    r = {"_meta": {"source": "compositional", "family": "held_and_or"}}
    with pytest.raises(ValueError, match="held-out"):
        validate_training([r], {"trainable_sources": ["compositional"]})

def test_unpinned_base_requires_full_sha_in_trial():
    from kev.experiment import validated_trial
    manifest = {"base_revisions": {"pinned": "a" * 40}, "trainable_sources": []}
    with pytest.raises(ValueError, match="base_revision"):
        validated_trial({"base": "other"}, manifest)
    with pytest.raises(ValueError, match="base_revision"):
        validated_trial({"base": "other", "base_revision": "main"}, manifest)
    assert validated_trial({"base": "other", "base_revision": "b" * 40}, manifest)["base_revision"] == "b" * 40
    with pytest.raises(ValueError, match="conflicts"):
        validated_trial({"base": "pinned", "base_revision": "b" * 40}, manifest)

def test_none_pair_is_minimal_and_relabelled():
    from kev.data import none_pair, materialize
    req = {"state": "The shoes are the wrong size.", "questions": {"reason": {"type": "choice", "instructions": "Why?",
           "criteria": {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"}, "label": "size", "src": "t"}}}
    present, absent = none_pair(req, random.Random(3))
    pk, ak = list(present["questions"]["reason"]["criteria"]), list(absent["questions"]["reason"]["criteria"])
    assert len(pk) == 4 and present["questions"]["reason"]["label"] == "size"
    assert [k for k in pk if k != "size"] == ak                     # same order, true option removed, nothing else moved
    assert absent["questions"]["reason"]["label"] == ak[-1] or absent["questions"]["reason"]["label"] in ak
    assert absent["questions"]["reason"]["label"] not in req["questions"]["reason"]["criteria"]
    materialize(present); materialize(absent)
    assert none_pair({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "label": True, "src": "t"}}}, random.Random(0)) == []

def test_missing_partition_is_fetched_and_verified(tmp_path, monkeypatch):
    import json
    from kev import suite as S
    evals = tmp_path / "evals" / "x" / "decision-x"; evals.mkdir(parents=True)
    payload = b'{"state": "s", "questions": {}, "_meta": {}}\n'
    import hashlib
    S.write_json(evals / "manifest.json", {"files": {"train.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}})
    served = tmp_path / "served.jsonl"; served.write_bytes(payload)
    calls = []
    def fake_download(repo, path, repo_type, revision):
        calls.append((repo, path, repo_type, revision)); return str(served)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert len(S.load_split(evals, "train")) == 1
    assert calls == [(S.SUITES_DATASET, "x/decision-x/train.jsonl", "dataset", S.SUITES_REVISION)]
    # a tampered mirror is rejected by the manifest hash
    (evals / "train.jsonl").unlink(); served.write_bytes(b'{"tampered": 1}\n')
    with pytest.raises(ValueError, match="checksum"):
        S.load_split(evals, "train")

def test_private_suite_fetches_from_its_own_mirror(tmp_path, monkeypatch):
    import hashlib
    from huggingface_hub.errors import RepositoryNotFoundError
    from kev import suite as S
    evals = tmp_path / "evals" / "held" / "docs-x"; evals.mkdir(parents=True)
    payload = b'{"state": "s", "questions": {}, "_meta": {}}\n'
    S.write_json(evals / "manifest.json", {"mirror": {"dataset": S.PRIVATE_DATASET, "revision": "abc123"},
                                           "files": {"test.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}})
    served = tmp_path / "served.jsonl"; served.write_bytes(payload)
    calls = []
    def fake_download(repo, path, repo_type, revision):
        calls.append((repo, path, repo_type, revision)); return str(served)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert len(S.load_split(evals, "test", allow_test=True)) == 1
    assert calls == [(S.PRIVATE_DATASET, "held/docs-x/test.jsonl", "dataset", "abc123")]
    # without access the Hub says "not found"; the loader says why, instead of a bare 404
    (evals / "test.jsonl").unlink()
    import httpx
    def denied(*a, **k): raise RepositoryNotFoundError("404 Client Error", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", denied)
    with pytest.raises(PermissionError, match="kev-private-evals"):
        S.load_split(evals, "test", allow_test=True)

def test_remote_predictor_maps_system_one_answers_and_retries(monkeypatch):
    import io, json
    from kev.predictors import RemotePredictor
    rec = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}, "label": "a", "src": "t"},
                                       "y": {"type": "noul", "instructions": "i", "label": True, "src": "t"}}}
    calls = []
    class Resp:
        def __init__(self, body): self.body = body
        def read(self): return json.dumps(self.body).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def urlopen(req, timeout):
        calls.append(json.loads(req.data))
        if len(calls) == 1: raise OSError("503")
        return Resp({"model": "openjev-x", "answers": {"q": {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}, "y": {"type": "noul", "noul": 0.2}}, "usage": {"input_tokens": 12}})
    p = RemotePredictor("http://example.test/", retries=2); monkeypatch.setattr("urllib.request.urlopen", urlopen); monkeypatch.setattr("time.sleep", lambda s: None)
    out = p(rec)
    assert out["probabilities"] == {"q": {"a": 0.7, "b": 0.3}, "y": {"true": 0.2, "false": 0.8}} and p.served_model == "openjev-x" and len(calls) == 2
    assert calls[0]["model"] == "kev-latest" and "label" not in json.dumps(calls[0])        # labels never leave the machine

def test_concurrent_predictions_keep_record_order_and_sequential_failure_semantics(tmp_path):
    """A predictor with `concurrency` > 1 is scored on a thread pool; the rows, predictions.jsonl order and coverage must be
    what the sequential loop produces, and an exception surfaces at the failing record's position with the same coverage."""
    import threading, time
    from kev.benchmark import evaluate_records
    from kev.model import ContextOverflow
    from kev.suite import read_json
    records = [frozen_request(i) for i in range(6)]
    keys = list(records[0]["questions"]["reason"]["criteria"])

    class Predictor:
        def __init__(self, concurrency, fail=None):
            self.concurrency, self.fail, self.in_flight, self.peak, self.lock = concurrency, fail, 0, 0, threading.Lock()
        def __call__(self, record):
            with self.lock:
                self.in_flight += 1; self.peak = max(self.peak, self.in_flight)
            time.sleep(0.02)
            with self.lock: self.in_flight -= 1
            i = int(record["_meta"]["id"].split("-")[1])
            if i == self.fail: raise ContextOverflow("too long")
            p = 0.5 + 0.05 * i
            return {"probabilities": {"reason": {keys[0]: p, keys[1]: 1 - p, **{k: 0.0 for k in keys[2:]}}}, "latency_ms": float(i)}

    seq = Predictor(1); par = Predictor(4)
    r1, rows1 = evaluate_records(records, seq, tmp_path / "seq"); r2, rows2 = evaluate_records(records, par, tmp_path / "par")
    assert rows1 == rows2 and r1["coverage"] == r2["coverage"] and r1["latency_ms"] == r2["latency_ms"]
    assert (tmp_path / "seq" / "predictions.jsonl").read_bytes() == (tmp_path / "par" / "predictions.jsonl").read_bytes()
    assert seq.peak == 1 and par.peak > 1

    r3, _ = evaluate_records(records, Predictor(4, fail=2), tmp_path / "skip", skip_overlong=True)
    assert r3["coverage"]["evaluated_records"] == 5 and [r["id"] for r in read_json(tmp_path / "skip" / "rejected.json")] == ["item-2"]
    with pytest.raises(ContextOverflow):
        evaluate_records(records, Predictor(4, fail=2), tmp_path / "abort")
    failure = read_json(tmp_path / "abort" / "failure.json")
    assert failure["record_id"] == "item-2" and failure["coverage"]["evaluated_records"] == 2

def test_top_bins_and_confidence_bias():
    from kev.metrics import metrics
    rows = [{"p": [0.99, 0.01], "label": 0, "type": "noul"}, {"p": [0.99, 0.01], "label": 1, "type": "noul"}, {"p": [0.6, 0.4], "label": 0, "type": "noul"}]
    m = metrics(rows)
    assert m["top_bins"]["0.99"] == {"n": 2, "errors": 1, "error_rate": 0.5} and m["top_bins"]["0.9"]["n"] == 2
    assert abs(m["confidence_bias"] - ((0.99 + 0.99 + 0.6) / 3 - 2 / 3)) < 1e-9

def test_anchor_loss_aligns_by_key_and_skips_changed_option_sets():
    from kev.train import anchor_loss
    q = {"keys": ["b", "a"]}
    z = torch.tensor([0.0, 0.0])
    # teacher puts 0.9 on 'a'; student uniform -> KL(teacher||student) > 0 and the same for either key order
    l1 = anchor_loss(z, {"keys": ["a", "b"]}, {"a": 0.9, "b": 0.1}, "cpu"); l2 = anchor_loss(z, q, {"a": 0.9, "b": 0.1}, "cpu")
    assert l1 is not None and abs(l1.item() - l2.item()) < 1e-6 and l1.item() > 0
    assert anchor_loss(z, {"keys": ["a", "b", "none"]}, {"a": 0.9, "b": 0.1}, "cpu") is None      # none-option inserted -> skip
    assert anchor_loss(z, q, None, "cpu") is None
    peaked = torch.tensor([10.0, -10.0])                                                       # student already matches teacher's argmax key 'b'? keys=[b,a]: p(b)=1
    assert anchor_loss(peaked, q, {"b": 1.0, "a": 0.0}, "cpu").item() < 1e-3

def test_anchor_trial_validation():
    from kev.experiment import validated_trial
    m = {"base_revisions": {"m": "x"}, "trainable_sources": ["arc", "boolq"]}
    with pytest.raises(ValueError, match="anchor"):
        validated_trial({"base": "m", "anchor_w": 0.5}, m)
    with pytest.raises(ValueError, match="anchor_sources"):
        validated_trial({"base": "m", "anchor": "runs/anchors/x.json", "anchor_w": 0.5, "anchor_sources": "mmlu"}, m)
    assert validated_trial({"base": "m", "anchor": "runs/anchors/x.json", "anchor_w": 0.5, "anchor_sources": "arc"}, m)["anchor_w"] == 0.5


def test_frozen_suites_load_under_any_locale(tmp_path):
    """Issue #12: frozen partitions contain non-ASCII text and are sha256-checked byte for byte, so they must be read as
    UTF-8 whatever the platform's preferred encoding is (Windows cp936 in the report; an ASCII C locale here), and their
    line endings must survive checkout (.gitattributes pins *.json / *.jsonl to LF)."""
    import os, subprocess, sys
    from kev.suite import digest, write_json, write_jsonl
    suite = tmp_path / "evals" / "x" / "decision-x"; suite.mkdir(parents=True)
    record = {**frozen_request(), "state": "Zwölf Boxkämpfer — ‘quotes’ and é"}
    write_jsonl(suite / "development.jsonl", [record])
    write_json(suite / "manifest.json", {"files": {"development.jsonl": {"sha256": digest(suite / "development.jsonl"), "records": 1}}})
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C", "LANG": "C", "PYTHONIOENCODING": "utf-8"}   # stdio only; open() still defaults to the locale
    code = f"import locale; from kev.suite import load_split; r = load_split({str(suite)!r}, 'development'); print(locale.getpreferredencoding(False), r[0]['state'])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=os.path.dirname(os.path.dirname(__file__)), encoding="utf-8")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith(record["state"]) and "UTF-8" not in out.stdout.split()[0].upper(), out.stdout
    attributes = (pathlib.Path(__file__).resolve().parents[1] / ".gitattributes").read_text(encoding="utf-8")
    assert "*.jsonl text eol=lf" in attributes and "*.json text eol=lf" in attributes


def test_semif_external_rows_convert_to_typed_requests():
    """SemIf's build_*.py rows become Kev requests: WANLI as a 3-way choice in SemIf's per-row option order, TypeSafe rows keep
    their primitive with the reference/published distributions keyed by option id (Kev's option_text adds the `id: ` prefix
    SemIf bakes into descriptions, so it is stripped)."""
    from scripts.freeze_semif_external import convert
    wanli = {"id": "w1", "group_id": "g1", "split": "external_test", "family": "evidence_interpretation", "state": "premise", "question": "Assess: claim",
             "options": [{"id": "insufficient", "description": "insufficient: neither"}, {"id": "supported", "description": "supported: yes"}, {"id": "contradicted", "description": "contradicted: no"}],
             "label": 1, "provenance": {"source": "WANLI", "source_revision": "abc", "rights": "CC-BY-4.0"}}
    rec = convert(wanli, "wanli")
    q = rec["questions"]["decision"]
    assert q == {"type": "choice", "instructions": "Assess: claim", "criteria": {"insufficient": "neither", "supported": "yes", "contradicted": "no"}, "label": "supported", "src": "wanli_nli"}
    assert rec["_meta"]["id"] == "wanli/w1" and rec["_meta"]["group_id"] == "wanli/g1" and rec["_meta"]["variant"] == "clean"
    ts = {"id": "t1", "group_id": "c1", "split": "external_typesafe_selected", "family": "typesafe_customer_service", "state": '{"ticket": "hi"}', "question": "Angry?",
          "primitive": "noul", "options": [{"id": "true", "description": "true: yes"}, {"id": "false", "description": "false: no"}], "label": 1, "target_distribution": [0.1, 0.9],
          "published_models": {"typesafe": {"model": "typesafe:v13", "distribution": [0.2, 0.8]}}, "provenance": {"workflow": "customer_service", "snapshot_sha256": "0" * 64}}
    rec = convert(ts, "typesafe")
    assert rec["state"] == {"ticket": "hi"}
    assert rec["questions"]["decision"] == {"type": "noul", "instructions": "Angry?", "criteria": {"true": "yes", "false": "no"}, "label": False, "src": "typesafe_customer_service"}
    assert rec["_meta"]["target"] == {"true": 0.1, "false": 0.9} and rec["_meta"]["published"]["typesafe"]["p"] == {"true": 0.2, "false": 0.8}
    assert rec["_meta"]["row_sha256"] == record_digest({"state": rec["state"], "questions": rec["questions"]})


def test_typesafe_equal_case_agreement_and_tvd():
    """Rows average within a case, cases average equally; a row the model never answered scores agreement 0 / TVD 1."""
    from scripts.compare_typesafe import case_means, score
    assert score({"true": 0.7, "false": 0.3}, {"true": 1.0, "false": 0.0}) == (1.0, pytest.approx(0.3))
    records = [{"_meta": {"id": f"r{i}", "group_id": g}} for i, g in enumerate(["a", "a", "b"])]
    scores = {"r0": (1.0, 0.0), "r1": (0.0, 0.5)}  # case b unanswered
    out = case_means(scores, records)
    assert out["rows"] == 3 and out["cases"] == 2
    assert out["equal_case_modal_agreement"] == pytest.approx((0.5 + 0.0) / 2)
    assert out["equal_case_total_variation"] == pytest.approx((0.25 + 1.0) / 2)


@pytest.mark.parametrize("suite, rows, tasks", [("wanli-v1", 256, {"wanli_nli"}),
                                                 ("typesafe-v1", 102, {"typesafe_agent_trace_observability", "typesafe_customer_service", "typesafe_invoice_processing", "typesafe_security_incidents"})])
def test_semif_external_suites_are_frozen_as_scored(suite, rows, tasks):
    """The committed selections match SemIf's manifest sizes, are eval-only, carry the unique reference argmax the comparison relies on
    and record the context they were admitted under (TypeSafe documents need the serving context; 13 of 102 exceed even that)."""
    from kev.suite import load_split, read_manifest
    root = pathlib.Path(__file__).resolve().parents[1] / "evals" / "external" / suite
    manifest = read_manifest(root)
    records = load_split(root, "development")
    assert len(records) == rows and manifest["eval_only"] and manifest["holdout_sources"] == [] and set(manifest["tasks"]) == tasks
    assert len({r["_meta"]["id"] for r in records}) == rows and all(r["_meta"]["variant"] == "clean" for r in records)
    assert all(r["_meta"]["row_sha256"] == record_digest({"state": r["state"], "questions": r["questions"]}) for r in records)
    if suite == "typesafe-v1":
        assert len({r["_meta"]["group_id"] for r in records}) == 20
        for r in records:
            target = r["_meta"]["target"]
            top = sorted(target.values(), reverse=True)
            assert top[0] > top[1], r["_meta"]["id"]
            assert set(target) == set(r["questions"]["decision"]["criteria"])


def test_rotation_averaging_cancels_a_position_bias():
    import math
    from kev.api import question_keys
    from kev.predictors import RotationAveraged
    content = {"a": 1.0, "b": 0.0, "c": -1.0}; position = [2.0, 0.0, 0.0]         # the first slot is favoured by +2 logits

    def biased(record):
        out = {"probabilities": {}, "logits": {}, "inference_temperature": 1.0, "latency_ms": 1.0}
        for qid, q in record["questions"].items():
            keys = question_keys(q["type"], q.get("criteria"))
            z = {k: (content.get(k, 0.0) + position[i] if q["type"] == "choice" else float(i)) for i, k in enumerate(keys)}
            s = sum(math.exp(v) for v in z.values())
            out["logits"][qid] = z; out["probabilities"][qid] = {k: math.exp(v) / s for k, v in z.items()}
        return out

    record = {"state": "s", "questions": {"c": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}, "n": {"type": "noul"}}}
    shifted = RotationAveraged.rotated(record, 1)
    assert list(shifted["questions"]["c"]["criteria"]) == ["b", "c", "a"] and shifted["questions"]["n"] == record["questions"]["n"]
    avg = RotationAveraged(biased, 3)
    one, other = avg(record), avg(shifted)
    assert one["rotations"] == 3 and one["latency_ms"] == 3.0
    for k in "abc":                                                               # order no longer matters after a full cycle
        assert one["probabilities"]["c"][k] == pytest.approx(other["probabilities"]["c"][k])
    z = one["logits"]["c"]; assert z["a"] - z["b"] == pytest.approx(1.0) and z["b"] - z["c"] == pytest.approx(1.0)   # the bias is a constant
    assert one["probabilities"]["n"] == pytest.approx(biased(record)["probabilities"]["n"])   # noul untouched
    with pytest.raises(ValueError):
        RotationAveraged(biased, 1)


def _leaderboard_row(study, trial, acc, seed=0, cfg="c1"):
    return {"study": study, "trial": trial, "base": "B", "seed": seed, "config": {"k": cfg, "seed": seed}, "config_sha256": f"{cfg}:{seed}", "legacy": False,
            "gates": {"complete_coverage": True, "isolation_and_packing": True}, "transfer_acc": acc, "transfer_brier": 0.3,
            "dev_acc": 0.8, "suite_sha256": None, "transfer_suite_sha256": None}


def _transfer_rows(correct):
    from kev.api import question_keys
    return [{"id": f"r{i}", "question": "q", "group": f"g{i}", "source": "s", "task": "t", "type": "noul", "variant": "clean",
             "keys": question_keys("noul", None), "label": 1, "p": [0.2, 0.8] if ok else [0.8, 0.2]} for i, ok in enumerate(correct)]


def test_challenger_needs_a_noninferior_paired_interval(tmp_path, monkeypatch):
    from kev import autoresearch
    from kev.suite import write_json
    monkeypatch.setattr(autoresearch, "ROOT", tmp_path)
    n = 400
    champion_ok = [i % 5 != 0 for i in range(n)]                    # 0.80
    tiny_lead = [ok or i == 0 for i, ok in enumerate(champion_ok)]  # +1 question: positive delta, lower bound within the margin
    big_lead = [ok or i % 10 == 0 for i, ok in enumerate(champion_ok)]   # +10 pp
    for trial, correct in (("champ", champion_ok), ("tiny", tiny_lead), ("big", big_lead), ("worse", [False] * 40 + champion_ok[40:])):
        (tmp_path / "runs/s" / trial / "transfer").mkdir(parents=True)
        write_json(tmp_path / "runs/s" / trial / "transfer/rows.json", _transfer_rows(correct))
    rows = [_leaderboard_row("s", t, 0.8, cfg=t) for t in ("champ", "tiny", "big", "worse", "missing")]
    champion = autoresearch.as_incumbent([rows[0]])
    tiny, big, worse, missing = (autoresearch.challenge(champion, r, rows, samples=300) for r in rows[1:])
    assert tiny["aggregation"] == "macro" and tiny["delta"] > 0 and tiny["ci95"][0] >= -0.01 and tiny["accepted"]
    assert big["accepted"] and big["ci95"][0] > 0
    assert not worse["accepted"] and worse["delta"] < 0
    assert not missing["accepted"] and "error" in missing                          # recorded, never raised after a paid round
    after, decision = autoresearch.next_incumbent(champion, rows[2], rows)
    assert decision["accepted"] and after["trials"] == ["s/big"]
    after, decision = autoresearch.next_incumbent(champion, rows[3], rows)
    assert not decision["accepted"] and after is champion


def test_replication_joins_the_champion_instead_of_challenging_it():
    from kev.autoresearch import as_incumbent, next_incumbent
    rows = [_leaderboard_row("s", "a0", 0.80, seed=0, cfg="a"), _leaderboard_row("s", "a1", 0.84, seed=1, cfg="a")]
    after, decision = next_incumbent(as_incumbent(rows[:1]), rows[1], rows)
    assert decision is None and after["seeds"] == [0, 1] and after["transfer_acc"] == pytest.approx(0.82)


def test_incumbent_is_the_ledger_champion_until_challenged():
    from kev.autoresearch import incumbent
    rows = [_leaderboard_row("s", "a", 0.80, cfg="a"), _leaderboard_row("s", "b", 0.81, cfg="b"), _leaderboard_row("t", "a", 0.78, seed=1, cfg="a")]
    assert incumbent(rows[:2], "B", None, None)["trials"] == ["s/b"]                     # seeding: point estimate
    history = [{"base": "B", "incumbent_after": {"trials": ["s/a"]}}]
    assert incumbent(rows, "B", None, None, history)["trials"] == ["s/a", "t/a"]         # sticky champion, plus its later replication
    assert incumbent(rows[:2], "B", None, None, [{"base": "B", "incumbent_after": {"trials": ["gone/x"]}}])["trials"] == ["s/b"]   # stale ledger falls back


def test_max_state_lifts_row_and_packed_limits_together():
    from kev.experiment import validated_trial
    from kev.model import MAX_BRANCH, MAX_PACKED, MAX_STATE, MAX_TRAIN_STATE, SERVE_MAX_BRANCH, training_context
    from kev.suite import CONTEXT
    assert training_context() == {k: v for k, v in CONTEXT.items() if k != "truncate"} == {"max_state": MAX_STATE, "max_branch": MAX_BRANCH, "max_packed": MAX_PACKED}
    long = 12 * MAX_STATE                                                              # the 4.12 delta's state limit
    lifted = training_context(long)
    assert lifted["max_branch"] - MAX_BRANCH == lifted["max_packed"] - MAX_PACKED == long - MAX_STATE
    assert training_context(MAX_TRAIN_STATE)["max_branch"] == SERVE_MAX_BRANCH          # the served row limit still fits a training branch
    with pytest.raises(ValueError):
        training_context(MAX_TRAIN_STATE + 1)
    manifest = {"base_revisions": {"model": "pinned"}}
    assert validated_trial({"base": "model", "max_state": long}, manifest)["max_state"] == long
    assert "max_state" not in validated_trial({"base": "model"}, manifest)          # optional: existing recipe digests are unchanged
    for bad in (MAX_STATE - 1, MAX_TRAIN_STATE + 1, float(long), True):
        with pytest.raises(ValueError, match="max_state"):
            validated_trial({"base": "model", "max_state": bad}, manifest)


def test_none_pair_leaves_soft_target_questions_alone():
    import random
    from kev.data import none_pair
    q = {"type": "choice", "criteria": {"a": None, "b": None, "c": None}, "label": "a", "src": "s"}
    req = {"state": "x", "questions": {"q": q}}
    assert len(none_pair(req, random.Random(0))) == 2
    assert none_pair({"state": "x", "questions": {"q": {**q, "target": {"a": 0.5, "b": 0.5}}}}, random.Random(0)) == []


def test_served_fits_on_raw_rows_and_cluster_resamples_keep_groups_together():
    import numpy as np
    from kev.metrics import TEMPERATURE_FIT, cluster_resamples, fit_temperature, served, served_at, tempered_row
    rng = np.random.default_rng(1)
    rows = []
    for g in range(20):
        for q in range(2):
            z = rng.normal(0, 1, 3); y = int(rng.integers(0, 3)); z[y] += 2.0; p = np.exp(z - z.max()); p /= p.sum()
            rows.append({"id": f"r{g}", "question": f"q{q}", "source": "s" if g % 2 else "t", "group": f"g{g}", "task": "k", "type": "choice",
                         "variant": "clean", "label": y, "logits": z.tolist(), "p": p.tolist(), "inference_temperature": None})
    temperature, out = served(rows, rows)
    raw = [{**r, "inference_temperature": 1.0} for r in rows]
    assert temperature == fit_temperature(raw, **TEMPERATURE_FIT)                      # None = recorded raw
    assert out == [tempered_row(r, temperature) for r in raw] == served_at(rows, temperature)
    assert served(out, rows)[0] == temperature                                         # fitting on served rows restores raw logits first
    assert np.allclose([r["p"] for r in served_at(out, temperature)], [r["p"] for r in out])   # re-serving served rows does not temper twice
    for idx in cluster_resamples(rows, 20, 0):
        drawn = [rows[i]["group"] for i in idx]
        assert all(drawn.count(g) % 2 == 0 for g in set(drawn))                        # both questions of a record move together
        assert sum(rows[i]["source"] == "s" for i in idx) == 20                        # stratified: each source keeps its size


def test_jsonl_round_trips_unicode_line_separators(tmp_path):
    from kev.suite import read_jsonl, write_jsonl
    recs = [{"state": "line one\u2028line two"}, {"state": "next\x85record\u2029end"}, {"state": "plain"}]
    write_jsonl(tmp_path / "x.jsonl", recs)
    assert read_jsonl(tmp_path / "x.jsonl") == recs
