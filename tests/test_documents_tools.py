"""The documents-v1 / documents-v2 suite tooling (scripts/{build,label,freeze}_documents_v*.py): answer parsing, the
two-adjudicator agreement rule, the freeze gates and documents-v2's exclusion of every documents-v1 candidate. No network,
no API, no frozen data: every input is synthetic."""
import io
import json
import sys
import urllib.error

import pytest

from kev.suite import read_jsonl, write_json, write_jsonl
from scripts import build_documents_v1 as v1
from scripts import build_documents_v2 as v2
from scripts import freeze_documents_v1 as fz
from scripts import label_documents_v1 as lab
from scripts.label_documents_v1 import parse

REC = {"state": "My card was charged twice.", "questions": {
    "product": {"instructions": "Which product?", "criteria": {"card": "A credit card", "mortgage": "A mortgage"}, "label": "card"},
    "issue": {"instructions": "Which issue?", "criteria": {"fees_or_interest": "Fees or interest", "closing_the_account": "Closing your account"}, "label": "fees_or_interest"}}}


def test_parse_reads_the_json_object_inside_prose_and_rejects_labels_outside_the_criteria():
    clean = parse('{"product": {"label": "card", "reason": "a card"}, "issue": {"label": "fees_or_interest", "reason": "double charge"}}', REC)
    assert clean == {"product": {"label": "card", "reason": "a card"}, "issue": {"label": "fees_or_interest", "reason": "double charge"}}
    wrapped = parse('Sure. Here is my answer:\n```json\n{"product": {"label": "mortgage", "reason": "r"}, "issue": "closing_the_account"}\n```\nThanks.', REC)
    assert wrapped == {"product": {"label": "mortgage", "reason": "r"}, "issue": {"label": "closing_the_account", "reason": ""}}
    invented = parse('{"product": {"label": "student_loan", "reason": "r"}}', REC)
    assert invented == {"product": {"label": None, "reason": "r"}, "issue": {"label": None, "reason": ""}}   # unknown key, missing question
    assert parse("I cannot decide.", REC) is None and parse("{not json}", REC) is None and parse(None, REC) is None


def test_parse_treats_valid_json_of_the_wrong_shape_as_unlabelled_instead_of_raising():
    # unhashable, numeric and null labels and a non-string reason: each would have raised TypeError and aborted the run
    shapes = ['{"product": {"label": ["card"], "reason": 3}, "issue": {"label": {"k": 1}}}', '{"product": 7, "issue": null}',
              '{"product": {"label": null, "reason": ["why"]}, "issue": ["fees_or_interest"]}']
    for text in shapes:
        assert parse(text, REC) == {"product": {"label": None, "reason": ""}, "issue": {"label": None, "reason": ""}}, text


def verdict(item, v, label=None, reason="r"):
    return json.dumps({"id": item, "verdict": v, "label": label, "reason": reason})


def write_lines(path, lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_combine_decides_an_item_only_when_both_adjudications_agree(tmp_path):
    items = ["a#product", "b#product", "c#issue", "d#issue", "e#issue", "f#product"]
    write_jsonl(tmp_path / "adjudication_queue.jsonl", [{"id": i} for i in items])
    for name, lines in {"out": [verdict("a#product", "accept"), verdict("b#product", "relabel", "card"), verdict("c#issue", "relabel", "x"),
                                verdict("d#issue", "accept"), verdict("e#issue", "drop"), verdict("f#product", "accept")],
                        "out2": [verdict("a#product", "accept"), verdict("b#product", "relabel", "card"), verdict("c#issue", "relabel", "y"),
                                 verdict("d#issue", "drop"), verdict("e#issue", "drop")]}.items():   # f has no second verdict
        (tmp_path / "adjudication" / name).mkdir(parents=True)
        write_lines(tmp_path / "adjudication" / name / "shard-00.jsonl", lines)
    counts = fz.combine(tmp_path)
    rows = {r["id"]: r for r in read_jsonl(tmp_path / "adjudications.jsonl")}
    assert counts == {"agreed_accept": 1, "agreed_relabel": 1, "agreed_drop": 1, "disagreed_dropped": 2, "missing_dropped": 1}
    assert (rows["a#product"]["verdict"], rows["a#product"]["label"], rows["a#product"]["agreed"]) == ("accept", None, True)
    assert (rows["b#product"]["verdict"], rows["b#product"]["label"]) == ("relabel", "card")
    assert (rows["c#issue"]["verdict"], rows["c#issue"]["agreed"]) == ("drop", False)   # relabel to different labels is a disagreement
    assert rows["d#issue"]["verdicts"] == ["accept", "drop"] and rows["e#issue"]["agreed"] is True
    assert rows["f#product"]["verdicts"] == ["accept", None] and rows["f#product"]["reasons"][1] == "missing"


def test_read_verdicts_skips_prose_broken_json_and_unknown_verdicts(tmp_path):
    write_lines(tmp_path / "shard-00.jsonl", ["Here are my verdicts:", verdict("a#product", "accept"), '{"id": "b#product", "verdict": "acc', "",
                                             verdict("c#issue", "maybe"), "   " + verdict("d#issue", "relabel", "card"), "```"])
    write_lines(tmp_path / "shard-01.jsonl", [verdict("a#product", "drop")])   # a later shard wins
    write_lines(tmp_path / "notes.jsonl", [verdict("e#issue", "accept")])      # not a shard
    out = fz.read_verdicts(tmp_path)
    assert {k: v["verdict"] for k, v in out.items()} == {"a#product": "drop", "d#issue": "relabel"}


def work_dir(root, n_test=60, reviewer_rejects=0):
    """A tiny documents work dir with unanimous judges on test (no adjudication needed) and spot-check reviews that
    accept all but `reviewer_rejects` of the sample."""
    records = [{"state": f"complaint {i}", "questions": {"product": {"type": "choice", "instructions": "Which product?",
                "criteria": {"card": "A card", "mortgage": "A mortgage"}, "label": "card", "src": "cfpb_product"}},
                "_meta": {"id": f"cfpb/{i}", "source": "cfpb", "length_bucket": "short", "chars": 11}} for i in range(n_test)]
    (root / "candidates").mkdir(parents=True)
    write_jsonl(root / "candidates" / "test.jsonl", records)
    write_json(root / "candidates" / "build.json", {"repo": "r", "revision": "v"})
    (root / "labels" / "test").mkdir(parents=True)
    for m in fz.JUDGES:
        write_jsonl(root / "labels" / "test" / (m.replace("/", "__") + ".jsonl"), [{"id": r["_meta"]["id"], "answers": {"product": {"label": "card"}}} for r in records])
    sample = fz.spot_check_sample(records)
    write_jsonl(root / "spot_check_reviews.jsonl", [{"id": s["id"], "proposed_label": s["proposed_label"], "label": "mortgage" if i < reviewer_rejects else s["proposed_label"],
                                                     "verdict": "relabel" if i < reviewer_rejects else "accept"} for i, s in enumerate(sample)])
    return records


def freeze(work, out, min_agreement):
    test, _, report = fz.eval_split(work, "test", {})
    fz.freeze(work, out, {"test": test}, {"test": report}, version="documents-t", min_agreement=min_agreement, private=False)


def test_questions_awaiting_adjudication_block_the_freeze(tmp_path):
    work, out = tmp_path / "work", tmp_path / "evals" / "documents-t"
    work_dir(work)
    write_jsonl(work / "labels" / "test" / (fz.JUDGES[0].replace("/", "__") + ".jsonl"), [{"id": "cfpb/0", "answers": {"product": {"label": "mortgage"}}}])
    with pytest.raises(SystemExit, match="adjudications missing"):   # judge 0 disagreed on cfpb/0 and missed the rest
        freeze(work, out, 47)
    assert not out.parent.exists()


def test_a_suite_without_test_candidates_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="no test candidates"):
        fz.freeze(tmp_path, tmp_path / "evals" / "documents-t", {}, {}, version="documents-t", min_agreement=47, private=False)


def test_a_failed_private_upload_removes_the_partial_suite(tmp_path, monkeypatch):
    work, out = tmp_path / "work", tmp_path / "evals" / "documents-t"
    work_dir(work)
    def upload(out, names):
        assert (out / "test.jsonl").exists()   # the partitions were written; then the upload fails
        raise ConnectionError("hub unavailable")
    monkeypatch.setattr(fz, "upload_private", upload)
    test, _, report = fz.eval_split(work, "test", {})
    with pytest.raises(ConnectionError):
        fz.freeze(work, out, {"test": test}, {"test": report}, version="documents-t", min_agreement=47, private=True)
    assert not out.exists()


def test_a_failed_spot_check_freezes_nothing(tmp_path):
    work, out = tmp_path / "work", tmp_path / "evals" / "documents-t"
    work_dir(work, reviewer_rejects=4)
    before = sorted(p.relative_to(work) for p in work.rglob("*"))
    with pytest.raises(SystemExit, match="46/50 is below the registered 47/50"):
        freeze(work, out, 47)
    assert not out.exists() and not out.parent.exists()
    assert sorted(p.relative_to(work) for p in work.rglob("*")) == before   # --freeze never writes into the work dir


def test_spot_check_reviews_must_match_the_sample_drawn_from_the_frozen_test(tmp_path):
    work, out = tmp_path / "work", tmp_path / "evals" / "documents-t"
    work_dir(work)
    reviews = read_jsonl(work / "spot_check_reviews.jsonl")
    write_jsonl(work / "spot_check_reviews.jsonl", reviews[:-1])
    with pytest.raises(SystemExit, match="do not match the sample"):
        freeze(work, out, 47)
    assert not out.parent.exists()


def test_a_passing_freeze_writes_the_partitions_and_a_held_out_only_manifest(tmp_path):
    work, out = tmp_path / "work", tmp_path / "evals" / "documents-t"
    records = work_dir(work, reviewer_rejects=3)
    freeze(work, out, 47)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert read_jsonl(out / "test.jsonl") == records and manifest["files"]["test.jsonl"]["records"] == len(records)
    assert manifest["spot_check"]["agreement"] == "47/50" and len(manifest["spot_check"]["disagreements"]) == 3
    assert manifest["partitions"] == ["test"] and manifest["trainable_sources"] == []   # no train split: nothing trainable


def test_an_adjudicated_label_outside_the_options_is_an_error_not_an_assert(tmp_path):
    work = tmp_path / "work"
    work_dir(work)
    write_jsonl(work / "labels" / "test" / (fz.JUDGES[0].replace("/", "__") + ".jsonl"), [{"id": "cfpb/0", "answers": {"product": {"label": "mortgage"}}}])
    with pytest.raises(ValueError, match="not an option"):
        fz.eval_split(work, "test", {"cfpb/0#product": {"verdict": "relabel", "label": "student_loan"}})


def test_private_freeze_outside_evals_fails_before_writing(tmp_path):
    with pytest.raises(SystemExit, match="not under an evals/ directory"):
        fz.evals_relative(tmp_path / "suites" / "documents-t")
    assert fz.evals_relative(tmp_path / "evals" / "held" / "documents-t").as_posix() == "held/documents-t"


def source_row(cid, text, product="Credit card", issue="Fees or interest"):
    return {"complaint_id": cid, "product": product, "sub_product": None, "issue": issue, "sub_issue": None,
            "complaint_what_happened": text, "company": "Bank", "date_received": "2020-01-01"}


def test_documents_v2_excludes_every_v1_candidate_by_text_and_by_id():
    body = "The bank charged a fee I never agreed to. " * 10   # 430 characters: the short bucket
    v1_row = v1.prepare(source_row(1, body))
    v1_records = [v1.record(v1_row, v1.RIGHTS)]
    rows = [source_row(2, "  " + body.upper() + "  "),         # same text up to case and whitespace: excluded by hash
            source_row(1, body + " Different ending."),        # new text, but v1's complaint id: excluded by id
            source_row(3, body + " Fresh."), source_row(4, body + " Fresh."),   # a duplicate within v2 counts once
            source_row(5, "too short"), source_row(6, body + " Other.", product="Not a product")]
    recs, stats = v2.select(rows, v1_records, "documents-v2")
    assert [r["_meta"]["id"] for r in recs] == ["cfpb/3"] and stats == {"excluded_or_duplicate": 3}
    assert recs[0]["questions"].keys() == {"product", "issue"} and recs[0]["questions"]["issue"]["label"] == "fees_or_interest"
    assert recs[0]["_meta"]["provenance"]["rights"] == v2.RIGHTS and v2.PER_CELL == v1.SPLITS["test"]


def test_documents_v2_takes_at_most_the_v1_test_count_per_cell():
    rows = [source_row(i, f"Complaint number {i}: " + "the fee was wrong. " * 20) for i in range(v2.PER_CELL + 5)]
    recs, _ = v2.select(rows, [], "documents-v2")
    assert len(recs) == v2.PER_CELL and len({r["_meta"]["id"] for r in recs}) == v2.PER_CELL
    assert recs == v2.select(rows, [], "documents-v2")[0]   # seeded: the same rows give the same draw


def test_questions_add_the_issue_only_when_the_raw_issue_is_canonical():
    assert list(v1.questions("card", "fees_or_interest")) == ["product", "issue"]
    assert list(v1.questions("card", None)) == ["product"]
    assert v1.questions("card", None)["product"]["criteria"].keys() == v1.PRODUCTS.keys()



def label_work(root, n):
    (root / "candidates").mkdir(parents=True)
    write_jsonl(root / "candidates" / "test.jsonl", [{**REC, "_meta": {"id": f"cfpb/{i}"}} for i in range(n)])


def run_labeller(monkeypatch, root, fake_call, *extra):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test")
    monkeypatch.setattr(lab, "call", fake_call)
    monkeypatch.setattr(sys, "argv", ["label", "--split", "test", "--models", "m/one", "--workers", "1", "--work", str(root), *extra])
    lab.main()


def test_the_spend_ledger_is_on_disk_after_every_labelled_result(tmp_path, monkeypatch):
    label_work(tmp_path, 5)
    seen = []   # the ledger total on disk when each call starts
    def fake_call(model, rec, key):
        path = tmp_path / "labels" / "spend.json"
        seen.append(json.loads(path.read_text(encoding="utf-8"))["total"] if path.exists() else 0.0)
        if len(seen) == 4: raise RuntimeError("the process dies here")
        return {"answers": None, "cost": 0.25, "raw": ""}
    with pytest.raises(RuntimeError):
        run_labeller(monkeypatch, tmp_path, fake_call)
    assert seen == [0.0, 0.25, 0.5, 0.75]   # one worker: each result is saved before the next call starts
    ledger = json.loads((tmp_path / "labels" / "spend.json").read_text(encoding="utf-8"))
    assert ledger["total"] == 0.75 and len(read_jsonl(tmp_path / "labels" / "test" / "m__one.jsonl")) == 3
    assert len(seen) == 4   # the crash stopped the queue: the fifth document was never sent
    assert sorted(p.name for p in (tmp_path / "labels").iterdir()) == ["spend.json", "test"]   # written atomically: no stray .tmp


def test_responses_without_a_cost_are_unknown_spend_and_stop_the_run(tmp_path, monkeypatch, capsys):
    label_work(tmp_path, 8)
    run_labeller(monkeypatch, tmp_path, lambda model, rec, key: {"answers": None, "cost": None, "raw": ""}, "--max-uncosted", "2")
    ledger = json.loads((tmp_path / "labels" / "spend.json").read_text(encoding="utf-8"))
    assert ledger["uncosted"] == 3 and ledger["total"] == 0.0   # one worker: nothing else was in flight when it stopped
    assert "3 uncosted" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="carried no usage.cost"):   # and a rerun refuses to start
        run_labeller(monkeypatch, tmp_path, lambda model, rec, key: {"answers": None, "cost": 0.1, "raw": ""}, "--max-uncosted", "2")


class Response(io.BytesIO):   # what urllib.request.urlopen returns: a readable context manager
    pass


def test_a_failed_call_is_not_cached_and_the_next_run_retries_it(tmp_path, monkeypatch):
    """The transport fails with a non-retryable status on the first run and answers on the second; the second run sends
    the document again and the answer it records is the one the freeze reads."""
    label_work(tmp_path, 1)
    sent = []
    def urlopen(req, timeout):
        sent.append(json.loads(req.data)["model"])
        if len(sent) == 1:
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b"model overloaded"))
        body = {"choices": [{"message": {"content": '{"product": {"label": "card", "reason": "a card"}, "issue": {"label": "fees_or_interest", "reason": "fee"}}'}}],
                "usage": {"cost": 0.01}}
        return Response(json.dumps(body).encode())
    monkeypatch.setattr(lab.urllib.request, "urlopen", urlopen)
    argv = ["label", "--split", "test", "--models", "m/one", "--workers", "1", "--work", str(tmp_path)]
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test")
    monkeypatch.setattr(sys, "argv", argv)
    lab.main()
    path = tmp_path / "labels" / "test" / "m__one.jsonl"
    assert read_jsonl(path)[0]["error"].startswith("HTTP 400") and lab.answered(path) == set()
    lab.main()   # the rerun retries the failed document
    rows = read_jsonl(path)
    assert sent == ["m/one", "m/one"] and len(rows) == 2 and lab.answered(path) == {"cfpb/0"}
    assert rows[1]["answers"]["product"]["label"] == "card" and rows[1]["cost"] == 0.01
    assert fz.answers(tmp_path, "test", ["m/one"])["m/one"]["cfpb/0"] == rows[1]   # the freeze reads the answer, not the failure
    lab.main()   # a third run has nothing left to send
    assert len(sent) == 2
