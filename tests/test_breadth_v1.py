"""The breadth-v1 builder's mappings (scripts/build_breadth_v1.py) and the Decision-Index-style scorer (scripts/breadth_report.py)
on small synthetic inputs, plus the frozen suite's structure. No weights, no network.
Run: uv run python -m pytest tests/test_breadth_v1.py -q
"""
import random
import struct
from pathlib import Path

import pytest

from kev.benchmark import labels
from kev.data import materialize
from kev.suite import PRIVATE_DATASET, load_split, read_json
from scripts import breadth_report as br
from scripts.build_breadth_v1 import (AREAS, BM25, DATASETS, MAX_OPTIONS, OOS, SGD_NONE, apibank_parse, ascii_board, bag_records, bfcl_candidate, cfcolor_record,
                                      chance, check_invariants, chess_options, clinc_options, contractnli_questions, decode_action_value, describe_move,
                                      fen_board, hellaswag_candidate, humicroedit_pair, musr_candidate, palette_hex, retrieval_choice, router_options,
                                      sata_candidate, sgd_candidate, sgd_turns, sgd_variant_services, tool_summary)

SUITE = Path(__file__).resolve().parents[1] / "evals" / "breadth-v1"
START = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def test_every_dataset_is_in_one_area_and_maps_to_valid_requests():
    assert set(AREAS) == {s["area"] for s in DATASETS.values()}
    assert all(s["metric"] in ("accuracy", "case_exact") for s in DATASETS.values())


def test_fen_board_and_ascii():
    board = fen_board(START)
    assert board["e4"] == "P" and board["e8"] == "k" and "e2" not in board and len(board) == 32
    rows = ascii_board(START).split("\n")
    assert rows[0] == "8 r n b q k b n r" and rows[4] == "4 . . . . P . . ." and rows[-1].strip() == "a b c d e f g h"


def test_describe_move():
    assert describe_move(START, "e7e5") == "black pawn e7-e5"
    fen = "r3k2r/8/8/3p4/4P3/8/8/R3K2R w KQkq - 0 1"
    assert describe_move(fen, "e4d5") == "white pawn e4-d5 capturing the pawn"
    assert describe_move(fen, "e1g1") == "white king e1-g1 (castling)"
    assert describe_move("8/4P3/8/8/8/8/8/k6K w - - 0 1", "e7e8q") == "white pawn e7-e8, promoting to queen"


def test_chess_options_keep_best_drop_near_ties_and_cap():
    moves = {f"a{i}b{i}": 0.1 + 0.01 * i for i in range(1, 9)} | {"e2e4": 0.9, "d2d4": 0.895, "g1f3": 0.5}
    moves |= {f"h{i}h{i + 1}": 0.2 for i in range(1, 8)}
    keys, best = chess_options(moves, random.Random(0))
    assert best == "e2e4" and "e2e4" in keys and "d2d4" not in keys          # within 0.01 of the best: left out
    assert len(keys) == MAX_OPTIONS and len(set(keys)) == len(keys)
    assert chess_options(moves, random.Random(0)) == (keys, best)            # deterministic
    assert chess_options({"e2e4": 0.5, "d2d4": 0.495}, random.Random(0)) is None


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n: return bytes(out)


def test_action_value_decoding_and_bag_reader(tmp_path):
    recs = []
    for fen, move, p in [(START, "e7e5", 0.51), ("8/8/8/8/8/8/8/k6K w - - 0 1", "h1g1", 0.5)]:
        recs.append(_varint(len(fen)) + fen.encode() + _varint(len(move)) + move.encode() + struct.pack(">d", p))
    ends, pos = [], 0
    for r in recs: pos += len(r); ends.append(pos)
    body = b"".join(recs)
    (tmp_path / "x.bag").write_bytes(body + struct.pack(f"<{len(ends)}q", *ends))     # the last end offset doubles as the index start
    out = [decode_action_value(r) for r in bag_records(tmp_path / "x.bag")]
    assert out == [(START, "e7e5", 0.51), ("8/8/8/8/8/8/8/k6K w - - 0 1", "h1g1", 0.5)]


DOMAINS = {"banking": [f"bank_{i}" for i in range(15)], "travel": [f"travel_{i}" for i in range(15)]}


def test_clinc_options_in_scope_and_out_of_scope():
    criteria, label = clinc_options("bank_3", DOMAINS, random.Random(1))
    keys = list(criteria)
    assert label == "bank_3" and len(keys) == 10 and keys[-1] == OOS
    assert all(k.startswith("bank_") for k in keys[:-1])                    # distractors from the gold's own domain
    criteria, label = clinc_options("oos", DOMAINS, random.Random(2))
    keys = list(criteria)
    assert label == OOS and len(keys) == 10 and len({k.split("_")[0] for k in keys[:-1]}) == 1


def test_contractnli_questions_round_robin_over_labels():
    ann = {"nda-1": {"choice": "Entailment"}, "nda-2": {"choice": "Entailment"}, "nda-3": {"choice": "NotMentioned"},
           "nda-4": {"choice": "Contradiction"}, "nda-5": {"choice": "Entailment"}, "nda-7": {"choice": "NotMentioned"}}
    hyps = {k: f"hypothesis {k}" for k in ann}
    qs = contractnli_questions(ann, hyps, random.Random(0), k=4)
    got = sorted(q["label"] for q in qs.values())
    assert got == ["contradiction", "entailment", "entailment", "not_mentioned"]     # sorted labels visited in turn; contradiction runs out
    assert all(set(q["criteria"]) == {"entailment", "contradiction", "not_mentioned"} for q in qs.values())
    assert "hypothesis nda-4" in qs["nda-4"]["instructions"]


def _dialogue():
    turn = lambda speaker, text, frames=(): {"speaker": speaker, "utterance": text, "frames": list(frames)}
    frame = lambda intent: {"service": "Alarm_1", "state": {"active_intent": intent}}
    return {"dialogue_id": "1_00001", "turns": [turn("USER", "hi", [frame("NONE")]), turn("SYSTEM", "hello"),
                                               turn("USER", "show my alarms", [frame("GetAlarms")]), turn("SYSTEM", "you have one"),
                                               turn("USER", "and the same again", [frame("GetAlarms")]), turn("SYSTEM", "ok"),
                                               turn("USER", "add one at 7", [frame("AddAlarm")])]}


def test_sgd_turns_strata_and_history():
    turns = sgd_turns(_dialogue())
    assert [t[3] for t in turns] == ["none", "changed", "same", "changed"]
    assert turns[1][4] == ["User: hi", "System: hello", "User: show my alarms"]


SCHEMA = [{"service_name": "Alarm_1", "description": "Manage alarms", "intents": [{"name": "GetAlarms", "description": "Get alarms"}, {"name": "AddAlarm", "description": "Set an alarm"}]}]
VARIANT = [{"service_name": "Alarm_11", "description": "Alarm manager", "intents": [{"name": "SeeAlarms", "description": "Show alarms"}, {"name": "SetAlarm", "description": "New alarm"}]}]


def test_sgd_variant_mapping_and_labels():
    schemas = {v: sgd_variant_services(SCHEMA, VARIANT) for v in ("v1", "v2", "v3", "v4", "v5")}
    schemas["original"] = {s["service_name"]: s for s in SCHEMA}
    turn = sgd_turns(_dialogue())[1]
    x = sgd_candidate("test", _dialogue(), turn, schemas)
    q = x["questions"]["intent"]
    assert SGD_NONE in q["criteria"] and len(q["criteria"]) == 3
    assert q["label"] in ("GetAlarms", "SeeAlarms") and (q["label"] == "GetAlarms") == (x["_meta"]["schema_variant"] == "original")
    with pytest.raises(ValueError):
        sgd_variant_services(SCHEMA, [{"service_name": "Bus_1", "intents": SCHEMA[0]["intents"]}])


def _fn(name):
    return {"name": name, "description": f"does {name}", "parameters": {"type": "dict", "properties": {}}}


def test_bfcl_nouls_follow_the_ground_truth():
    item = {"id": "multiple_1", "question": [[{"role": "user", "content": "area of a triangle"}]], "function": [_fn("tri.area"), _fn("circle.area")]}
    x = bfcl_candidate("multiple", item, [{"tri.area": {"base": [1]}}])
    assert [q["label"] for q in x["questions"].values()] == [True, False] and "tri.area" in x["questions"]["call_1"]["instructions"]
    irrelevant = bfcl_candidate("irrelevance", item, None)
    assert not any(q["label"] for q in irrelevant["questions"].values())
    assert bfcl_candidate("multiple", item, [{"other.fn": {}}]) is None
    assert bfcl_candidate("multiple", {**item, "function": [_fn(f"f{i}") for i in range(11)]}, []) is None


def test_sata_candidate_labels_and_drops():
    row = {"question": "<br>Multi Label Question: Who went?", "paragraph": "Paragraph: <b>Sent 1: </b>A and B went.", "answer groups": "['A', 'B']", "distractor groups": "['C']"}
    x = sata_candidate(3, row, random.Random(0))
    got = {q["instructions"]: q["label"] for q in x["questions"].values()}
    assert got == {'Is "A" a correct answer to the question?': True, 'Is "B" a correct answer to the question?': True, 'Is "C" a correct answer to the question?': False}
    assert x["state"] == {"passage": "Sent 1: A and B went.", "question": "Who went?"}
    assert sata_candidate(3, {**row, "distractor groups": "['a']", "answer groups": "['A']"}, random.Random(0)) is None       # answer == distractor
    assert sata_candidate(3, {**row, "distractor groups": str([f"d{i}" for i in range(9)])}, random.Random(0)) is None       # 11 candidates


def test_router_options_have_one_correct_model():
    scores = {f"m{i}": int(i == 4) for i in range(11)}
    keys, label = router_options(scores, random.Random(0))
    assert label == "m4" and len(keys) == MAX_OPTIONS and sum(scores[k] for k in keys) == 1
    assert router_options({"a": 1, "b": 1}, random.Random(0)) is None and router_options({"a": 0, "b": 0}, random.Random(0)) is None


def test_humicroedit_pair():
    row = {"original1": "Trump <meets/> Putin", "edit1": "hugs", "original2": "Trump meets <Putin/>", "edit2": "cat", "label": "2"}
    assert humicroedit_pair(row) == ("Trump meets Putin", "Trump hugs Putin", "Trump meets cat", "b")
    assert humicroedit_pair({**row, "label": "0"}) is None


def test_cfcolor_record_pairs_different_ratings_and_hides_them_from_history():
    rgb = {t: [t / 100] * 15 for t in range(1, 40)}
    train = [(t, 3) for t in range(1, 20)]
    test = [(20, 2), (21, 2), (22, 5)]
    state, q, meta = cfcolor_record(7, train, test, rgb, random.Random(0))
    assert meta["ratings"][0] != meta["ratings"][1] and 22 in meta["themes"]
    assert q["label"] == ("a" if meta["ratings"][0] > meta["ratings"][1] else "b")
    assert len(state["earlier_ratings"]) == 12
    assert cfcolor_record(7, train, [(20, 2), (21, 2)], rgb, random.Random(0)) is None
    assert palette_hex([1.0, 0.0, 0.5] * 5).split()[0] == "#ff0080"


def test_apibank_parse_and_tool_summary():
    row = {"instruction": 'Generate...\nAPI descriptions:\n{"name": "QueryStock", "description": "Stock price"}\n{"name": "Wiki", "description": "Search"}',
           "input": "User: price of NVDA?\nGenerate API Request:\n", "expected_output": "API-Request: [QueryStock(stock_code='NVDA')]"}
    assert apibank_parse(row) == ("User: price of NVDA?", {"QueryStock": "Stock price", "Wiki": "Search"}, "QueryStock")
    assert tool_summary('{"name": "weather", "description": "' + "x " * 400 + '"}')[1].endswith("…")
    assert tool_summary('{"api_call": "hub.load(1)", "functionality": "Detect"}') == ("hub.load(1)", "Detect")


def test_retrieval_choice_marks_the_gold():
    q = retrieval_choice("toolret", "Which?", "gold text", ["n1", "n2", "n3", "n4"], random.Random(0), "toolret", "tool")
    assert q["criteria"][q["label"]] == "gold text" and list(q["criteria"]) == [f"tool_{i}" for i in range(1, 6)]


def test_musr_and_hellaswag_candidates_materialize():
    m = musr_candidate("murder_mystery", 0, {"narrative": "A story.", "question": "Who?", "choices": "['Ann', 'Bob']", "answer_index": "1"})
    assert m["questions"]["answer"]["label"] == "opt_2" and m["questions"]["answer"]["criteria"]["opt_2"] == "Bob"
    hs = hellaswag_candidate(0, {"ind": 5, "activity_label": "Roofing", "ctx": "A man is on a roof. he", "endings": ["a", "b", "c", "d"], "label": "3",
                                 "source_id": "activitynet~v_x", "split_type": "indomain"})
    assert hs["questions"]["ending"]["label"] == "opt_4"
    for x in (m, hs): materialize(x)


def test_bm25_ranks_by_term_overlap_with_id_ties():
    bm = BM25(["apple banana", "banana cherry", "cherry durian", "apple banana"])
    assert bm.top("apple", 2) == [0, 3]
    assert bm.top("cherry durian", 1) == [2] and bm.top("zzz", 3) == []
    assert bm.top("banana", 4, skip=lambda i: i == 0) == [1, 3]


def test_chance():
    r = {"questions": {"a": {"type": "noul"}, "b": {"type": "choice", "criteria": {"x": None, "y": None, "z": None, "w": None}}}}
    assert chance(r, "accuracy") == [0.5, 0.25] and chance(r, "case_exact") == [0.125]

# ---------------------------------------------------------------- scoring helper


def _record(rid, source, qs):
    return {"_meta": {"id": rid, "source": source}, "questions": qs}


def _row(rid, qid, q, p):
    keys, y = labels(q)
    return {"id": rid, "question": qid, "variant": "clean", "label": y, "p": p, "keys": keys, "type": q["type"]}


def test_skill_formula():
    assert br.skill(0.25, 0.25) == 0 and br.skill(1.0, 0.25) == 1 and br.skill(0.1, 0.25) == 0
    assert br.skill(0.625, 0.25) == pytest.approx(0.5)


def test_dataset_score_counts_unanswered_as_wrong():
    q = {"type": "choice", "criteria": {"a": None, "b": None, "c": None, "d": None}, "label": "a"}
    recs = [_record(f"r{i}", "d", {"q": q}) for i in range(4)]
    rows = {("r0", "q"): _row("r0", "q", q, [0.7, 0.1, 0.1, 0.1]), ("r1", "q"): _row("r1", "q", q, [0.1, 0.7, 0.1, 0.1]),
            ("r2", "q"): _row("r2", "q", q, [0.9, 0.05, 0.03, 0.02])}
    s = br.dataset_score(recs, rows, "accuracy")
    assert s["score"] == 0.5 and s["answered"] == 0.75 and s["accuracy_answered"] == pytest.approx(2 / 3)
    assert s["chance"] == 0.25 and s["skill"] == pytest.approx((0.5 - 0.25) / 0.75)


def test_dataset_score_case_exact():
    n = {"type": "noul", "label": True}
    recs = [_record("r0", "d", {"a": n, "b": {**n, "label": False}}), _record("r1", "d", {"a": n, "b": n})]
    rows = {("r0", "a"): _row("r0", "a", n, [0.2, 0.8]), ("r0", "b"): _row("r0", "b", {**n, "label": False}, [0.6, 0.4]),
            ("r1", "a"): _row("r1", "a", n, [0.2, 0.8]), ("r1", "b"): _row("r1", "b", n, [0.9, 0.1])}
    s = br.dataset_score(recs, rows, "case_exact")
    assert s["units"] == 2 and s["score"] == 0.5 and s["chance"] == 0.25
    with pytest.raises(ValueError):
        br.dataset_score(recs, {("r0", "a"): {**rows[("r0", "a")], "label": 0}}, "case_exact")


def test_score_system_areas_and_index():
    manifest = {"datasets": {"x": {"area": "one", "metric": "accuracy"}, "y": {"area": "one", "metric": "accuracy"}, "z": {"area": "two", "metric": "accuracy"}},
                "areas": {"one": {"label": "One", "datasets": ["x", "y"]}, "two": {"label": "Two", "datasets": ["z"]}}}
    n = {"type": "noul", "label": True}
    recs = [_record("x0", "x", {"q": n}), _record("y0", "y", {"q": n}), _record("z0", "z", {"q": n})]
    rows = [_row("x0", "q", n, [0.1, 0.9]), _row("y0", "q", n, [0.9, 0.1]), _row("z0", "q", n, [0.4, 0.6])]
    out = br.score_system(recs, rows, manifest)
    assert out["areas"]["one"]["skill"] == 0.5 and out["areas"]["two"]["skill"] == 1.0
    assert out["overall"]["index"] == pytest.approx(75.0) and out["overall"]["raw_index"] == pytest.approx(75.0)
    assert out["datasets"]["y"]["calibration"]["n"] == 1 and "_rows" not in out["datasets"]["x"]
    assert "| One |" in br.markdown({"S": out}, manifest)

# ---------------------------------------------------------------- the frozen suite


def test_frozen_manifest_names_the_private_mirror():
    m = read_json(SUITE / "manifest.json")
    assert m["eval_only"] is True and m["trainable_sources"] == [] and m["locked"] == ["test"] and m["holdout_sources"] == []
    assert set(m["datasets"]) == set(DATASETS) and {d for a in m["areas"].values() for d in a["datasets"]} == set(DATASETS)
    assert m["mirror"]["dataset"] == PRIVATE_DATASET and len(m["mirror"]["revision"]) == 40
    overlap = read_json(SUITE / "overlap.json")
    assert overlap["offending_records"] == 0 and overlap["records"] == sum(f["records"] for f in m["files"].values())


def private_partitions():
    """Both partitions through kev.suite.load_split (fetched from the private mirror and hash-checked on first use), or a
    skip for an account without access to it."""
    try:
        return {"development": load_split(SUITE, "development"), "test": load_split(SUITE, "test", allow_test=True)}   # structure only, no scoring
    except PermissionError as error:
        pytest.skip(f"breadth-v1 partitions are private: {error}")


def test_frozen_partitions_structure():
    m = read_json(SUITE / "manifest.json")
    parts = private_partitions()
    for split, recs in parts.items():
        by = {}
        for r in recs: by.setdefault(r["_meta"]["source"], []).append(r)
        assert set(by) == set(DATASETS) and all(r["_meta"]["area"] == DATASETS[r["_meta"]["source"]]["area"] for r in recs)
        assert {d: len(v) for d, v in by.items()} == m["files"][f"{split}.jsonl"]["by_source"]
    check_invariants(parts)


def test_bootstrap_index_is_paired_and_brackets_the_point_estimate():
    manifest = {"datasets": {"x": {"area": "one", "metric": "accuracy"}}, "areas": {"one": {"label": "One", "datasets": ["x"]}}}
    n = {"type": "noul", "label": True}
    recs = [_record(f"x{i}", "x", {"q": n}) for i in range(40)]
    good = [_row(f"x{i}", "q", n, [0.1, 0.9] if i % 4 else [0.9, 0.1]) for i in range(40)]
    out = br.bootstrap_index(recs, {"A": good, "B": good}, manifest, samples=200)
    point = br.score_system(recs, good, manifest)["overall"]["index"]
    assert out["index"]["A"][0] <= point <= out["index"]["A"][1] and out["difference"]["B"] == [0.0, 0.0]
