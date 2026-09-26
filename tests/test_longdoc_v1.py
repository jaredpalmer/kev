"""evals/longdoc-v1: the synthetic generator's labels come from its solver over the stored facts, bundles land in their
length window and depth, CUAD questions carry CUAD's own labels, and the committed manifest is an eval-only private-mirror
suite in the serving context. No weights, no network, no tokenizer (token counts are faked at 4.1 characters each)."""
import json
import random
from pathlib import Path

import pytest

from kev.suite import SERVING_CONTEXT
from scripts import build_longdoc_v1 as B
from scripts import longdoc_v1_synthetic as S

ROOT = Path(__file__).resolve().parents[1]
count = lambda text: int(len(text) / S.CHARS_PER_TOKEN) + 1


def bundles(T, n, seed="t"):
    rng, out = random.Random(f"{seed}:{T}"), []
    while len(out) < n:
        got = S.generate(rng, T, S.DEPTHS[len(out) % 3], count)
        if got: out.append(got)
    return out


@pytest.fixture(scope="module")
def small():
    return {T: bundles(T, 3) for T in (4096, 16384)}


def test_labels_are_the_solver_over_round_tripped_facts(small):
    for T, got in small.items():
        for g in got:
            facts = json.loads(json.dumps(g["facts"]))
            for qid, spec in g["specs"].items():
                key = S.solve(facts, spec)
                assert key in g["questions"][qid]["criteria"]
                doc = facts["docs"][spec["doc"]]
                if spec["kind"] == "locate":
                    assert facts["docs"][spec["option_docs"][key]]["city"] == spec["city"]
                    assert sum(facts["docs"][i]["city"] == spec["city"] for i in spec["option_docs"].values()) == 1
                elif spec["kind"] == "multihop":
                    assert spec["option_values"][key] == doc["credits"][doc["sites"][spec["site"]]]
                else:
                    assert spec["option_values"][key] == doc["terms"].get(spec["term"], S.NOT_STATED)
                    assert (spec["option_values"][key] == S.NOT_STATED) == (not spec["stated"])
                if spec["kind"] != "locate":
                    assert list(spec["option_values"].values())[-1] == S.NOT_STATED   # always offered, always last


def test_bundles_land_in_their_window_and_depth(small):
    for T, got in small.items():
        lo, hi = int(0.84 * T), int(0.93 * T)
        for i, g in enumerate(got):
            assert lo <= g["state_tokens"] <= hi
            assert abs(g["specs"]["detail"]["depth"] - S.DEPTHS[i % 3]) <= 0.08
            assert g["specs"]["detail"]["stated"]
            spec = g["specs"]["absent"]
            if not spec["stated"]:   # the term it omits is stated by another agreement in the file
                assert any(spec["term"] in d["terms"] for j, d in enumerate(g["facts"]["docs"]) if j != spec["doc"])


def test_generation_is_deterministic():
    a, b = bundles(4096, 2, "same"), bundles(4096, 2, "same")
    assert [x["state"] for x in a] == [x["state"] for x in b] and [x["questions"] for x in a] == [x["questions"] for x in b]


def test_cuad_names_match_and_companies_group():
    assert B.title_key("MOELIS_CO_03_24_2014-EX-10.19-STRATEGIC ALLIANCE AGREEMENT") == B.title_key("MOELIS&CO_03_24_2014-EX-10.19-STRATEGIC ALLIANCE AGREEMENT.pdf")
    assert B.title_key("KALLOINC_11_03_2011-EX-10.1-STRATEGIC ALLIANCE AGREEMENT") == B.title_key("KALLOINC_11_03_2011-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.PDF'")
    assert B.company("PcquoteComInc_19990721_S-1A_EX-10.11_6377149_EX-10.11_Co-Branding Agreement2") == B.company("PcquoteComInc_19990721_S-1A_EX-10.11_6377149_EX-10.11_Co-Branding Agreement3")
    assert B.company("PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING AGREEMENT ") == "PACIRAPHARMACEUTICALSINC"


def test_cuad_questions_carry_cuad_labels():
    details = {k: f"Is there a {k} clause?" for k in "abcdefgh"}
    c = {"present": ["a", "b"], "absent": ["c", "d", "e", "f", "g", "h"], "law": "Delaware"}
    prevalence = {k: 0.5 for k in details}
    specs = B.cuad_questions("T", c, details, prevalence, ["Delaware", "New York", "Texas", "Florida"])
    by = {s["qid"]: s for s in specs}
    assert by["presence_found"]["category"] in c["present"] and by["presence_found"]["label"] is True
    assert by["presence_not_found"]["category"] in c["absent"] and by["presence_not_found"]["label"] is False
    assert by["category"]["label"] in c["present"] and sum(o in c["present"] for o in by["category"]["options"]) == 1
    assert by["governing_law"]["label"] == "Delaware" and len(set(by["governing_law"]["options"])) == 4
    assert specs == B.cuad_questions("T", c, details, prevalence, ["Delaware", "New York", "Texas", "Florida"])   # fixed per target
    qs = B.cuad_request("T", specs, details)
    assert qs["category"]["label"] in qs["category"]["criteria"] and qs["presence_found"]["type"] == "noul"


def test_cut_stops_at_a_paragraph_within_budget():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    piece = B.cut(text, 300, count)
    assert piece and count(piece) <= 300 and text.startswith(piece) and text[len(piece):].startswith("\n\n")


def test_manifest_is_an_eval_only_private_long_document_suite():
    path = ROOT / "evals/longdoc-v1/manifest.json"
    if not path.exists(): pytest.skip("suite not frozen in this checkout")
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m["eval_only"] and m["trainable_sources"] == [] and m["locked"] == ["test"]
    assert m["mirror"]["dataset"] == "jaredpalmer/kev-private-evals" and m["mirror"]["revision"]
    ctx = {k: v for k, v in m["context"].items() if k != "note"}
    assert ctx == SERVING_CONTEXT and m["tokenizer"]["model"] == "Qwen/Qwen3.8-27B"
    assert set(m["files"]) == {"development.jsonl", "test.jsonl"}
