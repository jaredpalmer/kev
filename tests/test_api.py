"""Conformance: the docs' example requests must round-trip through /v1/systemone with the documented shapes.
Run with the server up:  uv run --extra serve python -m pytest tests -q
"""
import math, os
import httpx, pytest

pytestmark = pytest.mark.server

BASE = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8008")
MAX_PROBE_TOKENS = 1 << 20   # largest advertised state limit over_length probes; a bigger one is skipped, not allocated

DEPARTMENT = {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}


def post(body):
    r = httpx.post(f"{BASE}/v1/systemone", json=body, timeout=120)
    assert r.headers["x-typesafe-request-id"]   # every TypeSafe client exposes it as response.request_id
    return r.status_code, r.json()


def test_choice_basic():
    code, r = post({"state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?", "model": "jev-latest",
                    "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT}}})
    assert code == 200 and r["model"] == "jev-latest"
    a = r["answers"]["department"]
    assert a["type"] == "choice" and a["choice"] in DEPARTMENT and set(a["probabilities"]) == set(DEPARTMENT)
    assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=0.03) and 0 <= a["confidence"] <= 1
    assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)
    assert set(r["usage"]) == {"input_tokens", "output_tokens"} | ({"state_tokens", "state_tokens_used"} if card()["truncate_states"] else set())   # the counts only where the server may truncate


def test_five_questions_and_null_descriptions():
    code, r = post({"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?", "model": "jev-latest",
                    "questions": {
                        "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT},
                        "return_reason": {"type": "choice", "instructions": "If the customer wants to return something, why?", "criteria": {"wrong_size": "The item doesn't fit", "wrong_item": "A different product was delivered", "damaged": "The item arrived broken or faulty", "changed_mind": "The item is fine, the customer no longer wants it", "other": "A return reason that fits none of the above"}},
                        "tone": {"type": "choice", "instructions": "What is the customer's tone?", "criteria": {"calm": None, "frustrated": None, "angry": None}},
                    }})
    assert code == 200 and set(r["answers"]) == {"department", "return_reason", "tone"}
    assert r["answers"]["tone"]["choice"] in {"calm", "frustrated", "angry"}


def test_structured_instructions_and_criteria():
    code, r = post({"state": "I sent the shoes back a week ago. When do I get my money?", "model": "jev-latest",
                    "questions": {"return_topic": {"type": "choice",
                                                   "instructions": {"question": "Which returns topic is the customer asking about?", "focus": "Classify the information the customer wants."},
                                                   "criteria": {"return_policy": {"what": "Whether and how an item can be returned", "not_for": "Progress of a return already sent", "examples": ["Can I return shoes I've worn once?", "How long do I have to return an order?"]},
                                                                "return_status": {"what": "Progress of a return already sent", "not_for": "Whether and how an item can be returned", "examples": ["Has my return arrived yet?", "When will my refund be paid?"]}}}}})
    assert code == 200 and r["answers"]["return_topic"]["choice"] in {"return_policy", "return_status"}


def test_noul_score_and_object_state():
    code, r = post({"state": {"document": "I was charged twice. Please fix this ASAP."}, "model": "jev-latest",
                    "questions": {"billing": {"type": "noul", "instructions": "Is this ticket about billing?", "criteria": {"true": "Explicitly about charges", "false": "Not about charges"}},
                                  "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["can wait", "this week", "today"]}}})
    assert code == 200
    n, s = r["answers"]["billing"], r["answers"]["urgency"]
    assert n == {"type": "noul", "noul": n["noul"]} and 0 <= n["noul"] <= 1
    assert s["type"] == "score" and 0 <= s["score"] <= 2 and s["legend"] == {"0": "can wait", "1": "this week", "2": "today"}
    assert set(s["probabilities"]) == {"0", "1", "2"} and 0 <= s["confidence"] <= 1
    assert math.isclose(s["score"], sum(int(k) * v for k, v in s["probabilities"].items()), abs_tol=0.05)


def test_optional_instructions_and_one_level_score():
    """The SDK omits `instructions` when it is not given and accepts a score with a single level."""
    code, r = post({"state": "I was charged twice.", "model": "jev-latest",
                    "questions": {"billing": {"type": "noul", "criteria": {"true": "About charges", "false": "Not about charges"}},
                                  "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["today"]}}})
    assert code == 200 and 0 <= r["answers"]["billing"]["noul"] <= 1
    assert r["answers"]["urgency"] == {"type": "score", "score": 0.0, "legend": {"0": "today"}, "probabilities": {"0": 1.0}, "confidence": 1.0}


def test_validation_422():
    assert post({"state": "x", "model": "m", "questions": {"q": {"type": "score", "instructions": "i", "criteria": []}}})[0] == 422
    assert post({"state": "x", "model": "m", "questions": {"q": {"type": "bogus", "instructions": "i"}}})[0] == 422
    assert post({"state": "x", "model": "m", "questions": {}})[0] == 422
    assert post({"state": "x", "model": "m", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(256)}}}})[0] == 422


def test_packed_equals_separate():
    """Answers must not depend on which sibling questions are in the request (branch isolation)."""
    qs = {"a": {"type": "noul", "instructions": "Is the weather described as nice?"},
          "b": {"type": "choice", "instructions": "Which season is it most likely?", "criteria": {"summer": None, "winter": None, "unknown": None}}}
    state = "The weather is nice today and the park is full of people."
    both = post({"state": state, "model": "m", "questions": qs})[1]["answers"]
    alone = post({"state": state, "model": "m", "questions": {"b": qs["b"]}})[1]["answers"]
    for k in both["b"]["probabilities"]:
        assert abs(both["b"]["probabilities"][k] - alone["b"]["probabilities"][k]) <= 0.011


def test_sdk_client():
    typesafe_sdk = pytest.importorskip("typesafe_sdk")
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    with TypeSafeClient(api_key="local", base_url=BASE, model="kev-latest") as client:
        resp = client.system_one(state={"document": "I was charged twice. Please fix this ASAP."},
                                 questions={"billing": Noul(instructions="Is this ticket about billing?"),
                                            "tone": Choice(instructions="What is the customer's tone?", criteria={"calm": None, "frustrated": None, "angry": None}),
                                            "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"])})
    assert 0 <= resp.nouls["billing"].noul <= 1
    assert resp.choices["tone"].choice in {"calm", "frustrated", "angry"}
    assert 0 <= resp.scores["urgency"].score <= 2
    assert resp.request_id and resp.usage.input_tokens > 0


def test_sdk_models():
    """models.list() parses only when every card carries name, description and release_date."""
    pytest.importorskip("typesafe_sdk")
    from typesafe_sdk import TypeSafeClient
    with TypeSafeClient(api_key="local", base_url=BASE, model="kev-latest") as client:
        cards = client.models.list().models
    assert {"kev-latest", "jev-latest"} <= {c.name for c in cards}   # jev-latest is the SDK's default model
    assert all(c.description and c.release_date for c in cards)


def test_sdk_async_client():
    pytest.importorskip("typesafe_sdk")
    import asyncio
    from typesafe_sdk import AsyncTypeSafeClient, Noul

    async def go():
        async with AsyncTypeSafeClient(api_key="local", base_url=BASE, model="kev-latest") as client:
            return await client.system_one(state="I was charged twice.", questions={"billing": Noul(instructions="Is this about billing?")})

    assert 0 <= asyncio.run(go()).nouls["billing"].noul <= 1


def card():
    return httpx.get(f"{BASE}/v1/models", timeout=30).json()["models"][0]


def over_length():
    """A request whose state is past the server's limit (/v1/models max_state_tokens): every word is at least one token."""
    limit = card()["max_state_tokens"]
    assert type(limit) is int and limit > 0, f"/v1/models must advertise max_state_tokens as an int > 0, got {limit!r}"
    if limit > MAX_PROBE_TOKENS: pytest.skip(f"max_state_tokens {limit:,} is over the {MAX_PROBE_TOKENS:,}-word probe the local allocation budget allows")
    return {"state": "word " * (limit + 16), "model": "kev-latest",
            "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}}


def test_over_length_state_is_refused_not_truncated():
    """The default server refuses a state past its limit with a 422 that names the count, the limit and the fixes; it
    never answers from a silently cut document. (A KEV_TRUNCATE_STATES=1 server would run a full-length pass here, which
    a CPU / MPS smoke server cannot hold, so this is checked on a default server only; tests/test_unit.py covers both.)"""
    if card()["truncate_states"]: pytest.skip("server started with KEV_TRUNCATE_STATES=1")
    limit = card()["max_state_tokens"]
    code, body = post(over_length())
    assert code == 422 and f"over the {limit:,}-token limit" in body["detail"] and "KEV_TRUNCATE_STATES=1" in body["detail"]


def test_sdk_surfaces_the_over_length_refusal():
    """The TypeSafe SDK raises the 422 as TypeSafeUnprocessableEntityError (no retry: 422 is not a retried status) whose
    message is the server's detail, request id included."""
    pytest.importorskip("typesafe_sdk")
    from typesafe_sdk import Noul, TypeSafeClient, TypeSafeUnprocessableEntityError
    if card()["truncate_states"]: pytest.skip("server started with KEV_TRUNCATE_STATES=1")
    with TypeSafeClient(api_key="local", base_url=BASE, model="kev-latest") as client:
        with pytest.raises(TypeSafeUnprocessableEntityError) as refused:
            client.system_one(state=over_length()["state"], questions={"billing": Noul(instructions="Is this about billing?")})
    assert refused.value.status == 422 and "tokens, over the" in str(refused.value) and refused.value.request_id


def test_truncating_server_marks_every_response():
    """A KEV_TRUNCATE_STATES=1 server says on every response whether it cut the state (`truncated`, usage.state_tokens /
    state_tokens_used), and the TypeSafe SDK still parses those responses (it ignores fields it does not model)."""
    if not card()["truncate_states"]: pytest.skip("server started without KEV_TRUNCATE_STATES=1")
    pytest.importorskip("typesafe_sdk")
    from typesafe_sdk import Noul, TypeSafeClient
    code, body = post({"state": "I was charged twice.", "model": "kev-latest", "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}})
    assert code == 200 and body["truncated"] is False and body["usage"]["state_tokens"] == body["usage"]["state_tokens_used"] > 1
    with TypeSafeClient(api_key="local", base_url=BASE, model="kev-latest") as client:
        resp = client.system_one(state="I was charged twice.", questions={"billing": Noul(instructions="Is this about billing?")})
    assert 0 <= resp.nouls["billing"].noul <= 1 and resp.usage.input_tokens == body["usage"]["input_tokens"]
