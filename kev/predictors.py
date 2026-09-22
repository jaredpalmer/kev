"""Predictors: callables record -> {"probabilities": {qid: {key: p}}, "latency_ms", "input_tokens", ...} for kev.benchmark.

LocalPredictor scores a checkpoint in-process; RemotePredictor any TypeSafe System One-compatible endpoint; JevPredictor
Jev itself through the AI SDK worker in playground/scripts (budget-capped).
"""
import json
import math
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import torch

from kev.api import question_keys
from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import api_request, materialize
from kev.device import sync
from kev.model import ContextOverflow
from kev.suite import CONTEXT


class LocalPredictor:
    def __init__(self, run, device, opts=LoadOptions(), context=CONTEXT):
        """opts.temperature=None scores with the temperature the checkpoint carries; 1.0 scores raw logits. context: the
        max_state / max_branch / max_packed a record must encode within (a suite manifest's `context`; the training
        context by default, the serving limits for external suites frozen without admission)."""
        if opts.temperature is not None and not (math.isfinite(opts.temperature) and opts.temperature > 0):
            raise ValueError("temperature must be finite and positive")
        checkpoint = Checkpoint(run)
        self.run = checkpoint.path
        if device == "cuda":
            # evaluation is fp32-exact: TF32 (10-bit mantissa) moves probabilities by ~1e-3, the isolation gate's tolerance
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.tok, self.model = checkpoint.load(device, opts)
        self.temperature = self.model.head.temperature
        self.device = device
        self.context = context

    @torch.no_grad()
    def __call__(self, record):
        enc = self.model.encode(self.tok, materialize(record), max_state=self.context["max_state"], max_branch=self.context["max_branch"], strict=True)
        if len(enc["ids"]) > self.context["max_packed"]:
            raise ContextOverflow(f"packed request exceeds the {self.context['max_packed']}-token limit")
        sync(self.device)
        start = time.perf_counter()
        logits = self.model.forward(enc)
        ps = [torch.softmax(z, -1).cpu() for z in logits]
        zs = [z.float().cpu() for z in logits]
        sync(self.device)
        keys = {qid: question_keys(q["type"], q.get("criteria")) for qid, q in record["questions"].items()}
        return {"probabilities": {qid: dict(zip(keys[qid], p.tolist())) for qid, p in zip(keys, ps)},
                "logits": {qid: dict(zip(keys[qid], z.tolist())) for qid, z in zip(keys, zs)},
                "inference_temperature": self.temperature,
                "latency_ms": 1000 * (time.perf_counter() - start), "input_tokens": len(enc["ids"])}


class RemotePredictor:
    """Score any TypeSafe System One-compatible endpoint (POST <base_url>/v1/systemone) on frozen records. Probabilities are
    taken from the response as returned (renormalised by validate_distribution like every other predictor). Records the
    server-reported model id so the manifest can pin what was scored."""

    def __init__(self, base_url, model="kev-latest", api_key="local", timeout=120, retries=3):
        self.base_url, self.model, self.api_key, self.timeout, self.retries = base_url.rstrip("/"), model, api_key, timeout, retries
        self.served_model = None

    def __call__(self, record):
        payload = json.dumps({**api_request(record), "model": self.model}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=payload, method="POST",
                                    headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"})
        last = None
        for attempt in range(self.retries):
            try:
                start = time.perf_counter()
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                latency = 1000 * (time.perf_counter() - start)
                break
            except Exception as error:   # 5xx / timeouts: retry with backoff; anything persistent surfaces as a rejected record
                last = error; time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"remote endpoint failed after {self.retries} attempts: {last}")
        self.served_model = body.get("model", self.served_model)
        probs = {}
        for qid, q in record["questions"].items():
            a = body["answers"][qid]
            if q["type"] == "noul": probs[qid] = {"true": float(a["noul"]), "false": 1 - float(a["noul"])}
            else: probs[qid] = {str(k): float(v) for k, v in a["probabilities"].items()}
        return {"probabilities": probs, "latency_ms": latency, "input_tokens": (body.get("usage") or {}).get("input_tokens")}


PRICE_PER_MILLION = 0.042   # Jev list price per million input tokens, for the budget cap


class JevPredictor:
    """Jev through the AI SDK worker (playground/scripts/jev-evaluate.mjs); every call is counted against a token budget."""
    def __init__(self, key, budget=0.1, max_calls=700):
        self.budget, self.max_calls = budget, max_calls
        self.calls, self.input_tokens, self.output_tokens, self.retries = 0, 0, 0, 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        worker = Path(__file__).resolve().parents[1] / "playground/scripts/jev-evaluate.mjs"
        self.process = subprocess.Popen(["node", str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                        env={**os.environ, "AI_GATEWAY_API_KEY": key})

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)

    def __call__(self, record):
        if self.calls >= self.max_calls or (self.input_tokens + 65536) * PRICE_PER_MILLION / 1e6 > self.budget:
            raise RuntimeError("Jev evaluation reached the request/token cost cap")
        request = api_request(record)
        for attempt in range(4):
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Jev SDK worker exited without a response")
            result = json.loads(line)
            self.calls += 1
            if "error" not in result:
                break
            status = result["error"]["status"]
            # bounded retry for hosted-side failures only; client errors (4xx) are real and must surface
            if attempt == 3 or (status is not None and status < 500):
                raise RuntimeError(f"Jev request failed: {result['error']['name']} (HTTP {status})")
            self.retries += 1
            time.sleep(2 ** attempt)
        usage = result["usage"]
        if usage.get("inputTokens") is None:
            raise RuntimeError("Jev returned no input token usage; cannot account for cost")
        self.input_tokens += usage["inputTokens"]
        self.output_tokens += usage.get("outputTokens") or 0
        probabilities = {}
        for qid, q in record["questions"].items():
            answer = result["answers"][qid]
            if q["type"] == "noul":
                probabilities[qid] = {"false": 1 - answer["probability"], "true": answer["probability"]}
            else:
                probabilities[qid] = answer["probabilities"]
        return {**result, "probabilities": probabilities}

    def accounting(self):
        return {"model": "typesafe-ai/jev", "model_revision": "Gateway alias; provider revision not exposed by SDK result",
                "started_at": self.started_at, "calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "retries_after_5xx": self.retries, "listed_input_usd_per_million": PRICE_PER_MILLION,
                "estimated_usd": self.input_tokens * PRICE_PER_MILLION / 1e6,
                "budget_usd": self.budget, "sdk": "ai@7.0.105", "zero_data_retention_requested": True}
