"""Predictors: callables record -> {"probabilities": {qid: {key: p}}, "latency_ms", "input_tokens", ...} for kev.benchmark.

LocalPredictor scores a checkpoint in-process; RemotePredictor any TypeSafe System One-compatible endpoint; JevPredictor
Jev itself through the AI SDK worker in playground/scripts (budget-capped).
"""
import contextlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from kev.api import question_keys
from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import api_request, materialize
from kev.device import sync
from kev.model import ROW_PASS_TOKENS, ContextOverflow, rows_of
from kev.suite import CONTEXT


LONG_ROW_KERNELS = "efficient"   # the label of a prediction (and its benchmark rows) that took the long-row kernels below

KERNEL_PACKAGES = ("torch", "transformers", "peft", "flash-linear-attention", "triton", "causal-conv1d", "mlx", "mlx-lm")
DELTANET_KERNELS = ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")   # the prefill hooks of transformers' GatedDeltaNet


@contextlib.contextmanager
def long_row_kernels():
    """The attention kernels a long row runs under (LocalPredictor): SDPA's flash and memory-efficient kernels, with the
    math kernel as the fallback. transformers asks SDPA for grouped-query attention (`enable_gqa`) when a call has no
    mask, which is the case for an unpadded state (kev.shared_prefix passes no mask so the state runs causal). Of the
    kernels allowed here only flash and math take `enable_gqa`, and flash has no fp32. So an fp32 state pass used to fall
    back to math, whose L x L scores do not fit: Kev-4B has 16 heads, so at 32k that is 16 x 32k^2 x 4 B = 69 GB per
    layer. Inside this context an fp32 call repeats its keys and values per query head instead (transformers' own path
    whenever a mask is given), and the memory-efficient kernel takes that: fp32 inputs and output, memory linear in L.
    On sm80+ its fp32 products are cutlass's OpMultiplyAddFastF32 (three TF32 products per product, fp32 accuracy and
    fp32 accumulation), not plain TF32. bf16 / fp16 calls keep `enable_gqa` and the flash kernel, so Kev-27B's long rows
    run as before."""
    from transformers.integrations import sdpa_attention
    grouped = sdpa_attention.use_gqa_in_sdpa
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key, value: key.dtype != torch.float32 and grouped(attention_mask, key, value)
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            yield
    finally:
        sdpa_attention.use_gqa_in_sdpa = grouped


def bound_implementation(fn):
    """The function transformers actually calls for one of its kernel hooks: `use_kernel_func_from_hub_with_fallback`
    wraps the PyTorch reference and binds the package's kernel (fla, causal-conv1d) as `implementation` when it imports;
    any other function (unwrapped, or patched in by a caller) is itself. -> "module.qualname"."""
    impl = inspect.getclosurevars(fn).nonlocals.get("implementation", fn) if inspect.isfunction(fn) else fn
    return f"{getattr(impl, '__module__', None)}.{getattr(impl, '__qualname__', type(impl).__name__)}"


def kernel_environment(model, device):
    """What a read's logits depend on besides kev's code and the checkpoint (runs/drift-v1/REPORT.md: #125's causal-conv1d
    kernel moved Kev-27B v1's reads by up to 0.06 in p with no code change): the scoring packages' versions, the GPU, the
    backbone dtype and, on a torch backbone, the attention implementation and (hybrid) the Gated DeltaNet layer's forward
    (kev.fused_qwen35 replaces it) and the convolution and chunked delta rule transformers bound. Reads only what is
    already loaded or imported. Two reads are comparable bit for bit only when this and the kev commit match. Recorded
    in report.json."""
    def installed(name):
        try: return version(name)
        except PackageNotFoundError: return None
    packages = {name: installed(name) for name in KERNEL_PACKAGES}
    packages["torch"] = torch.__version__   # with its build tag (2.8.0+cu128 on the CUDA wheels), which the metadata version drops
    npu = str(device).startswith("npu")
    env = {"packages": packages, "device": str(device),
           "gpu": torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda")
           else torch.npu.get_device_name(str(device)) if npu else None,
           "backend": model.backend, "dtype": model.dtype, "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT"),
           "attention": None, "deltanet": None}
    if model.backend != "torch": return env   # kev.mlx_model: mlx-lm's Metal kernels, named by the mlx / mlx-lm versions
    env["attention"] = getattr(model.lm.config, "_attn_implementation", None)
    layer = next((m for m in model.lm.modules() if type(m).__name__.endswith("GatedDeltaNet")), None) if model.hybrid else None
    if layer is not None:
        module = sys.modules[type(layer).__module__]
        env["deltanet"] = {"forward": bound_implementation(getattr(layer.forward, "__func__", layer.forward)),
                           **{name: bound_implementation(getattr(module, name)) for name in DELTANET_KERNELS if hasattr(module, name)}}
        if npu:   # kev.npu_qwen35.accelerate rebinds each layer's own chunk_gated_delta_rule, which the module names miss
            env["deltanet"]["chunk_gated_delta_rule"] = bound_implementation(layer.chunk_gated_delta_rule)
    return env


class LocalPredictor:
    """Scores a checkpoint in-process. Evaluation is fp32-exact in PyTorch (no TF32, no fused SDPA kernels on CUDA). On
    CUDA a hybrid backbone's Gated DeltaNet layers run flash-linear-attention's Triton kernels, whose dots are TF32, and a
    bf16 backbone is bf16 throughout, so reads repeat bit for bit only on the same kernel set (`self.environment`, from
    kernel_environment; AGENTS.md "What fp32-exact guarantees"). One more exception: on CUDA with the torch backend, a
    record whose longest row (state + one question) exceeds kev.model.ROW_PASS_TOKENS runs under SDPA's flash /
    memory-efficient kernels (long_row_kernels), because the exact math kernel's L x L score matrix does not fit (~275 GB
    per layer pass at 64k for Kev-4B's 16 heads). An fp32 backbone stays fp32 there: the memory-efficient kernel takes
    fp32 (Kev-4B / 9B at 8k-16k against the math kernel: max |dp| 6e-4, no flips in 238 questions;
    scripts/long_state_memory.py). Such a prediction carries `"kernels": LONG_ROW_KERNELS`, kev.benchmark copies it onto
    the record's rows and counts them in report.json's `long_rows`; shorter records carry nothing, so their rows and
    reports are unchanged. The efficient path is CUDA-only: on CPU / MPS attention stays eager and exact and still
    materialises L x L. A long record on a hybrid torch backbone also runs its state once, its questions continuing from
    it (kev.shared_prefix), instead of once per question; the MLX backend's forward already runs the state once."""

    def __init__(self, run, device, opts=LoadOptions(), context=CONTEXT):
        """opts.temperature=None scores with the temperature the checkpoint carries; 1.0 scores raw logits. context: the
        max_state / max_branch / max_packed a record must encode within (a suite manifest's `context`; the training
        context by default, the serving limits for external suites frozen without admission)."""
        if opts.temperature is not None and not (math.isfinite(opts.temperature) and opts.temperature > 0):
            raise ValueError("temperature must be finite and positive")
        checkpoint = self.checkpoint = Checkpoint(run)
        self.run = checkpoint.path
        if device == "cuda":
            # evaluation is fp32-exact: TF32 (10-bit mantissa) moves probabilities by ~1e-3, the isolation gate's tolerance
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.tok, self.model = checkpoint.load(device, opts)
        self.temperature = self.model.head.temperature
        self.environment = kernel_environment(self.model, device)
        self.device = device
        self.context = context

    @torch.no_grad()
    def __call__(self, record):
        enc = self.model.encode(self.tok, materialize(record), max_state=self.context["max_state"], max_branch=self.context["max_branch"], strict=True)
        if len(enc["ids"]) > self.context["max_packed"]:
            raise ContextOverflow(f"packed request exceeds the {self.context['max_packed']}-token limit")
        sync(self.device)
        start = time.perf_counter()
        state, _, rows = rows_of(enc)
        long = len(state) + max(len(r["ids"]) for r in rows) > ROW_PASS_TOKENS
        torch_backend = self.model.backend == "torch"
        efficient = long and torch_backend and self.device == "cuda"
        with long_row_kernels() if efficient else contextlib.nullcontext():
            if long and torch_backend and self.model.hybrid: logits = self.model.forward_batch([enc], shared_prefix=True)[0]
            else: logits = self.model.forward(enc)
        ps = [torch.softmax(z, -1).cpu() for z in logits]
        zs = [z.float().cpu() for z in logits]
        sync(self.device)
        keys = {qid: question_keys(q["type"], q.get("criteria")) for qid, q in record["questions"].items()}
        return {"probabilities": {qid: dict(zip(keys[qid], p.tolist())) for qid, p in zip(keys, ps)},
                "logits": {qid: dict(zip(keys[qid], z.tolist())) for qid, z in zip(keys, zs)},
                "inference_temperature": self.temperature,
                "latency_ms": 1000 * (time.perf_counter() - start), "input_tokens": len(enc["ids"]), **({"kernels": LONG_ROW_KERNELS} if efficient else {})}


REFUSAL_STATUSES = (400, 413, 422)   # an endpoint declining the request itself (kev.serve: 422 past its context; Jev: 400 / 413 / 422)


def http_detail(error):
    """The message of an HTTP error body: FastAPI's / TypeSafe's `detail` (or `error`) when it is JSON, else its text."""
    try: text = error.read().decode("utf-8", "replace")
    except Exception: return str(error.reason)
    try: body = json.loads(text)
    except ValueError: return text[:500] or str(error.reason)
    return str(body.get("detail") or body.get("error") or body) if isinstance(body, dict) else str(body)


class RemotePredictor:
    """Score any TypeSafe System One-compatible endpoint (POST <base_url>/v1/systemone) on frozen records. Probabilities are
    taken from the response as returned (renormalised by validate_distribution like every other predictor). Records the
    server-reported model id so the manifest can pin what was scored. `concurrency` is how many requests kev.benchmark may
    keep in flight at once (each call is independent: one request, its own retries); 1 scores sequentially. An endpoint
    that declines the request itself (REFUSAL_STATUSES: kev.serve's 422 for a state past its context) raises
    ContextOverflow at once, so kev.benchmark counts the record rejected under skip_overlong and stops otherwise; another
    client error stops at once too; 408, 429, 5xx, connection errors and timeouts are tried `retries` times in all."""

    def __init__(self, base_url, model="kev-latest", api_key="local", timeout=120, retries=3, concurrency=1):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.base_url, self.model, self.api_key, self.timeout, self.retries = base_url.rstrip("/"), model, api_key, timeout, retries
        self.concurrency = concurrency
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
            except urllib.error.HTTPError as error:
                if error.code in REFUSAL_STATUSES:   # the request itself (kev.serve: a state or question over the serving context): a rejected record, not retried
                    raise ContextOverflow(f"remote endpoint refused the request (HTTP {error.code}): {http_detail(error)}") from None
                if error.code not in (408, 429) and error.code < 500:   # 401, 403, 404, ...: the same answer every time; 408, 429 and 5xx are retried
                    raise RuntimeError(f"remote endpoint answered HTTP {error.code}: {http_detail(error)}") from None
                last = error
            except Exception as error:   # connection errors and timeouts: retried
                last = error
            if attempt + 1 < self.retries: time.sleep(2 ** attempt)
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


class RotationAveraged:
    """Test-time order averaging (PLAN.md round 4, item 4.4): score the record under the first `rotations` cyclic
    rotations of every Choice question's options (rotation r of a K-option question is r mod K; Noul and Score keep their
    order, which is part of their meaning) and average the logits per option key. The geometric mean of softmax
    probabilities is the softmax of the mean logits, so the returned probabilities and logits stay consistent; predictors
    that return no logits are averaged in log-probability space. Costs `rotations` forward passes per record."""

    def __init__(self, predictor, rotations):
        if rotations < 2:
            raise ValueError("rotation averaging needs at least 2 rotations")
        self.predictor, self.rotations = predictor, rotations
        self.temperature = getattr(predictor, "temperature", None)
        self.concurrency = getattr(predictor, "concurrency", 1)

    @staticmethod
    def rotated(record, r):
        def rotate(q):
            if q["type"] != "choice": return q
            keys = list(q["criteria"]); k = r % len(keys)
            return {**q, "criteria": {key: q["criteria"][key] for key in keys[k:] + keys[:k]}}
        return {**record, "questions": {qid: rotate(q) for qid, q in record["questions"].items()}}

    def __call__(self, record):
        widest = max((len(q["criteria"]) for q in record["questions"].values() if q["type"] == "choice"), default=1)
        preds = [self.predictor(self.rotated(record, r)) for r in range(min(self.rotations, widest))]
        field = "logits" if all("logits" in p for p in preds) else "probabilities"
        out = {"probabilities": {}, "latency_ms": sum(p["latency_ms"] for p in preds), "rotations": len(preds)}
        if field == "logits":
            out["logits"] = {}; out["inference_temperature"] = preds[0].get("inference_temperature", 1.0)
        for qid in record["questions"]:
            keys = list(preds[0]["probabilities"][qid])
            if field == "logits":
                z = {k: sum(p["logits"][qid][k] for p in preds) / len(preds) for k in keys}
            else:
                z = {k: sum(math.log(max(p["probabilities"][qid][k], 1e-12)) for p in preds) / len(preds) for k in keys}
            top = max(z.values()); e = {k: math.exp(v - top) for k, v in z.items()}; s = sum(e.values())
            out["probabilities"][qid] = {k: v / s for k, v in e.items()}
            if field == "logits": out["logits"][qid] = z
        return out


class JevRefused(ContextOverflow):
    """Jev declined a request because of the input itself (REFUSAL_STATUSES, or a hosted-side error on an oversize request;
    see JevPredictor). With count_refusals, kev.benchmark counts such a record as rejected (rejected.json) instead of stopping."""


# Jev's context in tokens (documented ~32k). Past it the AI Gateway answers with HTTP 400 *or* a 503 GatewayInternalServerError
# (evals/longdoc-v1: 212 x 400 and 28 x 503 on the 240 requests of ~57k-61k tokens; every request of <= ~31k tokens answered),
# so a hosted-side error on a request estimated past OVERSIZE x JEV_CONTEXT_TOKENS is the size, not an outage. The estimate is
# characters / 4 of the request JSON; the 1.5 margin keeps a ~30k-token legal text (~144k characters) out of it.
JEV_CONTEXT_TOKENS = 32768
OVERSIZE = 1.5


def oversize(request):
    """Whether a Jev request is well past Jev's context (estimated tokens = characters / 4)."""
    return len(json.dumps(request, ensure_ascii=False)) / 4 > OVERSIZE * JEV_CONTEXT_TOKENS


class JevPredictor:
    """Jev through the AI SDK worker (playground/scripts/jev-evaluate.mjs); every call is counted against a token budget.
    count_refusals: a request answered with a REFUSAL_STATUSES error, or with a hosted-side error (5xx, no status) while
    oversize(), raises JevRefused at once (counted in accounting()["refusals"], not fatal, not retried); any other client error
    (401, 403, 429, ...) still stops the read. attempts: tries per request on hosted-side failures of requests that are not
    oversize, backing off 1, 2, 4, ... seconds up to 30 between them."""
    def __init__(self, key, budget=0.1, max_calls=700, count_refusals=False, attempts=4):
        self.budget, self.max_calls, self.count_refusals, self.attempts = budget, max_calls, count_refusals, attempts
        self.calls, self.input_tokens, self.output_tokens, self.retries = 0, 0, 0, 0
        self.refusals = {}   # HTTP status -> count
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
        for attempt in range(self.attempts):
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
            hosted = status is None or status >= 500
            if self.count_refusals and (status in REFUSAL_STATUSES or hosted and oversize(request)):
                kind = str(status) if status in REFUSAL_STATUSES else f"{status} (oversize)"
                self.refusals[kind] = self.refusals.get(kind, 0) + 1
                raise JevRefused(f"Jev refused the request: {result['error']['name']} (HTTP {kind})")
            if attempt == self.attempts - 1 or (status is not None and status < 500):
                raise RuntimeError(f"Jev request failed: {result['error']['name']} (HTTP {status})")
            self.retries += 1
            time.sleep(min(2 ** attempt, 30))
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
                **({"refusals": dict(sorted(self.refusals.items()))} if self.count_refusals else {}),
                "estimated_usd": self.input_tokens * PRICE_PER_MILLION / 1e6,
                "budget_usd": self.budget, "sdk": "ai@7.0.105", "zero_data_retention_requested": True}
