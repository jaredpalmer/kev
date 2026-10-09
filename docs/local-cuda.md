# Local Fused CUDA Serving

A plain `serve` installation works without FLA, but uses the reference DeltaNet
implementation. `scripts/local_cuda.sh` installs the fused serving stack in a new
environment and executes real kernels before loading a checkpoint. It never
modifies an existing environment or the host driver.

Choose an environment and a cache directory on a filesystem with room for the
CUDA packages. Every package, Python, compiler and temporary cache created by the
script uses the selected directory. `HF_HOME` may point at an existing model cache.

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/local_cuda.sh setup \
  --env "$PWD/.venv-cuda" --cache /path/to/cache \
  --python /path/to/python3.13

HF_HOME=/path/to/existing/huggingface CUDA_VISIBLE_DEVICES=1 \
  bash scripts/local_cuda.sh serve \
  --env "$PWD/.venv-cuda" --cache /path/to/cache -- \
  --run jaredpalmer/kev-4b@v1.0 --host 127.0.0.1 --port 8010
```

The launcher explicitly enables bf16, fused kernels and CUDA graphs. A failed
kernel check or missing fused implementation stops startup rather than silently
selecting the slow path. To run the kernel check without loading weights, replace
`serve` with `check` and omit the arguments after `--`.

## Versions And Compatibility

The stack uses Torch 2.8.0's CUDA 12.6 wheel, FLA/fla-core 0.5.2 and Triton 3.7.1.
Torch 2.6's Inductor cannot import the newer Triton API. Torch 2.8 handles it, and
this is the same intentional Triton override used by Kev's Modal image: Torch's
package metadata pins Triton 3.4, while Kev's Hopper path requires >=3.7.1. A
subsequent dependency sync can undo the override. Use this separate environment
only for serving and verification; `uv pip check` will report that intentional
Torch/Triton metadata conflict.

The fused path uses FLA's Triton convolution, so the separate native
`causal-conv1d` wheel is optional here. The script does not build native CUDA
extensions or install a driver. CUDA minor-version compatibility alone does not
guarantee that a newer JIT kernel works with an older driver: the preflight runs
both causal convolution and Gated DeltaNet and checks finite outputs.

Keep CPU-only training tests in a separate environment without FLA. Transformers
5.19 binds an installed FLA implementation even for CPU inputs, which makes those
tests try to execute CUDA/Triton kernels on CPU tensors. Hiding CUDA devices alone
does not prevent the import. The CUDA serving and HTTP conformance tests run in
the fused environment.

## Measure Before Switching

Run the reference and candidate on the same GPU, preferably sequentially, using
the same checkpoint revision. Keep the original service on its existing port.

```bash
python -m scripts.http_serving_bench \
  --url http://127.0.0.1:8010 --out /path/to/candidate-http.json \
  --reps 20 --clients 1,4,8,16

python -m kev.benchmark --remote http://127.0.0.1:8010 \
  --suite evals/v7/decision-v7 --out /path/to/candidate-quality

KEV_BASE_URL=http://127.0.0.1:8010 python -m pytest tests/test_api.py -q
```

The HTTP benchmark reuses `scripts.serving_bench.py`'s four cases, reports new and
cached states separately, warms the batch shapes and uses unique ticket IDs for
each new-state request. Client p50/p95 includes network and queueing; `model_p50_ms`
is the server's reported model time. Use the existing benchmark's accuracy and
Brier metrics and compare probabilities on paired rows; fused bf16 arithmetic is
not bit-identical to the reference path. These fixed support cases establish
serving performance, not accuracy on a production workload.

## H20 Measurement

The [committed report](../runs/h20-local-fused/report.json) compares Torch 2.6's
reference path with this stack on the same H20, sequentially, on driver
535.161.08. Both use bf16, CUDA graphs, the same Kev commit and checkpoint. Model
latencies below are medians of 20 requests after warmup:

| Request | Reference new/cached, ms | Fused new/cached, ms | New-state speedup |
|---|---|---|---|
| Short text, 2 questions | 41.7 / 25.0 | 37.6 / 22.3 | 1.11x |
| Short text, 6 questions | 93.8 / 77.4 | 92.2 / 76.95 | 1.02x |
| 370-token text, 5 questions | 144.0 / 79.1 | 70.8 / 38.8 | 2.03x |
| 2,200-token text, 5 questions | 366.3 / 93.6 | 197.55 / 53.5 | 1.85x |

At 16 HTTP clients, six-question short-text throughput goes from 20.15 to 31.11
requests/s. The two-question case regresses slightly, from 48.40 to 47.09
requests/s; this stack is not faster for every shape. The first short request
costs 11.1 seconds of kernel compilation, excluded from the steady-state numbers.

On `decision-v7` development, all 1,204 records and 1,468 questions complete with
zero truncations or rejections. All 1,468 top-1 answers agree; clean accuracy is
0.871835 in both runs. Clean Brier is 0.182619 / 0.182377 and ECE is 0.013041 /
0.014479 (reference / fused). The largest candidate-probability difference is
0.0856, so equal labels do not imply equal probabilities or threshold decisions.
Use the report's full metrics when adopting confidence gates.

The reference HTTP benchmark completes with no graph-capture failure, but its
subsequent mixed quality run hits graph-capture OOM fallbacks. The fused server
has 206 captured graphs and zero failed captures or cache OOM retries after both
benchmarks and the API tests. This is bounded evidence for this workload, not a
guarantee for arbitrarily many shapes or longer documents.
