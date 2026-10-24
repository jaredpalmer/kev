"""Compare gathered and indexed recurrent states on the same loaded CUDA model.

Run through modal_app.py::script, or directly on CUDA with --out DIR. Both paths
use the same weights, encoded requests, FLA version and GPU. The baseline switch
retains the original recurrent gather and chunk_gated_delta_rule call.
"""
import argparse
import gc
import statistics
import time
from pathlib import Path

import torch

from kev.api import to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.cuda_graphs import CudaGraphs
from kev.device import allocated_bytes, empty_cache, sync
from kev.experiment import source_hashes
from kev.model import admit
from kev.predictors import kernel_environment
from kev.suite import write_json
from scripts.serving_bench import request


def mixed_throughput(model, tok, ck):
    """ABBA comparison of the server scheduler on the original seeded mixed workload."""
    import random
    from concurrent.futures import ThreadPoolExecutor
    from kev.data import materialize
    from kev.serve import Server
    from kev.suite import load_split

    records = random.Random(0).choices([materialize(r) for r in load_split('evals/v7/decision-v7', 'development')], k=256)

    def run(clients):
        def one(record):
            start = time.perf_counter(); server.probs(record)
            return (time.perf_counter() - start) * 1000
        with ThreadPoolExecutor(clients) as pool:
            start = time.perf_counter()
            latencies = sorted(pool.map(one, records))
            wall = time.perf_counter() - start
        return {'requests_per_s': len(records) / wall, 'p50_ms': statistics.median(latencies),
                'p99_ms': latencies[int(0.99 * (len(latencies) - 1))]}

    phases = []
    for indexed in (False, True, True, False):
        model.graphs = None
        gc.collect(); empty_cache('cuda')
        model.lm.indexed_delta_states = indexed
        model.graphs = CudaGraphs(model.lm, model.pad_id)
        server = Server(ck, tok, model, 'cuda', release_date='-')
        first = {n: run(n) for n in (32, 64)}
        server.wait_idle()
        result = {'indexed': indexed, 'first': first, 'steady': {n: run(n) for n in (32, 64)}, 'graphs': model.graphs.stats()}
        phases.append(result)
        server.close()
        print(result, flush=True)
    return phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--run', default='jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52')
    ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--indexed-first', action='store_true', help='reverse the measurement order to check order effects')
    ap.add_argument('--mixed-throughput', action='store_true', help='investigate scheduler throughput with an ABBA comparison')
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ck = Checkpoint(a.run)
    tok, model = ck.load('cuda', LoadOptions(dtype=torch.bfloat16, fused=True, cuda_graphs=True))
    report = {'run': a.run, 'environment': kernel_environment(model, 'cuda'), 'cuda': torch.version.cuda,
              'sources': source_hashes(), 'indexed_first': a.indexed_first, 'cases': {}}
    if a.mixed_throughput:
        report['mixed_throughput'] = mixed_throughput(model, tok, ck)
        write_json(Path(a.out) / 'report.json', report)
        return
    # Multiple identical requests share a bank entry; ticket numbers make distinct documents.
    cases = [(1, 1, False, 'warm'), (16, 1, False, 'warm'), (3, 3, False, 'warm'),
             (16, 16, False, 'warm'), (16, 16, True, 'warm'), (16, 16, False, 'cold'), (16, 16, False, 'mixed')]
    for requests, documents, long_branch, cache in cases:
        name = f'{requests}requests-{documents}docs-long{int(long_branch)}-{cache}'
        records = []
        for i in range(requests):
            req = request('6 questions, short state', 1 + i % documents)
            if long_branch:
                req = req.model_copy(deep=True)
                # Lengthen every question without changing its options.
                for q in req.questions.values(): q.instructions = (q.instructions or '') + ' Read the customer message carefully.' * 20
            records.append(admit(model, tok, to_record(req)[0]))
        outputs = {}
        result = {}
        for indexed in ((True, False) if a.indexed_first else (False, True)):
            label = 'indexed' if indexed else 'gathered'
            model.lm.indexed_delta_states = indexed
            model.graphs = None
            gc.collect(); empty_cache('cuda')
            model.graphs = CudaGraphs(model.lm, model.pad_id)
            start = time.perf_counter()
            _, prefixes = model.probs_batch(records, [None] * requests, [True] * requests)
            prefixes = [p if cache == 'warm' or (cache == 'mixed' and i % 2) else None for i, p in enumerate(prefixes)]
            for _ in range(3):
                model.probs_batch(records, prefixes, [False] * requests)
                model.graphs.capture_pending()
            sync('cuda')
            first_seconds = time.perf_counter() - start
            samples, gpu_samples = [], []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(a.reps):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start = time.perf_counter(); begin.record()
                probs, _ = model.probs_batch(records, prefixes, [False] * requests)
                end.record(); end.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
                gpu_samples.append(begin.elapsed_time(end))
            outputs[label] = [p.cpu() for ps in probs for p in ps]
            result[label] = {'wall_ms': samples, 'gpu_ms': gpu_samples, 'median_ms': statistics.median(samples),
                'median_gpu_ms': statistics.median(gpu_samples), 'requests_per_s': requests * 1000 / statistics.median(samples),
                'peak_allocated_bytes': allocated_bytes('cuda'), 'first_seconds': first_seconds, 'graphs': model.graphs.stats()}
            del prefixes
        differences = [float((p - q).abs().max()) for p, q in zip(outputs['gathered'], outputs['indexed'])]
        result['parity'] = {'questions': len(differences), 'max_dp': max(differences),
                            'flips': sum(int(p.argmax() != q.argmax()) for p, q in zip(outputs['gathered'], outputs['indexed']))}
        report['cases'][name] = result
        print(name, result, flush=True)
        write_json(Path(a.out) / 'report.json', report)
        if result['parity']['max_dp'] != 0:
            raise AssertionError(f'Indexed states changed outputs: {result["parity"]}')


if __name__ == '__main__': main()
