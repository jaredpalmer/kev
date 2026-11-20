"""CUDA-event measurement of the existing DeltaNet gather+copy expression."""
import argparse
import statistics
from pathlib import Path
import torch
from transformers import AutoConfig
from kev.checkpoint import Checkpoint
from kev.device import sync
from kev.suite import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--run', default='jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52')
    a = ap.parse_args()
    ck = Checkpoint(a.run)
    c = AutoConfig.from_pretrained(ck.meta.base, revision=ck.meta.base_revision).text_config
    layers = sum(t == 'linear_attention' for t in c.layer_types)
    torch.manual_seed(0)
    bank = [torch.randn(16, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
                        device='cuda', dtype=torch.float32) for _ in range(layers)]
    report = {'gpu': torch.cuda.get_device_name(), 'layers': layers, 'state_shape': list(bank[0].shape), 'cases': {}}
    for n in (1, 6, 8, 16, 32, 64, 96, 128):
        dst = [torch.empty((n, *s.shape[1:]), device='cuda', dtype=s.dtype) for s in bank]
        for documents in (1, 16):
            entry = torch.arange(n, device='cuda') % documents
            def body():
                for d, s in zip(dst, bank):
                    d.copy_(s.index_select(0, entry))
            for _ in range(10): body()
            sync('cuda')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): body()
            for _ in range(10): graph.replay()
            samples = []
            for _ in range(7):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100): graph.replay()
                end.record(); end.synchronize()
                samples.append(start.elapsed_time(end) / 100)
            report['cases'][f'{n}_rows_{documents}_documents'] = {'samples_ms': samples, 'median_ms': statistics.median(samples)}
            print(n, documents, report['cases'][f'{n}_rows_{documents}_documents'], flush=True)
            del graph
        del dst
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out) / 'report.json', report)

if __name__ == '__main__': main()
