"""Full-weight training (`kev.train --full_ft 1`): the whole text backbone and the pointer head are trained, not a LoRA.

The backbone's working weights are bf16 (what the forward and backward run in, and what the checkpoint stores). Plain
bf16 AdamW updates at lr ~1e-5 round away: a bf16 weight near 0.02 moves in steps of ~1e-4. So MasterAdamW keeps an
fp32 master copy of every weight plus fp32 AdamW moments, updates the masters exactly (torch's fused AdamW) and writes
bf16(master) back after each step. Where those 12 bytes per parameter live is the whole memory plan:

- one GPU (`offload`): in host memory. A 27B backbone is ~51 GB of bf16 weights plus ~51 GB of bf16 gradients on the
  H200, and ~310 GB of masters and moments in host RAM; each step streams one tensor at a time (gradient down, update on
  the CPU, bf16 weights up), so no fp32 gradient copy of the whole model is ever held. Technique from AutoJev
  (github.com/denis-pplx/autojev, src/autojev/optim.py, MIT: fp32 masters and AdamW in host memory beside bf16 weights on
  one H200); the per-tensor streaming and the clipping folded into the host pass are ours.
- several GPUs (torchrun, FSDP2 `shard`): each rank keeps its shard's masters and moments next to the shard on its GPU
  (a 27B over 8 H200s: ~6.4 GB of weights, ~6.4 GB of gradients and ~38 GB of optimizer state per GPU), so the step is a
  GPU kernel and the data-parallel ranks train 8 micro-batches at once.

Gradients accumulate in bf16 over the micro-batches of a step, as AutoJev's did. The pointer head is small and fp32; it
is replicated on every rank and its gradient summed across ranks before the step.
"""
import datetime
import os

import torch
import torch.distributed as dist
from torch.optim.adamw import adamw

SHARD_SIZE = "5GB"   # save_pretrained shards: model-00001-of-000NN.safetensors + model.safetensors.index.json


def local(t):
    """The part of a (possibly FSDP2-sharded, DTensor) tensor this rank owns."""
    return t.to_local() if hasattr(t, "to_local") else t


def sharded(t):
    return hasattr(t, "to_local")


class MasterAdamW(torch.optim.Optimizer):
    """AdamW on fp32 master copies of (bf16) parameters, in host memory (`offload`) or on the parameter's device. The
    param groups hold the model's own parameters, so lr schedulers work unchanged. step() clips to `max_grad_norm` by
    the global gradient norm (across ranks when sharded), then updates one tensor at a time and frees its gradient."""

    def __init__(self, groups, lr, weight_decay, offload, max_grad_norm=1.0, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(groups, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        self.max_grad_norm, self.grad_norm = max_grad_norm, None
        params = [p for g in self.param_groups for p in g["params"]]
        for p in params:
            master = local(p).detach().to("cpu" if offload else local(p).device, torch.float32, copy=True)
            self.state[p] = {"master": master, "exp_avg": torch.zeros_like(master), "exp_avg_sq": torch.zeros_like(master),
                             "step": torch.zeros((), device=master.device)}
        self.pinned = offload and any(p.is_cuda for p in params)
        if self.pinned:   # two pinned staging slots per direction and dtype: tensor i+1 moves while the CPU updates tensor i
            size = lambda dtype: max(local(p).numel() for p in params if p.dtype == dtype)
            self.down = {d: [torch.empty(size(d), dtype=d, pin_memory=True) for _ in range(2)] for d in {p.dtype for p in params}}
            self.up = {d: [torch.empty(size(d), dtype=d, pin_memory=True) for _ in range(2)] for d in self.down}
            self.scratch = torch.empty(max(local(p).numel() for p in params), dtype=torch.float32)
            self.uploads = [None, None]

    def _clip_scale(self, params):
        """Clip coefficient from the global L2 norm of all gradients (torch.nn.utils.clip_grad_norm_'s formula); sharded
        gradients are summed over ranks, replicated ones counted once."""
        sq = lambda ps: sum(torch.linalg.vector_norm(local(p.grad), dtype=torch.float32) ** 2 for p in ps)
        total = sq([p for p in params if sharded(p)]) + 0.0
        if dist.is_initialized() and torch.is_tensor(total): dist.all_reduce(total)
        self.grad_norm = float((total + sq([p for p in params if not sharded(p)])) ** 0.5)
        return min(1.0, self.max_grad_norm / (self.grad_norm + 1e-6)) if self.max_grad_norm else 1.0

    def _fetch(self, p, slot):
        """Start copying p's gradient to host memory; -> a callable returning it as fp32 once the copy has landed."""
        g = local(p.grad)
        if not self.pinned: return lambda: g.float()   # masters beside the weights (or everything on the CPU)
        buf = self.down[g.dtype][slot][:g.numel()].view_as(g)
        buf.copy_(g, non_blocking=True); done = torch.cuda.Event(); done.record()
        def ready():
            done.synchronize()
            return self.scratch[:g.numel()].view_as(g).copy_(buf)
        return ready

    def _put(self, p, master, slot):
        """Write bf16(master) into the working weights (through a pinned slot, asynchronously, when offloaded)."""
        w = local(p)
        if not self.pinned: w.copy_(master); return
        if self.uploads[slot] is not None: self.uploads[slot].synchronize()   # the slot's previous upload has left
        buf = self.up[w.dtype][slot][:w.numel()].view_as(w)
        buf.copy_(master); w.copy_(buf, non_blocking=True)
        self.uploads[slot] = torch.cuda.Event(); self.uploads[slot].record()

    @torch.no_grad()
    def step(self):
        work = [(g, p) for g in self.param_groups for p in g["params"] if p.grad is not None]
        if dist.is_initialized():
            for _, p in work:
                if not sharded(p): dist.all_reduce(p.grad)   # replicated (the pointer head): each rank holds its share
        scale = self._clip_scale([p for _, p in work])
        pending = self._fetch(work[0][1], 0) if work else None
        for i, (group, p) in enumerate(work):
            grad = pending()
            if i + 1 < len(work): pending = self._fetch(work[i + 1][1], (i + 1) % 2)
            if scale < 1: grad.mul_(scale)
            s = self.state[p]
            adamw([s["master"]], [grad], [s["exp_avg"]], [s["exp_avg_sq"]], [], [s["step"]], fused=True, amsgrad=False,
                  beta1=group["betas"][0], beta2=group["betas"][1], lr=group["lr"], weight_decay=group["weight_decay"],
                  eps=group["eps"], maximize=False)
            self._put(p, s["master"], i % 2)
            p.grad = None


# --- several GPUs -----------------------------------------------------------------------------------------------------

def init_distributed(device):
    """-> (rank, world size). Under torchrun (WORLD_SIZE > 1) joins the process group (NCCL on CUDA, gloo on CPU for tests)
    and makes this rank's GPU the current one, so "cuda" means it everywhere; otherwise (0, 1) and nothing is initialised.
    Either way turns on the CUDA allocator's expandable segments before the first allocation: micro-batch lengths vary
    from 200 to 8,000 tokens, and the first 27B probe on one H200 ran out of memory with 6.7 GB reserved but unused."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1: return 0, 1
    if device == "cuda": torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # rank 0 alone writes the gathered 27B checkpoint (~51 GB) while the others wait at a barrier: NCCL's 10 min default is too short
    dist.init_process_group("nccl" if device == "cuda" else "gloo", timeout=datetime.timedelta(minutes=60))
    return dist.get_rank(), world


def shard(model):
    """FSDP2 over the backbone: one unit per decoder layer plus the root (embeddings, final norm). Gradients are summed
    over ranks, not averaged (the trainer already divides each micro-batch's loss by the step's global record count). The
    pointer head stays replicated (MasterAdamW sums its gradient), starting from rank 0's initialisation."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mesh = init_device_mesh(str(model.device).split(":")[0], (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    units = [*model.lm.layers, model.lm]
    for unit in units: fully_shard(unit, mesh=mesh, mp_policy=policy)
    for unit in units: unit.set_gradient_divide_factor(1.0)
    for t in model.head.parameters(): dist.broadcast(t.data, 0)


def rank_share(items, rank, world):
    """This rank's equal share of one epoch's shuffled items: padded (by wrapping around) to a multiple of the world size,
    so every rank runs the same number of micro-batches and FSDP's collectives line up (as DistributedSampler does)."""
    if world == 1: return items
    padded = items + items[: -len(items) % world]
    return padded[rank::world]


def save_backbone(lm, out):
    """save_pretrained of the bf16 backbone: config.json + model-*.safetensors (+ index). Under FSDP2 every rank joins the
    full-state-dict gather (to rank 0's host memory) and rank 0 writes (transformers strips the FSDP class prefix)."""
    if not dist.is_initialized():
        lm.save_pretrained(out, max_shard_size=SHARD_SIZE)
        return
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    state = get_model_state_dict(lm, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if dist.get_rank() == 0:
        lm.save_pretrained(out, state_dict=state, max_shard_size=SHARD_SIZE)
    dist.barrier()
