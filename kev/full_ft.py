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

What a run writes besides its final checkpoint: resume points (ResumeWriter: the fp32 optimizer state, ~307 GB for a
27B, replaced as the run goes and removed at its end) and snapshots (SnapshotWriter: loadable bf16 checkpoints at
registered steps, the final checkpoint's files, ~51 GB for a 27B, kept).
"""
import copy
import datetime
import math
import os
import shutil
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim.adamw import adamw

from .device import sync
from .suite import read_json, write_json

MIN_TORCH = (2, 8)   # FSDPModule.set_gradient_divide_factor (shard) and the FSDP2 behaviour measured in PR #122 / #125


def unsupported_torch(version=torch.__version__):
    """Why this torch cannot train full weights, or None. pyproject allows torch >= 2.6; kev.full_ft needs MIN_TORCH."""
    have = tuple(int(part) for part in version.split("+")[0].split(".")[:2])
    if have < MIN_TORCH:
        return f"--full_ft 1 needs torch >= {'.'.join(map(str, MIN_TORCH))} (FSDPModule.set_gradient_divide_factor); this is torch {version}"
    return None


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
        if not math.isfinite(self.grad_norm):   # a NaN/inf gradient, even from a finite loss, stops here before any master moves (on every rank: the norm is global); min(1.0, nan) would otherwise apply it unclipped
            raise RuntimeError(f"non-finite gradient norm {self.grad_norm}; refusing the optimizer step")
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

    def state_dict(self):
        """This rank's optimizer state for a resume point: the group hyperparameters (the lr scheduler moves them) and, per
        parameter in order, its fp32 master, moments and step. The bf16 weights are not saved: they are bf16(master)."""
        return {"groups": [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups],
                "state": [self.state[p] for g in self.param_groups for p in g["params"]]}

    @torch.no_grad()
    def load_state_dict(self, saved):
        """Restore state_dict() in place (torch.optim.Optimizer's own would cast the fp32 masters to the parameters' bf16)
        and write bf16(master) back into the weights, which reproduces them bit for bit."""
        params = [p for g in self.param_groups for p in g["params"]]
        if len(saved["state"]) != len(params) or len(saved["groups"]) != len(self.param_groups):
            raise ValueError("resume point does not match this model's parameters")
        for group, hyper in zip(self.param_groups, saved["groups"]): group.update(hyper)
        for p, s in zip(params, saved["state"]):
            for key, value in s.items():
                if self.state[p][key].shape != value.shape: raise ValueError(f"resume point: {key} shape {tuple(value.shape)} for {tuple(self.state[p][key].shape)}")
                self.state[p][key].copy_(value)
            local(p).copy_(self.state[p]["master"])

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
    so every rank runs the same number of micro-batches and FSDP's collectives line up (as DistributedSampler does). The
    padding repeats the first items of the shuffled order, whatever their length (--length_sort 1 does not come through
    here: train.microbatch_plan pads each step from its shuffled records before cutting them)."""
    if world == 1: return items
    padded = items + items[: -len(items) % world]
    return padded[rank::world]


def global_sum(values):
    """Per-rank counters summed over the ranks (as they are on one process)."""
    if not dist.is_initialized(): return values
    total = torch.tensor(values, dtype=torch.float64, device="cuda" if dist.get_backend() == "nccl" else "cpu")
    dist.all_reduce(total)
    return total.tolist()


def global_max(values):
    """Per-rank values, the largest over the ranks (every rank calls it: a collective)."""
    if not dist.is_initialized(): return values
    top = torch.tensor(values, dtype=torch.float64, device="cuda" if dist.get_backend() == "nccl" else "cpu")
    dist.all_reduce(top, op=dist.ReduceOp.MAX)
    return top.tolist()


# --- resume points ----------------------------------------------------------------------------------------------------

LATEST = "latest.json"   # the resume point to continue from; written last (atomically) by rank 0, after every rank's file
WRITE_SHARE = 0.05       # at most this share of wall time may block on writing resume points
PEER_WAIT = 600          # seconds rank 0 waits, after writing its own file, for the other ranks' (they write in parallel)


def rank0_decides(value):
    """`value` as rank 0 sees it, on every rank (a decision every rank must act on together); itself on one process."""
    if not dist.is_initialized(): return value
    flag = torch.tensor(float(value), device="cuda" if dist.get_backend() == "nccl" else "cpu")
    dist.broadcast(flag, 0)
    return bool(flag)


def save_due(step, every_steps, every_minutes, since, blocked):
    """Whether to write a resume point after this optimizer step: every `every_steps` steps, or once `every_minutes`
    have passed since `since`, stretched so the time training has blocked on writes (`blocked` seconds per point) stays
    under WRITE_SHARE of the interval. Under torchrun rank 0's clock decides for every rank."""
    interval = max(60 * every_minutes, blocked / WRITE_SHARE)
    return rank0_decides(bool(every_steps and step % every_steps == 0) or bool(every_minutes and time.time() - since >= interval))


class ResumeWriter:
    """Writes resume points: each rank's optimizer state (its shard's fp32 masters and moments), the lr scheduler and the
    RNG states to `step-N/rank<r>.pt`, then rank 0's `latest.json` with the position (the data order, counters, the
    arguments it must be resumed with), then removes earlier points. A 27B's state is ~307 GB and the runs volume writes it
    at well under 1 GB/s, so where the state lives on the GPUs (FSDP2) `save` copies it to host memory and a thread
    writes it while training goes on (the next `save`, and `wait`, join it first); where it already lives in host memory
    (one GPU, offload) there is no room for a copy and `save` writes it before returning.
    `after`: the SnapshotWriter; latest.json waits for a snapshot still being written (and is not written if it failed),
    so a resume point is never committed before the snapshots of the steps it has passed: a continuation from it could
    not write them again."""

    def __init__(self, resume_dir, background):
        self.dir, self.background, self.thread, self.host, self.error = Path(resume_dir), background, None, None, None
        self.seconds = []   # how long each point took to write, in the background or not
        self.rank, self.world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)

    def save(self, step, opt, sched, position, after=None):
        self.wait()
        state = opt.state_dict()
        if self.background:   # host copies, allocated once and reused, so the optimizer may move on
            tensors = [t for s in state["state"] for t in s.values()]
            if self.host is None: self.host = [torch.empty(t.shape, dtype=t.dtype, device="cpu") for t in tensors]
            for h, t in zip(self.host, tensors): h.copy_(t, non_blocking=True)
            sync(tensors[0].device.type)
            copies = iter(self.host)
            state = {"groups": state["groups"], "state": [{k: next(copies) for k in s} for s in state["state"]]}
        rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}
        payload = {"optimizer": state, "scheduler": copy.deepcopy(sched.state_dict()), "rng": rng}
        if self.background:
            self.thread = threading.Thread(target=self._write, args=(step, payload, position, after), daemon=True); self.thread.start()
        else:
            self._write(step, payload, position, after)

    def wait(self):
        """Join the background write; its error, if any, is raised here."""
        if self.thread is not None: self.thread.join(); self.thread = None
        if self.error is not None: raise RuntimeError("writing the resume point failed") from self.error

    def _write(self, step, payload, position, after=None):
        started = time.time()
        try: self._write_point(step, payload, position, after); self.seconds.append(round(time.time() - started, 1))
        except BaseException as error:
            self.error = error
            if not self.background: raise

    def _write_point(self, step, payload, position, after=None):
        target = self.dir / f"step-{step:07d}"
        target.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target / f".rank{self.rank}.pt.tmp")
        os.replace(target / f".rank{self.rank}.pt.tmp", target / f"rank{self.rank}.pt")
        if self.rank: return
        deadline = time.time() + PEER_WAIT
        while missing := [r for r in range(self.world) if not (target / f"rank{r}.pt").exists()]:   # the ranks share one filesystem
            if time.time() > deadline:
                raise TimeoutError(f"resume point {target.name}: rank(s) {missing} did not finish writing within {PEER_WAIT // 60} min of rank 0; "
                                   "latest.json still names the previous point")
            time.sleep(2)
        if after is not None and (error := after.join()) is not None:   # the training thread raises it too (SnapshotWriter.wait)
            raise RuntimeError(f"a snapshot failed to write, so resume point {target.name} is not made the latest: a continuation from "
                               "the previous one writes that snapshot again") from error
        write_json(self.dir / LATEST, {"dir": target.name, "step": step, **position}, atomic=True)
        for old in self.dir.glob("step-*"):   # only earlier points: a rank done with this one may have started the next
            if int(old.name.removeprefix("step-")) < step: shutil.rmtree(old)


def load_resume(resume_dir, opt, sched, args):
    """-> the saved position (or None when there is no resume point), after restoring this rank's optimizer state, the
    weights, the scheduler and the RNG states. Refuses a point written with other arguments or another world size."""
    resume_dir = Path(resume_dir)
    if not (resume_dir / LATEST).exists(): return None
    position = read_json(resume_dir / LATEST)
    world, rank = dist.get_world_size() if dist.is_initialized() else 1, dist.get_rank() if dist.is_initialized() else 0
    if position["world"] != world or position["args"] != args:
        changed = sorted(k for k in set(args) | set(position["args"]) if args.get(k) != position["args"].get(k))
        raise ValueError(f"resume point {resume_dir} was written by {position['world']} rank(s) with other arguments: {changed or 'world size'}")
    saved = torch.load(resume_dir / position["dir"] / f"rank{rank}.pt", map_location="cpu", mmap=True, weights_only=True)
    opt.load_state_dict(saved["optimizer"]); sched.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["rng"]["torch"])
    if saved["rng"]["cuda"] is not None: torch.cuda.set_rng_state(saved["rng"]["cuda"])
    return position


def gather_backbone(lm):
    """Under FSDP2: the bf16 backbone's full state dict, gathered into rank 0's host memory (every rank must call this;
    the others get {}). On one process None: save_pretrained reads the model itself."""
    if not dist.is_initialized(): return None
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    return get_model_state_dict(lm, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def write_backbone(lm, out, state):
    """save_pretrained of the bf16 backbone: config.json + model-*.safetensors (+ index); `state` from gather_backbone
    (transformers strips the FSDP class prefix)."""
    lm.save_pretrained(out, state_dict=state, max_shard_size=SHARD_SIZE)


def save_backbone(lm, out):
    """The final checkpoint's backbone. Under FSDP2 every rank joins the gather and rank 0 writes while the others wait."""
    state = gather_backbone(lm)
    if not dist.is_initialized() or dist.get_rank() == 0: write_backbone(lm, out, state)
    if dist.is_initialized(): dist.barrier()


# --- snapshots --------------------------------------------------------------------------------------------------------

SNAPSHOT_INFO = "snapshot.json"   # written last (atomically) into a snapshot's checkpoint directory: the snapshot is complete


def snapshot_fractions(text):
    """--snapshot_fractions "0.25,0.5,0.75" -> (0.25, 0.5, 0.75); "" or "none" -> (). Each strictly between 0 and 1 (the
    final checkpoint is always written)."""
    if str(text).strip().lower() in ("", "none"): return ()
    try: values = tuple(sorted({float(part) for part in str(text).split(",")}))
    except ValueError: raise ValueError(f"snapshot fractions are comma-separated numbers, or none: {text!r}") from None
    if not all(0 < v < 1 for v in values): raise ValueError(f"snapshot fractions must lie strictly between 0 and 1: {text!r}")
    return values


def snapshot_steps(total, fractions=(), every=0):
    """The optimizer steps after which a run of `total` steps writes a snapshot: the first step at or past each fraction
    of `total`, and every `every` steps; never the last step (that is the final checkpoint)."""
    steps = {math.ceil(round(f * total, 6)) for f in fractions} | (set(range(every, total, every)) if every else set())
    return sorted(s for s in steps if 0 < s < total)


def too_many_snapshots(fractions=(), every=0, total=None):
    """Why this snapshot plan exceeds kev.budget.MAX_SNAPSHOTS (the disk a run's kept snapshots may take), or None.
    `total`: the run's optimizer steps, or an upper bound on them (a trial's max_steps); None = unknown, which an
    every-N plan cannot be checked against."""
    from .budget import MAX_SNAPSHOTS
    if every and total is None:
        return "snapshot_every_steps needs the run's step count to be bounded (max_steps), so its snapshot count can be checked"
    count = len(snapshot_steps(total, fractions, every)) if total is not None else len(fractions)
    if count > MAX_SNAPSHOTS:
        return (f"{count} snapshots planned (fractions {list(fractions)}, every {every} of {total} steps): at most kev.budget.MAX_SNAPSHOTS = "
                f"{MAX_SNAPSHOTS} fit the container disk next to the resume points (a full disk fails the trial)")
    return None


def snapshot_path(root, step):
    return Path(root) / f"step-{step:07d}" / "checkpoint"   # zero-padded like resume points (ResumeWriter)


def completed_snapshot_dirs(root):
    """{step: checkpoint directory} of the complete snapshots under `root` (a snapshot is complete once its SNAPSHOT_INFO
    exists); step-<N> directories are read by number, padded or not (the first snapshots, before padding, were not)."""
    return {int(p.parent.parent.name.removeprefix("step-")): p.parent for p in Path(root).glob(f"step-*/checkpoint/{SNAPSHOT_INFO}")}


def completed_snapshots(root):
    """Steps of the complete snapshots under `root`."""
    return sorted(completed_snapshot_dirs(root))


class SnapshotWriter:
    """Loadable bf16 checkpoints during a full-weight run, at registered optimizer steps, into `<root>/step-<N>/checkpoint`:
    exactly the final checkpoint's files (save_pretrained shards + head.pt + tokenizer) plus SNAPSHOT_INFO, written last,
    which marks the snapshot complete. Snapshots are kept (never deleted); a complete one is never written again (a
    continued run skips it) and an incomplete one (a write the container did not finish) is replaced.
    Under FSDP2 every rank joins the gather into rank 0's host memory (the only time training blocks), then rank 0 writes
    from that copy in a thread while training goes on; the next `save`, `wait` and a resume point's latest.json join it
    first. On one GPU the host holds the masters and there is no room for a copy, so `save` writes before returning."""

    def __init__(self, root, steps, background):
        self.root, self.steps, self.background, self.thread, self.error = Path(root), set(steps), background, None, None
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.written = []   # this process's snapshots: {"step", "blocking_seconds", "write_seconds"}

    def due(self, step):
        """Whether to write a snapshot after this step: a registered step without a complete snapshot (rank 0 decides)."""
        return step in self.steps and rank0_decides(step not in completed_snapshots(self.root))

    def missed(self, step):
        """Registered steps up to `step` (a resume point) without a complete snapshot: a continuation cannot write them."""
        done = set(completed_snapshots(self.root))
        return sorted(s for s in self.steps if s <= step and s not in done)

    def save(self, step, lm, finish, info):
        """Write the snapshot of this step: the backbone, then finish(directory) (head.pt and the tokenizer, as for the
        final checkpoint), then SNAPSHOT_INFO with `info` and the time it took."""
        self.wait()
        started = time.time()
        state = gather_backbone(lm)
        if self.rank: return
        blocked = time.time() - started
        if self.background:
            self.thread = threading.Thread(target=self._write, args=(step, lm, state, finish, info, blocked), daemon=True); self.thread.start()
        else:
            self._write(step, lm, state, finish, info, blocked)

    def join(self):
        """Wait for the snapshot being written, if any (any thread may); -> its error or None."""
        thread = self.thread
        if thread is not None: thread.join()
        return self.error

    def wait(self):
        """Join the background write; its error, if any, is raised here, on every rank at once (every rank calls wait:
        in save and at the end of training), so ranks 1..N do not go on into the next gather and hang on rank 0."""
        self.join(); self.thread = None
        if rank0_decides(self.error is not None):
            raise RuntimeError("writing a snapshot failed" + ("" if self.error else " on rank 0")) from self.error

    def _write(self, step, lm, state, finish, info, blocked):
        started = time.time()
        try:
            target = snapshot_path(self.root, step)
            if (target / SNAPSHOT_INFO).exists():   # complete (a racing writer, or an earlier attempt): never deleted or rewritten
                print(f"snapshot step {step}: {target} is complete already; left as it is", flush=True); return
            if target.exists(): shutil.rmtree(target)   # an incomplete write
            target.mkdir(parents=True)
            write_backbone(lm, target, state)
            finish(target)
            timing = {"step": step, "blocking_seconds": round(blocked + (0 if self.background else time.time() - started), 1),
                      "write_seconds": round(time.time() - started, 1)}
            write_json(target / SNAPSHOT_INFO, {**info, **timing}, atomic=True)
            self.written.append(timing)
            print(f"snapshot step {step}: {target} (training blocked {timing['blocking_seconds']} s, written in {timing['write_seconds']} s)", flush=True)
        except BaseException as error:
            self.error = error
            if not self.background: raise
