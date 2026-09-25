"""The resources a Modal trial container asks for and the admission bound on a study's cost: one home for
modal_app.run_trial/admit_study and kev.rounds.validate. Pure Python (no torch, no modal)."""
TRIAL_CPU, TRIAL_MEMORY = 4, (65536, 196608)   # cores, (request, limit) MiB
GPU_HOURLY = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "T4": 0.59}   # USD per GPU hour
# Full-weight trials (kev.train --full_ft, kev.full_ft). One GPU keeps the fp32 masters and AdamW moments in host memory:
# 12 bytes per parameter, ~310 GB for a 27B, plus the staging buffers and the loader; the CPU runs the AdamW step. Several
# GPUs (FSDP2) keep that state on the GPUs; host memory holds the copy of it a resume point is written from in the
# background (~307 GB over the ranks) and the gathered checkpoint rank 0 writes (~51 GB; a snapshot being written holds
# the same, one at a time). Disk: the runs volume stages what a container writes on its local disk, up to two resume
# points (the new one is complete before the old one goes), the checkpoint and the snapshots (kept: 3 by default, ~51 GB
# each for a 27B): 2 x 307 + 4 x 51 ≈ 820 GB, beyond Modal's default 512 GiB quota and inside 1 TiB with ~280 GB spare
# (about five more 27B snapshots, if the volume keeps committed files staged).
FULL_FT_SINGLE = 24, (368640, 409600)          # 360 / 400 GiB
FULL_FT_SHARDED = 16, (409600, 471040)         # 400 / 460 GiB
FULL_FT_DISK = 1048576                         # MiB of ephemeral disk (1 TiB); Modal bills disk as memory at 20:1
# A study's limits. Full-weight trials may run to Modal's 24 h cap per attempt and are retried after a timeout (each
# retry continues from the trial's last resume point, kev.experiment.continue_trial), so their bound counts every attempt
# and their budget cap covers one 8 x H200 day (a LoRA study keeps the $250 cap).
MAX_TIMEOUT = {False: 28800, True: 86400}   # a 24 h attempt fits the $1,000 cap on one H200 (3 x 24 h x ~$11/h); on
                                             # H200:8 (~$41/h) the cap allows ~8 h attempts, a day with the retries
MAX_BUDGET = {False: 250, True: 1000}
FULL_FT_RETRIES = 2
CPU_HOURLY, MEMORY_GIB_HOURLY = 0.04730, 0.008   # USD per core hour and per GiB hour (memory billed at the limit here)
# Checkpoint interpolation (scripts/interpolate_checkpoint.py, modal_app.py::interpolate): CPU only. The base backbone stays
# resident in bf16 (51 GB for Qwen3.8-27B, 25.6B parameters; loading it through transformers also stages the 2.5 GB LM
# head); the SFT side streams tensor by tensor, one output shard (<= 5 GB) is held until written, and the fp32 arithmetic
# runs in 16M-element chunks (~200 MB). Peak ~60 GB; the limit leaves room for the loader's transients.
INTERPOLATE_CPU, INTERPOLATE_MEMORY, INTERPOLATE_TIMEOUT = 8, (81920, 131072), 10800   # cores, 80 / 128 GiB, 3 h


def gpu_count(gpu):
    """Modal's "H200:8" -> 8; a bare type is one GPU."""
    return int(gpu.partition(":")[2] or 1)


def trial_resources(gpu, full_ft=False):
    """(cpu cores, (memory request, limit) MiB) for one trial container."""
    if not full_ft: return TRIAL_CPU, TRIAL_MEMORY
    return FULL_FT_SHARDED if gpu_count(gpu) > 1 else FULL_FT_SINGLE


def trial_disk(full_ft=False):
    """ephemeral_disk (MiB) for one trial container; None keeps Modal's default."""
    return FULL_FT_DISK if full_ft else None


def hourly_rate(gpu, full_ft=False):
    """USD per hour of one trial container of `gpu` ("H200", or "H200:8" for 8 in one container): GPU + CPU + memory at
    the resources trial_resources gives it."""
    kind = gpu.partition(":")[0]
    if kind not in GPU_HOURLY or gpu_count(gpu) < 1: raise ValueError(f"invalid GPU {gpu!r}")
    cpu, memory = trial_resources(gpu, full_ft)
    return GPU_HOURLY[kind] * gpu_count(gpu) + cpu * CPU_HOURLY + (memory[1] + (trial_disk(full_ft) or 0) / 20) / 1024 * MEMORY_GIB_HOURLY


def interpolation_bound(timeout=INTERPOLATE_TIMEOUT, jobs=1):
    """The most `jobs` interpolation containers (INTERPOLATE_CPU, INTERPOLATE_MEMORY; no GPU, no retries) can cost at
    `timeout` seconds each."""
    if timeout <= 0 or jobs < 1: raise ValueError("invalid timeout or job count")
    return (INTERPOLATE_CPU * CPU_HOURLY + INTERPOLATE_MEMORY[1] / 1024 * MEMORY_GIB_HOURLY) * timeout / 3600 * jobs


def compute_bound(gpu, timeout, trials, full_ft=False):
    """The most `trials` containers can cost if each runs to `timeout` seconds, counting a full-weight trial's retries."""
    if timeout <= 0 or trials < 1: raise ValueError("invalid GPU, timeout, or trial count")
    return hourly_rate(gpu, full_ft) * timeout / 3600 * trials * (1 + FULL_FT_RETRIES if full_ft else 1)
