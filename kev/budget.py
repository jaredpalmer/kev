"""The resources a Modal trial container asks for and the admission bound on a study's cost: one home for
modal_app.run_trial/admit_study and kev.rounds.validate. Pure Python (no torch, no modal)."""
TRIAL_CPU, TRIAL_MEMORY = 4, (65536, 196608)   # cores, (request, limit) MiB
GPU_HOURLY = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "T4": 0.59}   # USD per GPU hour
# Full-weight trials (kev.train --full_ft, kev.full_ft). One GPU keeps the fp32 masters and AdamW moments in host memory:
# 12 bytes per parameter, ~310 GB for a 27B, plus the staging buffers and the loader; the CPU runs the AdamW step. Several
# GPUs (FSDP2) keep that state on the GPUs, and host memory only has to hold the gathered checkpoint rank 0 writes (~51 GB).
FULL_FT_SINGLE = 24, (368640, 409600)          # 360 / 400 GiB
FULL_FT_SHARDED = 16, (131072, 262144)         # 128 / 256 GiB


def gpu_count(gpu):
    """Modal's "H200:8" -> 8; a bare type is one GPU."""
    return int(gpu.partition(":")[2] or 1)


def trial_resources(gpu, full_ft=False):
    """(cpu cores, (memory request, limit) MiB) for one trial container."""
    if not full_ft: return TRIAL_CPU, TRIAL_MEMORY
    return FULL_FT_SHARDED if gpu_count(gpu) > 1 else FULL_FT_SINGLE


def compute_bound(gpu, timeout, trials, full_ft=False):
    """The most `trials` containers of `gpu` ("H200", or "H200:8" for 8 in one container) can cost if each runs to
    `timeout` seconds (GPU + CPU + memory rates, at the resources trial_resources gives them)."""
    kind = gpu.partition(":")[0]
    if kind not in GPU_HOURLY or timeout <= 0 or trials < 1 or gpu_count(gpu) < 1:
        raise ValueError("invalid GPU, timeout, or trial count")
    cpu, memory = trial_resources(gpu, full_ft)
    return (GPU_HOURLY[kind] * gpu_count(gpu) + cpu * 0.04730 + memory[1] / 1024 * 0.008) * timeout / 3600 * trials
