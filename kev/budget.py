"""The resources a Modal trial container asks for and the admission bound on a study's cost: one home for
modal_app.run_trial/admit_study and kev.rounds.validate. Pure Python (no torch, no modal)."""
TRIAL_CPU, TRIAL_MEMORY = 4, (65536, 196608)   # cores, (request, limit) MiB
GPU_HOURLY = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "T4": 0.59}   # USD per GPU hour


def compute_bound(gpu, timeout, trials):
    """The most `trials` containers of `gpu` can cost if each runs to `timeout` seconds (GPU + CPU + memory rates)."""
    if gpu not in GPU_HOURLY or timeout <= 0 or trials < 1:
        raise ValueError("invalid GPU, timeout, or trial count")
    return (GPU_HOURLY[gpu] + TRIAL_CPU * 0.04730 + TRIAL_MEMORY[1] / 1024 * 0.008) * timeout / 3600 * trials
