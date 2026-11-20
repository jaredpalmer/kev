"""Use modal_app's image and benchmark functions in a fresh contributor workspace.

Register only this measurement, so unrelated mirroring functions do not require
huggingface-secret. Run with the repository's pinned Modal CLI under Python 3.13.
"""
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/root')
sys.path.insert(0, str(ROOT))
import modal_app as research

app = modal.App('kev-indexed-state-measurements')
image = research.image.add_local_file(ROOT / 'modal_app.py', '/root/modal_app.py')


@app.function(image=image, gpu='H100', cpu=2, memory=(32768, 131072), timeout=3600,
              volumes={research.RUNS_MOUNT: research.runs_volume, research.HF_MOUNT: research.hf_cache})
def measure(kind, name, run, flags):
    import modal_app
    if kind == 'serving': return modal_app.run_serving.local(run, name, flags)
    script = {'compare': 'indexed_state_bench.py', 'gather': 'deltanet_gather_bench.py'}[kind]
    return modal_app.run_script.local(script, name, f'--run {run} {flags}')


@app.local_entrypoint()
def main(kind: str, name: str, flags: str = '', run: str = 'jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52'):
    print(measure.remote(kind, name, run, flags))
