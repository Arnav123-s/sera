"""Run the inspected immutable packet verifier under the outer resource cap."""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "research/intake/v5-compact-20260916/expanded/SERA_Research_v5"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--max-jobs", type=int, default=8)
    a = p.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PACK / "code"))
    runner = runpy.run_path(str(PACK / "code/run_checks.py"))
    runner["run"](str(Path(a.output).resolve()), a.max_jobs)
