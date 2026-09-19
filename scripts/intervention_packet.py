"""Replay the supplied v18 packet without modifying its source or retraining it."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/intake/v18-understanding/SERA_v18"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    parser.add_argument("--portable", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--tail", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, "-X", "utf8", str(PACKET / "verify.py"), "--output", str(args.output.resolve())]
    if args.portable:
        command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/intervention_packet_portable.py"), "--output", str(args.output.resolve())]
    if args.diagnose:
        command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/intervention_packet_diagnosis.py"), "--output", str(args.output.resolve())]
    if args.tail:
        command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/intervention_packet_tail.py"), "--output", str(args.output.resolve())]
    if args.jobs:
        command.extend(["--jobs", *args.jobs])
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
