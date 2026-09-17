"""Classify a JSONL request inbox and extract entities with the saved SERA learner."""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--store", default="runs/sera-requests-live")
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    worker = ROOT / "runs" / ("annotation-command-" + str(uuid.uuid4()))
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_stream_curriculum_bounded.py"),
        "--seconds", str(args.seconds), "--output", str(worker),
        "--module", "experiments.stream_curriculum.batch", "--",
        "--input", str(Path(args.input).resolve()), "--output", str(Path(args.output).resolve()),
        "--store", str((ROOT / args.store).resolve()),
    ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    path = worker / "process.log"
    text = path.read_text(encoding="utf-8") if path.exists() else result.stderr + result.stdout
    if result.returncode:
        print(text)
    else:
        summary = json.loads(text.strip().splitlines()[-1])
        print(f"Annotated {summary['predictions']} requests; {summary['input_errors']} input errors.")
        print("Saved:", Path(args.output).resolve())
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
