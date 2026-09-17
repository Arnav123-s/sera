"""Use the saved learner to identify a request and its entities."""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("initialize", "ask", "result"))
    parser.add_argument("--id")
    parser.add_argument("--text")
    parser.add_argument("--checkpoint")
    parser.add_argument("--store", default="runs/sera-requests-live")
    parser.add_argument("--seconds", type=float, default=15.)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.operation in ("ask", "result") and not args.id:
        parser.error("Supply --id")
    if args.operation == "ask" and not args.text:
        parser.error("Supply --text")
    if args.operation == "initialize" and not args.checkpoint:
        parser.error("Supply --checkpoint")
    folder = ROOT / "runs" / ("request-command-" + str(uuid.uuid4()))
    extra = [args.operation, "--store", args.store]
    for name in ("id", "text", "checkpoint"):
        value = getattr(args, name)
        if value is not None:
            extra.extend(("--" + name, value))
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_stream_curriculum_bounded.py"),
        "--seconds", str(args.seconds), "--output", str(folder),
        "--module", "experiments.stream_curriculum.runtime", "--", *extra,
    ], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    log = folder / "process.log"
    raw = log.read_text(encoding="utf-8") if log.exists() else result.stderr + result.stdout
    if result.returncode:
        print(raw)
    else:
        answer = json.loads(raw.strip().splitlines()[-1])
        if args.json or "intent" not in answer:
            print(json.dumps(answer, indent=2, ensure_ascii=False))
        else:
            print("Request:", answer["text"])
            print("Learned intent:", answer["intent"].replace("_", " "))
            for entity in answer["entities"]:
                print(entity["type"].replace("_", " ") + ":", entity["value"])
            print("Model: saved request learner")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
