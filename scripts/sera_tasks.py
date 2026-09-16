"""Use the persistent numerical learner through the existing bounded supervisor."""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("init", "teach", "predict", "observe", "status", "migrate"))
    parser.add_argument("--directory", default="runs/sera-task-transfer")
    parser.add_argument("--task")
    parser.add_argument("--group", default="default")
    parser.add_argument("--csv")
    parser.add_argument("--tolerance", type=float, default=.05)
    parser.add_argument("--noise", type=float, default=0.)
    parser.add_argument("--seconds", type=float, default=30.)
    args = parser.parse_args()
    if args.operation in ("teach", "predict", "observe") and (not args.task or not args.csv):
        parser.error("This operation needs --task and --csv")
    output = ROOT/"runs"/("task-command-"+str(uuid.uuid4()))
    command = [sys.executable, str(ROOT/"scripts/run_task_transfer_bounded.py"),
               "--seconds", str(args.seconds), "--output", str(output),
               "--module", "experiments.task_transfer.runtime", "--", args.operation,
               "--directory", args.directory, "--group", args.group,
               "--tolerance", str(args.tolerance), "--noise", str(args.noise)]
    if args.task:
        command += ["--task", args.task]
    if args.csv:
        command += ["--csv", args.csv]
    result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    log = output/"process.log"
    if log.exists():
        print(log.read_text(encoding="utf-8").rstrip())
    else:
        print((result.stderr or result.stdout).rstrip(), file=sys.stderr)
    state = output/"state.json"
    if state.exists():
        receipt = json.loads(state.read_text())
        print(json.dumps({"supervision": receipt["status"], "charged_seconds": receipt["charged_seconds"],
                          "receipt": str(state)}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
