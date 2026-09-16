"""Ask, investigate, clarify, correct and execute in SERA's local simulator."""
import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("init", "ask", "work", "result", "pause", "resume", "rebase",
                                         "clarify", "observe", "execute", "teach", "status", "migrate"))
    p.add_argument("--store", default="runs/sera-inquiry")
    p.add_argument("--id")
    p.add_argument("--text")
    p.add_argument("--entity", default="orbit")
    p.add_argument("--anchor", choices=("current", "historical"), default="current")
    p.add_argument("--ticks", type=int, default=12)
    p.add_argument("--action", type=int, choices=(-1, 0, 1))
    p.add_argument("--actions", type=int, nargs="+", choices=(-1, 0, 1))
    p.add_argument("--phrase")
    p.add_argument("--seconds", type=float, default=30.)
    a = p.parse_args()
    if a.operation in ("ask", "result", "pause", "resume", "rebase", "clarify", "execute") and not a.id:
        p.error("This operation requires --id")
    if a.operation == "ask" and not a.text:
        p.error("Ask requires --text")
    if a.operation in ("observe", "teach") and a.action is None:
        p.error("This operation requires --action")
    if a.operation == "teach" and not a.phrase:
        p.error("Teach requires --phrase")
    if a.operation == "clarify" and not a.actions:
        p.error("Clarify requires --actions")
    directory = ROOT/"runs"/("inquiry-command-"+str(uuid.uuid4()))
    directory.mkdir()
    request = {k: v for k, v in vars(a).items() if k not in ("store", "seconds") and v is not None}
    request["operation"] = "initialize" if a.operation == "init" else a.operation
    (directory/"request.json").write_text(json.dumps(request, indent=2)+"\n", encoding="utf-8")
    command = [sys.executable, str(ROOT/"scripts/run_language_inquiry_bounded.py"),
               "--seconds", str(a.seconds), "--output", str(directory/"worker"),
               "--module", "experiments.language_inquiry.runtime", "--", "--store", a.store,
               "--request", str(directory/"request.json"), "--output", str(directory/"response.json")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if (directory/"response.json").exists():
        print((directory/"response.json").read_text(encoding="utf-8"))
    elif (directory/"worker/process.log").exists():
        print((directory/"worker/process.log").read_text(encoding="utf-8"))
    else:
        print(result.stderr or result.stdout)
    print("Receipt:", directory/"worker/state.json")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
