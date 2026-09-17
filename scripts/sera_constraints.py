"""Persistent language requests, continuous imagination and local evidence acquisition."""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def readable(response):
    if "jobs" in response:
        return "Saved investigations:\n" + "\n".join(
            f"  {name}: {job['status']} — {job['text']}" for name, job in response["jobs"].items())
    if "paid_observations" in response:
        return f"Learned the location of {response['entity']} from three local simulated sensor readings."
    status = response.get("status", "recorded")
    if status == "stale":
        return "This investigation refers to older evidence. Rebase it before continuing."
    result = response.get("result")
    if isinstance(result, dict):
        finding = ("Found a conditional completion." if result["status"] == "CONDITIONAL_WITNESS"
                   else "No completion found within this model and search budget.")
        return (f"{finding}\nProposed controls: {result['controls']}\n"
                f"Nominal target distance: {result['nominal_distance_m']:.4f} m; "
                f"largest parameter-branch distance: {result['worst_parameter_distance_m']:.4f} m.\n"
                "This uses the acquired motion model and continuous-control assumption. "
                "It does not establish that an event occurred or that the model applies physically.")
    return f"{status}: {response.get('request', 'Local state saved.')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("init", "ask", "work", "solve", "result", "acquire", "rebase", "observe", "status"))
    parser.add_argument("--id")
    parser.add_argument("--text")
    parser.add_argument("--entity", choices=("beacon", "marker", "anchor"))
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--action", type=int, choices=(-1, 0, 1))
    parser.add_argument("--store", default="runs/sera-constraints")
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--json", action="store_true", help="Show the complete machine-readable record")
    args = parser.parse_args()
    if args.operation in ("ask", "work", "solve", "result", "rebase") and not args.id:
        parser.error("This operation requires --id")
    if args.operation == "ask" and not args.text:
        parser.error("Ask requires --text")
    if args.operation == "acquire" and not args.entity:
        parser.error("Acquisition requires --entity")
    if args.operation == "observe" and args.action is None:
        parser.error("Observation requires --action")
    folder = ROOT / "runs" / ("constraint-command-" + str(uuid.uuid4()))
    folder.mkdir()
    request = {k: v for k, v in vars(args).items() if k not in ("store", "seconds", "json") and v is not None}
    request["operation"] = "initialize" if args.operation == "init" else args.operation
    (folder / "request.json").write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    process = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_constraint_inquiry_bounded.py"),
        "--seconds", str(args.seconds), "--output", str(folder / "worker"),
        "--module", "experiments.constraint_inquiry.runtime", "--", "--store", args.store,
        "--request", str(folder / "request.json"), "--output", str(folder / "response.json"),
    ], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    for path in (folder / "response.json", folder / "worker/process.log"):
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            print(raw if args.json or path.name != "response.json" else readable(json.loads(raw)))
            break
    else:
        print(process.stderr or process.stdout)
    print("Receipt:", folder / "worker/state.json")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
