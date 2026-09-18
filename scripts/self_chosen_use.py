"""Inspect or execute the current shared learner's independently certified discoveries."""

import argparse
import json
from pathlib import Path

import torch

from experiments.self_chosen.common import ROOT, read, write
from experiments.self_chosen.runtime import Session
from experiments.verified_completion.credit import decode
from workbench.storage import Store


def restore():
    saved = Store(ROOT / "runs/sera-self-discovery-live").read()
    if not saved or saved["schema"] != "sera.self-chosen-discovery.1":
        raise ValueError("No independently admitted self-discovery owner")
    session = Session(saved["parent"], saved=decode(saved["state"]))
    if session.identity() != saved["owner"]:
        raise ValueError("The retained owner identity changed")
    return session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("catalog", "solve", "tasks"))
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    if not args.output.resolve().is_relative_to(ROOT / "runs") or args.output.exists():
        raise ValueError("Use a fresh owned result file inside runs")
    session = restore()
    if args.action == "catalog":
        result = {"owner": session.identity(), "goal": session.goal, "questions": len(session.inventory),
                  "investigations": len(session.events), "points": sum(e["reward"] for e in session.events),
                  "discoveries": session.records}
    elif args.action == "solve":
        if args.input is None:
            raise ValueError("Provide a JSON list of typed observation tasks")
        requests = read(args.input)
        if not isinstance(requests, list) or not 1 <= len(requests) <= 256:
            raise ValueError("Use a finite list of one to 256 tasks")
        result = [{"id": r.get("id"), "owner": session.identity(), "result": session.solve(r["domain"], r["target"], r["observations"])} for r in requests]
    else:
        if args.input is None:
            raise ValueError("Provide existing language, reading, algebra or imagination tasks")
        from scripts.continuing_use import perform
        result = [perform(session, session.base, r) for r in read(args.input)]
    write(args.output, result)
    print(json.dumps({"output": str(args.output), "owner": session.identity(), "action": args.action}, indent=2))


if __name__ == "__main__":
    main()
