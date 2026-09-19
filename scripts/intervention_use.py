"""Use acquired intervention models alongside all retained SERA task routes."""

import argparse
import json
from pathlib import Path

import torch

from experiments.gap_inquiry import ROOT, read, write
from scripts.intervention_study import restore
from scripts.structural_use import perform as prior_perform


def perform(session, growth, request):
    kind = request.get("kind")
    if kind == "intervention_findings":
        result = [{"subject": subject, "original_goal": r["goal"], "status": r["status"],
                   "selected": r["qualification"]["selected"] if r["qualification"] else None,
                   "observations": len(r["observations"]),
                   "next_action": r["qualification"]["next_action"] if r["qualification"] else "Acquire an independent distinguishing outcome"}
                  for subject, r in session.subjects.items()]
    elif kind == "intervention_what_if":
        result = session.answer(request["subject"], request.get("program"))
    elif kind == "intervention_explain":
        subject = request["subject"]
        record = session.subjects[subject]
        answer = session.answer(subject)
        q = record["qualification"]
        result = {"kind": "EVIDENCE_RANKED_EXPLANATION", "original_goal": record["goal"],
                  "original_program": record["original_program"], "initial_alternatives": record["initial"],
                  "current_answer": answer,
                  "learned_parameters": {k: session.model(subject, k).weight.detach().tolist() for k in ("temporal", "pulse_loss")},
                  "parameter_meanings": ["irreversible rate per second", "static detuning width per second", "extra loss per applied pulse"],
                  "actuation_check": None if q is None else q["actuation"],
                  "selection": None if q is None else q["selection_scores"],
                  "independent_adequacy": None if q is None else q["adequacy"],
                  "scope": "Evidence compares supplied mechanisms under monitored interventions; alternative untested mechanisms remain recorded"}
    elif kind == "intervention_plan":
        subject = request["subject"]
        candidates = request["programs"]
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 64:
            raise ValueError("Declare a finite set of alternative pulse programs")
        branches = [session.answer(subject, p) for p in candidates]
        eligible = [b for b in branches if b["status"] == "QUALIFIED_SCOPE"]
        result = {"kind": "MODEL_BASED_PLAN", "objective": "maximize plus-X probability conditional on the recorded histories occurring",
                  "branches": branches, "selected": max(eligible, key=lambda b: b["plus_probability"]) if eligible else None,
                  "executed_actions": 0, "execution_requires": "independent confirmation of the applied pulse history"}
    else:
        return prior_perform(session.base, growth, request)
    return {"id": request.get("id"), "kind": kind, "owner": session.identity(), "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Use a fresh owned output path")
    audit = read(ROOT / "research-continuation/47_intervention_understanding/audit.json")
    session, growth = restore()
    if not audit["passed"] or audit["owner"] != session.identity():
        raise ValueError("Use the independently qualified intervention successor")
    tasks = read(args.input)
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 256:
        raise ValueError("Declare a finite request batch")
    before = session.state()
    results = [perform(session, growth, request) for request in tasks]
    if session.state() != before:
        raise ValueError("A hypothetical request changed factual evidence or weights")
    write(args.output, results)
    print(json.dumps({"owner": session.identity(), "requests": len(results), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
