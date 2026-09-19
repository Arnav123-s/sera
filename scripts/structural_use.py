"""Use qualified physical models alongside SERA's retained practical routes."""

import argparse
import json
from pathlib import Path

import torch

from experiments.gap_inquiry import ROOT, read, write
from scripts.counterfactual_live import perform as prior_perform
from scripts.structural_study import restore


def perform(session, growth, request):
    kind = request.get("kind")
    if kind == "field_findings":
        result = [{"subject": key, "goal": r["goal"], "observations": len(r["observations"]),
                   "representation": r["selected"], "qualification": r["qualification"],
                   "next_action": r["qualification"]["next_action"] if r["qualification"] else "Acquire independent evidence"}
                  for key, r in session.subjects.items()]
    elif kind in {"field_what_if", "field_plan", "field_explain", "field_trajectory"}:
        subject = request["subject"]
        view = session.view(subject)
        if kind == "field_what_if":
            result = view.imagine(request["state"])
        elif kind == "field_plan":
            result = view.plan(request["state"], request["target"])
        elif kind == "field_trajectory":
            result = view.trajectory(request["state"], request["control"], request.get("steps", 20), request.get("dt", .05))
        else:
            record = session.subjects[subject]
            result = {"kind": "EVIDENCE_RANKED_EXPLANATION", "original_goal": record["goal"],
                      "original_state": record["original_state"], "initial_answer": record.get("initial_answer"),
                      "alternatives": {k: session.model(subject, k).weight.detach().tolist() for k in ("radial", "directional")},
                      "qualification": record["qualification"], "scope": "Candidate descriptions compared with independent simulated or measured evidence",
                      "selected": record["selected"], "model": view.version}
    else:
        return prior_perform(session.base, growth, request)
    return {"id": request.get("id"), "kind": kind, "owner": session.identity(), "result": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    audit = read(ROOT / "research-continuation/46_structural_refinement/audit.json")
    session, growth = restore()
    if not audit["passed"] or session.identity() != audit["owner"]:
        raise ValueError("Use the independently qualified structural owner")
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Use a fresh owned output")
    tasks = read(a.input)
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 256:
        raise ValueError("Declare a finite request batch")
    before = session.state()
    results = [perform(session, growth, task) for task in tasks]
    if session.state() != before:
        raise ValueError("A hypothetical query changed learned state or observations")
    write(a.output, results)
    print(json.dumps({"owner": session.identity(), "requests": len(results), "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
