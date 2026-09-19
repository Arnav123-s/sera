"""Use retained investigations and ask checked what-if questions on the shared owner."""

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import torch

from experiments.counterfactual_check import after_discovery_words, validate_event
from experiments.counterfactual_core import admitted, at, operators, propose
from experiments.counterfactual_loop import commit, next_question
from experiments.gap_inquiry import ROOT, digest, read, sha, write
from scripts.counterfactual_study import OUT, RUN, restore_inquiry, verify_freeze
from scripts.solution_use import perform as previous_perform


def restore():
    verify_freeze()
    audit = read(OUT / "audit.json")
    session, growth = restore_inquiry()
    if not audit["passed"] or session.identity() != audit["owner"]:
        raise ValueError("Use the qualified counterfactual owner")
    return session, growth


def names(request, frontier):
    domain = request["domain"]
    def resolve(text):
        candidates = set()
        for q in frontier["questions"]:
            if q["domain"] != domain:
                continue
            for symbol, word in q["word_bindings"].items():
                if text == symbol or word and text.casefold() == word["term"].casefold():
                    candidates.add(symbol)
        if len(candidates) != 1:
            raise ValueError(f"Use an unambiguous retained symbol for {text}: {sorted(candidates)}")
        return next(iter(candidates))
    return resolve(request["axis"]), resolve(request["target"])


def perform(session, growth, request):
    kind = request.get("kind")
    if kind == "inquiry_findings":
        result = {"questions": len(session.records), "compact_rules": len(session.rules), "selected_policy": session.selected_policy,
                  "points": sum(c["points"] for c in session.credits.values()), "records": session.records,
                  "rules": session.rules, "saturation": session.saturation}
    elif kind == "inquiry_next":
        frontier = read(RUN / "frontier.json")
        question = next_question(frontier["questions"], session.records)
        result = {"next_question": question, "status": "NEW_INVESTIGATION" if question else "REPRESENTED_FRONTIER_EXHAUSTED",
                  "next_action": "Investigate the selected dependency" if question else
                  "Keep completed knowledge; expand the representation or supply a new observation source before awarding further discovery credit"}
    elif kind == "apply_inquiry_rule":
        result = session.apply_rule(request["rule"], request["initial"], request["multiplier"])
    elif kind == "what_if":
        frontier = read(RUN / "frontier.json")
        axis, target = names(request, frontier)
        question = next(q for q in frontier["questions"] if q["domain"] == request["domain"] and q["axis"] == axis and q["target"] == target)
        question = dict(question)
        if "context" in request:
            context = {k: str(Q(str(v))) for k, v in request["context"].items()}
            if set(context) != set(question["context"]):
                raise ValueError("A changed background needs a complete consistent world context")
            question["context"] = context
            question["id"] = digest({"parent_question": question["id"], "context": context})
        # Save the complete finite conjecture set before any proof or naming.
        event = commit(propose(question, operators(session.owner)), session.identity())
        directory = ROOT / "runs/CI-user-queries" / event["commitment"]
        proposal_path = directory / "proposal.json"
        if proposal_path.exists() and read(proposal_path) != event:
            raise ValueError("A prior committed query changed")
        if not proposal_path.exists():
            write(proposal_path, event)
        proof = validate_event(event, session.identity())
        write(directory / "proof.json", proof)
        if not proof["accepted"]:
            raise ValueError("Preserved a failed independent conditional check")
        scales = [Q(str(v)) for v in request.get("multipliers", ["1/2", "1", "2"])]
        portfolio = []
        for c in event["candidates"]:
            portfolio.append({"id": c["id"], "expression": c["analysis"]["expression"],
                              "predictions": [{"multiplier": str(z), "value": str(at(c["curve"], z)) if admitted(c, z) else None,
                                               "admitted": admitted(c, z)} for z in scales],
                              "assumptions": c["spec"], "conditions": c["conditions"], "analysis": c["analysis"],
                              "after_check_explanation": after_discovery_words(c)})
        result = {"question": question["question"], "context": question["context"], "alternatives": portfolio,
                  "proof_receipt": proof["id"], "proposal_file": proposal_path.relative_to(ROOT).as_posix(),
                  "source_sha256": sha(proposal_path), "status": "CHECKED_CONDITIONAL_PORTFOLIO", "physical_fact_added": False}
    else:
        return previous_perform(session.base, growth, request)
    return {"id": request.get("id"), "kind": kind, "owner": session.identity(), "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Use a fresh output in runs")
    tasks = read(args.input)
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 64:
        raise ValueError("Use one to 64 task objects")
    session, growth = restore()
    results = [perform(session, growth, request) for request in tasks]
    write(args.output, results)
    print(json.dumps({"tasks": len(results), "owner": session.identity(), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
