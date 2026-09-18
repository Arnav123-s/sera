"""Solve with all checked alternatives; investigate an unanswered task to closure."""

import argparse
import json
from pathlib import Path

import torch

from experiments.gap_inquiry import ROOT, digest, read, sha, write
from experiments.self_chosen import equations as eq
from experiments.solution_owner import SolutionSession
from experiments.task_transfer.runtime import lock
from scripts.portable_gap_use import perform as gap_perform
from scripts.portable_gap_use import restore as gap_restore
from workbench.storage import Store

OUT = ROOT / "research-continuation/44_solution_portfolios"


def restore(store="runs/sera-solution-progress-live"):
    path = (ROOT / store).resolve()
    if not path.is_relative_to(ROOT / "runs"):
        raise ValueError("Use a local owned solution store")
    state = Store(path).read()
    if not state or state["audit"] != sha(OUT / "audit.json") or state["final"] != sha(OUT / "final.json"):
        raise ValueError("A qualified current solution owner is required")
    if state["schema"] == "sera.solution-progress.store.1":
        from experiments.solution_credit import ReconciledSession
        if state["credit_audit"] != sha(OUT / "credit-audit.json") or state["parent_current"] != sha(ROOT / state["parent_store"] / "current.json"):
            raise ValueError("Changed credit repair or original parent")
        original, growth = restore(state["parent_store"])
        if original.identity() != state["parent_owner"]:
            raise ValueError("Changed original solution owner")
        return ReconciledSession(original, state["state"]), growth
    base, growth = gap_restore()
    if state["parent_owner"] != base.identity():
        raise ValueError("The actual parent owner changed")
    session = SolutionSession(base, state["state"])
    if session.identity() != state["owner"]:
        raise ValueError("Solution checkpoint identity changed")
    return session, growth


def perform(session, growth, request, persist=None):
    kind = request.get("kind")
    if kind == "solution_findings":
        result = {"questions": session.questions, "records": session.records, "credits": session.credits,
                  "default_curiosity_cycle": "imagine alternatives -> commit -> independently prove/check -> answer original goal -> unique credit"}
    elif kind in {"solve_solution", "curiosity"}:
        domain, target = request["domain"], request["target"]
        observations = request["observations"]
        result = session.solve(domain, target, observations)
        if persist and (kind == "curiosity" or result["status"] == "INVESTIGATION_REQUIRED"):
            missing = sorted(set(eq.layout(domain)) - set(observations) - {target})
            core = {"domain": domain, "target": target, "missing": missing}
            question = {**core, "id": digest(core), "question": f"Find all checked routes to {target} in {domain} from available observations"}
            result = session.autonomous_round(question, observations, persist)
    else:
        return gap_perform(session.base, growth, request)
    return {"id": request.get("id"), "kind": kind, "owner": session.identity(), "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", default="runs/sera-solution-progress-live")
    parser.add_argument("--read-only", action="store_true", help="Inspect existing answers without starting a new investigation")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Use a fresh owned output")
    tasks = read(args.input)
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 256:
        raise ValueError("Use one to 256 finite tasks")
    store = Store(ROOT / args.store)
    session, growth = restore(args.store)
    previous = store.read()
    def persist(state):
        nonlocal previous
        with lock(store.directory):
            if store.read() != previous:
                raise ValueError("Another writer changed this owner; preserve both investigations")
            previous = store.commit({**previous, "state": state, "owner": state["owner"]}, previous)
    results = []
    for request in tasks:
        try:
            result = perform(session, growth, request, None if args.read_only else persist)
        except (ValueError, KeyError, IndexError) as error:
            result = {"id": request.get("id"), "status": "PRESERVED_FOR_RESUMPTION", "reason": str(error)}
        results.append(result)
        write(args.output, results)
    print(json.dumps({"owner": session.identity(), "tasks": len(results), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
