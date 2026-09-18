"""Persistent internal discovery and executable use of newly proved relations."""

import argparse
import copy
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import torch

from experiments.learning_progress.bridge import LiveSession
from experiments.task_transfer.runtime import lock
from experiments.verified_completion.common import ROOT, model_hash, read, sha, write
from sera.session_state import model_identity
from workbench.storage import Store

from .core import WORDS, Discoveries, DiscoveryR1, exact, propose, source

OUT = ROOT / "research-continuation/35_operator_discovery"
RUN = ROOT / "runs/OD-study-001"


def contracts():
    return {"core": source(), "runtime": sha(Path(__file__)), "protocol": sha(OUT / "PROTOCOL.md")}


class DiscoverySession:
    def __init__(self, parent=None, saved=None):
        if saved and saved["contracts"] != contracts():
            raise ValueError("Changed discovery source; explicit migration required")
        self.parent_record = copy.deepcopy(saved["parent"] if saved else parent)
        self.base = LiveSession(self.parent_record)
        self.owner = DiscoveryR1.attach(self.base.owner)
        self.discovery = Discoveries(self.owner, saved["discovery"] if saved else None)
        self.base.refresh()
        if saved and model_identity(self.owner) != saved["owner"]:
            raise ValueError("Discovery checkpoint identity changed")

    def snapshot(self):
        self.base.refresh()
        return {"schema": "sera.operator-discovery.1", "contracts": contracts(), "parent": self.parent_record,
                "owner": model_identity(self.owner), "discovery": self.discovery.snapshot()}

    def solve(self, coefficients, *, velocity=0, position=0):
        if Q(velocity) or Q(position):
            raise ValueError("This new zero-anchored displacement route requires zero initial conditions")
        result = self.discovery.alternatives("II", coefficients)
        expected = list(map(str, exact("II", coefficients)))
        if any(r["result"] != expected for r in result):
            raise AssertionError("An executable discovery lost its independent certificate")
        return {"status": "CERTIFIED_DISCOVERED_ALTERNATIVES", "acceleration": coefficients,
                "position_polynomial": expected, "alternative_routes": result,
                "initial_conditions": {"velocity": "0", "position": "0"}, "factual_updates": 0}


def study():
    if (OUT / "results.json").exists():
        raise FileExistsError("Completed discovery study must not restart")
    RUN.mkdir(parents=True, exist_ok=True)
    parent = Store(ROOT / "runs/sera-learning-live").read()
    write(RUN / "parent.json", parent)
    session = DiscoverySession(parent)
    inherited = {k: v.clone() for k, v in session.owner.state_dict().items() if not k.startswith("discovery_")}
    for i in range(9):
        session.discovery.round(i)
        write(RUN / f"step-{i+1:02d}.json", session.snapshot())
        print(json.dumps({"step": i+1, "discoveries": len(session.discovery.records)}), flush=True)
    final = session.snapshot()
    # Independent replay begins from a genuine interrupted successor boundary.
    replay = DiscoverySession(saved=read(RUN / "step-04.json"))
    for i in range(4, 9):
        replay.discovery.round(i)
    resumed = replay.snapshot() == final
    if not resumed:
        raise AssertionError("Discovery restart lost exact state")
    del replay
    control_results = {}
    for name, depths in (("exhaustive", [1, 2, 3]), ("random", np.random.default_rng(3502).integers(1, 4, 9).tolist())):
        # Same computational owner; separate controlled scratch discoveries do
        # not enter the live registry or receive its credit.
        accepted, receipts, cost = [], [], 0
        from .core import canonical, certify, rational_rank
        for i, depth in enumerate(depths):
            proposals = propose(session.owner, depth, 35200+i)
            cost += 12*2**depth
            for proposal in proposals:
                coef = canonical(proposal["coefficients"])
                certificate = certify(coef)
                new = certificate["accepted"] and rational_rank(accepted+[coef]) > rational_rank(accepted)
                if new:
                    accepted.append(coef)
                receipts.append({"coefficients": coef, "new": new, "accepted": certificate["accepted"]})
        control_results[name] = {"depths": depths, "discoveries": len(accepted), "program_executions": cost, "receipts": receipts}
    write(RUN / "selection.json", {"contracts": contracts(), "discovery_ids": sorted(session.owner.discovery_rules),
                                  "state": sha(RUN / "step-09.json"), "final_opened": False})
    rng = np.random.default_rng(35991)
    evaluated = []
    for i in range(64):
        coefficients = [str(Q(int(rng.integers(-10, 11)), d)) for d in (5, 5, 2, 1)]
        for word in ("II", "III"):
            expected = list(map(str, exact(word, coefficients)))
            alternatives = session.discovery.alternatives(word, coefficients)
            if not alternatives or any(a["result"] != expected for a in alternatives):
                raise AssertionError("A fresh exact alternative failed")
            evaluated.append({"id": i, "target": word, "input": coefficients, "expected": expected, "alternatives": alternatives})
    kept = all(torch.equal(v, session.owner.state_dict()[k]) for k, v in inherited.items())
    if not kept:
        raise AssertionError("Internal discovery overwrote retained knowledge")
    example = session.solve(["2"])
    store = Store(ROOT / "runs/sera-discovery-live")
    if store.read() is not None:
        raise FileExistsError("Preserve an existing discovery owner")
    store.commit(session.snapshot(), None)
    records = [{**r, "display": " + ".join(f"({c}) {w}" for c, w in zip(r["coefficients"], WORDS, strict=True) if Q(c)) + " = 0"}
               for r in session.discovery.records]
    result = {"contracts": contracts(), "owner": model_identity(session.owner), "model_hash": model_hash(session.owner),
              "novel_to_learner_relations": len(records), "records": records, "events": session.discovery.events,
              "controls": control_results, "evaluated": evaluated, "fresh_polynomials": 64,
              "checked_alternatives": sum(len(r["alternatives"]) for r in evaluated),
              "exact_resume": resumed, "inherited_tensors": len(inherited), "inherited_equal": kept,
              "source_gate": session.base.base.base.grounded.gate["autonomous"], "example": example,
              "research_scope": "new to this retained library; generated from learned integration under a supplied finite grammar; exact degree-bounded identities",
              "web_queries_by_learner": 0, "supplied_completed_equations": 0}
    write(OUT / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "events", "controls", "evaluated"}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("study", "status", "solve"))
    parser.add_argument("--coefficients", default="2")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "study":
        study()
        return
    path = ROOT / "runs/sera-discovery-live"
    with lock(path):
        session = DiscoverySession(saved=Store(path).read())
        result = session.solve(args.coefficients.split(",")) if args.action == "solve" else {
            "owner": model_identity(session.owner), "discoveries": session.discovery.records,
            "source": "retained executable knowledge", "web_queries": 0}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
