"""Continue the real owner, choose gaps and retain independently checked discoveries."""

import argparse
import copy
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.operator_discovery.core import WORDS as OLD_WORDS
from experiments.operator_discovery.core import DiscoveryR1
from experiments.operator_discovery.runtime import DiscoverySession
from experiments.task_transfer.runtime import lock
from experiments.verified_completion.common import ROOT, digest, read, sha, write
from sera.session_state import model_identity
from workbench.storage import Store

from .core import (
    DEGREE,
    DEPTH,
    TOKENS,
    WORDS,
    AcquisitionGap,
    canonical,
    certify,
    consequence_span,
    elimination_proposals,
    execute,
    numerical_proposals,
    reference,
    signatures,
)

OUT = ROOT / "research-continuation/36_autonomous_discovery"
RUN = ROOT / "runs/AD-study-002"


def contracts():
    return {"core": sha(Path(__file__).with_name("core.py")), "runtime": sha(Path(__file__)),
            "protocol": sha(OUT / "PROTOCOL.md")}


class AutonomousR1(DiscoveryR1):
    @classmethod
    def attach(cls, owner):
        if type(owner) is not DiscoveryR1:
            raise ValueError("Continue the actual saved discovery owner")
        owner.__class__ = cls
        owner.autonomous_rules = nn.ParameterDict()
        owner.autonomous_policy = nn.Parameter(owner.discovery_policy.detach().clone())
        owner.register_buffer("autonomous_denominators", torch.tensor([120, 120], dtype=torch.int64))
        owner.autonomous_contracts = contracts()
        return owner

    def export_config(self):
        return {**super().export_config(), "autonomous_discovery": {
            "contracts": self.autonomous_contracts, "words": list(WORDS), "degree": DEGREE,
            "rule_ids": sorted(self.autonomous_rules)}}


class AutonomousSession:
    def __init__(self, parent=None, saved=None):
        if saved and saved["contracts"] != contracts():
            raise ValueError("Changed research contract; preserve the earlier state")
        self.parent_record = copy.deepcopy(saved["parent"] if saved else parent)
        self.base = DiscoverySession(saved=self.parent_record)
        self.owner = AutonomousR1.attach(self.base.owner)
        self.prior = []
        for record in self.base.discovery.records:
            row = [Q(0)]*len(WORDS)
            for word, c in zip(OLD_WORDS, record["coefficients"], strict=True):
                row[WORDS.index(word)] = Q(c)
            self.prior.append({"coefficients": list(map(str, row))})
        available = set(TOKENS[k] for k in self.owner.study_maps if k in TOKENS) | {"M"}
        represented = {c for r in self.prior for w, q in zip(WORDS, r["coefficients"], strict=True) if Q(q) for c in w}
        self.state = copy.deepcopy(saved["state"]) if saved else {
            "available_primitives": sorted(available), "unconnected_primitives": sorted(available-represented),
            "remaining_frontier": list(range(1, DEPTH+1)), "records": [], "events": [],
            "original_goal": "Find and independently check connections in retained knowledge without supplied target equations",
        }
        if available != {"I", "M", "S"}:
            raise ValueError("The retained primitive inventory changed")
        self.rng = torch.Generator().manual_seed(3601)
        if saved:
            self.rng.set_state(torch.tensor(saved["rng"], dtype=torch.uint8))
            with torch.no_grad():
                self.owner.autonomous_policy.copy_(torch.tensor(saved["policy"], dtype=torch.float64))
                self.owner.autonomous_denominators.copy_(torch.tensor(saved["denominators"], dtype=torch.int64))
            for record in self.state["records"]:
                self.register(record)
        self.base.base.refresh()
        if saved and model_identity(self.owner) != saved["owner"]:
            raise ValueError("Changed autonomous owner")

    def register(self, record):
        if not certify(record["coefficients"])["accepted"]:
            raise ValueError("Unverified discovery cannot become retained knowledge")
        self.owner.autonomous_rules[record["id"]] = nn.Parameter(
            torch.tensor([float(Q(v)) for v in record["coefficients"]], dtype=torch.float64), requires_grad=False)

    def advance(self):
        frontier = self.state["remaining_frontier"]
        if not frontier:
            raise ValueError("Finite frontier exhausted; preserve the completed run")
        bonus = torch.tensor([np.log1p(sum(any(c in w for c in self.state["unconnected_primitives"])
                                                for w in WORDS if len(w) == d))
                              for d in range(1, DEPTH+1)], dtype=torch.float64)
        logits = self.owner.autonomous_policy + bonus
        mask = torch.tensor([d in frontier for d in range(1, DEPTH+1)])
        probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        action = int(torch.multinomial(probabilities.detach(), 1, generator=self.rng))
        depth = action+1
        sequence = len(self.state["events"])
        names, columns, generation = signatures(self.owner, depth, 36100+sequence)
        # Neither proposal procedure can see checker semantics or accepted
        # equations; each receives only imagined executions and program names.
        proposals = {"numerical_dependency": numerical_proposals(names, columns),
                     "exact_elimination": elimination_proposals(names, columns)}
        span = consequence_span(self.prior+self.state["records"])
        receipts, points = [], 0
        for method, rows in proposals.items():
            for row in rows:
                certificate = certify(row)
                new = certificate["accepted"] and span.add(row)
                identifier = digest({"coefficients": list(map(str, row)), "scope": certificate["scope"]})
                if new:
                    record = {"id": identifier, "coefficients": list(map(str, row)),
                              "certificate": certificate, "method": method, "investigation": sequence,
                              "origin": "fitted from retained learned-operator executions"}
                    self.register(record)
                    self.state["records"].append(record)
                    # Deduplicate valid composed consequences, not only aliases.
                    span = consequence_span(self.prior+self.state["records"])
                    points += 1
                receipts.append({"method": method, "coefficients": list(map(str, row)),
                                 "certificate": certificate, "new_discovery": new, "id": identifier})
        procedure_points = len(generation.get("procedure_updates", []))
        reward = points+procedure_points-.01*len(names)
        loss = -reward*probabilities[action].log()
        self.owner.autonomous_policy.grad = None
        loss.backward()
        with torch.no_grad():
            self.owner.autonomous_policy.add_(self.owner.autonomous_policy.grad.clamp(-2, 2), alpha=-.05)
        frontier.remove(depth)
        event = {"sequence": sequence, "chosen_depth": depth, "frontier_before": sorted(frontier+[depth]),
                 "choice_probabilities": probabilities.detach().tolist(), "identified_gap": self.state["unconnected_primitives"],
                 "generation": generation, "proposals": receipts, "discovery_points": points,
                 "reward": reward, "procedure_improvement_points": procedure_points}
        self.state["events"].append(event)
        return event

    def snapshot(self):
        self.base.base.refresh()
        return {"schema": "sera.autonomous-discovery.1", "contracts": contracts(), "parent": self.parent_record,
                "owner": model_identity(self.owner), "state": copy.deepcopy(self.state),
                "rng": self.rng.get_state().tolist(), "policy": self.owner.autonomous_policy.detach().tolist(),
                "denominators": self.owner.autonomous_denominators.tolist()}

    def use(self, record, p):
        if record["id"] not in self.owner.autonomous_rules:
            raise ValueError("No retained executable discovery")
        coefficient = [Q(float(v)).limit_denominator(1000000) for v in self.owner.autonomous_rules[record["id"]]]
        if list(map(str, canonical(coefficient))) != record["coefficients"]:
            raise ValueError("Stored executable coefficients changed")
        target = max((w for w, c in zip(WORDS, coefficient, strict=True) if c), key=lambda w: (len(w), w))
        position = WORDS.index(target)
        cache, counts = {}, {"primitive_calls": 0}
        alternative = [Q(0)]*13
        for i, word in enumerate(WORDS):
            if i != position and coefficient[i]:
                computed = execute(self.owner, word, p, cache, counts)
                alternative = [a-coefficient[i]/coefficient[position]*b for a, b in zip(alternative, computed, strict=True)]
        direct = execute(self.owner, target, p, cache, counts)
        expected = reference(target, p)
        if alternative != direct or direct != expected:
            raise AssertionError("A learned route failed independent execution")
        return {"rule": record["id"], "target": target, "input": p,
                "result": list(map(str, alternative)), "independent_match": True, **counts}


def study():
    if (OUT / "results.json").exists() or (RUN / "step-01.json").exists():
        raise FileExistsError("Preserve completed or interrupted autonomous work")
    RUN.mkdir(parents=True, exist_ok=True)
    parent = Store(ROOT / "runs/sera-discovery-live").read()
    write(RUN / "parent.json", parent)
    session = AutonomousSession(parent=parent)
    inherited = {k: v.clone() for k, v in session.owner.state_dict().items() if not k.startswith("autonomous_")}
    while session.state["remaining_frontier"]:
        event = session.advance()
        write(RUN / f"step-{len(session.state['events']):02d}.json", session.snapshot())
        print(json.dumps({"depth": event["chosen_depth"], "new_connections": event["discovery_points"],
                          "remaining_frontier": session.state["remaining_frontier"]}), flush=True)
    final = session.snapshot()
    write(RUN / "selection.json", {"contracts": contracts(), "state": sha(RUN / "step-03.json"),
                                  "final_opened": False, "discoveries": len(session.state["records"])})
    replay = AutonomousSession(saved=read(RUN / "step-01.json"))
    while replay.state["remaining_frontier"]:
        replay.advance()
    resumed = replay.snapshot() == final
    if not resumed:
        raise AssertionError("Autonomous choices did not resume exactly")
    del replay
    rng = np.random.default_rng(36991)
    evaluations = []
    for _ in range(32):
        p = list(map(str, rng.integers(-7, 8, DEGREE+1).tolist()))
        for record in session.state["records"]:
            try:
                evaluations.append(session.use(record, p))
            except AcquisitionGap as error:
                evaluations.append({"rule": record["id"], "input": p, "independent_match": False,
                                    "status": "NEEDS_ACQUISITION", "reason": str(error)})
    if not all(torch.equal(v, session.owner.state_dict()[k]) for k, v in inherited.items()):
        raise AssertionError("Autonomous discovery changed earlier knowledge")
    if session.snapshot() != final:
        raise AssertionError("Final evaluation modified learning state")
    live = Store(ROOT / "runs/sera-autonomous-live")
    if live.read() is not None:
        raise FileExistsError("Preserve an existing live owner")
    live.commit(final, None)
    result = {"contracts": contracts(), "owner": model_identity(session.owner), "state": session.state,
              "evaluations": evaluations, "checked_routes": sum(e["independent_match"] for e in evaluations),
              "attempted_routes": len(evaluations), "fresh_polynomials": 32,
              "exact_resume": resumed, "inherited_tensors": len(inherited), "inherited_equal": True,
              "final_learning_state_unchanged": True, "stop_reason": "finite_frontier_exhausted",
              "web_queries": 0, "supplied_target_equations": 0, "human_interventions_after_freeze": 0,
              "source_gate": session.base.base.base.base.grounded.gate["autonomous"],
              "procedure_improvement_points": sum(e["procedure_improvement_points"] for e in session.state["events"]),
              "learned_reconstruction_limits": session.owner.autonomous_denominators.tolist(),
              "example": evaluations[0] if evaluations else None}
    write(OUT / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"state", "evaluations"}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("study", "status", "use"))
    parser.add_argument("--coefficients", default="2")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "study":
        study()
        return
    path = ROOT / "runs/sera-autonomous-live"
    with lock(path):
        session = AutonomousSession(saved=Store(path).read())
        if args.action == "use":
            if not session.state["records"]:
                raise ValueError("The learner has no independently admitted discovery")
            result = session.use(session.state["records"][0], args.coefficients.split(","))
        else:
            result = {"owner": model_identity(session.owner), "state": session.state}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
