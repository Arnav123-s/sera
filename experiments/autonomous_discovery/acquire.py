"""Acquire an identified operator gap from retained constraint feedback, then investigate."""

import argparse
import copy
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import torch

from experiments.self_study.algebra import check, independent, vector
from experiments.verified_completion.common import ROOT, read, sha, write
from experiments.verified_completion.credit import decode, encode
from sera.session_state import model_identity
from workbench.storage import Store

from .core import DEGREE, DEPTH, TOKENS, AcquisitionGap
from .runtime import AutonomousR1, AutonomousSession

OUT = ROOT / "research-continuation/37_constraint_acquisition"
RUN = ROOT / "runs/AC-study-001"


def contracts():
    return {"source": sha(Path(__file__)), "protocol": sha(OUT / "PROTOCOL.md")}


def construct(owner, domain, p):
    """Fit a candidate from generic residual probes, never a supplied output."""
    original = owner.study_maps[domain].propose(p)
    initial = check(domain, p, original)
    if initial["accepted"]:
        return {"input": p, "candidate": original, "acquired": False, "feedback_queries": 1}
    # The anchored-zero boundary and linear verification residual are already
    # part of the retained operator contract. Every candidate direction is an
    # execution of the retained integral on a representation basis element.
    start = vector(original)
    start[0] = Q(0)
    directions = []
    for degree in range(DEGREE+DEPTH):
        basis = [0]*degree+[1]
        value = owner.study_maps["integral"].propose(basis)
        if not check("integral", basis, value)["accepted"]:
            raise AcquisitionGap("Retained candidate generator requires acquisition")
        directions.append(vector(value))
    initial_residual = list(map(Q, check(domain, p, list(map(str, start)))["residual"]))
    columns = []
    for direction in directions:
        trial = [a+b for a, b in zip(start, direction, strict=True)]
        residue = list(map(Q, check(domain, p, list(map(str, trial)))["residual"]))
        columns.append([a-b for a, b in zip(residue, initial_residual, strict=True)])
    design = torch.tensor(np.array(columns, dtype=float).T, dtype=torch.float64)
    target = -torch.tensor(list(map(float, initial_residual)), dtype=torch.float64)
    coefficients = torch.linalg.lstsq(design, target, driver="gelsd", rcond=1e-10).solution
    coefficients = [Q(float(v)).limit_denominator(120) for v in coefficients]
    candidate = [a+sum(c*d[i] for c, d in zip(coefficients, directions, strict=True)) for i, a in enumerate(start)]
    candidate = list(map(str, candidate))
    if not check(domain, p, candidate)["accepted"] or not independent(domain, p, candidate):
        raise AcquisitionGap("Constructed candidate did not pass independent verification")
    return {"input": p, "original": original, "candidate": candidate, "acquired": True,
            "directions": [list(map(str, d)) for d in directions], "coefficients": list(map(str, coefficients)),
            "initial_residual": list(map(str, initial_residual)), "feedback_queries": len(directions)+3,
            "independent_check": True, "feedback_kind": "retained constraint residual; automated supervision"}


class AcquiredR1(AutonomousR1):
    def export_config(self):
        return {**super().export_config(), "constraint_acquisition": contracts()}


class AcquisitionSession:
    def __init__(self, parent=None, saved=None):
        if saved and saved["contracts"] != contracts():
            raise ValueError("Changed acquisition contract")
        self.parent = copy.deepcopy(saved["parent"] if saved else parent)
        self.base = AutonomousSession(saved=self.parent)
        self.owner = self.base.owner
        self.owner.__class__ = AcquiredR1
        self.acquisition = copy.deepcopy(saved["acquisition"]) if saved else []
        if saved:
            self.owner.study_maps.load_state_dict(decode(saved["maps"]))
            self.base.state = copy.deepcopy(saved["state"])
            self.base.rng.set_state(torch.tensor(saved["rng"], dtype=torch.uint8))
            with torch.no_grad():
                self.owner.autonomous_policy.copy_(torch.tensor(saved["policy"], dtype=torch.float64))
                self.owner.autonomous_denominators.copy_(torch.tensor(saved["denominators"], dtype=torch.int64))
            for record in self.base.state["records"]:
                self.base.register(record)
        self.base.base.base.refresh()
        if saved and model_identity(self.owner) != saved["owner"]:
            raise ValueError("Changed acquired owner")

    def acquire(self):
        if self.acquisition:
            raise ValueError("Do not repeat completed acquisition")
        missing = self.base.state["unconnected_primitives"]
        for domain, token in TOKENS.items():
            if token not in missing:
                continue
            learner = self.owner.study_maps[domain]
            for degree in range(DEGREE+DEPTH):
                p = [0]*degree+[1]
                record = construct(self.owner, domain, p)
                record.update(domain=domain, degree=degree)
                if record["acquired"]:
                    # A basis correction updates the actual operator. The
                    # sufficient statistics remain consistent with its new map.
                    with torch.no_grad():
                        learner.weight[degree].copy_(torch.tensor(list(map(float, vector(record["candidate"]))), dtype=torch.float64))
                        learner.gram[degree, degree] += 1
                        learner.rhs.copy_(learner.gram @ learner.weight)
                        learner.observations.add_(1)
                    if not check(domain, p, learner.propose(p))["accepted"]:
                        raise AssertionError("Acquired correction was not executable")
                self.acquisition.append(record)
        self.base.state["remaining_frontier"] = list(range(1, DEPTH+1))
        self.base.state["original_goal"] = "Continue the previously blocked internal discovery after verified prerequisite acquisition"
        self.base.state["prerequisite_credit"] = {
            "points": sum(r["acquired"] for r in self.acquisition),
            "scope": "newly independently verified basis corrections; awarded once per acquisition record",
            "parent_owner": self.parent["owner"], "contracts": contracts(),
        }

    def snapshot(self):
        self.base.base.base.refresh()
        return {"schema": "sera.constraint-acquisition.1", "contracts": contracts(), "parent": self.parent,
                "owner": model_identity(self.owner), "maps": encode(self.owner.study_maps.state_dict()),
                "state": copy.deepcopy(self.base.state), "acquisition": copy.deepcopy(self.acquisition),
                "rng": self.base.rng.get_state().tolist(), "policy": self.owner.autonomous_policy.detach().tolist(),
                "denominators": self.owner.autonomous_denominators.tolist()}


def study():
    if (OUT / "results.json").exists() or (RUN / "acquired.json").exists():
        raise FileExistsError("Preserve completed or interrupted acquisition")
    parent = Store(ROOT / "runs/sera-autonomous-live").read()
    RUN.mkdir(parents=True, exist_ok=True)
    write(RUN / "parent.json", parent)
    session = AcquisitionSession(parent=parent)
    old_maps = copy.deepcopy(session.owner.study_maps)
    original_event_count = len(session.base.state["events"])
    protected = {k: v.clone() for k, v in session.owner.state_dict().items()
                 if not k.startswith(("study_maps.", "autonomous_"))}
    old_passes = [(d, j) for d, m in old_maps.items() for j in range(11)
                  if check(d, [0]*j+[1], m.propose([0]*j+[1]))["accepted"]]
    session.acquire()
    write(RUN / "acquired.json", session.snapshot())
    while session.base.state["remaining_frontier"]:
        event = session.base.advance()
        write(RUN / f"step-{len(session.base.state['events'])-original_event_count:02d}.json", session.snapshot())
        print(json.dumps({"depth": event["chosen_depth"], "discoveries": event["discovery_points"],
                          "procedure_points": event["procedure_improvement_points"]}), flush=True)
    final = session.snapshot()
    write(RUN / "selection.json", {"contracts": contracts(), "state": sha(RUN / "step-03.json"),
                                  "final_opened": False})
    resumed = AcquisitionSession(saved=read(RUN / "step-02.json"))
    while resumed.base.state["remaining_frontier"]:
        resumed.base.advance()
    if resumed.snapshot() != final:
        raise AssertionError("Acquisition and discovery did not resume exactly")
    del resumed
    rng = np.random.default_rng(37991)
    evaluations, acquisition_tests = [], []
    for _ in range(32):
        p = list(map(str, rng.integers(-7, 8, DEGREE+1).tolist()))
        for domain in TOKENS:
            before = check(domain, p, old_maps[domain].propose(p))["accepted"]
            proposed = session.owner.study_maps[domain].propose(p)
            after = check(domain, p, proposed)["accepted"] and independent(domain, p, proposed)
            acquisition_tests.append({"domain": domain, "input": p, "before": before, "after": after})
        for record in session.base.state["records"]:
            try:
                evaluations.append(session.base.use(record, p))
            except AcquisitionGap as error:
                evaluations.append({"rule": record["id"], "input": p, "independent_match": False,
                                    "status": "NEEDS_ACQUISITION", "reason": str(error)})
    retained = all(check(d, [0]*j+[1], session.owner.study_maps[d].propose([0]*j+[1]))["accepted"] for d, j in old_passes)
    protected_equal = all(torch.equal(v, session.owner.state_dict()[k]) for k, v in protected.items())
    if not retained or not protected_equal or session.snapshot() != final:
        raise AssertionError("Retention or frozen-evaluation state changed")
    live = Store(ROOT / "runs/sera-acquisition-live")
    if live.read() is not None:
        raise FileExistsError("Preserve a previous acquisition owner")
    live.commit(final, None)
    result = {"contracts": contracts(), "owner": model_identity(session.owner), "state": session.base.state,
              "acquisition": session.acquisition, "acquisition_tests": acquisition_tests, "evaluations": evaluations,
              "checked_routes": sum(r["independent_match"] for r in evaluations), "attempted_routes": len(evaluations),
              "new_relations": len(session.base.state["records"]), "exact_resume": True,
              "protected_tensors": len(protected), "protected_equal": protected_equal,
              "retained_old_basis_cases": old_passes, "retention_passed": retained,
              "denominators": session.owner.autonomous_denominators.tolist(), "fresh_polynomials": 32,
              "prior_events": original_event_count, "web_queries": 0, "supplied_target_equations": 0,
              "human_interventions_after_freeze": 0, "feedback": "existing mathematical constraints with residuals; automated supervision",
              "source_gate": session.base.base.base.base.base.grounded.gate["autonomous"]}
    write(OUT / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"state", "acquisition", "acquisition_tests", "evaluations"}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("study", "use"))
    parser.add_argument("--coefficients", default="2")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "study":
        study()
        return
    session = AcquisitionSession(saved=Store(ROOT / "runs/sera-acquisition-live").read())
    if not session.base.state["records"]:
        raise AcquisitionGap("No admitted relationship is available")
    print(json.dumps(session.base.use(session.base.state["records"][0], args.coefficients.split(",")), indent=2))


if __name__ == "__main__":
    main()
