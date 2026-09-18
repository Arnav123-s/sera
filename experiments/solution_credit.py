"""Reconcile solution reward with executable knowledge outside the route registry.

The completed PS-001 run is immutable. This additive successor keeps its checked
answers, revokes credit for already executable primitives, and preserves both
reward versions. The affected procedure predictor stays archived for analysis.
"""

import copy
import json
from functools import lru_cache

import sympy as sp
import torch

from experiments.discovery_frontier import native
from experiments.gap_inquiry import ROOT, digest, sha
from experiments.solution_owner import SolutionR1, SolutionSession
from experiments.verified_completion.credit import decode, encode


def contracts():
    return {"source": sha(ROOT / "experiments/solution_credit.py"),
            "rule": "Known executions and dominated input/guard coverage earn no discovery reward"}


@lru_cache(maxsize=8192)
def guard_factors(domain, denominator):
    from experiments.self_chosen.equations import symbolic_world
    from experiments.solution_check import symbolic
    expression = symbolic(json.loads(denominator), symbolic_world(domain))
    factors = []
    for value, _ in sp.factor_list(expression)[1]:
        if str(value) in {"t", "m"}:
            continue  # These are positive in the retained domain contract.
        symbols = sorted(value.free_symbols, key=str)
        factors.append(str(sp.Poly(value, *symbols).monic().as_expr()))
    return frozenset(factors)


def dominates(previous, candidate):
    if previous["id"] == candidate["id"] or (previous["domain"], previous["target"]) != (candidate["domain"], candidate["target"]):
        return False
    left, right = set(previous["requires"]), set(candidate["requires"])
    if not left <= right:
        return False
    a = guard_factors(previous["domain"], json.dumps(previous["denominator"]))
    b = guard_factors(candidate["domain"], json.dumps(candidate["denominator"]))
    return a <= b and (left < right or a < b)


def known_execution(candidate, records=(), solutions=()):
    domain, target, available = candidate["domain"], candidate["target"], set(candidate["requires"])
    if any(r["question"]["domain"] == domain and r["proposal"]["target"] == target
           and "coefficients" in r["proposal"] and set(r["proposal"]["requires"]) <= available
           and set(r["proposal"].get("nonzero", [])) <= {"t", "m"} for r in records):
        return True
    if any(dominates(r["candidate"], candidate) for r in solutions):
        return True
    if domain == "polynomials":
        required = {"c0", "c1", "c2", "t"}
        return target in {"p", "i", "s"} and required <= available
    if not domain.startswith("motion_"):
        return False
    degree = int(domain[-1])
    coefficients = {f"a{k}" for k in range(degree + 1)}
    required = {"f": coefficients | {"m", "t"}, "v": coefficients | {"v0", "t"},
                "x": coefficients | {"x0", "v0", "t"}}
    # A constant acceleration evaluates without a time observation.
    if degree == 0:
        required["f"] = {"a0", "m"}
    if target in required and required[target] <= available:
        return True
    return native({"domain": domain}, {"target": target, "requires": sorted(available)})


class CreditR1(SolutionR1):
    def export_config(self):
        return {**super().export_config(), "solution_credit_reconciliation": self.credit_contract}


class ReconciledSession(SolutionSession):
    def __init__(self, original, saved=None):
        self.__dict__.update(original.__dict__)
        self.original_owner = original.identity()
        self.owner.__class__ = CreditR1
        self.owner.credit_contract = contracts()
        self.records, self.credits = copy.deepcopy((original.records, original.credits))
        self.questions = copy.deepcopy(original.questions)
        self.corrections = []
        for key, record in self.records.items():
            if record["points"] > 0 and known_execution(record["candidate"], self.base.base.base.records, self.records.values()):
                prior = record["points"]
                record["points"] = 0.
                record["novel_input_route"] = False
                record["credit_correction"] = "already executable procedure or input/guard coverage dominated by another checked route"
                self.credits[record["receipt"]["id"]]["points"] = 0.
                self.corrections.append({"candidate": key, "prior_points": prior, "points": 0.,
                                         "reason": record["credit_correction"]})
        # The old predictor was trained on the overcounted target. Preserve its
        # checkpoint in the original owner; do not tune against opened holdout.
        torch.nn.init.zeros_(self.owner.solution_policy.weight)
        torch.nn.init.zeros_(self.owner.solution_policy.bias)
        self.optimizer = torch.optim.Adam(self.owner.solution_policy.parameters(), lr=.02)
        if saved is not None:
            if saved["parent_owner"] != self.original_owner or saved["contracts"] != contracts():
                raise ValueError("Changed credit reconciliation lineage")
            self.records, self.credits, self.questions = copy.deepcopy((saved["records"], saved["credits"], saved["questions"]))
            self.pending = copy.deepcopy(saved["pending"])
            self.corrections = copy.deepcopy(saved["corrections"])
            for key, record in self.records.items():
                if key not in self.owner.solution_models:
                    self._weights(key, record["candidate"])
            self.owner.solution_policy.load_state_dict(decode(saved["policy"]))
            self.optimizer.load_state_dict(decode(saved["optimizer"]))
            if self.identity() != saved["owner"]:
                raise ValueError("Reconciled owner failed exact restoration")

    def state(self):
        return {"schema": "sera.solution-credit-reconciliation.1", "parent_owner": self.original_owner,
                "contracts": contracts(), "records": copy.deepcopy(self.records), "credits": copy.deepcopy(self.credits),
                "questions": copy.deepcopy(self.questions), "pending": copy.deepcopy(self.pending),
                "corrections": copy.deepcopy(self.corrections), "policy": encode(self.owner.solution_policy.state_dict()),
                "optimizer": encode(self.optimizer.state_dict()), "owner": self.identity()}

    def autonomous_round(self, question, observations, persist):
        """Keep exhaustive proposal/check behavior and correct feedback accounting."""
        from experiments import solution_check

        # The base lifecycle validates proofs before adding records. A zero
        # novelty tag is a credit adjustment, never a correctness adjustment.
        original_records = set(self.records)
        saved_policy = copy.deepcopy(self.owner.solution_policy.state_dict())
        saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
        staged = []
        def save_boundary(state):
            if state["pending"] is not None:
                persist(state)
            else:
                staged.append(state)
        answer = super().autonomous_round(question, observations, save_boundary)
        if not staged:
            return answer
        corrected = False
        for key in set(self.records) - original_records:
            record = self.records[key]
            if record["points"] and known_execution(record["candidate"], self.base.base.base.records, self.records.values()):
                corrected = True
                prior = record["points"]
                record["points"] = 0.
                self.credits[record["receipt"]["id"]]["points"] = 0.
                self.corrections.append({"candidate": key, "prior_points": prior, "points": 0., "reason": "already executable or dominated input/guard coverage"})
        if corrected:
            # Replace the affected update using the same frozen learning rule,
            # before the completed cycle is committed. No held-out data enters.
            self.owner.solution_policy.load_state_dict(saved_policy)
            self.optimizer.load_state_dict(saved_optimizer)
            from experiments.solution_owner import features
            event = self.questions[question["id"]]["proposals"]
            for attempt in event["attempts"]:
                ids = {c["id"] for c in attempt["proposals"]} & (set(self.records) - original_records)
                reward = sum(self.records[key]["points"] for key in ids)
                x = torch.tensor(features(question, attempt["method"], attempt["degree"]), dtype=torch.float64)
                self.optimizer.zero_grad(set_to_none=True)
                (self.owner.solution_policy(x).squeeze() - reward).square().backward()
                self.optimizer.step()
        total = sum(self.records[key]["points"] for key in set(self.records) - original_records)
        self.questions[question["id"]]["reward"] = total
        for key in set(self.records) - original_records:
            r = self.records[key]
            solution_check.validate(r["receipt"], r["question"], r["candidate"], r["commitment"], r["predictor"])
        persist(self.state())
        return answer | {"reward": total, "owner": self.identity(), "credit_reconciled": True}


def correction_digest(records):
    return digest([{k: r[k] for k in ("candidate", "prior_points", "points", "reason")} for r in records])
