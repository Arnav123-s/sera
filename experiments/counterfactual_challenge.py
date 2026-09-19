"""Additive completion repair: challenge a narrow consensus with retained alternatives.

The original CI-001 implementation and failed validation remain byte-identical.
This wrapper changes closure, while reusing its proposal, measurement and checker
interfaces. It runs only in the explicitly selected successor process.
"""

import copy

import torch
from torch import nn

from experiments import counterfactual_loop as original_loop
from experiments.counterfactual_core import EXPANSIONS, POWERS, propose
from experiments.counterfactual_loop import commit
from experiments.counterfactual_owner import CounterfactualR1, InquirySession, contracts
from experiments.gap_inquiry import ROOT, digest, sha
from experiments.verified_completion.credit import decode, encode

ORIGINAL_INVESTIGATE = original_loop.investigate
QUALIFICATION = ROOT / "research-continuation/45_counterfactual_inquiry/qualification"


def guard_contract():
    return {"source": sha(ROOT / "experiments/counterfactual_challenge.py"),
            "protocol": sha(QUALIFICATION / "PROTOCOL.md"), "budget": original_loop.MAX_STEPS,
            "rule": "Challenge the full retained power basis before closing a narrow consensus"}


def investigate(event, matrices, evidence_source, policy, weights, seed, sink, saved=None):
    if saved and saved.get("known_basis_challenged"):
        if saved["initial_commitment"] != event["commitment"] or saved["policy"] != policy or saved["weights"] != list(weights):
            raise ValueError("Changed qualified resumption")
        return copy.deepcopy(saved)
    result = ORIGINAL_INVESTIGATE(event, matrices, evidence_source, policy, weights, seed, sink, saved)
    if not set(EXPANSIONS) <= set(result["events"][-1]["powers"]):
        challenge = commit(propose(event["question"], matrices, (*POWERS, *EXPANSIONS)), event["predictor"])
        pending = copy.deepcopy(result)
        for name in ("id", "answer", "survivors", "next_action"):
            pending.pop(name, None)
        pending["events"].append(challenge)
        pending["followups"].append({"kind": "CHALLENGE_NARROW_CONSENSUS", "original_goal": pending["original_goal"],
                                      "trigger": "Apparent completion does not test omitted retained mechanisms",
                                      "powers_added": list(EXPANSIONS), "external_hints": 0})
        pending["status"] = "INVESTIGATING"
        sink(copy.deepcopy(pending))
        result = ORIGINAL_INVESTIGATE(event, matrices, evidence_source, policy, weights, seed, sink, pending)
    result["known_basis_challenged"] = True
    result["id"] = digest({k: v for k, v in result.items() if k != "id"})
    sink(copy.deepcopy(result))
    return result


def install():
    # Explicit process-local adapter. Original file bytes, archives and entry
    # points remain available for reproducing the rejected predecessor.
    original_loop.investigate = investigate


class ChallengedR1(CounterfactualR1):
    def export_config(self):
        return {**super().export_config(), "completion_challenge": self.completion_challenge_contract}


class ChallengedSession(InquirySession):
    def __init__(self, previous, saved=None):
        self.prior_inquiry_owner = previous.identity()
        self.base, self.parent_owner = previous.base, self.prior_inquiry_owner
        self.owner = previous.owner
        if type(self.owner) is not CounterfactualR1:
            raise ValueError("Continue the preserved experimental inquiry owner")
        self.owner.__class__ = ChallengedR1
        self.owner.completion_challenge_contract = guard_contract()
        self.owner.challenge_policy = nn.Linear(8, 1, bias=False, dtype=torch.float64)
        nn.init.zeros_(self.owner.challenge_policy.weight)
        self.records, self.credits = {}, {}
        self.rules = copy.deepcopy(previous.rules)
        self.pending, self.selected_policy = None, "entropy"
        self.xtx, self.xty = torch.eye(8, dtype=torch.float64), torch.zeros(8, dtype=torch.float64)
        self.training_receipts = []
        self.saturation = {"finished_questions": [], "next_action": "Complete the qualified prospective frontier"}
        if saved:
            if saved["parent_owner"] != self.parent_owner or saved["contracts"] != contracts() or saved["challenge_contract"] != guard_contract():
                raise ValueError("Changed challenged owner lineage")
            for name in ("records", "rules", "credits", "pending", "selected_policy", "training_receipts", "saturation"):
                setattr(self, name, copy.deepcopy(saved[name]))
            self.xtx, self.xty = decode(saved["xtx"]), decode(saved["xty"])
            self.owner.challenge_policy.load_state_dict(decode(saved["challenge_policy"]))
            for name, values in (("inquiry_beliefs", saved["beliefs"]), ("inquiry_rules", saved["rule_weights"])):
                for key, value in values.items():
                    getattr(self.owner, name)[key] = nn.Parameter(decode(value), requires_grad=False)
            if self.identity() != saved["owner"]:
                raise ValueError("Challenged owner failed exact restore")

    def policy_weights(self):
        return self.owner.challenge_policy.weight.detach().flatten().tolist()

    def learn_procedure(self, result, replay):
        if not result.get("known_basis_challenged") or not replay["accepted"] or replay["investigation"] != result["id"] or result["id"] in self.training_receipts:
            raise ValueError("Unverified or repeated challenged procedure credit")
        if replay["predictor"] != result["predictor"] or replay["original_goal"] != result["original_goal"]:
            raise ValueError("Stale procedure evidence")
        for item in result["training"]:
            x = torch.tensor(item["features"], dtype=torch.float64)
            self.xtx += torch.outer(x, x)
            self.xty += x * item["verified_later_target"]
        with torch.no_grad():
            self.owner.challenge_policy.weight.copy_(torch.linalg.solve(self.xtx, self.xty).unsqueeze(0))
        self.training_receipts.append(result["id"])

    def retain(self, result, *args):
        if not result.get("known_basis_challenged"):
            raise ValueError("Challenge omitted retained mechanisms before qualification")
        return super().retain(result, *args)

    def state(self):
        return {**super().state(), "schema": "sera.counterfactual-challenged-owner.1", "challenge_contract": guard_contract(),
                "challenge_policy": encode(self.owner.challenge_policy.state_dict())}
