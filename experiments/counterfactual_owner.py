"""Counterfactual knowledge and investigation learning on the existing shared R1."""

import copy
from fractions import Fraction as Q

import torch
from torch import nn

from experiments.counterfactual_core import at, generalized_shape
from experiments.gap_inquiry import ROOT, digest, sha
from experiments.solution_credit import CreditR1
from experiments.verified_completion.credit import decode, encode
from sera.session_state import model_identity

OUT = ROOT / "research-continuation/45_counterfactual_inquiry"


def contracts():
    paths = ["experiments/counterfactual_" + s + ".py" for s in ("core", "check", "loop", "owner")]
    paths.append("research-continuation/45_counterfactual_inquiry/PROTOCOL.md")
    return {p: sha(ROOT / p) for p in paths}


class CounterfactualR1(CreditR1):
    @classmethod
    def attach(cls, owner):
        if type(owner) is not CreditR1:
            raise ValueError("Continue the actual reconciled owner")
        owner.__class__ = cls
        owner.inquiry_policy = nn.Linear(8, 1, bias=False, dtype=torch.float64)
        nn.init.zeros_(owner.inquiry_policy.weight)
        owner.inquiry_beliefs = nn.ParameterDict()
        owner.inquiry_rules = nn.ParameterDict()
        owner.inquiry_contract = contracts()
        return owner

    def export_config(self):
        return {**super().export_config(), "counterfactual": self.inquiry_contract,
                "belief_keys": sorted(self.inquiry_beliefs), "inquiry_rule_keys": sorted(self.inquiry_rules)}


class InquirySession:
    def __init__(self, base, saved=None):
        self.base, self.parent_owner = base, base.identity()
        self.owner = CounterfactualR1.attach(base.owner)
        self.records, self.rules, self.credits = {}, {}, {}
        self.pending, self.selected_policy = None, "entropy"
        self.xtx = torch.eye(8, dtype=torch.float64)
        self.xty = torch.zeros(8, dtype=torch.float64)
        self.training_receipts = []
        self.saturation = {"finished_questions": [], "next_action": "Select an unexplored retained dependency"}
        if saved is not None:
            if saved["parent_owner"] != self.parent_owner or saved["contracts"] != contracts():
                raise ValueError("Changed counterfactual lineage or implementation")
            for name in ("records", "rules", "credits", "pending", "selected_policy", "training_receipts", "saturation"):
                setattr(self, name, copy.deepcopy(saved[name]))
            self.xtx, self.xty = decode(saved["xtx"]), decode(saved["xty"])
            self.owner.inquiry_policy.load_state_dict(decode(saved["policy"]))
            for key, value in saved["beliefs"].items():
                self.owner.inquiry_beliefs[key] = nn.Parameter(decode(value), requires_grad=False)
            for key, value in saved["rule_weights"].items():
                self.owner.inquiry_rules[key] = nn.Parameter(decode(value), requires_grad=False)
            if self.identity() != saved["owner"]:
                raise ValueError("Counterfactual weights failed exact restoration")

    def identity(self):
        return model_identity(self.owner)

    def policy_weights(self):
        return self.owner.inquiry_policy.weight.detach().flatten().tolist()

    def learn_procedure(self, result, replay):
        if not replay["accepted"] or replay["investigation"] != result["id"] or result["id"] in self.training_receipts:
            raise ValueError("Unverified or repeated procedure credit")
        if replay["predictor"] != result["predictor"] or replay["original_goal"] != result["original_goal"]:
            raise ValueError("Stale procedure evidence")
        for item in result["training"]:
            x = torch.tensor(item["features"], dtype=torch.float64)
            self.xtx += torch.outer(x, x)
            self.xty += x * item["verified_later_target"]
        with torch.no_grad():
            self.owner.inquiry_policy.weight.copy_(torch.linalg.solve(self.xtx, self.xty).unsqueeze(0))
        self.training_receipts.append(result["id"])

    def retain(self, result, proofs, replay, matrices, artifact):
        from experiments.counterfactual_check import after_discovery_words, check_generalization
        goal = result["original_goal"]
        if goal in self.records or result["id"] in self.credits:
            raise ValueError("Repeated investigation cannot earn fresh credit")
        if digest({k: v for k, v in result.items() if k != "id"}) != result["id"]:
            raise ValueError("Changed investigation commitment")
        if not replay["accepted"] or replay["investigation"] != result["id"] or replay["predictor"] != result["predictor"]:
            raise ValueError("A replay bound to this decision, goal and predictor is required")
        if replay["original_goal"] != goal or any(o.get("source") != replay["source"] for o in result["observations"]):
            raise ValueError("Evidence source or original goal changed")
        if len(proofs) != len(result["events"]) or any(not p["accepted"] or p["commitment"] != e["commitment"]
                                                     for p, e in zip(proofs, result["events"], strict=True)):
            raise ValueError("Every committed alternative must pass independent checking")
        if any(p["id"] != digest({k: v for k, v in p.items() if k != "id"}) or p["predictor"] != result["predictor"] for p in proofs):
            raise ValueError("Changed proof receipt or predictor")
        if self.pending and self.pending["predictor"] != result["predictor"]:
            raise ValueError("Pending investigation predictor changed")
        event = result["events"][-1]
        candidates = event["candidates"]
        survivors = set(result["survivors"])
        self.owner.inquiry_beliefs[goal] = nn.Parameter(torch.tensor([float(c["id"] in survivors) for c in candidates], dtype=torch.float64), requires_grad=False)
        added, points, names = [], 0., {}
        known_shapes = {digest(r["claim"]["shape"]) for r in self.rules.values()}
        for c in candidates:
            if c["id"] not in survivors:
                continue
            claim = generalized_shape(matrices, c, event["question"]["target"])
            names[c["id"]] = after_discovery_words(c)
            if not claim or claim["tautological_control"]:
                continue
            proof = check_generalization(claim)
            if not proof["accepted"]:
                raise ValueError("A proposed background-independent relationship failed proof")
            key = digest(claim)
            if key in self.rules:
                continue
            weights = [float(Q(x)) for x in claim["shape"]["n"] + claim["shape"]["d"]]
            self.owner.inquiry_rules[key] = nn.Parameter(torch.tensor(weights, dtype=torch.float64), requires_grad=False)
            shape = digest(claim["shape"])
            # Constant and controlled outcomes cannot farm discovery points.
            variable = len(claim["shape"]["n"]) > 1 or len(claim["shape"]["d"]) > 1
            reward = 1. if variable and shape not in known_shapes else 0.
            known_shapes.add(shape)
            self.rules[key] = {"claim": claim, "proof": proof, "original_goal": goal, "source": replay["source"],
                               "candidate": c["id"], "points": reward, "naming": names[c["id"]]}
            added.append(key)
            points += reward
        record = {"domain": event["question"]["domain"], "axis": event["question"]["axis"], "target": event["question"]["target"],
                  "question": event["question"]["question"], "status": result["status"], "investigation": result["id"],
                  "artifact": artifact, "survivors": sorted(survivors), "rules_added": added, "points": points,
                  "observations": len(result["observations"]), "followups": result["followups"], "naming": names,
                  "kind": "Verified conditional knowledge and simulated evidence"}
        self.records[goal] = record
        self.credits[result["id"]] = {"points": points, "original_goal": goal, "predictor": result["predictor"],
                                       "source": replay["source"], "proofs": [p["id"] for p in proofs], "rules": added}
        self.saturation["finished_questions"].append(goal)
        self.pending = None
        return record

    def apply_rule(self, key, initial, multiplier):
        record = self.rules[key]
        if Q(str(initial)) == 0:
            raise ValueError("Normalized retained knowledge requires a nonzero initial target")
        original = record["claim"]["shape"]
        weights = [str(Q(float(v)).limit_denominator(1000000)) for v in self.owner.inquiry_rules[key]]
        if weights != original["n"] + original["d"]:
            raise ValueError("Rule weights changed after qualification")
        n = len(original["n"])
        factor = at({"n": weights[:n], "d": weights[n:]}, Q(str(multiplier)))
        if factor is None:
            raise ValueError("The proposed scale is singular")
        return {"conditional_value": str(Q(str(initial)) * factor), "condition": record["claim"]["condition"],
                "required_mechanism": record["claim"]["spec"], "proof": record["proof"], "physical_fact_added": False}

    def state(self):
        return {"schema": "sera.counterfactual-owner.1", "parent_owner": self.parent_owner, "contracts": contracts(),
                **{name: copy.deepcopy(getattr(self, name)) for name in ("records", "rules", "credits", "pending", "selected_policy", "training_receipts", "saturation")},
                "xtx": encode(self.xtx), "xty": encode(self.xty), "policy": encode(self.owner.inquiry_policy.state_dict()),
                "beliefs": {k: encode(v.detach()) for k, v in self.owner.inquiry_beliefs.items()},
                "rule_weights": {k: encode(v.detach()) for k, v in self.owner.inquiry_rules.items()}, "owner": self.identity()}
