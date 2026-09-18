"""On-policy credit with explicit decision, predictor, goal and independent audit binding."""

import copy

import numpy as np
import torch

from .common import digest, model_hash
from .task import score


def encode(value):
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "dtype": str(value.dtype).removeprefix("torch."), "value": value.tolist()}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[k, encode(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": [encode(v) for v in value]}
    return value


def decode(value):
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "tensor":
        dtypes = {"float64": torch.float64, "float32": torch.float32, "int64": torch.int64, "uint8": torch.uint8}
        result = torch.tensor(value["value"], dtype=dtypes[value["dtype"]])
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite saved tensor")
        return result
    if kind == "dict":
        return {k: decode(v) for k, v in value["items"]}
    if kind in {"tuple", "list"}:
        return (tuple if kind == "tuple" else list)(decode(v) for v in value["items"])
    raise ValueError("Invalid saved numerical state")


class Credit:
    def __init__(self, policy, saved=None):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        self.baseline, self.steps, self.receipts = 0., 0, []
        self.rng = torch.Generator().manual_seed(32401)
        if saved is not None:
            policy.load_state_dict(decode(saved["weights"]))
            self.optimizer.load_state_dict(decode(saved["optimizer"]))
            self.rng.set_state(decode(saved["rng"]))
            self.baseline, self.steps = saved["baseline"], saved["steps"]
            self.receipts = copy.deepcopy(saved["receipts"])
            if model_hash(policy) != saved["policy"] or self.steps != len(self.receipts):
                raise ValueError("Credit checkpoint mismatch")
            baseline, used = 0., set()
            for receipt in self.receipts:
                raw = {k: v for k, v in receipt.items() if k != "identity"}
                reward = self.reward(receipt["before"], receipt["after"], receipt["audit"]["value"], receipt["cost"])
                if digest(raw) != receipt["identity"] or reward != receipt["reward"] or receipt["audit"]["event_id"] in used:
                    raise ValueError("Credit replay integrity failed")
                used.add(receipt["audit"]["event_id"])
                baseline = .98*baseline+.02*reward
            if baseline != self.baseline:
                raise ValueError("Credit baseline lost its history")

    @staticmethod
    def reward(before, after, outcome, cost):
        if not np.isfinite([*before, *after, outcome, cost]).all() or min(before[1], after[1]) <= 0 or cost < 0:
            raise ValueError("Invalid independent predictive score")
        return float(score(*after, outcome)-score(*before, outcome)-cost)

    def decision(self, features, goal, predictor):
        f = np.asarray(features, dtype=np.float64)
        if f.shape != (9, 8) or not np.isfinite(f).all():
            raise ValueError("Nine bounded candidate records required")
        with torch.no_grad():
            probabilities = self.policy(torch.from_numpy(f)).softmax(-1)
            action = int(torch.multinomial(probabilities, 1, generator=self.rng))
        row = {"goal": goal, "predictor_before": predictor, "policy": model_hash(self.policy),
               "features": f.tolist(), "probabilities": probabilities.tolist(), "action": action,
               "sequence": self.steps, "sampling": "on-policy categorical"}
        return row | {"identity": digest(row)}

    def apply(self, decision, receipt):
        body = {k: v for k, v in decision.items() if k != "identity"}
        if digest(body) != decision["identity"] or decision["policy"] != model_hash(self.policy):
            raise ValueError("Changed or stale policy decision")
        if receipt["decision"] != decision["identity"] or receipt["goal"] != decision["goal"]:
            raise ValueError("Credit belongs to another decision or goal")
        if receipt["predictor_before"] != decision["predictor_before"]:
            raise ValueError("Credit belongs to another predictor")
        if any(r["audit"]["event_id"] == receipt["audit"]["event_id"] or r["decision"] == receipt["decision"] for r in self.receipts):
            raise ValueError("Duplicate decision or audit credit")
        raw = {k: v for k, v in receipt.items() if k != "identity"}
        if digest(raw) != receipt["identity"] or receipt["reward"] != self.reward(receipt["before"], receipt["after"], receipt["audit"]["value"], receipt["cost"]):
            raise ValueError("Corrupt independent receipt")
        if receipt["audit"]["purpose"] != "independent_audit" or receipt["audit"]["event_id"] in receipt["acquisition_ids"]:
            raise ValueError("Acquisition evidence cannot reward itself")
        params, optimizer = copy.deepcopy(self.policy.state_dict()), copy.deepcopy(self.optimizer.state_dict())
        try:
            logits = self.policy(torch.tensor(decision["features"], dtype=torch.float64))
            loss = -(receipt["reward"]-self.baseline)*logits.log_softmax(-1)[decision["action"]]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 2., error_if_nonfinite=True)
            self.optimizer.step()
            if not all(torch.isfinite(p).all() for p in self.policy.parameters()):
                raise ValueError("Nonfinite policy update")
        except Exception:
            self.policy.load_state_dict(params)
            self.optimizer.load_state_dict(optimizer)
            raise
        self.baseline = .98*self.baseline+.02*receipt["reward"]
        self.steps += 1
        self.receipts.append(copy.deepcopy(receipt))
        return {"policy_before": decision["policy"], "policy_after": model_hash(self.policy),
                "step": self.steps, "reward": receipt["reward"], "factual_updates": 0}

    def snapshot(self):
        return {"weights": encode(self.policy.state_dict()), "optimizer": encode(self.optimizer.state_dict()),
                "rng": encode(self.rng.get_state()), "baseline": self.baseline, "steps": self.steps,
                "receipts": copy.deepcopy(self.receipts), "policy": model_hash(self.policy)}
