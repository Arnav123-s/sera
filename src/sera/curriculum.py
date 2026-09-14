"""A learned intervention-selection policy fitted to measured support/query outcomes."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sera.connected import METHODS


class ImprovementPolicy(nn.Module):
    def __init__(self, features=8, width=32):
        super().__init__()
        self.features, self.width = features, width
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.network = nn.Sequential(nn.Linear(features, width), nn.Tanh(), nn.Linear(width, len(METHODS)))

    def export_config(self):
        return {"type": "controller", "features": self.features, "width": self.width}

    def forward(self, features):
        return self.network((features - self.center) / self.scale.clamp_min(0.05))

    @torch.no_grad()
    def choose(self, features, *, resettable=True):
        self.eval()
        values = self(torch.as_tensor(features, dtype=torch.float32)[None])[0]
        if not resettable:
            values[METHODS.index("program")] = -torch.inf
        return METHODS[int(values.argmax())], {method: float(values[i])
                                              for i, method in enumerate(METHODS)
                                              if torch.isfinite(values[i])}


def utility(result, baseline, work, *, target):
    """Query improvement, empirical old-task regression and declared operation costs."""
    new_gain = result["tasks"][target]["score"] - baseline["tasks"][target]["score"]
    losses = [max(0, baseline["tasks"][name]["score"] - result["tasks"][name]["score"])
              for name in baseline["tasks"] if name != target]
    old_loss = max(losses, default=0.0)
    return float(new_gain - 2 * old_loss - 0.002 * np.log1p(work["operation_sum"]))


def fit_policy(model, training, validation, *, steps=500, seed=0):
    if not training or not validation or steps < 1:
        raise ValueError("Policy learning needs separate development and validation episodes")
    if {row["episode_id"] for row in training} & {row["episode_id"] for row in validation}:
        raise ValueError("Meta-training and validation episodes overlap")
    for row in [*training, *validation]:
        if row.get("evidence_kind") != "verified_outcome":
            raise ValueError("Policy targets require measured, verified intervention outcomes")
        if not row.get("query_dataset_id") or not row.get("support_dataset_id"):
            raise ValueError("Policy training needs explicit support/query provenance")
        if row["query_dataset_id"] == row["support_dataset_id"]:
            raise ValueError("Policy support and query evidence overlap")
        if len(row["features"]) != model.features or not np.isfinite(row["features"]).all():
            raise ValueError("Invalid diagnostic feature vector")
        if not row["outcomes"] or not set(row["outcomes"]).issubset(METHODS):
            raise ValueError("Unknown or empty intervention outcomes")
        if not all(np.isfinite(outcome["utility"]) for outcome in row["outcomes"].values()):
            raise ValueError("Nonfinite policy utility")
    def prepare(rows):
        x = torch.tensor([row["features"] for row in rows])
        y = torch.tensor([[row["outcomes"].get(method, {}).get("utility", 0.0)
                           for method in METHODS] for row in rows])
        mask = torch.tensor([[method in row["outcomes"] for method in METHODS] for row in rows])
        return x, y, mask
    x, targets, mask = prepare(training)
    vx, vy, vm = prepare(validation)
    model.center.copy_(x.mean(0))
    model.scale.copy_(x.std(0, unbiased=False).clamp_min(0.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    best, best_state, history = float("inf"), None, []
    torch.manual_seed(seed)
    for step in range(steps):
        model.train()
        loss = F.mse_loss(model(x)[mask], targets[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                validation_loss = float(F.mse_loss(model(vx)[vm], vy[vm]))
            history.append({"step": step + 1, "train_mse": float(loss.detach()),
                            "validation_mse": validation_loss})
            if validation_loss < best:
                best, best_state = validation_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return {"steps": steps, "training_episodes": len(training),
            "validation_episodes": len(validation), "validation_mse": best, "history": history,
            "scope": "A learned selector over six supplied interventions; it does not invent update algorithms"}
