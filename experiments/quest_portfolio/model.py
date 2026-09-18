"""A finite-method selector registered on the continuing CompletionR1 owner."""

import copy

import torch
from torch import nn

from experiments.verified_completion.model import CompletionR1

from .common import contracts


class MethodPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 24, dtype=torch.float64), nn.Tanh(), nn.Linear(24, 1, dtype=torch.float64))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class QuestR1(CompletionR1):
    @classmethod
    def attach(cls, owner, seed=3301):
        if type(owner) is not CompletionR1:
            raise ValueError("Continue the actual CompletionR1 owner")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            policy = MethodPolicy()
        owner.__class__ = cls
        owner.quest_policy = policy
        owner.quest_config = {"contracts": contracts(), "seed": seed, "methods": "supplied typed mechanics"}
        return owner

    def export_config(self):
        return {**super().export_config(), "verified_quests": copy.deepcopy(self.quest_config)}
