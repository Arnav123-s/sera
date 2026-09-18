"""Learning-procedure reward prediction registered on the continuing owner."""

import copy

import torch
from torch import nn

from experiments.quest_portfolio.model import QuestR1

from .common import PREFIXES, RATES, SKILLS, contracts


class ProgressR1(QuestR1):
    @classmethod
    def attach(cls, owner, seed):
        if type(owner) is not QuestR1:
            raise ValueError("Continue the actual QuestR1 owner")
        for name, parameter in owner.named_parameters():
            parameter.requires_grad_(name.startswith(PREFIXES))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            policy = nn.Sequential(nn.Linear(12, 24, dtype=torch.float64), nn.Tanh(),
                                   nn.Linear(24, 1, dtype=torch.float64))
            nn.init.zeros_(policy[-1].weight)
            nn.init.zeros_(policy[-1].bias)
        owner.__class__ = cls
        owner.learning_policy = policy
        owner.learning_config = {"seed": seed, "contracts": contracts()}
        return owner

    def export_config(self):
        return {**super().export_config(), "learning_progress": copy.deepcopy(self.learning_config)}


def knowledge(owner):
    return {k: v.detach().clone() for k, v in owner.state_dict().items() if k.startswith(PREFIXES)}


def apply_knowledge(owner, values):
    current = knowledge(owner)
    if set(values) != set(current) or any(v.shape != current[k].shape or v.dtype != current[k].dtype
                                        or not torch.isfinite(v).all() for k, v in values.items()):
        raise ValueError("Changed continuing knowledge schema")
    owner.load_state_dict(values, strict=False)


def candidates(scores, accuracy, recent, remaining):
    rows = []
    for i in range(len(SKILLS)):
        for j in range(len(RATES)):
            rows.append([float(k == i) for k in range(len(SKILLS))] +
                        [scores[i], accuracy[i], recent[i], remaining[i]/8, j/2, 1.])
    return torch.tensor(rows, dtype=torch.float64)
