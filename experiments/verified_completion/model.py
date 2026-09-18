"""A learned investigator and joint completion weights on the retained owner."""

import copy

import numpy as np
import torch
from torch import nn

from experiments.book_learning.ground_model import GroundR1

from .common import contracts


class Investigator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32, dtype=torch.float64), nn.Tanh(),
                                 nn.Linear(32, 1, dtype=torch.float64))
        self.register_buffer("center", torch.zeros(8, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(8, dtype=torch.float64))

    def forward(self, features):
        return self.net((features-self.center)/self.scale).squeeze(-1)


class JointCompletion(nn.Module):
    def __init__(self, mean, covariance):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float64), requires_grad=False)
        self.register_buffer("covariance", torch.tensor(covariance, dtype=torch.float64))
        self.validate()

    def validate(self):
        m, c = self.mean.detach().numpy(), self.covariance.numpy()
        if m.shape != (3,) or c.shape != (3, 3) or not np.isfinite(m).all() or not np.isfinite(c).all():
            raise ValueError("Bounded finite joint completion required")
        if not np.allclose(c, c.T, atol=1e-12) or np.linalg.eigvalsh(c).min() <= 0:
            raise ValueError("Positive symmetric joint covariance required")

    def predict(self, query, noise=0.):
        x = np.asarray(query, dtype=np.float64)
        if x.shape != (3,) or not np.isfinite(x).all() or not np.isfinite(noise) or noise < 0:
            raise ValueError("Finite typed query required")
        return float(x@self.mean.detach().numpy()), float(x@self.covariance.numpy()@x+noise)

    @torch.no_grad()
    def observe(self, design, value, variance):
        x = np.asarray(design, dtype=np.float64)
        if x.shape != (3,) or not np.isfinite(x).all() or not np.isfinite([value, variance]).all() or variance <= 0:
            raise ValueError("Eligible finite observation required")
        m, c = self.mean.numpy().copy(), self.covariance.numpy().copy()
        gain = c@x/(x@c@x+variance)
        next_m = m+gain*(value-x@m)
        transform = np.eye(3)-np.outer(gain, x)
        next_c = transform@c@transform.T+variance*np.outer(gain, gain)
        candidate = JointCompletion(next_m, (next_c+next_c.T)/2)
        self.mean.copy_(candidate.mean)
        self.covariance.copy_(candidate.covariance)


class CompletionR1(GroundR1):
    @classmethod
    def attach(cls, owner, seed=3211):
        if type(owner) is not GroundR1:
            raise ValueError("Continue the actual GroundR1 owner")
        for parameter in owner.parameters():
            parameter.requires_grad_(False)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            policy = Investigator()
        owner.__class__ = cls
        owner.completion_policy = policy
        owner.completion_clouds = nn.ModuleDict()
        owner.completion_config = {"source": contracts(), "family": "a=b*u-d*v+c",
                                   "evidence_scope": "model_conditional_practice", "seed": seed}
        return owner

    def export_config(self):
        return {**super().export_config(), "verified_completion": copy.deepcopy(self.completion_config)}
