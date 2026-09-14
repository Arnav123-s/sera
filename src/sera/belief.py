"""Trainable classical baselines and an observation-aliased controlled process."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sera.contracts import EvidenceKind, Provenance
from sera.experience import Trajectory


class ClassicalBelief(nn.Module):
    def __init__(self, states=22, proposal_width=48):
        super().__init__()
        if not 4 <= states <= 128:
            raise ValueError("Classical belief size exceeds the declared bounds")
        self.states, self.proposal_width = states, proposal_width
        self.transition_raw = nn.Parameter(torch.randn(4, states, states) * .1)
        self.emission_raw = nn.Parameter(torch.randn(states, 4) * .1)
        self.initial_raw = nn.Parameter(torch.zeros(states))
        self.proposal = nn.Sequential(nn.Linear(states + 4, proposal_width), nn.Tanh(),
                                      nn.Linear(proposal_width, 4))

    def export_config(self):
        return {"type": "classical_belief", "states": self.states, "proposal_width": self.proposal_width}

    def operators(self):
        return self.transition_raw.softmax(-1), self.emission_raw.softmax(-1)

    def initial(self, batch):
        return self.initial_raw.softmax(-1).expand(batch, -1)

    def control(self, belief, actions, operators):
        return torch.bmm(belief[:, None], operators[0][actions]).squeeze(1)

    def branches(self, belief, operators):
        return belief[:, None] * operators[1].T[None]

    def probabilities(self, belief, operators):
        return belief @ operators[1]

    def observe(self, belief, colors, operators):
        branch = self.branches(belief, operators)[torch.arange(len(colors)), colors.clamp_min(0)]
        probability = branch.sum(-1)
        posterior = branch / probability.clamp_min(1e-20)[:, None]
        return torch.where((colors >= 0)[:, None], posterior, belief), probability

    def proposal_logits(self, belief, goals):
        return self.proposal(torch.cat((belief, F.one_hot(goals, 4)), -1))

    def sequence(self, observations, actions, goals):
        operators = self.operators()
        belief, _ = self.observe(self.initial(len(observations)), observations[:, 0], operators)
        predictions, proposals = [], []
        for index in range(actions.shape[1]):
            proposals.append(self.proposal_logits(belief, goals))
            belief = self.control(belief, actions[:, index], operators)
            predictions.append(self.probabilities(belief, operators))
            belief, _ = self.observe(belief, observations[:, index + 1], operators)
        return torch.stack(predictions, 1), torch.stack(proposals, 1), belief

    @torch.no_grad()
    def validity(self):
        transition, emission = self.operators()
        return {"transition_normalization": float((transition.sum(-1) - 1).abs().max()),
                "emission_normalization": float((emission.sum(-1) - 1).abs().max())}


class RecurrentPredictor(nn.Module):
    def __init__(self, width=21, proposal_width=48):
        super().__init__()
        self.width, self.proposal_width = width, proposal_width
        self.cell = nn.GRUCell(9, width)
        self.readout = nn.Linear(width + 4, 4)
        self.proposal = nn.Sequential(nn.Linear(width + 4, proposal_width), nn.Tanh(),
                                      nn.Linear(proposal_width, 4))

    def export_config(self):
        return {"type": "recurrent_predictor", "width": self.width, "proposal_width": self.proposal_width}

    def sequence(self, observations, actions, goals):
        hidden = self.readout.weight.new_zeros(len(observations), self.width)
        previous = hidden.new_zeros(len(observations), 4)
        predictions, proposals = [], []
        for index in range(actions.shape[1]):
            visible = (observations[:, index] >= 0)[:, None]
            sensor = F.one_hot(observations[:, index].clamp_min(0), 4) * visible
            hidden = self.cell(torch.cat((sensor, previous, visible), -1).float(), hidden)
            condition = F.one_hot(actions[:, index], 4)
            predictions.append(self.readout(torch.cat((hidden, condition), -1)).softmax(-1))
            proposals.append(self.proposal(torch.cat((hidden, F.one_hot(goals, 4)), -1)))
            previous = condition
        return torch.stack(predictions, 1), torch.stack(proposals, 1), hidden

    @torch.no_grad()
    def validity(self):
        return {"nonfinite_parameters": float(any(not torch.isfinite(p).all() for p in self.parameters()))}


def real_parameter_count(model, *, likelihood_only=False):
    return sum(value.numel() * (2 if value.is_complex() else 1)
               for name, value in model.named_parameters()
               if not likelihood_only or not name.startswith("proposal."))


def aliased_trajectories(*, seed, count=256, length=12, split="support", structure="ordinary"):
    """Eight hidden states emit four labels; only the observation/action sequence is retained.

    Initial labels reveal a bit. Two phases share label 2; label 3 also hides the bit.
    Action 2 flips the bit, action 0 advances, action 1 waits and action 3 skips a phase.
    Long repeated-action structures are excluded from ordinary training sequences.
    """
    if structure not in {"ordinary", "repeated", "random"}:
        raise ValueError("Unknown held-out action-structure family")
    rng = np.random.default_rng(seed)
    records = []
    for index in range(count):
        bit, phase, actions = int(rng.integers(2)), 0, []
        observations = [bit]
        for step in range(length):
            if structure == "repeated" and step % 4:
                action = actions[-1]
            else:
                action = int(rng.integers(4))
            if structure == "ordinary" and len(actions) >= 2 and actions[-2:] == [action, action]:
                action = (action + int(rng.integers(1, 4))) % 4
            if action == 2:
                bit = 1 - bit
            phase = (phase + (0 if action == 1 else 2 if action == 3 else 1)) % 4
            observations.append(bit if phase == 0 else 3 if phase == 3 else 2)
            actions.append(action)
        records.append(Trajectory("aliased-bit-phase-v1", tuple(observations), tuple(actions),
                                  tuple(float(color == 1) for color in observations[1:]), 1,
                                  Provenance("aliased-observation-simulator", f"{split}/{seed}/{index}",
                                             EvidenceKind.SYNTHETIC), split))
    return records


@torch.no_grad()
def score_predictor(model, records, *, work=None):
    from sera.r1 import tensors
    model.eval()
    nll, correct, brier, return_correct, count = 0.0, 0.0, 0.0, 0.0, 0
    returns = 0
    for start in range(0, len(records), 64):
        batch = records[start:start + 64]
        observations, actions, _, goals, _ = tensors(batch)
        probabilities, _, _ = model.sequence(observations, actions, goals)
        target = observations[:, 1:]
        visible = target >= 0
        selected = probabilities.gather(-1, target.clamp_min(0)[..., None]).squeeze(-1)
        nll += float(-selected[visible].clamp_min(1e-12).log().sum())
        correct += float((probabilities.argmax(-1)[visible] == target[visible]).sum())
        brier += float((probabilities - F.one_hot(target.clamp_min(0), 4)).square().sum(-1)[visible].sum())
        returning = visible & (target < 2)
        return_correct += float((probabilities.argmax(-1)[returning] == target[returning]).sum())
        returns += int(returning.sum())
        count += int(visible.sum())
        if work is not None:
            work.add("predictor_evaluation_transitions", int(visible.sum()))
    return {"accuracy": correct / count, "nll": nll / count, "brier": brier / count,
            "revealed_bit_accuracy": return_correct / max(returns, 1),
            "scored_transitions": count, "revealed_bit_transitions": returns}
