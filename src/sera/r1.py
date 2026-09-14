"""Connected recurrent world prediction, imagined planning and learning from admitted feedback."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sera.data import seed_for
from sera.experience import EvidenceReplay
from sera.models import ModelConfig, StatefulModel


def context_vector(identifier):
    # Public environment identity, not its transition table or hidden physical state.
    raw = hashlib.sha256(identifier.encode()).digest()[:8]
    return torch.tensor([(x / 127.5) - 1 for x in raw], dtype=torch.float32)


class RecurrentWorldModel(nn.Module):
    def __init__(self, width=48, heads=2, memory_dim=8, kind="delta", planning_horizon=3):
        super().__init__()
        self.settings = dict(width=width, heads=heads, memory_dim=memory_dim, kind=kind)
        self.planning_horizon = planning_horizon
        if kind in {"reference", "lowrank_hybrid"}:
            from sera.lowrank import ReferenceMemory
            self.memory = ReferenceMemory(width, heads, memory_dim, compact=kind == "lowrank_hybrid")
        else:
            self.memory = StatefulModel(ModelConfig(**self.settings))
            self.memory.encoder = nn.Sequential(nn.Linear(22, width), nn.Tanh())
            self.memory.decoder = nn.Identity()
        self.transition = nn.Sequential(nn.Linear(width + 4, width * 2), nn.Tanh(),
                                        nn.Linear(width * 2, 4))
        self.fusion = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh())
        self.reward = nn.Sequential(nn.Linear(width + 8, width), nn.Tanh(), nn.Linear(width, 1))

    def export_config(self):
        return {"type": "r1", **self.settings, "planning_horizon": self.planning_horizon}

    def initial(self, batch):
        return self.memory.initial_state(batch)

    def observe(self, state, colors, previous_actions, rewards, goals, contexts):
        # A missing color is zeros plus an explicit availability flag; it is not a hidden ID.
        visible = (colors >= 0).float()[:, None]
        sensor = F.one_hot(colors.clamp_min(0), 4).float() * visible
        action = F.one_hot(previous_actions.clamp_min(0), 4).float()
        action = action * (previous_actions >= 0).float()[:, None]
        features = torch.cat((sensor, action, rewards[:, None], F.one_hot(goals, 4),
                              contexts, visible), -1)
        encoded = self.memory.encoder(features)
        hidden, next_state = self.memory.step(state, encoded, write=torch.ones(len(colors), 1))
        return self.fusion(torch.cat((encoded, hidden), -1)), next_state

    def predict(self, hidden, actions, goals):
        condition = torch.cat((hidden, F.one_hot(actions, 4)), -1)
        return self.transition(condition), self.reward(
            torch.cat((condition, F.one_hot(goals, 4)), -1)
        ).squeeze(-1)

    def forward(self, observations, actions, rewards, goals, contexts, *, reset_memory=False):
        state = self.initial(len(observations))
        previous = torch.full((len(observations),), -1, dtype=torch.long)
        received = torch.zeros(len(observations))
        predictions, reward_predictions = [], []
        for t in range(actions.shape[1]):
            if reset_memory:
                state = self.initial(len(observations))
            hidden, state = self.observe(state, observations[:, t], previous, received, goals, contexts)
            logits, reward = self.predict(hidden, actions[:, t], goals)
            predictions.append(logits)
            reward_predictions.append(reward)
            previous, received = actions[:, t], rewards[:, t]
        return torch.stack(predictions, 1), torch.stack(reward_predictions, 1)


def tensors(records):
    if not records or len({len(r.actions) for r in records}) != 1:
        raise ValueError("A batch needs equally long admitted trajectories")
    return (torch.tensor([r.observations for r in records]),
            torch.tensor([r.actions for r in records]),
            torch.tensor([r.rewards for r in records]),
            torch.tensor([r.goal for r in records]),
            torch.stack([context_vector(r.world_id) for r in records]))


def fit(model, evidence: EvidenceReplay, *, steps=200, batch_size=32, seed=0,
        learning_rate=0.003, replay=None, work=None):
    if not isinstance(evidence, EvidenceReplay) or not evidence.records or steps < 1:
        raise ValueError("Learning requires nonempty admitted evidence and a finite update budget")
    rng = np.random.default_rng(seed_for("r1-update-order", seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses = []
    model.train()
    for _ in range(steps):
        novel_n = batch_size if replay is None or not replay.records else batch_size // 2
        records = [evidence.records[i] for i in rng.integers(len(evidence.records), size=novel_n)]
        if novel_n != batch_size:
            records += [replay.records[i] for i in rng.integers(len(replay.records),
                                                              size=batch_size - novel_n)]
        obs, actions, rewards, goals, contexts = tensors(records)
        logits, predicted_rewards = model(obs, actions, rewards, goals, contexts)
        targets = obs[:, 1:]
        valid = targets >= 0
        prediction_loss = F.cross_entropy(logits[valid], targets[valid]) if valid.any() else logits.sum() * 0
        reward_loss = F.binary_cross_entropy_with_logits(predicted_rewards, rewards)
        loss = prediction_loss + 0.5 * reward_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite connected world-model loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if work is not None:
            work.add("optimizer_steps")
            work.add("update_trajectory_draws", len(records))
            work.add("update_transition_draws", actions.numel())
            work.add("novel_trajectory_draws", novel_n)
            work.add("replay_trajectory_draws", batch_size - novel_n)
    model.eval()
    return {"initial_batch_loss": losses[0], "final_batch_loss": losses[-1],
            "mean_last_20_loss": float(np.mean(losses[-20:])), "steps": steps,
            "admitted_examples": len(evidence.records), "history": losses}


@torch.no_grad()
def score(model, records, truth, *, reset_memory=False):
    model.eval()
    correct, nll, rewards, brier, entropy = [], [], [], [], []
    for start in range(0, len(records), 128):
        batch = records[start:start + 128]
        inputs = tensors(batch)
        logits, reward = model(*inputs, reset_memory=reset_memory)
        targets = torch.tensor(truth[start:start + len(batch), 1:])
        p = logits.softmax(-1)
        correct.extend((p.argmax(-1) == targets).float().mean(-1).tolist())
        nll.extend((-p.gather(-1, targets[..., None]).clamp_min(1e-12).log()).mean((1, 2)).tolist())
        brier.extend((p - F.one_hot(targets, 4)).square().sum(-1).mean(-1).tolist())
        rewards.extend((reward.sigmoid() - inputs[2]).square().mean(-1).tolist())
        entropy.extend((-(p * p.clamp_min(1e-12).log()).sum(-1)).mean(-1).tolist())
    return {"accuracy": float(np.mean(correct)), "nll": float(np.mean(nll)),
            "brier": float(np.mean(brier)), "reward_mse": float(np.mean(rewards)),
            "entropy": float(np.mean(entropy)), "episodes": len(records)}, np.asarray(correct)


class WorldSession:
    def __init__(self, model, world_id, goal, initial_observation):
        self.model = model.eval()
        self.world_id, self.goal = world_id, goal
        self.context = context_vector(world_id)[None]
        self.state = model.initial(1)
        self.hidden = None
        self.observe(initial_observation, -1, 0.0)

    @torch.no_grad()
    def observe(self, color, action, reward):
        self.hidden, self.state = self.model.observe(
            self.state, torch.tensor([color]), torch.tensor([action]), torch.tensor([reward]),
            torch.tensor([self.goal]), self.context
        )

    @torch.no_grad()
    def plan(self, *, horizon=None, beam=8, work=None):
        horizon = self.model.planning_horizon if horizon is None else horizon
        if not 1 <= horizon <= 8 or not 1 <= beam <= 64:
            raise ValueError("Invalid planning bounds")
        pending = [(self.state, self.hidden, (), 0.0)]
        best = None
        for depth in range(horizon):
            expanded = []
            for state, hidden, prefix, _ in pending:
                for action in range(4):
                    logits, reward = self.model.predict(hidden, torch.tensor([action]),
                                                        torch.tensor([self.goal]))
                    color = logits.argmax(-1)
                    probability = float(logits.softmax(-1)[0, self.goal])
                    value = (probability + float(reward.sigmoid()[0])) / 2 - 0.035 * depth
                    actions = prefix + (action,)
                    if best is None or value > best[0]:
                        best = (value, actions)
                    next_hidden, next_state = self.model.observe(
                        state, color, torch.tensor([action]), reward.sigmoid(),
                        torch.tensor([self.goal]), self.context
                    )
                    expanded.append((next_state, next_hidden, actions, value))
                    if work is not None:
                        work.add("planning_model_transitions")
            pending = sorted(expanded, key=lambda row: row[-1], reverse=True)[:beam]
        return best[1]


def updated(model, evidence, **kwargs):
    candidate = copy.deepcopy(model)
    record = fit(candidate, evidence, **kwargs)
    return candidate, record
