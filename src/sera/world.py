"""A fully observed finite control laboratory with learned action predictions.

This is a small R1 world-model integration, not a Dreamer reimplementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sera.data import CONTROL_TABLE, seed_for


class FiniteWorld:
    def __init__(self, table=None, *, environment_id="ordered-control-v1"):
        self._table = np.array(CONTROL_TABLE if table is None else table, dtype=np.int64, copy=True)
        if (
            self._table.ndim != 2
            or self._table.size == 0
            or np.min(self._table) < 0
            or np.max(self._table) >= len(self._table)
        ):
            raise ValueError("Invalid world transition table")
        self.environment_id = environment_id
        self.states, self.actions = self._table.shape

    def transition(self, state, action):
        if not 0 <= state < self.states or not 0 <= action < self.actions:
            raise ValueError("State or action outside environment")
        return int(self._table[state, action])

    def query(self, actions):
        state = 0
        for action in actions:
            state = self.transition(state, action)
        return state


class ActionWorldModel(nn.Module):
    def __init__(self, states=4, actions=4, width=32):
        super().__init__()
        self.states, self.actions = states, actions
        self.network = nn.Sequential(
            nn.Linear(states + actions, width), nn.Tanh(), nn.Linear(width, states)
        )

    def forward(self, states, actions):
        encoded = torch.cat((F.one_hot(states, self.states), F.one_hot(actions, self.actions)), -1)
        return self.network(encoded.float())

    @torch.no_grad()
    def distribution(self, state, action):
        self.eval()
        device = next(self.parameters()).device
        return (
            self(torch.tensor([state], device=device), torch.tensor([action], device=device))
            .softmax(-1)[0]
            .cpu()
            .numpy()
        )


@dataclass(frozen=True)
class Plan:
    actions: tuple[int, ...]
    predicted_states: tuple[int, ...]
    expansions: int
    reached: bool


def plan(
    distribution,
    *,
    start: int,
    goal: int,
    actions: int,
    max_horizon=8,
    max_expansions=64,
    confidence=0.5,
):
    if max_horizon < 0 or max_expansions < 1 or not 0 <= confidence <= 1:
        raise ValueError("Invalid planning limits")
    pending = deque([(start, (), (start,))])
    visited, expansions = {start}, 0
    while pending:
        state, prefix, history = pending.popleft()
        if state == goal:
            return Plan(prefix, history, expansions, True)
        if len(prefix) >= max_horizon:
            continue
        for action in range(actions):
            if expansions >= max_expansions:
                return Plan((), (), expansions, False)
            probabilities = np.asarray(distribution(state, action))
            if (
                probabilities.ndim != 1
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.isclose(probabilities.sum(), 1)
            ):
                raise ValueError("Invalid predictive distribution")
            expansions += 1
            target = int(probabilities.argmax())
            if probabilities[target] < confidence or target in visited:
                continue
            visited.add(target)
            pending.append((target, prefix + (action,), history + (target,)))
    return Plan((), (), expansions, False)


def world_experiment(seed=0, steps=300):
    if steps < 1:
        raise ValueError("Training steps must be positive")
    torch.manual_seed(seed_for("world-init", seed))
    world = FiniteWorld()
    model = ActionWorldModel(world.states, world.actions)
    rng = np.random.default_rng(seed_for("world-train", seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    observed_pairs = set()
    for _ in range(steps):
        states = rng.integers(0, world.states, 64)
        actions = rng.integers(0, world.actions, 64)
        targets = [world.transition(s, a) for s, a in zip(states, actions)]
        observed_pairs.update(zip(states.tolist(), actions.tolist()))
        loss = F.cross_entropy(
            model(torch.tensor(states), torch.tensor(actions)), torch.tensor(targets)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pairs = [(s, a) for s in range(world.states) for a in range(world.actions)]
    correct = sum(
        int(model.distribution(s, a).argmax()) == world.transition(s, a) for s, a in pairs
    )
    records = []
    random_rng = np.random.default_rng(seed_for("world-eval", seed))
    for start in range(world.states):
        for goal in range(world.states):
            if start == goal:
                continue
            learned = plan(model.distribution, start=start, goal=goal, actions=world.actions)
            oracle = plan(
                lambda s, a: np.eye(world.states)[world.transition(s, a)],
                start=start,
                goal=goal,
                actions=world.actions,
            )
            state = start
            for action in learned.actions:
                state = world.transition(state, action)
            random_state, random_steps = start, 0
            for action in random_rng.integers(0, world.actions, 8):
                random_steps += 1
                random_state = world.transition(random_state, int(action))
                if random_state == goal:
                    break
            records.append(
                {
                    "start": start,
                    "goal": goal,
                    "actions": list(learned.actions),
                    "model_expansions": learned.expansions,
                    "learned_success": learned.reached and state == goal,
                    "oracle_success": oracle.reached,
                    "oracle_steps": len(oracle.actions),
                    "random_success": random_state == goal,
                    "random_steps": random_steps,
                }
            )
    result = {
        "seed": seed,
        "training_steps": steps,
        "training_transitions": steps * 64,
        "observed_pairs": len(observed_pairs),
        "possible_pairs": len(pairs),
        "transition_accuracy": correct / len(pairs),
        "plans": records,
        "learned_success": sum(r["learned_success"] for r in records) / len(records),
        "oracle_success": sum(r["oracle_success"] for r in records) / len(records),
        "random_success": sum(r["random_success"] for r in records) / len(records),
        "scope": "Observable state IDs; all finite transitions may be seen in training; no unseen-dynamics claim",
    }
    return model, result
