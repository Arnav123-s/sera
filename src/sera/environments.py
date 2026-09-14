"""Small observable-color worlds; hidden state and truth stay in the environment/evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sera.contracts import EvidenceKind, Provenance
from sera.data import seed_for
from sera.experience import Trajectory


@dataclass(frozen=True)
class WorldSpec:
    identifier: str
    table: tuple[tuple[int, ...], ...]
    colors: tuple[int, ...] = (0, 1, 2, 3)
    resettable: bool = True

    def __post_init__(self):
        if len(self.table) != 4 or any(len(row) != 4 for row in self.table):
            raise ValueError("This laboratory uses four states and four actions")
        if sorted(self.colors) != list(range(4)) or any(
            type(v) is not int or not 0 <= v < 4 for row in self.table for v in row
        ):
            raise ValueError("Invalid world table or sensor permutation")

    def sensor_transition(self, color, action):
        """Environment/oracle operation. Learners use observed trajectories instead."""
        state = self.colors.index(int(color))
        return self.colors[self.table[state][int(action)]]

    def execute(self, start_color, actions):
        color = int(start_color)
        observations = [color]
        for action in actions:
            color = self.sensor_transition(color, action)
            observations.append(color)
        return tuple(observations)


def make_world(seed, *, family="permutation", resettable=True):
    rng = np.random.default_rng(seed_for("world-generator-v2", seed))
    if family == "rotation":
        offsets = rng.permutation(4)
        table = [[int((s + delta) % 4) for delta in offsets] for s in range(4)]
    elif family == "permutation":
        columns = [rng.permutation(4) for _ in range(4)]
        # One cycle guarantees reachability without granting that knowledge to the learner.
        columns[0] = (np.arange(4) + 1) % 4
        table = np.stack(columns, -1).tolist()
    elif family == "reset":
        table = [[int((s + 1) % 4), int((s - 1) % 4), 0, 2] for s in range(4)]
        table = np.asarray(table)[:, rng.permutation(4)].tolist()
    else:
        raise ValueError("Unknown world generator family")
    return WorldSpec(f"{family}-{seed}", tuple(tuple(row) for row in table),
                     tuple(int(c) for c in rng.permutation(4)), resettable)


def collect(spec, *, seed, count=64, length=8, mask_rate=0.0, split="support", work=None):
    if count < 1 or length < 1 or not 0 <= mask_rate <= 1:
        raise ValueError("Invalid collection budget")
    rng = np.random.default_rng(seed_for(f"{split}/{spec.identifier}", seed))
    records, truths = [], []
    for i in range(count):
        start, goal = (int(x) for x in rng.integers(0, 4, 2))
        actions = tuple(int(a) for a in rng.integers(0, 4, length))
        truth = spec.execute(start, actions)
        observed = (truth[0],) + tuple(
            -1 if rng.random() < mask_rate else color for color in truth[1:]
        )
        provenance = Provenance(f"simulator:{spec.identifier}", f"{split}/{seed}/{i}",
                                EvidenceKind.SYNTHETIC)
        records.append(Trajectory(spec.identifier, observed, actions,
                                  tuple(float(color == goal) for color in truth[1:]),
                                  goal, provenance, split))
        truths.append(truth)
    if work is not None:
        work.add(f"{split}_episodes", count)
        work.add(f"{split}_environment_actions", count * length)
        work.add(f"{split}_sensor_observations", sum(sum(x >= 0 for x in r.observations)
                                                   for r in records))
    # Truth is returned separately for scoring; admitted Trajectory contains only actual sensors.
    return records, np.asarray(truths)
