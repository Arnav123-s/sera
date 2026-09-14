"""Procedural tasks with explicit instructions and separate RNG namespaces.

The final token is a query. Its value is hidden. Targets never enter the encoder.
These are synthetic symbolic tasks, not a natural-language benchmark.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

TASKS = ("marked_retrieval", "latest_binding", "ordered_control", "majority", "earliest_binding")
INPUT_DIM = 20
N_CLASSES = 4
# Four deliberately noncommuting operations, indexed [state, action].
CONTROL_TABLE = np.array([[1, 0, 0, 1], [0, 2, 0, 2], [3, 1, 3, 3], [2, 3, 3, 0]])


# The legacy dataset namespace is immutable: changing it would change the study data.
# It is a versioned RNG tag, independent of the project or import name.
def seed_for(namespace: str, seed: int, index: int = 0) -> int:
    blob = f"sable/data-v1/{namespace}/{seed}/{index}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little") % (2**63 - 1)


@dataclass
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor
    tasks: torch.Tensor
    actions: np.ndarray
    initial_states: np.ndarray
    dataset_id: str

    def to(self, device: str) -> "Batch":
        return Batch(
            self.inputs.to(device),
            self.targets.to(device),
            self.tasks.to(device),
            self.actions,
            self.initial_states,
            self.dataset_id,
        )


def make_batch(
    n: int, length: int, seed: int, *, split: str, index: int = 0, task: int | None = None
) -> Batch:
    if n < 1 or length < 3 or not split or (task is not None and task not in range(5)):
        raise ValueError("Invalid dataset dimensions, split, or task")
    rng = np.random.default_rng(seed_for(split, seed, index))
    steps = length - 1
    keys = rng.integers(0, 4, (n, steps))
    values = rng.integers(0, 4, (n, steps))
    marks = rng.integers(0, steps, n)
    # A query key is chosen uniformly, then one occurrence is guaranteed.
    query = rng.integers(0, 4, n)
    keys[np.arange(n), rng.integers(0, steps, n)] = query
    initial = rng.integers(0, 4, n)
    task_ids = rng.integers(0, 4, n) if task is None else np.full(n, task)
    x = np.zeros((n, length, INPUT_DIM), np.float32)
    x[:, :steps, :4] = np.eye(4)[keys]
    x[:, :steps, 4:8] = np.eye(4)[values]
    x[:, -1, :4] = np.eye(4)[query]
    x[:, :, 8:13] = np.eye(5)[task_ids][:, None, :]
    x[np.arange(n), marks, 13] = 1
    x[:, -1, 14] = 1
    x[:, :, 15] = np.arange(length) / (length - 1)
    x[:, 0, 16:20] = np.eye(4)[initial]
    labels = np.empty(n, np.int64)
    for row, mode in enumerate(task_ids):
        if mode == 0:
            labels[row] = values[row, marks[row]]
        elif mode in {1, 4}:
            occurrences = np.flatnonzero(keys[row] == query[row])
            labels[row] = values[row, occurrences[-1 if mode == 1 else 0]]
        elif mode == 2:
            state = initial[row]
            for action in values[row]:
                state = CONTROL_TABLE[state, action]
            labels[row] = state
        else:
            labels[row] = np.bincount(values[row], minlength=4).argmax()
    digest = hashlib.sha256(x.tobytes() + labels.tobytes()).hexdigest()
    return Batch(
        torch.from_numpy(x),
        torch.from_numpy(labels),
        torch.from_numpy(task_ids),
        values,
        initial,
        digest,
    )
