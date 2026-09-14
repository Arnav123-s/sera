"""Acquire bounded transition programs from experiments, then store verified skills.

Programs are validated data in a finite interpreter. There is no eval, Python
code generation, subprocess execution, network capability, or arbitrary import.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from sera.storage import digest, write_json


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class TransitionProgram:
    environment_id: str
    initial_state: int
    table: tuple[tuple[int, ...], ...]
    max_steps: int = 10_000

    def __post_init__(self):
        n = len(self.table)
        if not self.environment_id or not 1 <= n <= 256 or self.max_steps < 1:
            raise ValueError("Invalid program domain or size")
        width = len(self.table[0])
        if not 1 <= width <= 32 or not 0 <= self.initial_state < n:
            raise ValueError("Invalid action alphabet or initial state")
        if any(len(row) != width for row in self.table):
            raise ValueError("Transition table must be rectangular")
        if any(type(v) is not int or not 0 <= v < n for row in self.table for v in row):
            raise ValueError("Transition must name a valid state")

    @property
    def identifier(self):
        return digest(asdict(self))

    def execute(self, actions, *, initial_state=None, environment_id=None):
        if environment_id is not None and environment_id != self.environment_id:
            raise ValueError("Skill domain mismatch")
        state = self.initial_state if initial_state is None else int(initial_state)
        if not 0 <= state < len(self.table):
            raise ValueError("Unknown starting state")
        for step, action in enumerate(actions):
            if step >= self.max_steps:
                raise BudgetExhausted("Program instruction budget exhausted")
            if not isinstance(action, (int, np.integer)) or not 0 <= action < len(self.table[0]):
                raise ValueError("Action outside declared alphabet")
            state = self.table[state][action]
        return state


@dataclass(frozen=True)
class Discovery:
    program: TransitionProgram
    oracle_queries: int
    executed_actions: int
    observations: tuple[dict, ...]


def discover(
    query: Callable[[tuple[int, ...]], int],
    *,
    environment_id: str,
    actions: int,
    max_queries: int = 100,
    max_states: int = 32,
) -> Discovery:
    """Breadth-first active identification with repeated-query determinism checks.

    Assumes resettable dynamics and observable contiguous integer state IDs.
    Query costs include replaying the access sequence. Stronger feedback than
    the supervised neural task is made explicit in every experiment report.
    """
    if not 1 <= actions <= 32 or not 1 <= max_states <= 256 or max_queries < 1:
        raise ValueError("Invalid discovery budget")
    observations, count, executed = [], 0, 0

    def ask(sequence):
        nonlocal count, executed
        if count >= max_queries:
            raise BudgetExhausted("Active experiment query budget exhausted")
        state = query(sequence)
        count += 1
        executed += len(sequence)
        if type(state) is not int or not 0 <= state < max_states:
            raise ValueError("Oracle state is outside the declared observable finite-state class")
        observations.append({"actions": list(sequence), "observed_state": state})
        return state

    initial = ask(())
    access = {initial: ()}
    pending = deque([initial])
    transitions = {}
    while pending:
        state = pending.popleft()
        for action in range(actions):
            experiment = access[state] + (action,)
            observed = ask(experiment)
            if ask(experiment) != observed:
                raise ValueError("Repeated experiments contradict deterministic dynamics")
            transitions[state, action] = observed
            if observed not in access:
                access[observed] = experiment
                pending.append(observed)
    if set(access) != set(range(len(access))):
        raise ValueError("This program representation requires contiguous observed state IDs")
    table = tuple(tuple(transitions[s, a] for a in range(actions)) for s in range(len(access)))
    return Discovery(
        TransitionProgram(environment_id, initial, table), count, executed, tuple(observations)
    )


def verify_program(program, oracle, *, seed: int, samples: int = 256, lengths=(12, 24, 96)):
    if samples < 1 or not lengths or min(lengths) < 1:
        raise ValueError("Invalid verification dimensions")
    rng = np.random.default_rng(seed)
    results, transcripts = {}, []
    for length in lengths:
        correct = 0
        for sequence in rng.integers(0, len(program.table[0]), (samples, length)):
            inputs = tuple(int(a) for a in sequence)
            observed = oracle(inputs)
            correct += program.execute(inputs) == observed
            transcripts.append([list(inputs), observed])
        results[str(length)] = {
            "correct": correct,
            "samples": samples,
            "accuracy": correct / samples,
        }
    return {
        "by_length": results,
        "dataset_id": digest(transcripts),
        "passed": all(r["correct"] == samples for r in results.values()),
    }


class SkillLibrary:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def admit(self, discovery: Discovery, verification: dict):
        if not verification.get("passed") or not verification.get("dataset_id"):
            raise ValueError("A skill needs successful independent execution evidence")
        by_length = verification.get("by_length", {})
        if not by_length or any(
            v["samples"] < 1 or v["correct"] != v["samples"] for v in by_length.values()
        ):
            raise ValueError("Incomplete skill verification record")
        program = discovery.program
        record = {
            "schema_version": 1,
            "skill_id": program.identifier,
            "signature": "(actions: sequence[int], initial_state: int) -> int",
            "program": asdict(program),
            "verification": verification,
            "oracle_queries": discovery.oracle_queries,
            "executed_actions": discovery.executed_actions,
            "assumptions": [
                "deterministic",
                "resettable",
                "observable state IDs",
                "fixed action alphabet",
                "same environment version",
            ],
            "dependencies": [],
            "evidence": list(discovery.observations),
        }
        write_json(self.root / f"{program.identifier}.json", record)
        return program.identifier

    def load(self, identifier):
        if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Invalid skill identifier")
        record = json.loads((self.root / f"{identifier}.json").read_text(encoding="utf-8"))
        p = record["program"]
        program = TransitionProgram(
            p["environment_id"],
            p["initial_state"],
            tuple(tuple(row) for row in p["table"]),
            p["max_steps"],
        )
        if record["schema_version"] != 1 or program.identifier != identifier:
            raise ValueError("Skill content does not match its identifier")
        return program
