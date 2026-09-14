"""Explicit failure hypotheses and task-relevant acquisition from observed support."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from sera.contracts import EvidenceKind, Provenance
from sera.experience import EvidenceReplay, Trajectory

FAILURES = ("solved", "missing_evidence", "missing_procedure", "world_model", "search", "invalid_specification")


@dataclass(frozen=True)
class Diagnosis:
    category: str
    observed_prediction_accuracy: float
    goal_success: float
    transition_coverage: float
    previous_score: float
    previous_attempts: int
    remaining_budget: float
    rationale: str

    def record(self):
        return asdict(self)


def diagnose(prediction_accuracy, goal_success, support, *, previous_score=0.0,
             previous_attempts=0, remaining_budget=1.0, valid_specification=True):
    observed = {(before, action) for row in support.records
                for before, action, after in zip(row.observations, row.actions, row.observations[1:])
                if before >= 0 and after >= 0}
    coverage = len(observed) / 16
    if not valid_specification:
        category, rationale = "invalid_specification", "The task violates its declared input or resource contract"
    elif coverage < .75:
        category, rationale = "missing_evidence", "Observed support does not cover enough state/action consequences"
    elif prediction_accuracy < .85:
        category, rationale = "world_model", "Predictions disagree with observed consequences despite available support"
    elif goal_success < .85 and previous_attempts and prediction_accuracy > .95:
        category, rationale = "search", "Accurate short predictions have not produced a successful plan across attempts"
    elif goal_success < .85:
        category, rationale = "missing_procedure", "Available prediction competence is insufficient for the requested goal"
    else:
        category, rationale = "solved", "Observed prediction and goal checks pass the current diagnostic thresholds"
    return Diagnosis(category, float(prediction_accuracy), float(goal_success), coverage,
                     float(previous_score), int(previous_attempts), float(remaining_budget), rationale)


def choose_curriculum(diagnosis):
    return {"solved": "retention-check", "missing_evidence": "uncovered-transitions",
            "world_model": "prediction-counterexamples", "missing_procedure": "program-composition",
            "search": "longer-horizon-goals", "invalid_specification": "repair-task-contract"}[diagnosis.category]


def acquire_targeted(spec, evidence, *, count=64, length=8, seed=0, work=None,
                     model=None, curriculum="uncovered-transitions"):
    """Choose poorly covered or inconsistent observable transitions; never inspect the table."""
    if not spec.resettable:
        raise ValueError("Targeted transition experiments require declared reset access")
    if type(count) is not int or not 1 <= count <= 4096 or type(length) is not int or not 1 <= length <= 64:
        raise ValueError("Invalid targeted acquisition budget")
    rng = np.random.default_rng(seed)
    support = EvidenceReplay(evidence.records)
    visits, outcomes = Counter(), {}
    for row in support.records:
        if row.world_id != spec.identifier:
            continue
        for before, action, after in zip(row.observations, row.actions, row.observations[1:]):
            if before >= 0 and after >= 0:
                visits[(before, action)] += 1
                outcomes.setdefault((before, action), Counter())[after] += 1
    records, selections = [], []
    model_errors = {}
    if model is not None and curriculum == "prediction-counterexamples":
        import torch

        from sera.r1 import WorldSession
        with torch.no_grad():
            for (before, action), values in outcomes.items():
                session = WorldSession(model, spec.identifier, 0, before)
                logits, _ = model.predict(session.hidden, torch.tensor([action]), torch.tensor([0]))
                mode = max(values, key=values.get)
                model_errors[(before, action)] = 1 - float(logits.softmax(-1)[0, mode])
                if work is not None:
                    work.add("active_priority_model_predictions")
    for index in range(count):
        priorities = []
        for pair in ((s, a) for s in range(4) for a in range(4)):
            values = outcomes.get(pair, {})
            disagreement = (1 - max(values.values()) / sum(values.values())) if values else 1.0
            priorities.append((1 / (1 + visits[pair]) + disagreement + model_errors.get(pair, 0), pair))
        maximum = max(value for value, _ in priorities)
        ties = [pair for value, pair in priorities if abs(value - maximum) < 1e-12]
        start, first_action = ties[int(rng.integers(len(ties)))]
        goal = int(rng.integers(4))
        actions = (first_action,) + tuple(int(x) for x in rng.integers(4, size=length - 1))
        observations = spec.execute(start, actions)
        row = Trajectory(spec.identifier, observations, actions,
                         tuple(float(color == goal) for color in observations[1:]), goal,
                         Provenance(f"targeted-execution:{spec.identifier}", f"{seed}/{index}", EvidenceKind.VERIFIED),
                         "active-support")
        for before, action, after in zip(observations, actions, observations[1:]):
            visits[(before, action)] += 1
            outcomes.setdefault((before, action), Counter())[after] += 1
        records.append(row)
        selections.append({"start": start, "first_action": first_action, "priority": maximum,
                           "evidence": row.identifier})
        if work is not None:
            work.add("active_environment_resets")
            work.add("active_environment_actions", length)
            work.add("active_sensor_observations", length + 1)
    return records, {"policy": "observed-coverage-disagreement-and-prediction-error", "curriculum": curriculum,
                     "priority_refresh": "Predictions are frozen during each acquisition batch",
                     "selections": selections,
                     "observed_transition_coverage": len(visits) / 16}
