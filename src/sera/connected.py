"""One solver connects learned dynamics, acquired programs and an improvement policy."""

from __future__ import annotations

import copy
import itertools
import time

import numpy as np
import torch

from sera.data import seed_for
from sera.environments import collect
from sera.experience import EvidenceReplay
from sera.r1 import RecurrentWorldModel, WorldSession, fit, score
from sera.r2 import ControlledInstrument, fit_instrument, flatten_program, search_program
from sera.solver import Work
from sera.storage import digest

METHODS = ("none", "update", "replay", "evidence", "planning", "program")


def restore_component(config):
    settings = dict(config)
    kind = settings.pop("type")
    if kind == "r1":
        return RecurrentWorldModel(**settings)
    if kind == "r2":
        return ControlledInstrument(**settings)
    if kind == "controller":
        from sera.curriculum import ImprovementPolicy
        return ImprovementPolicy(**settings)
    raise ValueError("Unknown component schema")


def world_programs(solver, world_id):
    return {key: record for key, record in solver.skills.items()
            if record.get("kind") == "action_program" and record["world_id"] == world_id}


@torch.no_grad()
def execute_goal(solver, spec, start, goal, *, mask_rate=0.0, seed=0, max_actions=6,
                 work=None, force_reactive=False, use_library=True, stop_on_success=True):
    """The public behavior path resolves admitted programs before neural planning."""
    rng = np.random.default_rng(seed)
    session = WorldSession(solver.components["r1"], spec.identifier, goal, start)
    library = world_programs(solver, spec.identifier) if use_library else {}
    choices = [record for record in library.values()
               if record["start"] == start and record["goal"] == goal]
    chosen = min(choices, key=lambda r: len(r["actions"])) if choices else None
    proposed = list(flatten_program(chosen["body"], library)) if chosen else []
    color, actions, sensors, rewards = start, [], [start], []
    for _ in range(max_actions):
        if color == goal and stop_on_success:
            break
        if proposed:
            action = proposed.pop(0)
        else:
            action = session.plan(horizon=1 if force_reactive else None, work=work)[0]
        color = spec.sensor_transition(color, action)
        reward = float(color == goal)
        sensor = -1 if rng.random() < mask_rate else color
        session.observe(sensor, action, reward)
        actions.append(action)
        sensors.append(sensor)
        rewards.append(reward)
        if work is not None:
            work.add("control_environment_actions")
            work.add("control_sensor_observations", int(sensor >= 0))
    return {"success": color == goal, "actions": actions, "observations": sensors,
            "rewards": rewards, "used_skill": chosen is not None,
            "world_id": spec.identifier, "start": start, "goal": goal}


def planned_evidence(solver, spec, *, seed, count=32, length=8, work=None):
    from sera.contracts import EvidenceKind, Provenance
    from sera.experience import Trajectory
    rng = np.random.default_rng(seed_for("planned-practice", seed))
    replay = EvidenceReplay()
    for index in range(count):
        start, goal = (int(x) for x in rng.integers(0, 4, 2))
        result = execute_goal(solver, spec, start, goal, max_actions=length,
                               work=work, stop_on_success=False, use_library=False)
        replay.admit(Trajectory(spec.identifier, tuple(result["observations"]),
                                tuple(result["actions"]), tuple(result["rewards"]), goal,
                                Provenance(f"planned-execution:{spec.identifier}",
                                           f"{seed}/{index}", EvidenceKind.VERIFIED), "planned-support"))
    return replay


def autonomous_round(store, spec, prior_specs, replay, *, seed, samples=1024, steps=32):
    """Load current solver, diagnose, select a learned intervention, execute and verify it."""
    incumbent = store.load()
    if "controller" not in incumbent.components:
        raise ValueError("Autonomous improvement requires a trained policy component")
    work = Work()
    support, _ = collect(spec, seed=seed, count=32, length=8, split="support", work=work)
    evidence = EvidenceReplay(support)
    features = diagnostic_features(incumbent, spec, seed=seed, support=evidence, work=work)
    method, utilities = incumbent.components["controller"].choose(features, resettable=spec.resettable)
    candidate, construction = intervene(incumbent, spec, evidence, replay,
                                         method=method, seed=seed, steps=steps, work=work)
    specs = list({s.identifier: s for s in [*prior_specs, spec]}.values())
    def evaluator(solver, fresh_seed):
        return evaluate_worlds(solver, specs, seed=fresh_seed, samples=samples, work=work)
    result = store.consider(candidate, evaluator, work=work,
                             description={"policy": "learned_intervention_values",
                                          "method": method, "world": spec.identifier,
                                          "features": features.tolist(), "predicted_utilities": utilities})
    return result, construction, evidence


def control_table(solver, spec, *, work=None, force_reactive=False, use_library=True):
    return [execute_goal(solver, spec, start, goal, work=work,
                         force_reactive=force_reactive, use_library=use_library)
            for start in range(4) for goal in range(4) if start != goal]


def evaluate_worlds(solver, specs, *, seed, samples=128, length=12, mask_rate=0.4, work=None):
    """Each bounded paired score represents an independent sampled trajectory/problem."""
    results, scores, identifiers = {}, {}, []
    for spec in specs:
        records, truth = collect(spec, seed=seed, count=samples, length=length,
                                 mask_rate=mask_rate, split="promotion-world", work=work)
        prediction, prediction_scores = score(solver.components["r1"], records, truth)
        # Exact enumeration caches deterministic unmasked control outcomes. Fresh random
        # pairs below are IID draws from this explicitly finite evaluation distribution.
        controls = control_table(solver, spec, work=work)
        rng = np.random.default_rng(seed_for(f"promotion-control/{spec.identifier}", seed))
        selected = rng.integers(0, len(controls), size=samples)
        control_scores = np.asarray([float(controls[i]["success"]) for i in selected])
        scores[spec.identifier] = (prediction_scores + control_scores) / 2
        results[spec.identifier] = {"prediction": prediction,
                                   "control_success": float(control_scores.mean()),
                                   "all_pair_control_success": sum(r["success"] for r in controls) / 12,
                                   "score": float(scores[spec.identifier].mean()), "control_cases": controls}
        identifiers.extend(r.identifier for r in records)
        identifiers.append(digest([spec.identifier, selected.tolist()]))
    return {"tasks": results, "macro_score": float(np.mean([x.mean() for x in scores.values()])),
            "dataset_id": digest(identifiers),
            "scoring": "Equal prediction-accuracy and goal-success weights per sampled episode; equal world weights",
            "control_scope": "Finite deterministic start/goal pairs; future-world generalization is reported separately"}, scores


def intervene(solver, spec, evidence, replay, *, method, seed=0, steps=32, work=None):
    """Actual candidate construction. The learning-policy targets come from measured results."""
    if method not in METHODS:
        raise ValueError("Unknown learning intervention")
    work = Work() if work is None else work
    started = time.perf_counter()
    candidate = copy.deepcopy(solver)
    training, programs = None, []
    if method in {"update", "replay", "evidence"}:
        support = EvidenceReplay(evidence.records)
        if method == "evidence":
            extra, _ = collect(spec, seed=seed, count=64, length=len(support.records[0].actions),
                               split="extra-support", work=work)
            for record in extra:
                support.admit(record)
        training = fit(candidate.components["r1"], support, steps=steps, batch_size=32,
                       seed=seed, replay=replay if method == "replay" else None, work=work)
    elif method == "planning":
        candidate.components["r1"].planning_horizon = 3
    elif method == "program":
        if not spec.resettable:
            raise ValueError("Reset access is required for this bounded program intervention")
        key = f"r2:{spec.identifier}"
        torch.manual_seed(seed_for("r2-init", seed))
        instrument = (candidate.components[key] if key in candidate.components
                      else ControlledInstrument())
        successful = []
        library = world_programs(candidate, spec.identifier)
        # Bootstrap verified execution credit using a supplied bounded fixed search.
        for start, goal in itertools.permutations(range(4), 2):
            found = search_program(spec, start, goal, budget=4, work=work)
            if found["success"]:
                successful.append(found["successful_trace"])
                record = {"kind": "action_program", **found["record"]}
                library[found["skill_id"]] = record
        training = fit_instrument(instrument, evidence, plans=successful,
                                  steps=steps, seed=seed, work=work)
        for start, goal in itertools.permutations(range(4), 2):
            found = search_program(spec, start, goal, model=instrument, budget=4, work=work)
            if found["success"]:
                successful.append(found["successful_trace"])
                record = {"kind": "action_program", **found["record"]}
                library[found["skill_id"]] = record
                programs.append({"start": start, "goal": goal, "actions": record["actions"],
                                 "attempts": found["attempts"]})
        # Verified post-training traces supply another genuine update to the proposal model.
        if successful:
            fit_instrument(instrument, evidence, plans=successful, steps=max(1, steps // 4),
                           seed=seed + 1, work=work)
        candidate.components[key] = instrument
        candidate.skills.update(library)
    work.seconds += time.perf_counter() - started
    return candidate.eval(), {"method": method, "training": training,
                              "programs": programs, "work": work.record()}


def diagnostic_features(solver, spec, *, seed, support, previous_score=0.0, work=None):
    records, _ = collect(spec, seed=seed, count=24, length=8, mask_rate=0.4,
                         split="diagnostic", work=work)
    # The competence estimator uses observable labels only, not missing sensor values.
    # Recompute the diagnostic accuracy on actually visible next events.
    from sera.r1 import tensors
    with torch.no_grad():
        logits, reward = solver.components["r1"](*tensors(records))
        observed = torch.tensor([r.observations[1:] for r in records])
        visible = observed >= 0
        observed_accuracy = float((logits.argmax(-1)[visible] == observed[visible]).float().mean())
        probabilities = logits.softmax(-1)
        entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean())
        reward_mse = float((reward.sigmoid() - torch.tensor([r.rewards for r in records])).square().mean())
    control = control_table(solver, spec, work=work)
    features = [observed_accuracy, sum(r["success"] for r in control) / 12,
                entropy / np.log(4), reward_mse,
                min(len(support.records), 128) / 128, float(spec.resettable),
                previous_score, solver.components["r1"].planning_horizon / 8]
    return np.asarray(features, dtype=np.float32)
