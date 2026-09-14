"""One solver connects learned dynamics, acquired programs and an improvement policy."""

from __future__ import annotations

import copy
import itertools
import json
import time

import numpy as np
import torch

from sera.data import seed_for
from sera.environments import collect
from sera.experience import EvidenceReplay
from sera.r1 import RecurrentWorldModel, WorldSession, fit, score
from sera.r2 import ControlledInstrument, fit_instrument, flatten_program, search_program
from sera.solver import Work
from sera.storage import digest, write_json

METHODS = ("none", "update", "replay", "evidence", "planning", "program")
EXTENDED_METHODS = METHODS + ("targeted", "adapter", "scratch")


def restore_component(config, components=None):
    settings = dict(config)
    kind = settings.pop("type")
    if kind == "r1":
        return RecurrentWorldModel(**settings)
    if kind == "shared_r1":
        from sera.shared import SharedR1
        settings.setdefault("encoding", "legacy-v1")
        settings.setdefault("structured_addresses", False)
        return SharedR1(**settings)
    if kind == "independent_typed_control":
        from sera.shared import IndependentTypedControl
        return IndependentTypedControl(restore_component(settings["core"]))
    if kind in {"shared_typed_view", "shared_sequence_view"}:
        from sera.shared import SharedSequenceView, SharedTypedView
        name = settings.pop("owner")
        if settings or components is None or name not in components:
            raise ValueError("Restore the registered shared owner before its interfaces")
        return (SharedTypedView if kind == "shared_typed_view" else SharedSequenceView)(components[name], name)
    if kind == "r2":
        return ControlledInstrument(**settings)
    if kind == "controller":
        from sera.curriculum import ImprovementPolicy
        return ImprovementPolicy(**settings)
    if kind in {"classical_belief", "recurrent_predictor"}:
        from sera.belief import ClassicalBelief, RecurrentPredictor
        return (ClassicalBelief if kind == "classical_belief" else RecurrentPredictor)(**settings)
    if kind == "typed_reasoner":
        from sera.typed_learning import TypedReasoner
        if settings.pop("event_schema") != 1:
            raise ValueError("Unsupported typed event schema")
        return TypedReasoner(**settings)
    raise ValueError("Unknown component schema")


def world_programs(solver, world_id):
    return {key: record for key, record in solver.skills.items()
            if record.get("kind") == "action_program" and record["world_id"] == world_id}


@torch.no_grad()
def execute_goal(solver, spec, start, goal, *, mask_rate=0.0, seed=0, max_actions=6,
                 work=None, force_reactive=False, use_library=True, stop_on_success=True):
    """The public behavior path resolves admitted programs before neural planning."""
    rng = np.random.default_rng(seed)
    session = WorldSession(solver.components["r1"], spec.identifier, goal, start,
                           model_version=solver.version)
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


def autonomous_round(store, spec, prior_specs, replay, *, seed, samples=1024, steps=32,
                     total_update_budget=256, admission_contract="separate-retention-v2"):
    """Load current solver, diagnose, select a learned intervention, execute and verify it."""
    incumbent = store.load()
    if "controller" not in incumbent.components:
        raise ValueError("Autonomous improvement requires a trained policy component")
    work = Work()
    history_path = store.root / "learning-history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    previous = [row for row in history if row["world"] == spec.identifier]
    used = sum(row["update_steps"] for row in history)
    available = max(0, total_update_budget - used)
    remaining = available / max(total_update_budget, 1)
    support, _ = collect(spec, seed=seed, count=32, length=8, split="support", work=work)
    evidence = EvidenceReplay(support)
    features, diagnosis = diagnostic_features(incumbent, spec, seed=seed, support=evidence, work=work,
                                               previous_score=previous[-1]["diagnostic_score"] if previous else 0.0,
                                               previous_attempts=len(previous), remaining_budget=remaining,
                                               feature_count=incumbent.components["controller"].features,
                                               return_diagnosis=True)
    method, utilities = incumbent.components["controller"].choose(features, resettable=spec.resettable,
                                                                 remaining_budget=available)
    candidate, construction = intervene(incumbent, spec, evidence, replay,
                                         method=method, seed=seed, steps=max(1, min(steps, available)), work=work,
                                         diagnosis=diagnosis)
    specs = list({s.identifier: s for s in [*prior_specs, spec]}.values())
    from sera.capability_evaluation import evaluate_capabilities, required_capabilities
    if admission_contract not in {"world-composite-v1", "separate-retention-v2"}:
        raise ValueError("Unknown admission contract")
    required = required_capabilities(incumbent, specs) if admission_contract == "separate-retention-v2" else None
    from sera.shared import SharedR1
    shared = isinstance(incumbent.components["r1"], SharedR1)
    if shared and admission_contract == "separate-retention-v2":
        from sera.shared_evaluation import shared_capabilities
        required = shared_capabilities(incumbent, specs, world_learning=True)
    def evaluator(solver, fresh_seed):
        if shared and required is not None:
            from sera.shared_evaluation import evaluate_shared
            return evaluate_shared(solver, specs, seed=fresh_seed, samples=samples, retained_samples=samples,
                                    work=work, world_learning=True)
        if required is not None:
            return evaluate_capabilities(solver, specs, seed=fresh_seed, samples=samples, work=work)
        return evaluate_worlds(solver, specs, seed=fresh_seed, samples=samples, work=work)
    result = store.consider(candidate, evaluator, work=work,
                             required_capabilities=required,
                             description={"policy": "learned_intervention_values",
                                          "method": method, "world": spec.identifier,
                                          "features": features.tolist(), "predicted_utilities": utilities,
                                          "diagnosis": diagnosis.record(), "remaining_update_budget": available})
    for row in evidence.records:
        replay.admit(row)
    replay.save(store.root / "experience.json")
    if shared and (store.root / "evidence-current.json").exists():
        from sera.shared_learning import SharedEvidence
        pointer = json.loads((store.root / "evidence-current.json").read_text(encoding="utf-8"))
        shared_evidence = SharedEvidence.load(store.root / "evidence" / pointer["revision"])
        shared_evidence.world = replay
        revision = "e" + digest([pointer, sorted(replay.identifiers)])[:16]
        with store.writing():
            shared_evidence.save(store.root / "evidence" / revision)
            write_json(store.root / "evidence-current.json", {"revision": revision})
            store.journal.append("shared_world_evidence_admitted", {"parent_revision": pointer["revision"], "revision": revision})
    actual_steps = construction.get("update_steps", 0)
    history.append({"world": spec.identifier, "method": method, "diagnosis": diagnosis.record(),
                    "diagnostic_score": float((features[0] + features[1]) / 2),
                    "update_steps": actual_steps, "candidate": result.get("candidate", result.get("version")),
                    "work": work.record(), "admitted_replay_records": len(replay.records)})
    write_json(history_path, history)
    return result, construction, evidence


def control_table(solver, spec, *, work=None, force_reactive=False, use_library=True):
    return [execute_goal(solver, spec, start, goal, work=work,
                         force_reactive=force_reactive, use_library=use_library)
            for start in range(4) for goal in range(4) if start != goal]


def evaluate_worlds(solver, specs, *, seed, samples=128, length=12, mask_rate=0.4, work=None,
                    separate_retention=False):
    """Each bounded paired score represents an independent sampled trajectory/problem."""
    results, scores, identifiers, capabilities, capability_ids = {}, {}, [], {}, {}
    for spec in specs:
        records, truth = collect(spec, seed=seed, count=samples, length=length,
                                 mask_rate=mask_rate, split="promotion-world", work=work)
        prediction, prediction_scores = score(solver.components["r1"], records, truth, work=work)
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
        if separate_retention:
            identity = digest([[r.identifier for r in records], selected.tolist()])
            for facet, values in (("prediction", prediction_scores), ("control", control_scores)):
                name = f"world/{spec.identifier}/{facet}"
                capabilities[name] = values
                capability_ids[name] = identity
    if separate_retention:
        from sera.evaluation import CapabilityScores
        scores = CapabilityScores(scores, capabilities, capability_ids)
    return {"tasks": results, "macro_score": float(np.mean([x.mean() for x in scores.values()])),
            "dataset_id": digest(identifiers),
            "scoring": "Equal prediction-accuracy and goal-success weights per sampled episode; equal world weights",
            "control_scope": "Finite deterministic start/goal pairs; future-world generalization is reported separately"}, scores


def intervene(solver, spec, evidence, replay, *, method, seed=0, steps=32, work=None, diagnosis=None):
    """Actual candidate construction. The learning-policy targets come from measured results."""
    if method not in EXTENDED_METHODS:
        raise ValueError("Unknown learning intervention")
    work = Work() if work is None else work
    started = time.perf_counter()
    candidate = copy.deepcopy(solver)
    training, programs, acquisition, mutation, update_steps = None, [], None, None, 0
    if method in {"update", "replay", "evidence", "targeted", "adapter", "scratch"}:
        support = EvidenceReplay(evidence.records)
        if method == "evidence":
            extra, _ = collect(spec, seed=seed, count=64, length=len(support.records[0].actions),
                               split="extra-support", work=work)
            for record in extra:
                support.admit(record)
                evidence.admit(record)
        elif method == "targeted":
            from sera.diagnostics import acquire_targeted, choose_curriculum
            extra, acquisition = acquire_targeted(spec, support, count=64,
                                                   length=len(support.records[0].actions), seed=seed, work=work,
                                                   model=candidate.components["r1"],
                                                   curriculum=choose_curriculum(diagnosis) if diagnosis else "uncovered-transitions")
            for record in extra:
                support.admit(record)
                evidence.admit(record)
        elif method == "adapter":
            if candidate.components["r1"].adapter is None:
                from sera.mutations import Mutation, mutate
                candidate, mutation = mutate(candidate, Mutation("insert_adapter", "r1", 4), seed=seed)
        elif method == "scratch":
            settings = dict(candidate.components["r1"].settings)
            torch.manual_seed(seed_for("scratch-world", seed))
            from sera.shared import SharedR1, replace_shared_owner
            if isinstance(candidate.components["r1"], SharedR1):
                settings = dict(candidate.components["r1"].export_config())
                settings.pop("type")
                settings.pop("programs")
                replace_shared_owner(candidate, SharedR1(**settings))
            else:
                candidate.components["r1"] = RecurrentWorldModel(**settings,
                                                                  planning_horizon=candidate.components["r1"].planning_horizon)
        training = fit(candidate.components["r1"], support, steps=steps, batch_size=32,
                       seed=seed, replay=replay if method == "replay" else None, work=work,
                       update_mode="adapter" if method == "adapter" else "all")
        update_steps = steps
    elif method == "planning":
        candidate.components["r1"].planning_horizon = 3
    elif method == "program":
        if not spec.resettable:
            raise ValueError("Reset access is required for this bounded program intervention")
        key = f"r2:{spec.identifier}"
        torch.manual_seed(seed_for("r2-init", seed))
        instrument = (candidate.components[key] if key in candidate.components
                      else ControlledInstrument(dimension=8, rank=2, event_kind="kraus"))
        successful = []
        library = world_programs(candidate, spec.identifier)
        # Bootstrap verified execution credit using a supplied bounded fixed search.
        for start, goal in itertools.permutations(range(4), 2):
            found = search_program(spec, start, goal, library=library, budget=4, work=work)
            for row in found["traces"]:
                evidence.admit(row)
            if found["success"]:
                successful.append(found["successful_trace"])
                record = {"kind": "action_program", **found["record"]}
                library[found["skill_id"]] = record
        instrument_steps = max(1, steps - max(1, steps // 5))
        training = fit_instrument(instrument, evidence, plans=successful,
                                  steps=instrument_steps, seed=seed, work=work)
        for start, goal in itertools.permutations(range(4), 2):
            found = search_program(spec, start, goal, model=instrument, library=library, budget=4, work=work)
            for row in found["traces"]:
                evidence.admit(row)
            if found["success"]:
                successful.append(found["successful_trace"])
                record = {"kind": "action_program", **found["record"]}
                library[found["skill_id"]] = record
                programs.append({"start": start, "goal": goal, "actions": record["actions"],
                                 "attempts": found["attempts"]})
        # Verified post-training traces supply another genuine update to the proposal model.
        if successful and steps > instrument_steps:
            credit_steps = steps - instrument_steps
            fit_instrument(instrument, evidence, plans=successful, steps=credit_steps,
                           seed=seed + 1, work=work)
        else:
            credit_steps = 0
        update_steps = instrument_steps + credit_steps
        candidate.components[key] = instrument
        candidate.skills.update(library)
    work.seconds += time.perf_counter() - started
    from sera.diagnostics import choose_curriculum
    return candidate.eval(), {"method": method, "training": training, "update_steps": update_steps,
                              "programs": programs, "acquisition": acquisition, "mutation": mutation,
                              "support_record_ids": sorted(evidence.identifiers),
                              "curriculum": choose_curriculum(diagnosis) if diagnosis is not None else None,
                              "work": work.record()}


def diagnostic_features(solver, spec, *, seed, support, previous_score=0.0, work=None,
                        previous_attempts=0, remaining_budget=1.0, feature_count=8,
                        return_diagnosis=False):
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
    from sera.diagnostics import FAILURES, diagnose
    diagnosis = diagnose(observed_accuracy, features[1], support, previous_score=previous_score,
                          previous_attempts=previous_attempts, remaining_budget=remaining_budget)
    if feature_count == 16:
        features += [float(diagnosis.category == name) for name in FAILURES]
        features += [float(np.clip(remaining_budget, 0, 1)), min(previous_attempts, 16) / 16]
    elif feature_count != 8:
        raise ValueError("Unknown diagnostic feature schema")
    if work is not None:
        work.add("diagnostic_model_transitions", sum(len(row.actions) for row in records))
    result = np.asarray(features, dtype=np.float32)
    return (result, diagnosis) if return_diagnosis else result
