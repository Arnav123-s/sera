"""Declared connected R1/R2 studies, counterfactual policy learning and persistent generations."""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sera.connected import (
    METHODS,
    autonomous_round,
    control_table,
    diagnostic_features,
    evaluate_worlds,
    execute_goal,
    intervene,
    planned_evidence,
)
from sera.curriculum import ImprovementPolicy, fit_policy, utility
from sera.data import seed_for
from sera.engine import improve
from sera.environments import collect, make_world
from sera.evaluation import evaluate
from sera.experience import EvidenceReplay
from sera.models import ModelConfig
from sera.r1 import RecurrentWorldModel, fit, score
from sera.r2 import ControlledInstrument, fit_instrument, search_program
from sera.solver import Solver, SolverStore, Work
from sera.storage import digest, write_json
from sera.training import TrainConfig, environment, train


def compact_report(report):
    return {"macro_score": report["macro_score"], "dataset_id": report["dataset_id"],
            "tasks": {name: {key: value for key, value in row.items() if key != "control_cases"}
                      for name, row in report["tasks"].items()}}


def policy_episode(base, base_spec, old_replay, *, seed, index, split, output, steps=16):
    episode_id = f"{split}/{seed}/{index}"
    episode_seed = seed_for(f"meta-episode/{split}", seed, index)
    known = split != "meta-test" and index % 4 == 0
    family = "reset" if split == "meta-test" else ("rotation" if index % 2 == 0 else "permutation")
    spec = base_spec if known else make_world(seed * 1000 + index +
                                             {"meta-train": 10000, "meta-validation": 20000,
                                              "meta-test": 30000}[split], family=family,
                                             resettable=index % 3 != 1)
    work = Work()
    support, _ = collect(spec, seed=episode_seed, count=16 if index % 2 else 64, length=8,
                         mask_rate=0.4 if index % 2 else 0.0, split="meta-support", work=work)
    evidence = EvidenceReplay(support)
    initial = copy.deepcopy(base)
    initial.components["r1"].planning_horizon = 1
    features = diagnostic_features(initial, spec, seed=episode_seed, support=evidence, work=work)
    specs = list({s.identifier: s for s in [base_spec, spec]}.values())
    query_seed = seed_for(f"{split}-query", seed, index)
    baseline_work = Work()
    baseline, _ = evaluate_worlds(initial, specs, seed=query_seed, samples=64, length=12,
                                  work=baseline_work)
    outcomes = {}
    for method in METHODS:
        if method == "program" and not spec.resettable:
            continue
        trial_work = Work()
        candidate, construction = intervene(initial, spec, evidence, old_replay,
                                             method=method, seed=episode_seed, steps=steps,
                                             work=trial_work)
        construction_cost = trial_work.record()
        evaluation_work = Work()
        result, _ = evaluate_worlds(candidate, specs, seed=query_seed, samples=64, length=12,
                                    work=evaluation_work)
        # Charge the extra planning work used by the changed policy as well as construction.
        extra = max(0, evaluation_work.counts.get("planning_model_transitions", 0)
                    - baseline_work.counts.get("planning_model_transitions", 0))
        trial_work.add("additional_evaluation_planning", extra)
        outcomes[method] = {"utility": utility(result, baseline, trial_work.record(), target=spec.identifier),
                            "result": compact_report(result), "construction_work": construction_cost,
                            "evaluation_work": evaluation_work.record(), "charged_work": trial_work.record(),
                            "training": construction["training"], "programs": construction["programs"]}
    row = {"episode_id": episode_id, "episode_seed": episode_seed,
           "support_record_ids": sorted(evidence.identifiers),
           "world": asdict(spec), "features": features.tolist(),
           "evidence_kind": "verified_outcome", "support_dataset_id": digest(sorted(evidence.identifiers)),
           "query_dataset_id": baseline["dataset_id"], "baseline": compact_report(baseline),
           "diagnosis_and_acquisition_work": work.record(), "outcomes": outcomes}
    write_json(output / f"{split}-{index}.json", row)
    print(f"{episode_id}: measured {len(outcomes)} interventions", flush=True)
    return row


def policy_summary(policy, rows):
    cases = []
    for row in rows:
        chosen, predicted = policy.choose(row["features"], resettable=row["world"]["resettable"])
        best = max(row["outcomes"], key=lambda method: row["outcomes"][method]["utility"])
        cases.append({"episode_id": row["episode_id"], "chosen": chosen, "posthoc_best": best,
                      "utility": row["outcomes"][chosen]["utility"],
                      "regret": row["outcomes"][best]["utility"] - row["outcomes"][chosen]["utility"],
                      "prediction": predicted,
                      "score": row["outcomes"][chosen]["result"]["macro_score"],
                      "baseline_score": row["baseline"]["macro_score"]})
    fixed = {method: float(np.mean([row["outcomes"].get(method, row["outcomes"]["none"])["utility"]
                                    for row in rows])) for method in METHODS}
    return {"cases": cases, "mean_utility": float(np.mean([r["utility"] for r in cases])),
            "mean_regret": float(np.mean([r["regret"] for r in cases])),
            "mean_score": float(np.mean([r["score"] for r in cases])),
            "mean_baseline_score": float(np.mean([r["baseline_score"] for r in cases])),
            "fixed_policy_utilities": fixed,
            "scope": "Reset-family worlds withheld from policy development; supplied method set; unavailable fixed methods fall back to none"}


def instrument_study(spec, replay, *, seed, output, steps=200):
    work = Work()
    training_plans, heldout = [], []
    for start in range(4):
        for goal in range(4):
            if start == goal:
                continue
            if (start + goal) % 3 == 0:
                heldout.append((start, goal))
            else:
                found = search_program(spec, start, goal, budget=32, work=work)
                if found["success"]:
                    training_plans.append(found["successful_trace"])
    torch.manual_seed(seed_for("controlled-instrument-init", seed))
    model = ControlledInstrument()
    untrained = copy.deepcopy(model)
    training = fit_instrument(model, replay, plans=training_plans, steps=steps, seed=seed, work=work)
    trials = []
    for method, guide in (("fixed", None), ("untrained_instrument", untrained), ("learned_instrument", model)):
        for start, goal in heldout:
            trial_work = Work()
            started = time.perf_counter()
            result = search_program(spec, start, goal, model=guide, budget=4, work=trial_work)
            trial_work.seconds = time.perf_counter() - started
            trials.append({"method": method, "start": start, "goal": goal, "success": result["success"],
                           "attempts": result["attempts"],
                           "actions": result.get("record", {}).get("actions"), "work": trial_work.record()})
    record = {"seed": seed, "world": asdict(spec), "training_programs": len(training_plans),
              "heldout_pairs": heldout, "training": training, "training_work": work.record(),
              "trials": trials,
              "scope": "Unseen start/goal combinations in the trained world; all transition types can appear in likelihood training; four execution attempts per search"}
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.export_config(), "state": model.state_dict()}, output / "instrument.pt")
    write_json(output / "instrument-study.json", record)
    return record


def run_seed(output: Path, *, seed, steps=900, meta_train=8, meta_validation=4, meta_test=4,
             inner_steps=16, instrument_steps=200):
    if (output / "study.json").exists():
        raise FileExistsError("Study exists; select a new output directory")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    work = Work()
    neural = train(ModelConfig(), TrainConfig(seed=seed, steps=500), output / "symbolic-training")
    work.add("symbolic_optimizer_steps", 500)
    work.add("symbolic_training_examples", 500 * 64)
    spec = make_world(seed + 50000, family="rotation")
    rows, _ = collect(spec, seed=seed, count=512, length=8, mask_rate=0.3,
                       split="initial-support", work=work)
    replay = EvidenceReplay(rows)
    replay.save(output / "initial-evidence.json")
    torch.manual_seed(seed_for("connected-r1-init", seed))
    r1 = RecurrentWorldModel(planning_horizon=1)
    validation, validation_truth = collect(spec, seed=seed, count=128, length=12, mask_rate=0.4,
                                           split="validation")
    initial_metrics, _ = score(r1, validation, validation_truth)
    r1_training = fit(r1, replay, steps=steps, seed=seed, work=work)
    base = Solver(neural, components={"r1": r1}).eval()
    r1_evaluations = {}
    for name, length, mask in (("visible", 12, 0.0), ("partial", 12, 0.4), ("long", 24, 0.4)):
        queries, truth = collect(spec, seed=seed + 1, count=256, length=length, mask_rate=mask,
                                  split=f"test-{name}", work=work)
        r1_evaluations[name], _ = score(r1, queries, truth)
        r1_evaluations[f"{name}_reset_memory"], _ = score(r1, queries, truth, reset_memory=True)
    planning_solver = copy.deepcopy(base)
    planning_solver.components["r1"].planning_horizon = 3
    r1_controls = {"reactive": control_table(base, spec, work=work, force_reactive=True),
                   "planned": control_table(planning_solver, spec, work=work),
                   "partial_planned": [execute_goal(planning_solver, spec, start, goal,
                                                      mask_rate=0.4, seed=seed + start * 4 + goal,
                                                      work=work)
                                        for start in range(4) for goal in range(4) if start != goal]}
    print(f"R1 seed={seed}: partial={r1_evaluations['partial']['accuracy']:.3f}; long={r1_evaluations['long']['accuracy']:.3f}", flush=True)
    rows_by_split = {}
    for split, count in (("meta-train", meta_train), ("meta-validation", meta_validation)):
        rows_by_split[split] = [policy_episode(base, spec, replay, seed=seed, index=index, split=split,
                                               output=output / "policy-episodes", steps=inner_steps)
                                for index in range(count)]
    torch.manual_seed(seed_for("controller-init", seed))
    policy = ImprovementPolicy()
    policy_training = fit_policy(policy, rows_by_split["meta-train"], rows_by_split["meta-validation"], seed=seed)
    # Freeze the policy before producing any final meta-test intervention outcomes.
    frozen_policy = copy.deepcopy(policy.state_dict())
    rows_by_split["meta-test"] = [policy_episode(base, spec, replay, seed=seed, index=index,
                                                split="meta-test", output=output / "policy-episodes",
                                                steps=inner_steps) for index in range(meta_test)]
    assert all(torch.equal(value, frozen_policy[name]) for name, value in policy.state_dict().items())
    policy_test = policy_summary(policy, rows_by_split["meta-test"])
    base.components["controller"] = policy
    store = SolverStore(output / "solver")
    store.initialize(base)
    replay.save(store.root / "experience.json")
    write_json(store.root / "worlds.json", [asdict(spec)])
    symbolic_generations = [improve(base, store.root, seed=seed, task=task, samples=2048)
                            for task in (2, 4)]
    # A full R1 plan -> actual feedback -> admitted replay -> candidate update -> fresh gate.
    feedback_spec = make_world(70000 + seed, family="permutation", resettable=False)
    feedback_work = Work()
    feedback = planned_evidence(store.load(), feedback_spec, seed=seed, count=32, work=feedback_work)
    feedback.save(output / "planned-evidence.json")
    updated, feedback_training = intervene(store.load(), feedback_spec, feedback, replay,
                                            method="replay", seed=seed, steps=120, work=feedback_work)
    def feedback_evaluator(solver, fresh_seed):
        return evaluate_worlds(solver, [spec, feedback_spec], seed=fresh_seed, samples=1024,
                                work=feedback_work)
    feedback_result = store.consider(updated, feedback_evaluator, work=feedback_work,
                                      description={"method": "planned-feedback-with-replay",
                                                   "world": feedback_spec.identifier})
    known = [spec, feedback_spec]
    generations = []
    for generation in range(2):
        target = make_world(90000 + seed * 10 + generation, family="reset")
        result, construction, evidence = autonomous_round(store, target, known, replay,
                                                           seed=seed + generation, samples=1024,
                                                           steps=inner_steps)
        generations.append({"result": result, "construction": construction, "world": asdict(target)})
        evidence.save(output / f"generation-{generation}-evidence.json")
        known.append(target)
        print(f"Autonomous seed={seed} generation={generation}: {construction['method']} -> {result['status']}", flush=True)
    final = store.load()
    write_json(store.root / "worlds.json", [asdict(s) for s in known])
    symbolic_after, _ = evaluate(final, seed=seed + 5, split="final-symbolic", samples=512, tasks=range(5))
    # Exercise behavioral rollback in a copy, keeping the live solver at its admitted version.
    copied = output / "rollback-check"
    shutil.copytree(store.root, copied)
    rollback_store = SolverStore(copied)
    rollback_before = rollback_store.load()
    parent = rollback_store.current_record()["parent"]
    expected = rollback_store.load(parent)
    rollback_store.rollback()
    restored = rollback_store.load()
    assert restored.identity() == expected.identity()
    rollback_record = {"before": rollback_before.version, "restored": restored.version,
                       "expected_parent_identity": expected.identity(),
                       "restored_identity": restored.identity(), "passed": True}
    r2_spec = make_world(60000 + seed, family="permutation")
    r2_rows, _ = collect(r2_spec, seed=seed, count=256, length=8, mask_rate=0.3,
                         split="instrument-support", work=work)
    r2_evidence = EvidenceReplay(r2_rows)
    r2_evidence.save(output / "r2-evidence.json")
    r2 = instrument_study(r2_spec, r2_evidence, seed=seed, output=output / "r2", steps=instrument_steps)
    total_seconds = time.perf_counter() - started
    result = {"seed": seed, "environment": environment(), "world": asdict(spec),
              "r1_initial": initial_metrics, "r1_training": r1_training, "r1_evaluation": r1_evaluations,
              "r1_controls": r1_controls,
              "policy_training": policy_training, "policy_test": policy_test,
              "symbolic_generations": symbolic_generations,
              "feedback_training": feedback_training, "feedback_round": feedback_result,
              "autonomous_generations": generations, "current_version": final.version,
              "current_identity": final.identity(), "symbolic_final": symbolic_after,
              "rollback": rollback_record, "r2": r2, "initial_work": work.record(),
              "total_wall_seconds": total_seconds,
              "teaching": {"symbolic_tasks": ["marked retrieval", "latest binding", "ordered actions", "majority"],
                           "acquired_symbolic_rule": "earliest binding",
                           "world_events": "four sensor colors, four actions, visible goal, external reward, missing-sensor flag and public world identity",
                           "latent_state_access": "hidden physical state and transition table never enter neural training; program search receives authorized reset/execution access",
                           "policy_methods": list(METHODS)}}
    write_json(output / "study.json", result)
    return result


def connected_study(output, *, seeds=(0, 1, 2), steps=900, **kwargs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": list(seeds), "steps": steps, "configuration": kwargs,
                "environment": environment(), "purpose": "Connected R1/R2 learning and persistent improvement"}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Study manifest differs; use a fresh directory")
    else:
        write_json(manifest_path, manifest)
    results = []
    for seed in seeds:
        completed = output / str(seed) / "study.json"
        if completed.exists():
            results.append(json.loads(completed.read_text(encoding="utf-8")))
        else:
            results.append(run_seed(output / str(seed), seed=seed, steps=steps, **kwargs))
    write_json(output / "summary.json", {"manifest": manifest, "runs": results})
    return results
