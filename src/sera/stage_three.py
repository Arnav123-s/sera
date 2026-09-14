"""Finite, source-frozen follow-up studies for the original architecture audit."""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from sera.accounting import Costs
from sera.archive import retrieve_specialist, useful_archive
from sera.belief import (
    ClassicalBelief,
    RecurrentPredictor,
    aliased_trajectories,
    real_parameter_count,
    score_predictor,
)
from sera.connected import autonomous_round, control_table, evaluate_worlds, intervene
from sera.curriculum import ImprovementPolicy, fit_policy
from sera.environments import collect, make_world
from sera.experience import EvidenceReplay
from sera.models import ModelConfig
from sera.mutations import Mutation, mutate
from sera.r1 import RecurrentWorldModel, WorldSession, fit, score
from sera.r2 import ControlledInstrument, fit_instrument, search_program, search_transformation
from sera.solver import Solver, SolverStore, Work, tensor_digest
from sera.storage import digest, write_json
from sera.study import policy_episode, policy_summary
from sera.training import TrainConfig, environment, source_hash, train
from sera.typed_learning import TypedEvidence, TypedReasoner, fit_typed, score_typed, typed_examples

POLICY_METHODS = ("none", "update", "replay", "evidence", "targeted", "planning", "adapter", "program")


@dataclass(frozen=True)
class StageConfig:
    predictor_seconds: float = 24
    symbolic_steps: int = 500
    typed_steps: int = 1400
    world_steps: int = 360
    reference_steps: int = 120
    adaptation_steps: int = 32
    outer_inner_steps: int = 8
    outer_episodes: int = 12
    evaluation_samples: int = 128
    predictor_rates: tuple = (.015, .015, .015, .015, .015)

    def __post_init__(self):
        if min(self.predictor_seconds, self.symbolic_steps, self.typed_steps, self.world_steps, self.reference_steps,
               self.adaptation_steps, self.outer_inner_steps, self.evaluation_samples) <= 0:
            raise ValueError("Study budgets must be positive")
        if self.outer_episodes < 3 or self.outer_episodes % 3 or len(self.predictor_rates) != 5:
            raise ValueError("The outer study uses three equal generations and five predictor controls")


def predictors(root, seed, config, costs, work):
    support = aliased_trajectories(seed=44000 + seed, count=256, length=12, split="support")
    validation = aliased_trajectories(seed=45000 + seed, count=64, length=24, split="validation", structure="repeated")
    tests = {name: aliased_trajectories(seed=46000 + seed, count=256, length=length,
                                       split=f"test-{name}", structure=structure)
             for name, structure, length in (("ordinary", "ordinary", 12), ("repeated", "repeated", 24),
                                              ("long-random", "random", 40))}
    factories = {"projective": lambda: ControlledInstrument(),
                 "complex": lambda: ControlledInstrument(dimension=8, rank=2, event_kind="kraus", event_rank=2),
                 "real": lambda: ControlledInstrument(dimension=8, rank=4, event_kind="kraus", event_rank=4, complex_valued=False),
                 "hmm": lambda: ClassicalBelief(22), "gru": lambda: RecurrentPredictor(21)}
    results = {}
    for (name, factory), rate in zip(factories.items(), config.predictor_rates):
        with costs.phase(f"predictor/{name}", work):
            torch.manual_seed(100000 + seed)
            model = factory()
            learned = fit_instrument(model, EvidenceReplay(support), steps=100000, max_seconds=config.predictor_seconds,
                                       learning_rate=rate, validation=validation, validation_interval=50, seed=seed, work=work)
            results[name] = {"config": model.export_config(), "training": learned,
                             "likelihood_real_parameters": real_parameter_count(model, likelihood_only=True),
                             "total_real_parameters": real_parameter_count(model),
                             "tests": {key: score_predictor(model, rows, work=work) for key, rows in tests.items()}}
            torch.save({"config": model.export_config(), "state": model.state_dict()}, root / f"predictor-{name}.pt")
        print(f"seed {seed}: {name} repeated accuracy {results[name]['tests']['repeated']['accuracy']:.3f}", flush=True)
    return {"models": results, "support_ids": [row.identifier for row in support],
            "validation_ids": [row.identifier for row in validation],
            "test_ids": {name: [row.identifier for row in rows] for name, rows in tests.items()},
            "protocol": "Same admitted draws and wall-clock training cap including checkpoint validation. General real/complex and HMM/GRU approximately match likelihood parameters; unused proposal heads are reported separately. Projective is a smaller restricted control. Repeated structures were used in development validation; final draws are distinct. Long-random length 40 is extrapolation."}


def typed_study(seed, config, costs, work):
    with costs.phase("typed-learning", work):
        torch.manual_seed(200000 + seed)
        model = TypedReasoner()
        support = TypedEvidence(typed_examples(seed=210000 + seed, count=192))
        validation = typed_examples(seed=220000 + seed, count=48, split="validation")
        learned = fit_typed(model, support, validation=validation, steps=config.typed_steps, seed=seed, work=work)
        from sera.typed_programs import induce
        acquisition = {task: induce(support.records, validation, task=task, work=work)
                       for task in ("modular_sum", "byte_sum")}
        model.programs = {task: record["record"] for task, record in acquisition.items() if record["accepted"]}
        results = {}
        integrated = {}
        for family in ("ordinary", "extended", "structure"):
            rows = typed_examples(seed=230000 + seed, count=config.evaluation_samples, split=f"test-{family}", family=family)
            results[family] = score_typed(model, rows, work=work)
            integrated[family] = score_typed(model, rows, work=work, use_programs=True)
        examples = typed_examples(seed=240000 + seed, count=1, split="test-examples", family="structure")
        outputs = [{"task": row.task, "observations": [asdict(obs) for obs in row.observations],
                    "expected": row.target, "predicted": model.predict(row.observations, row.task)} for row in examples]
    return model, {"training": learned, "tests": results, "with_programs": integrated,
                   "program_acquisition": acquisition, "examples": outputs,
                   "scope": "Seven synthetic tasks across five typed modalities. Tiny byte strings, 4x4 patches and 16-sample audio are representation tests, not natural-language, vision or speech benchmarks."}


def oracle_controls(spec, work):
    outcomes = []
    for start, goal in itertools.permutations(range(4), 2):
        answer = None
        for length in range(1, 4):
            for actions in itertools.product(range(4), repeat=length):
                work.add("oracle_transition_accesses", length)
                if spec.execute(start, actions)[-1] == goal:
                    answer = list(actions)
                    break
            if answer is not None:
                break
        outcomes.append({"start": start, "goal": goal, "success": answer is not None, "actions": answer})
    return outcomes


def world_studies(root, seed, config, costs, work):
    spec = make_world(310000 + seed, family="permutation")
    with costs.phase("symbolic-sequence-teaching", work):
        from sera.evaluation import evaluate
        neural = train(ModelConfig(kind="delta"), TrainConfig(steps=config.symbolic_steps, seed=seed), root / "symbolic")
        work.add("symbolic_optimizer_steps", config.symbolic_steps)
        work.add("symbolic_update_examples", config.symbolic_steps * 64)
        symbolic, _ = evaluate(neural, seed=710000 + seed, split="test-stage-three-symbolic", samples=config.evaluation_samples)
    with costs.phase("world-teaching", work):
        rows, _ = collect(spec, seed=seed, count=256, length=8, mask_rate=.3, work=work)
        replay = EvidenceReplay(rows)
        torch.manual_seed(320000 + seed)
        model = RecurrentWorldModel()
        learned = fit(model, replay, steps=config.world_steps, seed=seed, work=work)
        base = Solver(neural, components={"r1": model})
    with costs.phase("world-planning-controls", work):
        rows, truth = collect(spec, seed=330000 + seed, count=config.evaluation_samples, length=20,
                               mask_rate=.4, split="test-world", work=work)
        prediction = score(model, rows, truth, work=work)[0]
        reset = score(model, rows, truth, reset_memory=True, work=work)[0]
        controls = {"reactive": control_table(base, spec, force_reactive=True, work=work),
                    "learned": control_table(base, spec, work=work), "oracle": oracle_controls(spec, work)}
    adaptations = []
    novel = make_world(340000 + seed, family="reset")
    old_test, old_truth = collect(spec, seed=seed, count=config.evaluation_samples, length=12,
                                 split="test-retention", mask_rate=.4, work=work)
    new_test, new_truth = collect(novel, seed=seed, count=config.evaluation_samples, length=12,
                                 split="test-adaptation", mask_rate=.4, work=work)
    for count in (8, 32, 128):
        support, _ = collect(novel, seed=seed, count=count, length=8, mask_rate=.2, work=work)
        for method in ("none", "update", "adapter", "replay", "scratch"):
            with costs.phase(f"adaptation/{count}/{method}", work):
                candidate, construction = intervene(base, novel, EvidenceReplay(support), replay,
                                                     method=method, steps=config.adaptation_steps, seed=seed, work=work)
                adaptations.append({"support_count": count, "method": method, "construction": construction,
                                     "old": score(candidate.components["r1"], old_test, old_truth, work=work)[0],
                                     "new": score(candidate.components["r1"], new_test, new_truth, work=work)[0]})
    references = []
    reference_models = {}
    for routing in ("all", "top1"):
        with costs.phase(f"reference-training/{routing}", work):
            torch.manual_seed(350000 + seed)
            reference = RecurrentWorldModel(width=256, heads=8, memory_dim=32, kind="reference", routing=routing)
            teaching = fit(reference, replay, steps=config.reference_steps, batch_size=8, seed=seed, work=work)
            reference_models[routing] = reference
        for rank in (2, 4, 8):
            with costs.phase(f"reference-rank/{routing}/{rank}", work):
                probe = copy.deepcopy(reference)
                probe.memory.density.rank = rank
                probe.settings["density_rank"] = rank
                probe.memory.begin_diagnostics()
                start = time.perf_counter()
                result = score(probe, old_test, old_truth, work=work)[0]
                seconds = time.perf_counter() - start
                diagnostics = probe.memory.end_diagnostics()
                masses = [mass for step in diagnostics if step["density"] is not None
                          for row in step["density"]["discarded_mass"] for mass in row]
                branches = {name: sum(step["branch_rows_executed"][name] for step in diagnostics)
                            for name in ("delta", "rotor", "density")}
                state = probe.initial(1)
                references.append({"routing": routing, "rank": rank, "training": teaching if rank == 4 else None,
                                   "prediction": result, "state_bytes": sum(v.numel() * v.element_size() for v in state.values()),
                                   "evaluation_seconds": seconds, "branch_rows_executed": branches,
                                   "discarded_mass_max": max(masses, default=0),
                                   "discarded_mass_mean": float(np.mean(masses)) if masses else 0,
                                   "diagnostic_events": len(diagnostics)})
    torch.save({"config": reference_models["top1"].export_config(), "state": reference_models["top1"].state_dict()}, root / "reference-top1.pt")
    result = {"world": asdict(spec), "symbolic": symbolic, "training": learned, "prediction": prediction, "memory_reset": reset,
              "planning": controls, "adaptation": adaptations, "reference": references,
              "reference_protocol": "Two separately trained full-dimension models, fixed optimizer steps. Rank 2/4/8 evaluation shares rank-4-trained weights; changes also affect initial density factors. This is a fixed-weight rank sensitivity study. Hard top-1 routing uses an explicitly declared surrogate gradient. Single-step discarded mass does not bound cumulative conditional error."}
    return base, spec, replay, result


def program_study(root, seed, config, costs, work):
    with costs.phase("program-acquisition-and-controls", work):
        spec = make_world(410000 + seed, family="permutation")
        support, _ = collect(spec, seed=seed, count=128, length=8, work=work)
        evidence, library = EvidenceReplay(support), {}
        # Acquire earlier two-step skills through actual search; they become callable tokens.
        for start, goal in itertools.permutations(range(4), 2):
            result = search_program(spec, start, goal, budget=20, max_length=2, work=work)
            for row in result["traces"]:
                evidence.admit(row)
            if result["success"] and len(result["record"]["actions"]) == 2:
                library[result["skill_id"]] = {"kind": "action_program", **result["record"]}
        # Teacher demonstrations cover whole two-action transformations. Their
        # inducing search never receives the teacher's generating program.
        for demonstrated_actions in itertools.product(range(4), repeat=2):
            targets_for_starts = [spec.execute(start, demonstrated_actions)[-1] for start in range(4)]
            work.add("program_demonstration_actions", 8)
            found = search_transformation(spec, targets_for_starts, budget=20, action_budget=160,
                                           max_length=2, candidate_budget=20, work=work)
            if not found["success"] or len(found["attempts"][-1]["actions"]) != 2:
                continue
            from sera.contracts import EvidenceKind, Provenance
            from sera.experience import Trajectory
            actions = tuple(found["attempts"][-1]["actions"])
            verified = [spec.execute(start, actions) for start in range(4)]
            work.add("program_composition_verification_actions", 8)
            new_traces = []
            for start, observations in enumerate(verified):
                goal = targets_for_starts[start]
                row = Trajectory(spec.identifier, observations, actions,
                                 tuple(float(value == goal) for value in observations[1:]), goal,
                                 Provenance("transformation-verification", digest([spec.identifier, start, actions]), EvidenceKind.VERIFIED),
                                 "program-support")
                evidence.admit(row)
                new_traces.append(row)
            record = {"kind": "action_program", "schema_version": 2, "world_id": spec.identifier,
                      "signature": "color + action_program -> color",
                      "start": 0, "goal": targets_for_starts[0], "actions": list(actions), "body": found["body"],
                      "dependencies": [], "dependency_versions": {},
                      "input_domain": {"world_id": spec.identifier, "verified_starts": list(range(4)), "reset_required": True},
                      "examples": [{"input": start, "output": targets_for_starts[start]} for start in range(4)],
                      "tests": [{"input": start, "expected": targets_for_starts[start], "observed": verified[start][-1]}
                                for start in range(4)], "evidence": [row.identifier for row in new_traces],
                      "failure_cases": found["attempts"][:-1], "scope": "Verified whole deterministic transformation"}
            library[digest(record)] = record
        guides = {}
        for name, factory in (("general", lambda: ControlledInstrument(dimension=8, rank=2, event_kind="kraus")),
                              ("classical", lambda: ClassicalBelief(22))):
            torch.manual_seed(420000 + seed)
            guide = factory()
            learned = fit_instrument(guide, evidence, steps=300, seed=seed, work=work)
            guides[name] = (guide, learned)
        training_maps = {tuple(spec.execute(start, actions)[-1] for start in range(4))
                        for length in (1, 2) for actions in itertools.product(range(4), repeat=length)}
        targets = {}
        macro_actions = sorted({tuple(row["actions"]) for row in library.values()})
        for actions in (a + b + (c,) for a in macro_actions for b in macro_actions for c in range(4)):
            mapping = tuple(spec.execute(start, actions)[-1] for start in range(4))
            work.add("heldout_problem_generation_actions", 4 * len(actions))
            if mapping not in training_maps:
                targets.setdefault(mapping, actions)
        cases = []
        if not targets:
            raise ValueError("Declared program holdout has no unseen transformations")
        for targets_for_starts, source_actions in list(sorted(targets.items()))[:8]:
            for guide_name, model in (("fixed", None), *[(name, pair[0]) for name, pair in guides.items()]):
                for enabled in (False, True):
                    trial = Work()
                    result = search_transformation(spec, targets_for_starts, library=library if enabled else {},
                                                     model=model, budget=8, action_budget=128, max_length=3,
                                                     candidate_budget=128, work=trial)
                    for key, count in trial.counts.items():
                        work.add(key, count)
                    cases.append({"guide": guide_name, "library": enabled, "source_actions": list(source_actions),
                                  "work": trial.record(), **result})
        write_json(root / "acquired-programs.json", library)
    return guides["general"][0], {"world": asdict(spec), "library": library, "cases": cases,
            "training": {name: learned for name, (_, learned) in guides.items()},
            "heldout_transformations": len(targets),
            "protocol": "Whole four-input transformation targets excluded from all primitive programs of length one or two. Queries compose two acquired macros plus an action. Same eight executed candidates and 128 executed environment-action cap, with at most three program tokens and 128 constructed candidates. Library calls permit longer primitive expansions; this is measured reuse under a token budget, not an equal primitive-length comparison. Candidate scoring uses likelihood, so classical/general match likelihood parameters; guides are per-world trained, not world-transfer models."}


def outer_study(root, seed, config, costs, work, base, spec, replay):
    episode_root = root / "policy-episodes"
    episode_root.mkdir()
    groups = {}
    for split, count in (("meta-train", config.outer_episodes), ("meta-validation", 3), ("meta-test", 10)):
        groups[split] = []
        for index in range(count):
            with costs.phase(f"outer-episode/{split}/{index}"):
                row = policy_episode(base, spec, replay, seed=seed, index=index, split=split,
                                       output=episode_root, steps=config.outer_inner_steps,
                                       methods=POLICY_METHODS, feature_count=16, samples=32)
            groups[split].append(row)
    torch.manual_seed(510000 + seed)
    policy = ImprovementPolicy(features=16, methods=POLICY_METHODS)
    generations = []
    frozen = []
    for generation in range(3):
        with costs.phase(f"outer-policy-update/{generation}", work):
            before = tensor_digest(policy)
            training = groups["meta-train"][:(generation + 1) * config.outer_episodes // 3]
            learned = fit_policy(policy, training, groups["meta-validation"], steps=400, seed=seed + generation)
            work.add("outer_policy_optimizer_steps", 400)
            work.add("outer_policy_training_episode_draws", 400 * len(training))
            work.add("outer_policy_validation_episode_draws", 21 * len(groups["meta-validation"]))
            # Save each eta before any anchor/frontier scores are computed.
            frozen.append(copy.deepcopy(policy))
            torch.save({"config": policy.export_config(), "state": policy.state_dict()}, root / f"outer-policy-{generation}.pt")
            generations.append({"generation": generation, "before": before, "after": tensor_digest(policy),
                                "training": learned, "training_episode_ids": [row["episode_id"] for row in training]})
    with costs.phase("outer-sealed-evaluation", work):
        for generation, candidate in enumerate(frozen):
            anchor = groups["meta-test"][:4]
            frontier = groups["meta-test"][4 + 2 * generation:6 + 2 * generation]
            generations[generation]["anchor"] = policy_summary(candidate, anchor)
            generations[generation]["frontier"] = policy_summary(candidate, frontier)
            # Fixed, uniform and difficulty curricula use identical recorded counterfactual outcomes.
            for name, rows in (("anchor", anchor), ("frontier", frontier)):
                uniform = float(np.mean([np.mean([v["utility"] for v in row["outcomes"].values()]) for row in rows]))
                difficulty = float(np.mean([row["outcomes"]["none" if row["features"][0] > .85 else "update"]["utility"] for row in rows]))
                diagnosis_values = []
                for row in rows:
                    category = row["diagnosis"]["category"]
                    choice = {"solved": "none", "missing_evidence": "targeted", "world_model": "update",
                              "missing_procedure": "planning", "search": "planning", "invalid_specification": "none"}[category]
                    diagnosis_values.append(row["outcomes"].get(choice, row["outcomes"]["none"])["utility"])
                generations[generation][name]["uniform_expected_utility"] = uniform
                generations[generation][name]["difficulty_utility"] = difficulty
                generations[generation][name]["fixed_diagnosis_utility"] = float(np.mean(diagnosis_values))
    return policy, {"generations": generations, "episode_counts": {name: len(rows) for name, rows in groups.items()},
                    "measured_interventions": sum(len(row["outcomes"]) for rows in groups.values() for row in rows),
                    "protocol": "Three actual eta updates, cumulative meta-training and separate validation, a fixed four-world anchor and two new frontier worlds per generation. All eta checkpoints are frozen before anchor/frontier scoring. Counterfactual outcome enumeration is charged as research cost; selected runtime cost is reported per outcome. The underlying inner solver is frozen across this outer comparison; persistent solver generations are evaluated separately. No anchor/frontier labels enter eta training."}


def persistence_study(root, seed, config, costs, work, base, spec, replay, typed, policy):
    base = copy.deepcopy(base)
    base.components["typed"] = typed
    base.components["controller"] = policy
    store = SolverStore(root / "solver")
    with costs.phase("solver-initialize-and-session", work):
        store.initialize(base)
        replay.save(store.root / "experience.json")
        write_json(store.root / "worlds.json", [asdict(spec)])
        session = WorldSession(store.load().components["r1"], spec.identifier, 2, 0, owner="stage-three-session")
        session.observe(spec.sensor_transition(0, 0), 0, 0.)
        session.save(root / "live-session.json")
    rounds = []
    known = [spec]
    for generation in range(3):
        world = make_world(610000 + seed * 100 + generation, family="reset" if generation == 2 else "permutation")
        with costs.phase(f"persistent-round/{generation}", work):
            result, construction, _ = autonomous_round(store, world, known, replay, seed=seed + generation,
                                                        samples=1024, steps=config.adaptation_steps, total_update_budget=96)
        rounds.append({"result": result, "construction": construction, "replay_records": len(replay.records)})
        known.append(world)
        write_json(store.root / "worlds.json", [asdict(world) for world in known])
    screens = []
    with costs.phase("typed-mutation-screening", work):
        proposal_work = Work()
        current = store.load()
        if current.components["r1"].adapter is None:
            candidate, mutation = mutate(current, Mutation("insert_adapter", value=4), seed=seed)
        else:
            candidate, mutation = copy.deepcopy(current), {"operation": "continue-existing-adapter"}
        screen_rows, screen_truth = collect(known[-1], seed=seed, count=64, split="validation-mutation", work=work)
        before = score(current.components["r1"], screen_rows, screen_truth, work=work)[0]
        support = EvidenceReplay([row for row in replay.records if row.world_id == known[-1].identifier])
        fit(candidate.components["r1"], support, steps=config.adaptation_steps, update_mode="adapter", seed=seed, work=proposal_work)
        candidate.validate()
        after = score(candidate.components["r1"], screen_rows, screen_truth, work=work)[0]
        screen = {"mutation": mutation, "validity": True, "development_before": before,
                  "development_after": after, "passed_development": after["accuracy"] >= before["accuracy"]}
        if screen["passed_development"]:
            def evaluator(solver, fresh_seed):
                return evaluate_worlds(solver, known, seed=fresh_seed, samples=1024, work=proposal_work)
            screen["admission"] = store.consider(candidate, evaluator, work=proposal_work,
                                                 description={"stage": "typed-adapter-mutation", "mutation": mutation})
        screens.append(screen)
        for key, count in proposal_work.counts.items():
            work.add(key, count)
    with costs.phase("useful-archive-retrieval", work):
        development = {world.identifier: collect(world, seed=seed, count=32, split="validation-archive", work=work)
                       for world in known}
        def competence(solver):
            return {name: score(solver.components["r1"], rows, truth, work=work)[0]["accuracy"]
                    for name, (rows, truth) in development.items()}
        archive = useful_archive(store, competence, dataset_id=digest([row.identifier for rows, _ in development.values() for row in rows]), work=work)
        selected = retrieve_specialist(store, archive, known[-1].identifier)
        selected_identity = selected.identity()
        # A retrieved lineage produces a real candidate; current remains the rollback parent.
        evidence = EvidenceReplay([row for row in replay.records if row.world_id == known[-1].identifier])
        proposal_work = Work()
        candidate, construction = intervene(selected, known[-1], evidence, replay, method="replay",
                                              steps=config.outer_inner_steps, seed=seed, work=proposal_work)
        def evaluator(solver, fresh_seed):
            return evaluate_worlds(solver, known, seed=fresh_seed, samples=1024, work=proposal_work)
        lineage = store.consider(candidate, evaluator, work=proposal_work,
                                 description={"stage": "retrieved-lineage", "source_identity": selected_identity})
        for key, count in proposal_work.counts.items():
            work.add(key, count)
    store.journal.verify()
    return {"rounds": rounds, "mutations": screens, "archive": archive,
            "retrieved_lineage": {"source_identity": selected_identity, "construction": construction, "admission": lineage},
            "current": store.current_record(), "replay_records": len(replay.records),
            "replay_dataset_id": digest(sorted(replay.identifiers))}


def run_stage_three(output, *, seeds=(0, 1, 2), config=None):
    config = StageConfig() if config is None else config
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh study directory to preserve evidence identity")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    manifest = {"schema_version": 1, "seeds": list(seeds), "config": asdict(config), "environment": environment(),
                "archive_sha256": "7e9268f8a0281190551ecfdadf8270eab989190e62e4621c03b93590c3932cce",
                "protocol": "Bounded architecture-audit follow-up. Per-seed training is independent. No final-result-driven configuration changes within this invocation."}
    write_json(output / "manifest.json", manifest)
    total = Costs()
    results = []
    try:
        for seed in seeds:
            root = output / str(seed)
            root.mkdir()
            costs, work = Costs(), Work()
            result = {"seed": seed}
            try:
                result["predictors"] = predictors(root, seed, config, costs, work)
                typed, result["typed"] = typed_study(seed, config, costs, work)
                base, spec, replay, result["world"] = world_studies(root, seed, config, costs, work)
                instrument, result["programs"] = program_study(root, seed, config, costs, work)
                base.components[f"r2:{result['programs']['world']['identifier']}"] = instrument
                base.skills.update(result["programs"]["library"])
                policy, result["outer"] = outer_study(root, seed, config, costs, work, base, spec, replay)
                result["persistence"] = persistence_study(root, seed, config, costs, work, base, spec, replay, typed, policy)
            finally:
                result["costs"], result["work"] = costs.record(), work.record()
                write_json(root / "study.json", result)
            if source_hash() != manifest["environment"]["source_sha256"]:
                raise ValueError("Executable source changed during the frozen study")
            results.append(result)
            print(f"seed {seed}: all stages completed", flush=True)
    finally:
        write_json(output / "accounting.json", total.record())
    write_json(output / "summary.json", {"manifest": manifest, "runs": results, "costs": total.record()})
    return results
