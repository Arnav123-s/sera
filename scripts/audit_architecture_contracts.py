"""Targeted source-alignment probes; these are not new generalization experiments."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sera.connected import control_table
from sera.environments import WorldSpec, collect, make_world
from sera.experience import EvidenceReplay
from sera.r1 import RecurrentWorldModel, WorldSession
from sera.r2 import ControlledInstrument, fit_instrument, search_program
from sera.solver import SolverStore, Work, tensor_digest
from sera.storage import digest, write_json
from sera.training import source_hash


def instrument_reduction():
    maximum_posterior = maximum_probability = maximum_state = 0.0
    for seed in (17, 41, 83):
        for complex_valued in (False, True):
            torch.manual_seed(seed)
            model = ControlledInstrument(complex_valued=complex_valued)
            operators = model.operators()
            projectors = operators[1]
            raw = torch.randn(16, 4, 4, dtype=model.action_raw.dtype)
            rho = raw @ raw.mH
            rho = rho / rho.diagonal(dim1=-2, dim2=-1).sum(-1).real[:, None, None]
            for color in range(4):
                posterior, _ = model.observe(rho, torch.full((16,), color), operators)
                maximum_posterior = max(maximum_posterior, float((posterior - projectors[color]).abs().max()))
            transition = torch.stack([model.probabilities(model.control(projectors, torch.full((4,), action), operators), operators)
                                      for action in range(4)])
            belief = torch.full((1, 4), .25)
            state = model.initial(1)
            rng = np.random.default_rng(seed)
            for _ in range(128):
                action = int(rng.integers(4))
                color = int(rng.integers(-1, 4))
                state = model.control(state, torch.tensor([action]), operators)
                probabilities = model.probabilities(state, operators)
                predicted = belief @ transition[action]
                maximum_probability = max(maximum_probability, float((probabilities - predicted).abs().max()))
                state, _ = model.observe(state, torch.tensor([color]), operators)
                belief = F.one_hot(torch.tensor([color]), 4).float() if color >= 0 else predicted
                reconstructed = (belief[..., None, None] * projectors[None]).sum(1)
                maximum_state = max(maximum_state, float((state - reconstructed).abs().max()))
    return {"random_seeds": [17, 41, 83], "real_and_complex": True, "steps_per_case": 128,
            "posterior_equals_observed_projector_max_error": maximum_posterior,
            "classical_filter_probability_max_error": maximum_probability,
            "classical_filter_reconstructed_state_max_error": maximum_state,
            "interpretation": "In the current rank-one projective event basis, visible events reset predictive memory and missing events admit an exact four-state classical belief-filter representation, up to floating-point error"}


def training_contract():
    torch.manual_seed(17)
    world = make_world(991)
    support, _ = collect(world, seed=0, count=8, length=3)
    sealed, _ = collect(world, seed=1, count=1, length=3, split="query-audit")
    try:
        EvidenceReplay(sealed)
        replay_rejected = False
    except ValueError:
        replay_rejected = True
    model = ControlledInstrument()
    before = tensor_digest(model)
    try:
        fit_instrument(model, EvidenceReplay(support), plans=sealed, steps=1)
        accepted, reason = True, None
    except ValueError as error:
        accepted, reason = False, str(error)
    return {"sealed_replay_rejected": replay_rejected, "sealed_program_credit_accepted": accepted,
            "parameters_changed": before != tensor_digest(model), "reason": reason,
            "scope": "Synthetic audit counterexample; no evidence that the published study passed sealed records through this path"}


def proposal_contract():
    world = make_world(123)
    work = Work()
    try:
        search_program(world, 0, 1, budget=1, max_length=6, work=work)
        rejected, reason = False, None
    except ValueError as error:
        rejected, reason = True, str(error)
    bounded_work = Work()
    search_program(world, 0, 1, budget=1, max_length=5, work=bounded_work)
    return {"overlength_fixed_search_rejected": rejected, "reason": reason, "work": work.record(),
            "valid_length_five_budget_one_work": bounded_work.record(),
            "scope": "Length-five and length-six probes only; no unbounded or large-memory search executed"}


def integration_contracts():
    from sera import connected
    tree = ast.parse(Path(connected.__file__).read_text(encoding="utf-8"))
    calls = [{"line": node.lineno, "passes_library": any(k.arg == "library" for k in node.keywords)}
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "search_program"]
    session = WorldSession(RecurrentWorldModel(), "audit-world", 1, 0)
    return {"connected_search_calls": sorted(calls, key=lambda x: x["line"]),
            "world_session_has": {name: hasattr(session, name)
                                  for name in ("owner", "model_version", "encoder_version", "schema_version", "save", "load")}}


def valid_behavior_reference():
    torch.manual_seed(31)
    support, _ = collect(make_world(991), seed=0, count=8, length=3)
    model = ControlledInstrument()
    # Repeated admitted traces deliberately check preservation of sampling multiplicity.
    plans = support[:4] + [support[0], support[0]]
    fit = fit_instrument(model, EvidenceReplay(support), plans=plans, steps=2, seed=31)
    outcomes = []
    for world_seed in (123, 456):
        world = make_world(world_seed)
        for start in range(4):
            for goal in range(4):
                if start == goal:
                    continue
                for length in (1, 3, 5):
                    for budget in (1, 4, 12):
                        result = search_program(world, start, goal, budget=budget, max_length=length)
                        outcomes.append({"success": result["success"], "attempts": result["attempts"],
                                         "record": result.get("record"),
                                         "traces": [r.identifier for r in result["traces"]]})
    return {"training_seed": 31, "optimizer_steps": 2, "plan_count_including_repeats": len(plans),
            "training_history": fit["history"], "parameter_sha256": tensor_digest(model),
            "fixed_search_cases": len(outcomes), "fixed_search_outcomes_sha256": digest(outcomes),
            "scope": "Deterministic regression fixture, excluded from learning-performance estimates"}


def skill_ablation(study):
    results = []
    for seed in (0, 1, 2):
        root = study / str(seed) / "solver"
        solver = SolverStore(root).load()
        worlds = json.loads((root / "worlds.json").read_text(encoding="utf-8"))
        for row in worlds:
            spec = WorldSpec(row["identifier"], tuple(tuple(r) for r in row["table"]),
                             tuple(row["colors"]), row["resettable"])
            if not any(record.get("world_id") == spec.identifier for record in solver.skills.values()):
                continue
            with_library = control_table(solver, spec)
            without_library = control_table(solver, spec, use_library=False)
            results.append({"seed": seed, "world": spec.identifier,
                            "with_library_successes": sum(r["success"] for r in with_library),
                            "without_library_successes": sum(r["success"] for r in without_library),
                            "pairs": 12})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.no_grad():
        reduction = instrument_reduction()
    result = {"source_sha256": source_hash(), "instrument_reduction": reduction,
              "program_credit_contract": training_contract(), "proposal_budget_contract": proposal_contract(),
              "integration_contracts": integration_contracts(),
              "valid_behavior_reference": valid_behavior_reference(),
              "skill_library_ablation": skill_ablation(args.study) if args.study else [],
              "scope": "Post-publication architecture audit, mathematical and behavioral probes; not confirmatory generalization evidence"}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
