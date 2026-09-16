"""Qualification-gated persistence of the same owner, old facts and new interaction."""

import argparse
import copy
import gzip
import json
from pathlib import Path

import torch

from experiments.guarded_consolidation.core import Library, Work
from sera.event_ir import Event
from sera.generative import GenerativeSharedR1, SharedGenerativeSession
from sera.session_state import model_identity
from sera.shared import replace_shared_owner
from sera.world_graph import ExecutionBudget, LiveSituation, SharedOwnerRef

from .control import decide
from .core import ContinuingSession, InteractiveR1, fingerprint, sensor
from .environment import World
from .study import RELEASE, ROOT, check_contract, parent, read, sha, write

OUTPUT = RELEASE/"integration"


def validate_extension(original, successor):
    if type(successor) is not InteractiveR1 or successor.interaction_parent != model_identity(original):
        raise ValueError("Extension has the wrong original owner")
    if successor.interaction_source != fingerprint():
        raise ValueError("Extension interpreter changed")
    if set(dict(original.named_parameters())) != set(dict(successor.named_parameters())):
        raise ValueError("Additional trainable parameters are outside this extension")
    expected = InteractiveR1.extend(original, successor.interaction_window, successor.interaction_detect_change)
    if set(expected.state_dict()) != set(successor.state_dict()):
        raise ValueError("Unsupported state schema")
    if GenerativeSharedR1.export_config(successor) != original.export_config():
        raise ValueError("Original executable configuration changed")
    for name, tensor in original.state_dict().items():
        if not torch.equal(tensor, successor.state_dict()[name]):
            raise ValueError("Original neural/generator tensors changed")


def reattach_legacy(original, interaction):
    validate_extension(original.components["r1"], interaction.owner)
    baseline = original.identity()
    predecessor_record = read(ROOT/"research-continuation/17_transfer/integration/factual-situation.json")
    library_record = read(ROOT/"research-continuation/17_transfer/integration/finite-library.json")
    previous = SharedGenerativeSession.restore(original, predecessor_record)
    old_work = Work()
    Library.restore(library_record, SharedOwnerRef.from_solver(original), old_work, proof_backend="indexed")
    successor = copy.deepcopy(original)
    replace_shared_owner(successor, interaction.owner)
    reference = SharedOwnerRef.from_solver(successor)
    rejected = {}
    for name, call in {
        "factual_state": lambda: SharedGenerativeSession.restore(successor, predecessor_record),
        "finite_library": lambda: Library.restore(library_record, reference, Work(), proof_backend="indexed"),
    }.items():
        try:
            call()
        except ValueError as error:
            rejected[name] = str(error)
        else:
            raise ValueError("A stale whole-owner dependency was silently accepted")
    payload = predecessor_record["payload"]
    situation = LiveSituation(payload["situation_id"], interaction.owner.generator_graph(), reference,
                              budget=ExecutionBudget.restore(predecessor_record["budget"]), max_events=payload["max_events"])
    for event in payload["observations"]:
        situation._admit(Event.from_record(event), label=False, replay=True)
    for event in payload["labels"]:
        situation._admit(Event.from_record(event), label=True, replay=True)
    legacy = SharedGenerativeSession.restore(successor, situation.snapshot())
    rebinding = copy.deepcopy(library_record)
    rebinding["owner_sha256"] = reference.identity
    proof = Work()
    library = Library.restore(rebinding, reference, proof, proof_backend="indexed")
    phases = [-1.2, -.4, .2, .9, 1.4]
    a, b = previous.predict(phases), legacy.predict(phases)
    if any(not torch.equal(a[k], b[k]) for k in ("mean", "covariance", "class_means", "class_probabilities")):
        raise ValueError("Original executable generator changed")
    if original.identity() != baseline or successor.components["typed"].owner is not interaction.owner:
        raise ValueError("Original owner changed or the shared route detached")
    return successor, legacy, library, {"status": "PASS", "stale_bindings_rejected": rejected,
        "old_factual_observations": len(payload["observations"]), "old_corrections": len(payload["labels"]),
        "prior_cumulative_budget": predecessor_record["budget"], "restored_cumulative_budget": legacy.situation.budget.record(),
        "predecessor_proof_work": old_work.counts, "new_proof_work": proof.counts,
        "original_tensors_bitwise_equal": True, "original_generator_outputs_bitwise_equal": True,
        "parent_unchanged": True, "statistical_certificates_transferred": 0, "learned_latent_migration": False}


def integrate():
    directory = RELEASE/"A08-FINAL-001"
    check_contract(directory)
    audit = read(directory/"independent-audit.json")
    rule = read(RELEASE/"integration-rule.json")
    if not audit["gate"]["component_pass"]:
        raise ValueError("Prospective gate failed; preserve the challenger and A06 current owner")
    OUTPUT.mkdir(exist_ok=False)
    original = parent()
    chosen = rule["selection"].split(";")[0]
    raw = json.loads(gzip.decompress((directory/"lifetimes"/(chosen+".json.gz")).read_bytes()))
    selected = raw["snapshot"]
    interaction = ContinuingSession.restore(original.components["r1"], selected)
    write(OUTPUT/"selected-session.json", selected)
    # Four further paid interactions demonstrate actual continuation, without
    # changing any policy, retrospective gate or completed evaluation record.
    world = World(raw["spec"])
    state = raw["rows"][-1]["after_assessor"]
    world.angle, world.velocity, world.time = state["angle"], state["velocity"], state["time"]
    continuation = []
    for _ in range(4):
        plan = decide(interaction, world.goals(4), "mpc")
        interaction.validate_plan(plan)
        before = world.assessment()
        world.step(plan["action"])
        observed = world.measure()
        interaction.admit(sensor(observed, len(interaction.events), f"{raw['spec']['seed']}:{world.time}"), plan["action"])
        continuation.append({"before_assessor": before, "plan": plan, "observation": observed.tolist(),
                             "after_assessor": world.assessment()})
    successor, legacy, library, report = reattach_legacy(original, interaction)
    live = interaction.snapshot()
    write(OUTPUT/"interaction-session.json", live)
    write(OUTPUT/"factual-situation.json", legacy.snapshot())
    write(OUTPUT/"finite-library.json", library.record())
    write(OUTPUT/"four-new-interactions.json", {"rows": continuation, "paid_new_observations": 4,
          "paid_new_actions": 4, "world_at_end": world.assessment(), "spec": raw["spec"],
          "comparison_to_final_gate": "No retuning, new qualification claim or extra independent final world"})
    # Compact complete-owner reconstruction: exact immutable A06 parent plus
    # registered interaction state, factual lineage and unchanged formal programs.
    manifest = {"schema": 1, "base_solver": original.identity(), "solver_sha256": successor.identity(),
                "owner_sha256": model_identity(interaction.owner), "chosen": chosen,
                "files": {p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()},
                "source": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
                "report": report, "new_observation_count": len(interaction.events),
                "new_action_count": interaction.work["executed_actions"], "interaction_work": interaction.work,
                "status": "QUALIFIED_EXPERIMENTAL_COMPONENT", "operational_replacement": False}
    write(OUTPUT/"owner.json", manifest)
    restored_solver, restored_session, restored_library = restore()
    if restored_solver.identity() != successor.identity() or restored_session.snapshot()["buffers"] != live["buffers"]:
        raise ValueError("Integrated owner changed on full restoration")
    check = restored_library.answer(restored_library.store.current.definition("affine").identity,
                                    {"a": 4, "b": 5, "c": 6}, Work())
    if check != 0:
        raise ValueError("Original guarded finite skill did not survive")
    write(OUTPUT/"reload.json", {"status": "PASS", "solver_sha256": successor.identity(), "finite_answer": check,
                                "interaction_observations": len(restored_session.events),
                                "restore_work": restored_session.work})
    print(json.dumps({"status": "PASS", "solver": successor.identity(), "report": report,
                      "interaction_observations": len(interaction.events), "further_paid_interactions": 4}, indent=2))


def restore():
    manifest = read(OUTPUT/"owner.json")
    for name, expected in manifest["files"].items():
        if sha(OUTPUT/name) != expected:
            raise ValueError("Integrated state bytes changed")
    for name, expected in manifest["source"].items():
        if sha(Path(__file__).parent/name) != expected:
            raise ValueError("Integrated interaction runtime changed")
    solver = parent()
    if solver.identity() != manifest["base_solver"]:
        raise ValueError("Immutable parent changed")
    interaction = ContinuingSession.restore(solver.components["r1"], read(OUTPUT/"interaction-session.json"))
    replace_shared_owner(solver, interaction.owner)
    if solver.identity() != manifest["solver_sha256"]:
        raise ValueError("Restored solver identity differs")
    legacy = SharedGenerativeSession.restore(solver, read(OUTPUT/"factual-situation.json"))
    work = Work()
    library = Library.restore(read(OUTPUT/"finite-library.json"), SharedOwnerRef.from_solver(solver), work, proof_backend="indexed")
    if len(legacy.situation.observations) != 16 or len(legacy.situation.labels) != 1:
        raise ValueError("Original factual acquisition was lost")
    return solver, interaction, library


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("integrate", "status"))
    args = parser.parse_args()
    if args.mode == "integrate":
        integrate()
    else:
        solver, session, library = restore()
        print(json.dumps({"solver": solver.identity(), "observations": len(session.events), "work": session.work,
                          "status": "QUALIFIED_EXPERIMENTAL_COMPONENT", "guarded_programs": len(library.store.current.definitions)}, indent=2))
