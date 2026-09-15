"""Preserve factual events and finite semantics under a narrowly changed owner.

This is exact replay/reproof. It is not learned latent migration, a relaxation of
the old guards, or a statistical certificate for the new neural route.
"""

import copy

import torch

from experiments.guarded_consolidation.core import Library, Work
from sera.event_ir import Event
from sera.generative import SharedGenerativeSession
from sera.shared import replace_shared_owner
from sera.world_graph import ExecutionBudget, LiveSituation, SharedOwnerRef


def changed_readout_only(original, successor):
    before, after = original.state_dict(), successor.state_dict()
    if set(before) != set(after):
        raise ValueError("Owner schema changed; readout migration is inapplicable")
    changed = []
    for name in before:
        if before[name].dtype != after[name].dtype or before[name].shape != after[name].shape:
            raise ValueError("Owner representation changed")
        if not torch.equal(before[name], after[name]):
            if not name.startswith("typed_numeric."):
                raise ValueError("Update exceeds the qualified motion readout")
            changed.append(name)
    if not changed:
        raise ValueError("No learned readout change to migrate")
    # Configuration changes other than trainability flags must be explicit.
    a, b = original.export_config(), successor.export_config()
    a.pop("generator_requires_grad", None)
    b.pop("generator_requires_grad", None)
    if a != b:
        raise ValueError("Executable configuration changed")
    return changed


def migrate(parent_solver, successor_owner, factual_record, library_record):
    parent_identity = parent_solver.identity()
    changed = changed_readout_only(parent_solver.components["r1"], successor_owner)
    old_ref = SharedOwnerRef.from_solver(parent_solver)
    # Validate each original dependency before permitting any rebinding.
    predecessor = SharedGenerativeSession.restore(parent_solver, factual_record)
    original_work = Work()
    Library.restore(library_record, old_ref, original_work, proof_backend="indexed")
    successor = copy.deepcopy(parent_solver)
    replace_shared_owner(successor, copy.deepcopy(successor_owner))
    new_ref = SharedOwnerRef.from_solver(successor)
    if new_ref.identity == old_ref.identity:
        raise ValueError("A changed readout must acquire a new owner identity")
    stale = {}
    for name, restore in {
        "factual_state": lambda: SharedGenerativeSession.restore(successor, factual_record),
        "finite_library": lambda: Library.restore(library_record, new_ref, Work(), proof_backend="indexed"),
    }.items():
        try:
            restore()
        except ValueError as error:
            stale[name] = str(error)
        else:
            raise ValueError("Old dependency record was accepted after an owner change")
    # Replay the unchanged factual log against the new identity. Numerical
    # statistics are then independently recomputed by SharedGenerativeSession.
    payload = factual_record["payload"]
    situation = LiveSituation(payload["situation_id"], successor.components["r1"].generator_graph(), new_ref,
                              budget=ExecutionBudget.restore(factual_record["budget"]), max_events=payload["max_events"])
    for event in payload["observations"]:
        situation._admit(Event.from_record(event), label=False, replay=True)
    for event in payload["labels"]:
        situation._admit(Event.from_record(event), label=True, replay=True)
    resumed = SharedGenerativeSession.restore(successor, situation.snapshot())
    # The formal finite programs do not call the neural readout. Rebinding their
    # owner is allowed only after the scoped-change check and complete reproof.
    rebound_record = copy.deepcopy(library_record)
    rebound_record["owner_sha256"] = new_ref.identity
    reproof = Work()
    library = Library.restore(rebound_record, new_ref, reproof, proof_backend="indexed")
    phases = [-1.23, -.47, .19, .81, 1.37]
    old_prediction = predecessor.predict(phases)
    new_prediction = resumed.predict(phases)
    if any(not torch.equal(old_prediction[k], new_prediction[k]) for k in ("mean", "covariance", "class_means", "class_probabilities")):
        raise ValueError("Generator behavior changed during readout migration")
    if successor.neural.owner is not successor.components["typed"].owner or successor.neural.owner is not new_ref.owner:
        raise ValueError("Common owner identity was lost")
    if parent_solver.identity() != parent_identity:
        raise ValueError("Migration modified its immutable predecessor")
    report = {"old_owner": old_ref.identity, "new_owner": new_ref.identity,
              "changed_tensors": changed, "stale_records_rejected": stale,
              "factual_observations_replayed": len(payload["observations"]),
              "factual_corrections_replayed": len(payload["labels"]),
              "generator_outputs_bitwise_equal": True,
              "original_proof_work": original_work.counts, "successor_proof_work": reproof.counts,
              "replay_and_generator_check_work": resumed.situation.budget.record(),
              "original_cumulative_budget": factual_record["budget"],
              "predecessor_validation_added_operations": {
                  k: v-factual_record["budget"]["counts"].get(k, 0)
                  for k, v in predecessor.situation.budget.counts.items()},
              "successor_replay_added_operations": {
                  k: v-factual_record["budget"]["counts"].get(k, 0)
                  for k, v in resumed.situation.budget.counts.items()},
              "finite_graph_unchanged": library.store.current.identity == library_record["graph_sha256"],
              "parent_solver_unchanged": True,
              "method": "Restricted readout delta, factual replay/statistics verification and exhaustive finite reproof",
              "statistical_certificates_transferred": 0, "learned_latent_migration": False}
    return successor, resumed, library, report
