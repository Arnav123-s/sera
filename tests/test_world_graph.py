"""Ownership/lifecycle fixtures, not learned grounding or capability evaluations."""

import copy
import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from sera.contracts import EvidenceKind, Observation, Provenance
from sera.event_ir import Event, EventRole, EventSchema
from sera.shared import SharedR1, make_shared_solver, replace_shared_owner
from sera.storage import digest
from sera.world_graph import (
    DependencyRef,
    ExecutableDefinition,
    ExecutableGraph,
    ExecutionBudget,
    GraphStore,
    LiveSituation,
    MechanismRef,
    SharedOwnerRef,
    interpreter_fingerprint,
)


@pytest.fixture
def schema():
    return EventSchema("position", 1, "numeric", 2, "m", entity_slots=1)


@pytest.fixture
def owner():
    torch.manual_seed(7)
    return SharedR1(width=8, heads=2, memory_dim=2)


@pytest.fixture
def graph(schema):
    interpreter = interpreter_fingerprint({"fixture.py": b"version one"})
    base = ExecutableDefinition.build("base", {"operation": "fixture", "value": 1},
                                      schema, interpreter)
    middle = ExecutableDefinition.build("middle", {"operation": "call"}, schema, interpreter,
                                        dependencies=(base.reference,))
    top = ExecutableDefinition.build("top", {"operation": "call"}, schema, interpreter,
                                     dependencies=(middle.reference,))
    other = ExecutableDefinition.build("other", {"operation": "fixture", "value": 8},
                                       schema, interpreter)
    return ExecutableGraph(schema, interpreter, (base, middle, top, other))


def event(schema, position=0, *, values=(1., 2.), available=None, record_id=None,
          kind=EvidenceKind.OBSERVATION, role=EventRole.OBSERVATION):
    return Event(schema, position, values,
                 Provenance("fixture", record_id or f"event-{position}", kind),
                 available, ("entity-a",), role)


def revised_base(graph):
    return ExecutableDefinition.build("base", {"operation": "fixture", "value": 2},
                                      graph.schema, graph.interpreter_sha256)


def test_event_defensively_owns_lists_and_erases_unavailable_content(schema):
    values, mask, entities = [4., 999.], [True, False], ["entity-a"]
    first = Event(schema, 0, values, Provenance("fixture", "x", EvidenceKind.OBSERVATION),
                  mask, entities)
    values[0], mask[0], entities[0] = 88., False, "entity-b"
    second = Event(schema, 0, (4., float("nan")), first.provenance,
                   (True, False), ("entity-a",))
    assert first == second and first.values == (4., 0.)
    assert first.identity == second.identity
    raw = first.record()
    raw["values"][0] = 77
    assert first.values[0] == 4.
    with pytest.raises(FrozenInstanceError):
        first.sequence = 4
    assert Event.from_record(first.record()) == first


def test_schema_units_masks_and_observation_conversion_are_explicit(schema):
    with pytest.raises(ValueError, match="units"):
        EventSchema("bad", 1, "numeric", 2)
    with pytest.raises(ValueError, match="booleans"):
        event(schema, available=(1, 0))
    with pytest.raises(ValueError, match="finite"):
        event(schema, values=(float("nan"), 0))
    observed = Observation("numeric", (1., 2.), 0,
                           Provenance("fixture", "obs", EvidenceKind.OBSERVATION), units="m")
    assert Event.from_observation(observed, schema, entity_refs=("entity-a",)).values == (1., 2.)
    with pytest.raises(ValueError, match="conversion"):
        Event.from_observation(replace(observed, units="cm"), schema, entity_refs=("entity-a",))
    with pytest.raises(ValueError, match="conversion"):
        Event.from_observation(replace(observed, scale=100), schema, entity_refs=("entity-a",))


def test_definition_and_graph_snapshots_have_no_nested_alias(schema):
    source = {"runtime.py": b"runtime v1"}
    interpreter = interpreter_fingerprint(source)
    body = {"nodes": [["input", 0]], "data": {"labels": [1, 2]}}
    definition = ExecutableDefinition.build("base", body, schema, interpreter)
    definitions = [definition]
    graph = ExecutableGraph(schema, interpreter, definitions)
    before = graph.identity
    body["data"]["labels"][0] = 999
    definitions.clear()
    definition.body["nodes"].clear()
    graph.record()["definitions"][0]["body"]["data"]["labels"].clear()
    source["runtime.py"] = b"runtime v2"
    assert graph.identity == before and len(graph.definitions) == 1
    assert definition.body["data"]["labels"] == [1, 2]
    assert interpreter_fingerprint(source) != interpreter
    assert ExecutableGraph.from_record(json.loads(json.dumps(graph.record()))) == graph


def test_dependency_and_interpreter_tampering_fail(graph):
    broken = replace(graph.definition("middle"), dependencies=(DependencyRef("base", "0" * 64),))
    with pytest.raises(ValueError, match="stale"):
        ExecutableGraph(graph.schema, graph.interpreter_sha256, (graph.definition("base"), broken))
    with pytest.raises(ValueError, match="interpreter"):
        ExecutableGraph(graph.schema, "0" * 64, graph.definitions)
    with pytest.raises(ValueError, match="Duplicate"):
        ExecutableGraph(graph.schema, graph.interpreter_sha256, (graph.definitions[0],) * 2)


def test_replacement_invalidates_transitive_dependents_and_preserves_unrelated(graph):
    store = GraphStore(graph)
    result = store.replace((revised_base(graph),), expected_parent=graph.identity)
    assert result.changed == ("base",)
    assert result.invalidated == ("middle", "top")
    assert {item.name for item in store.current.definitions} == {"base", "other"}
    assert store.current.definition("other") is graph.definition("other")
    assert graph.definition("base").body["value"] == 1
    with pytest.raises(ValueError, match="parent"):
        store.replace(expected_parent=graph.identity)


def test_dependent_rebuild_requires_fresh_content_references(graph):
    store = GraphStore(graph)
    base = revised_base(graph)
    with pytest.raises(ValueError, match="stale"):
        store.replace((base, graph.definition("middle")), expected_parent=graph.identity)
    assert store.current is graph
    middle = replace(graph.definition("middle"), dependencies=(base.reference,))
    top = replace(graph.definition("top"), dependencies=(middle.reference,))
    result = store.replace((base, middle, top), expected_parent=graph.identity,
                           validator=lambda candidate: candidate.definition("top") == top)
    assert result.invalidated == () and len(store.current.definitions) == 4


def test_rejection_and_interruption_are_transactional(graph):
    store = GraphStore(graph)
    with pytest.raises(ValueError, match="admission"):
        store.replace((revised_base(graph),), expected_parent=graph.identity,
                      validator=lambda _: False)
    assert store.current is graph

    def interrupted(_):
        raise InterruptedError("fixture cancellation")

    with pytest.raises(InterruptedError):
        store.replace((revised_base(graph),), expected_parent=graph.identity, validator=interrupted)
    assert store.current is graph
    removed = store.replace(remove=("base",), expected_parent=graph.identity)
    assert removed.invalidated == ("base", "middle", "top")
    assert removed.graph.definitions == (graph.definition("other"),)


def test_parameter_free_reference_checks_object_owner_and_registered_replacement(owner):
    solver = make_shared_solver(owner)
    reference = SharedOwnerRef.from_solver(solver)
    assert list(reference.parameters()) == [] and reference.state_dict() == {}
    assert copy.deepcopy(reference) is reference
    assert reference.owner is owner
    clone = copy.deepcopy(owner)
    assert reference.identity == SharedOwnerRef(clone).identity
    with pytest.raises(ValueError, match="another parameter owner"):
        reference.require_current(clone)
    parameter_ids = {id(p) for p in solver.parameters()}
    assert parameter_ids == {id(p) for p in owner.parameters()}
    replace_shared_owner(solver, clone)
    with pytest.raises(ValueError, match="rebind"):
        reference.require_current()
    assert SharedOwnerRef.from_solver(solver).owner is clone


def test_mechanism_pins_complete_graph_owner_schema_and_interpreter(graph, owner):
    reference = SharedOwnerRef(owner)
    mechanism = MechanismRef.bind(graph, reference)
    mechanism.validate(graph, reference)
    assert mechanism.dependencies == graph.dependencies
    changed = GraphStore(graph).replace((revised_base(graph),), expected_parent=graph.identity).graph
    with pytest.raises(ValueError, match="Mechanism"):
        mechanism.validate(changed, reference)
    with torch.no_grad():
        next(owner.parameters()).add_(.1)
    with pytest.raises(ValueError, match="parameters"):
        mechanism.validate(graph, reference)


def test_live_situation_rejects_wrong_schema_order_duplicate_and_prediction(graph, owner):
    live = LiveSituation("one", graph, SharedOwnerRef(owner))
    with pytest.raises(ValueError, match="sequence"):
        live.observe(event(graph.schema, 1))
    with pytest.raises(ValueError, match="factual"):
        live.observe(event(graph.schema, kind=EvidenceKind.PREDICTION))
    with pytest.raises(ValueError, match="schema"):
        live.observe(event(replace(graph.schema, units="cm")))
    assert live.observations == ()
    live.observe(event(graph.schema))
    with pytest.raises(ValueError, match="already"):
        live.observe(event(graph.schema, 1, record_id="event-0"))
    with pytest.raises(ValueError, match="Labels"):
        live.admit_label(event(graph.schema, record_id="label", kind=EvidenceKind.PREDICTION,
                               role=EventRole.TARGET))
    with pytest.raises(ValueError, match="fully available"):
        live.admit_label(event(graph.schema, record_id="label", kind=EvidenceKind.VERIFIED,
                               available=(True, False), role=EventRole.TARGET))
    live.admit_label(event(graph.schema, record_id="label", kind=EvidenceKind.VERIFIED,
                           role=EventRole.TARGET))
    assert len(live.observations) == len(live.labels) == 1


def test_imagination_cannot_mutate_live_state_or_admit_labels(graph, owner):
    live = LiveSituation("one", graph, SharedOwnerRef(owner))
    live.observe(event(graph.schema))
    before, weights = live.snapshot()["sha256"], live.owner.identity
    first, sibling = live.fork(), live.fork()
    first.assume(event(graph.schema, 1, role=EventRole.HYPOTHETICAL,
                       kind=EvidenceKind.PREDICTION, record_id="imagination"))
    assert live.snapshot()["sha256"] == before and live.owner.identity == weights
    assert sibling.assumptions == () and first.observations == live.observations
    first.snapshot()["payload"]["observations"][0]["values"][0] = 999
    assert live.observations[0].values == (1., 2.)
    for branch in (first, sibling):
        with pytest.raises(ValueError, match="cannot admit"):
            branch.observe(event(graph.schema, 1))
        with pytest.raises(ValueError, match="cannot admit"):
            branch.admit_label(event(graph.schema, kind=EvidenceKind.VERIFIED,
                                     role=EventRole.TARGET, record_id="label"))
    with pytest.raises(ValueError, match="snapshot"):
        LiveSituation.restore(first.snapshot(), graph, live.owner, situation_id="one")


def test_shared_budget_rejects_negative_counts_and_does_not_refresh_on_branch(graph, owner):
    budget = ExecutionBudget(2)
    with pytest.raises(ValueError, match="nonnegative"):
        budget.add("bad", -100)
    with pytest.raises(ValueError, match="integer"):
        budget.add("bad", True)
    live = LiveSituation("one", graph, SharedOwnerRef(owner), budget=budget)
    live.observe(event(graph.schema))
    branch = live.fork()
    assert branch.budget is live.budget
    with pytest.raises(InterruptedError):
        branch.assume(event(graph.schema, 1, role=EventRole.HYPOTHETICAL,
                            kind=EvidenceKind.PREDICTION))
    assert branch.assumptions == () and len(live.observations) == 1
    with pytest.raises(InterruptedError):
        live.observe(event(graph.schema, 1))
    assert len(live.observations) == 1


def test_snapshot_restoration_preserves_evidence_and_charges_replay(graph, owner):
    live = LiveSituation("one", graph, SharedOwnerRef(owner))
    live.observe(event(graph.schema))
    live.admit_label(event(graph.schema, kind=EvidenceKind.VERIFIED,
                           role=EventRole.TARGET, record_id="label"))
    frozen = live.snapshot()
    restored = LiveSituation.restore(json.loads(json.dumps(frozen)), graph, live.owner,
                                     situation_id="one")
    assert restored.snapshot()["sha256"] == frozen["sha256"]
    assert restored.observations == live.observations and restored.labels == live.labels
    assert restored.budget.counts == {"factual_observations": 1, "verified_labels": 1,
                                      "replayed_observations": 1, "replayed_labels": 1}
    live.observe(event(graph.schema, 1))
    restored.observe(event(graph.schema, 1))
    assert restored.snapshot()["sha256"] == live.snapshot()["sha256"]
    frozen["payload"]["observations"][0]["values"][0] = 999
    assert restored.observations[0].values == (1., 2.)
    with pytest.raises(ValueError, match="integrity"):
        LiveSituation.restore(frozen, graph, live.owner, situation_id="one")


def test_restoration_rejects_wrong_situation_model_and_budget_tampering(graph, owner):
    live = LiveSituation("one", graph, SharedOwnerRef(owner))
    live.observe(event(graph.schema))
    frozen = live.snapshot()
    with pytest.raises(ValueError, match="identity"):
        LiveSituation.restore(frozen, graph, live.owner, situation_id="another")
    with pytest.raises(ValueError, match="budget"):
        LiveSituation.restore(frozen, graph, live.owner, situation_id="one",
                              budget=ExecutionBudget(4))
    broken = copy.deepcopy(frozen)
    broken["budget"]["counts"]["factual_observations"] = -2
    broken["budget_sha256"] = digest(broken["budget"])
    with pytest.raises(ValueError, match="counts"):
        LiveSituation.restore(broken, graph, live.owner, situation_id="one")
    clone = copy.deepcopy(owner)
    with torch.no_grad():
        next(clone.parameters()).add_(.1)
    with pytest.raises(ValueError, match="identity"):
        LiveSituation.restore(frozen, graph, SharedOwnerRef(clone), situation_id="one")


def test_identity_migration_replays_admitted_facts_without_reinterpreting_schema(graph, owner):
    live = LiveSituation("one", graph, SharedOwnerRef(owner))
    live.observe(event(graph.schema))
    live.admit_label(event(graph.schema, record_id="label", kind=EvidenceKind.VERIFIED,
                           role=EventRole.TARGET))
    before = live.snapshot()["sha256"]
    successor_graph = GraphStore(graph).replace((revised_base(graph),),
                                                expected_parent=graph.identity).graph
    clone = copy.deepcopy(owner)
    with torch.no_grad():
        next(clone.parameters()).add_(.1)
    successor = live.migrate(successor_graph, SharedOwnerRef(clone),
                             expected_mechanism=live.mechanism.identity)
    assert successor.observations == live.observations and successor.labels == live.labels
    assert successor.mechanism != live.mechanism
    assert successor.migration["method"] == "same-schema-event-replay-v1"
    assert live.snapshot()["sha256"] == before
    assert successor.budget is live.budget
    assert successor.budget.counts == {"factual_observations": 1, "verified_labels": 1,
                                       "replayed_observations": 1, "replayed_labels": 1}
    incompatible = ExecutableGraph(replace(graph.schema, version=2), graph.interpreter_sha256)
    with pytest.raises(ValueError, match="Incompatible"):
        live.migrate(incompatible, live.owner, expected_mechanism=live.mechanism.identity)
    with pytest.raises(ValueError, match="parent"):
        live.migrate(graph, live.owner, expected_mechanism="0" * 64)


def test_live_model_mutation_fails_before_new_evidence_is_admitted(graph, owner):
    live = LiveSituation("one", graph, SharedOwnerRef(owner))
    live.observe(event(graph.schema))
    with torch.no_grad():
        next(owner.parameters()).add_(.1)
    with pytest.raises(ValueError, match="parameters"):
        live.observe(event(graph.schema, 1))
    assert len(live.observations) == 1


def test_raw_tensor_mutation_cannot_bypass_the_full_owner_fingerprint(graph, owner):
    live = LiveSituation("one", graph, SharedOwnerRef(owner))
    live.observe(event(graph.schema))
    before = live.owner.identity
    parameter = next(owner.parameters())
    version = parameter._version
    parameter.data.add_(.125)
    assert parameter._version == version
    assert live.owner.identity != before
    with pytest.raises(ValueError, match="parameters"):
        live.observe(event(graph.schema, 1))
    assert len(live.observations) == 1


@pytest.mark.parametrize("operation", ("restore", "migrate"))
def test_exhausted_replay_keeps_old_facts_and_exposes_no_partial_successor(graph, owner, operation):
    budget = ExecutionBudget(4)
    live = LiveSituation("one", graph, SharedOwnerRef(owner), budget=budget)
    live.observe(event(graph.schema))
    live.observe(event(graph.schema, 1))
    live.admit_label(event(graph.schema, record_id="label", kind=EvidenceKind.VERIFIED,
                           role=EventRole.TARGET))
    before = live.snapshot()
    successors = []
    with pytest.raises(InterruptedError, match="budget"):
        if operation == "restore":
            successors.append(LiveSituation.restore(before, graph, live.owner,
                                                     situation_id="one", budget=budget))
        else:
            changed = GraphStore(graph).replace((revised_base(graph),),
                                                expected_parent=graph.identity).graph
            successors.append(live.migrate(changed, live.owner,
                                            expected_mechanism=live.mechanism.identity))
    assert successors == []
    assert live.snapshot()["sha256"] == before["sha256"]
    assert len(live.observations) == 2 and len(live.labels) == 1
    assert budget.counts == {"factual_observations": 2, "verified_labels": 1,
                              "replayed_observations": 2}
