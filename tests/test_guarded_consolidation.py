import copy

import pytest
import torch

from experiments.guarded_consolidation.core import (
    SCHEMA,
    Library,
    Work,
    compile_definition,
    interpreter_identity,
    teaching_trace,
)
from experiments.guarded_consolidation.study import TRAIN_INPUTS, base
from sera.shared import SharedR1
from sera.storage import digest
from sera.world_graph import ExecutableDefinition, ExecutableGraph, GraphStore, SharedOwnerRef


@pytest.fixture
def fixture():
    interpreter, work = interpreter_identity(), Work()
    relation = base("affine", interpreter)
    traces = [teaching_trace(relation.body["left"], inputs, work, str(i)) for i, inputs in enumerate(TRAIN_INPUTS)]
    definition = compile_definition(relation, traces, SCHEMA, interpreter, work)
    owner = SharedOwnerRef(SharedR1(width=8, heads=2, memory_dim=2))
    library = Library(GraphStore(ExecutableGraph(SCHEMA, interpreter, (relation, definition))), owner, work)
    return library, relation, definition, traces, interpreter


def test_acquired_program_parameterizes_values_and_handles_new_bindings(fixture):
    library, relation, definition, _, _ = fixture
    assert definition.body["program"] == ["mul", ["sub", ["input", "c"], ["input", "b"]], ["inv", ["input", "a"]]]
    assert definition.body["proof"]["accepted_inputs"] == 1210
    assert library.answer(relation.identity, dict(a=4, b=5, c=6), Work()) == 3
    assert "answers" not in definition.body


@pytest.mark.parametrize("inputs", [dict(a=0, b=5, c=6), dict(a=True, b=5, c=6), dict(a=4, b=-1, c=6), dict(a=4, b=5, c=11)])
def test_necessary_assumptions_are_preserved(fixture, inputs):
    library, relation, _, _, _ = fixture
    assert library.answer(relation.identity, inputs, Work()) is None


def test_omitting_inverse_guard_creates_an_incorrect_answer(fixture):
    library, relation, _, _, _ = fixture
    assert library.answer(relation.identity, dict(a=0, b=5, c=6), Work(), omit_guard=True) == 0


def test_predictions_and_forged_observations_cannot_teach(fixture):
    _, relation, _, traces, interpreter = fixture
    changed = copy.deepcopy(traces)
    changed[0]["evidence"] = "imagined"
    with pytest.raises(ValueError, match="Imagined"):
        compile_definition(relation, changed, SCHEMA, interpreter, Work())
    changed = copy.deepcopy(traces)
    changed[0]["observed_solutions"] = [9]
    with pytest.raises(ValueError, match="Forged"):
        compile_definition(relation, changed, SCHEMA, interpreter, Work())


def test_correction_removes_old_shortcut_and_repair_is_a_new_program(fixture):
    library, relation, definition, _, interpreter = fixture
    revised = base("affine", interpreter, corrected=True)
    update = library.store.replace((revised,), expected_parent=library.store.current.identity)
    assert update.invalidated == (definition.name,)
    assert library.answer(revised.identity, dict(a=4, b=5, c=6), Work()) is None
    traces = [teaching_trace(revised.body["left"], inputs, Work(), str(i)) for i, inputs in enumerate(TRAIN_INPUTS)]
    new = compile_definition(revised, traces, SCHEMA, interpreter, Work())
    library.store.replace((new,), expected_parent=library.store.current.identity)
    assert library.answer(revised.identity, dict(a=4, b=5, c=6), Work()) == 0


def test_restore_reproves_and_owner_change_invalidates(fixture):
    library, relation, definition, _, interpreter = fixture
    restored = Library.restore(library.record(), library.owner, Work())
    assert restored.record() == library.record()
    body = definition.body
    body["program"] = ["input", "c"]
    body["nonzero_conditions"] = []
    forged = ExecutableDefinition.build(definition.name, body, SCHEMA, interpreter, dependencies=definition.dependencies)
    graph = ExecutableGraph(SCHEMA, interpreter, (relation, forged))
    record = {"owner_sha256": library.owner_id, "graph": graph.record(), "graph_sha256": graph.identity}
    with pytest.raises(ValueError, match="proof"):
        Library.restore(record, library.owner, Work())
    with torch.no_grad():
        next(library.owner.owner.parameters()).add_(1)
    with pytest.raises(ValueError, match="Owner changed"):
        library.answer(relation.identity, dict(a=4, b=5, c=6), Work())


def test_indexed_proof_checks_identical_domain_without_repeating_forward_answers(fixture):
    from experiments.guarded_consolidation.core import prove
    from experiments.guarded_consolidation.indexed_proof import prove_indexed
    _, relation, definition, _, _ = fixture
    old, new = Work(), Work()
    tree, left = definition.body["program"], relation.body["left"]
    assert prove(tree, left, old) == prove_indexed(tree, left, new)
    assert new.operations < old.operations
    assert new.counts["proof_index_insertions"] == 11**3
    assert new.counts["proof_index_lookups"] == 1210
    with pytest.raises(ValueError, match="incorrect or ambiguous"):
        prove_indexed(["input", "c"], left, Work())


def test_indexed_compile_and_restore_retain_identical_executable(fixture):
    library, relation, original, traces, interpreter = fixture
    optimized = compile_definition(relation, traces, SCHEMA, interpreter, Work(), proof_backend="indexed")
    assert optimized == original
    assert Library.restore(library.record(), library.owner, Work(), proof_backend="indexed").record() == library.record()


def test_indexed_proof_rejects_a_forward_relation_that_depends_on_rhs():
    from experiments.guarded_consolidation.indexed_proof import prove_indexed
    # Reusing a table built at c=0 is unsound for x+c=c. That equation's
    # solution is zero, whereas a forged answer of c passes the stale table.
    with pytest.raises(ValueError, match="right-hand side"):
        prove_indexed(["input", "c"], ["add", "x", "c"], Work())


def test_an_internally_consistent_graph_with_stale_interpreter_is_rejected(fixture):
    library, relation, _, _, _ = fixture
    stale = digest("old runtime")
    other = ExecutableDefinition.build(relation.name, relation.body, SCHEMA, stale)
    with pytest.raises(ValueError, match="Stale executable interpreter"):
        Library(GraphStore(ExecutableGraph(SCHEMA, stale, (other,))), library.owner, Work())
