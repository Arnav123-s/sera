import copy

import pytest
import torch

from sera.binding import binding_cases
from sera.data import make_batch
from sera.environments import collect, make_world
from sera.r1 import tensors
from sera.shared import SharedR1, make_shared_solver
from sera.shared_archive import load_shared_checkpoint, save_shared_checkpoint
from sera.solver import SolverStore
from sera.typed_learning import typed_examples
from sera.typed_protocol import audit_partitions


@pytest.mark.parametrize("kind", ["delta", "reference"])
def test_three_interfaces_update_one_memory_and_restore_its_identity(tmp_path, kind):
    torch.manual_seed(17)
    core = SharedR1(width=32, heads=2, memory_dim=4, kind=kind)
    solver = make_shared_solver(core)
    parameter = core.memory.delta.key.weight if kind == "reference" else core.memory.cores["delta"].key.weight
    observations, _ = collect(make_world(16), seed=14, count=3, length=4)
    typed = [r for r in typed_examples(seed=18, count=3) if r.task == "binding"]
    seq = make_batch(3, 6, 19, split="support", task=1)
    for operation in (lambda: core(*tensors(observations))[0].square().mean(),
                      lambda: solver.components["typed"]([r.observations for r in typed], [r.task for r in typed])["categorical"].square().mean(),
                      lambda: solver.neural(seq.inputs).square().mean()):
        solver.zero_grad()
        operation().backward()
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert not solver.neural.state_dict() and not solver.components["typed"].state_dict()
    assert sum(p.numel() for p in solver.parameters()) == sum(p.numel() for p in core.parameters())
    clone = copy.deepcopy(solver)
    assert clone.neural.owner is clone.components["typed"].owner is clone.components["r1"]
    assert clone.components["r1"] is not core
    store = SolverStore(tmp_path)
    store.initialize(solver)
    restored = store.load()
    assert store.current_record()["schema_version"] == 3
    assert restored.neural.owner is restored.components["typed"].owner is restored.components["r1"]
    assert restored.identity() == solver.identity()
    torch.testing.assert_close(restored(seq.inputs), solver(seq.inputs), atol=0, rtol=0)
    torch.testing.assert_close(restored.components["typed"]([r.observations for r in typed], [r.task for r in typed])["categorical"],
                               solver.components["typed"]([r.observations for r in typed], [r.task for r in typed])["categorical"], atol=0, rtol=0)


def test_reference_dimensions_and_detached_view_rejection():
    core = SharedR1(kind="reference")
    states = core.initial(1)
    assert core.core_state_bytes() == 35840
    assert states["delta"].shape == (1, 8, 32, 32)
    assert states["rotor"].shape == (1, 128, 2)
    assert states["density"].shape == (1, 4, 16, 4)
    torch.testing.assert_close(states["density"].abs().square().sum((-2, -1)), torch.ones(1, 4))
    solver = make_shared_solver(core)
    object.__setattr__(solver.components["typed"], "owner", copy.deepcopy(core))
    with pytest.raises(ValueError, match="detached"):
        solver.validate()


def test_binding_rules_are_explicit_and_overwrites_change_the_answer():
    for rule in ("earliest", "latest"):
        partitions = {split: binding_cases(seed=12, count=32, rule=rule, split=split) for split in ("support", "validation", "test")}
        assert audit_partitions(partitions)["cross_partition_overlap"] == 0
        for rows in partitions.values():
            for row in rows:
                query = row.observations[-1].values[:4].index(1)
                answers = [obs.values[4:8].index(1) for obs in row.observations[:-1] if obs.values[query] == 1]
                assert answers[0] != answers[-1]
                assert row.target == answers[0 if rule == "earliest" else -1]
                assert row.observations[-1].values[4:8] == (0, 0, 0, 0)


def test_adapter_preserves_initial_behavior_and_parent_referenced_archive(tmp_path):
    model = SharedR1(width=32, heads=2, memory_dim=4)
    rows = binding_cases(seed=15, count=4)
    before = model.forward_typed([r.observations for r in rows], [r.task for r in rows])
    base = tmp_path / "base.pt"
    save_shared_checkpoint(base, model)
    adapted = copy.deepcopy(model)
    adapted.add_adapter(rank=4)
    after = adapted.forward_typed([r.observations for r in rows], [r.task for r in rows])
    torch.testing.assert_close(before["categorical"], after["categorical"], atol=0, rtol=0)
    child = tmp_path / "adapter.pt"
    record = save_shared_checkpoint(child, adapted, parent=base)
    assert record["stored_tensors"] == 2
    restored = load_shared_checkpoint(child)
    torch.testing.assert_close(restored.forward_typed([r.observations for r in rows], [r.task for r in rows])["categorical"], after["categorical"], atol=0, rtol=0)
    base.write_bytes(b"invalid parent")
    with pytest.raises(ValueError, match="parent changed"):
        load_shared_checkpoint(child)


def test_masked_payload_is_unobservable_and_shared_adapter_migration_keeps_aliases():
    from dataclasses import replace

    from sera.mutations import Mutation, mutate
    core = SharedR1(width=32, heads=2, memory_dim=4)
    row = binding_cases(seed=43, count=1)[0]
    query = row.observations[-1]
    hidden = replace(query, values=tuple([*query.values[:4], 999., -999., 71., 33., *query.values[8:]]))
    first = core.forward_typed([row.observations], [row.task])["categorical"]
    second = core.forward_typed([(*row.observations[:-1], hidden)], [row.task])["categorical"]
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    changed, record = mutate(make_shared_solver(core), Mutation("insert_adapter", "r1", 4), seed=4)
    assert record["function_preserving_at_initialization"]
    assert changed.neural.owner is changed.components["typed"].owner is changed.components["r1"]
    torch.testing.assert_close(changed.components["typed"]([row.observations], [row.task])["categorical"], first, atol=0, rtol=0)


def test_corrective_replay_integrity_roles_and_stale_parent_are_enforced(tmp_path):
    from dataclasses import replace

    from sera.experience import EvidenceReplay
    from sera.shared_learning import SharedEvidence
    from sera.typed_learning import TypedEvidence
    world, _ = collect(make_world(1), seed=1, count=2, length=4, split="support")
    rows = binding_cases(seed=12, count=2)
    evidence = SharedEvidence(TypedEvidence(rows), EvidenceReplay(world), 17)
    evidence.save(tmp_path / "evidence")
    loaded = SharedEvidence.load(tmp_path / "evidence")
    assert loaded.typed.identifiers == evidence.typed.identifiers
    assert loaded.world.identifiers == evidence.world.identifiers
    with pytest.raises(ValueError, match="Conflicting"):
        loaded.admit_typed([replace(rows[0], target=(rows[0].target+1) % 4)])
    with pytest.raises(ValueError, match="validation or sealed"):
        loaded.admit_typed(binding_cases(seed=22, count=2, split="validation"))
    core = SharedR1(width=32, heads=2, memory_dim=4)
    store = SolverStore(tmp_path / "store")
    store.initialize(make_shared_solver(core))
    with pytest.raises(ValueError, match="incumbent changed"):
        store.consider(make_shared_solver(core), lambda *_: pytest.fail("Should not expose a suite"), expected_parent="stale")
    assert not (tmp_path / "store" / "versions" / "v1.pt").exists()
