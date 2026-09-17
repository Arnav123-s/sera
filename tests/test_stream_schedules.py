"""Scheduling evidence, bounded replay and exact interrupted updates."""

import copy
import json

import pytest
import torch

from experiments.stream_curriculum import mixed
from experiments.stream_curriculum.audit import same
from experiments.stream_curriculum.data import ROOT, first_rows
from experiments.stream_curriculum.model import apply, delta
from experiments.stream_curriculum.runtime import RequestSession
from experiments.stream_curriculum.sequential import Reservoir
from experiments.stream_curriculum.study import update
from sera.session_state import model_identity
from workbench.storage import Store


def test_mixed_view_is_an_exact_source_permutation_and_reuses_it(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    values = [{"id": str(i), "partition": "train", "text": f"request {i}"} for i in range(101)]
    (prepared / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
    monkeypatch.setattr(mixed, "ROOT", tmp_path)
    monkeypatch.setattr(mixed, "PREPARED", prepared)
    manifest = mixed.prepare(2641)
    records = [json.loads(line) for line in (tmp_path / manifest["view"]).read_text().splitlines()]
    assert sorted(records, key=lambda r: int(r["id"])) == values
    assert records != values
    assert mixed.prepare(2641) == manifest


def test_reservoir_bounds_unique_ids_rng_and_atomic_split_guard():
    rows = [{"id": str(i), "partition": "train"} for i in range(400)]
    memory = Reservoir(2651, capacity=7)
    memory.admit(rows[:200])
    saved = memory.snapshot()
    restored = Reservoir(2651, capacity=7, saved=saved)
    for instance in (memory, restored):
        instance.admit(rows[:200])
        instance.admit(rows[200:])
    assert memory.snapshot() == restored.snapshot()
    assert len(memory.rows) == 7 and len(memory.seen) == 400
    assert memory.sample(rows[:16]) == restored.sample(rows[:16])
    before = memory.snapshot()
    with pytest.raises(ValueError):
        memory.admit([{"id": "new", "partition": "train"}, {"id": "held", "partition": "dev"}])
    assert memory.snapshot() == before


def test_replay_update_survives_optimizer_and_memory_restore(tmp_path):
    record = Store(ROOT / "runs/sera-requests").read()
    if record is None:
        pytest.skip("Requires the local acquired learner")
    session = RequestSession(record["checkpoint"], record)
    original = model_identity(session.owner)
    left, right = copy.deepcopy(session.owner), copy.deepcopy(session.owner)
    vocabulary = left.stream_config["vocabulary"]
    opts = [torch.optim.Adam([p for p in owner.parameters() if p.requires_grad], lr=.003) for owner in (left, right)]
    memories = [Reservoir(2651), Reservoir(2651)]
    rows = first_rows("train", 48)
    for memory in memories:
        memory.admit(rows[:16])
    for current in (rows[16:32], rows[32:48]):
        update(left, opts[0], current + memories[0].sample(current), vocabulary)
        memories[0].admit(current)
    current = rows[16:32]
    update(right, opts[1], current + memories[1].sample(current), vocabulary)
    memories[1].admit(current)
    path = tmp_path / "paused.pt"
    torch.save({"delta": delta(right), "optimizer": opts[1].state_dict(), "memory": memories[1].snapshot(),
                "rng": torch.get_rng_state()}, path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    right = copy.deepcopy(session.owner)
    apply(right, saved["delta"])
    opt = torch.optim.Adam([p for p in right.parameters() if p.requires_grad], lr=.003)
    opt.load_state_dict(saved["optimizer"])
    memory = Reservoir(2651, saved=saved["memory"])
    torch.set_rng_state(saved["rng"])
    current = rows[32:48]
    update(right, opt, current + memory.sample(current), vocabulary)
    memory.admit(current)
    assert same(delta(left), delta(right))
    assert same(opts[0].state_dict(), opt.state_dict())
    assert memory.snapshot() == memories[0].snapshot()
    assert model_identity(session.owner) == original
