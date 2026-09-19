"""Tests for the schedule, streams, masked loss, failing-closed measurement and checkpoints.

These use a small stand-in module with the same interface as the descendant's
language route, so they stay cheap. The connected route itself is exercised
against the real owner by ``scripts/owner_language_checks.py`` under the
supervisor.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

from experiments.owner_language import training as tr  # noqa: E402
from experiments.owner_language.tasks import Tokens  # noqa: E402

SCRATCH = LAB_ROOT / "runs/owner-learning-001/test-scratch"
BUILT = (LAB_ROOT / "runs/owner-learning-001/corpus/items.jsonl").is_file()
needs_corpus = pytest.mark.skipif(not BUILT, reason="Corpus has not been built in this checkout")


@pytest.fixture
def scratch():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=SCRATCH))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class Stand(nn.Module):
    """The same masked-loss contract as the descendant, without the owner."""

    def __init__(self, size=64, width=16):
        super().__init__()
        self.embedding = nn.Embedding(size, width, padding_idx=0)
        self.head = nn.Linear(width, size)

    def lex_logits(self, ids):
        return self.head(self.embedding(ids)), ids.ne(0)

    def lex_next_token_loss(self, ids):
        logits, _ = self.lex_logits(ids)
        predicted = logits[:, :-1].reshape(-1, logits.shape[-1])
        target = ids[:, 1:].reshape(-1)
        counted = int((target != 0).sum())
        if counted == 0:
            raise ValueError("A batch must contain at least one predicted token")
        return F.cross_entropy(predicted, target, ignore_index=0, reduction="sum") / counted, counted


def test_pad_batch_right_pads_and_keeps_the_first_token_real():
    ids = tr.pad_batch([[5, 6, 7], [8], [9, 10]])
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0], [9, 10, 0]]
    assert bool(ids[:, 0].all())


def test_masked_loss_ignores_padding_positions():
    torch.manual_seed(0)
    model = Stand()
    rows = [[5, 6, 7, 8, 9], [10, 11], [12, 13, 14]]
    loss, counted = model.lex_next_token_loss(tr.pad_batch(rows))
    assert counted == sum(len(row) - 1 for row in rows)
    singles = [model.lex_next_token_loss(torch.tensor([row]))[0].item() for row in rows]
    combined = sum(value * (len(row) - 1) for value, row in zip(singles, rows)) / counted
    assert combined == pytest.approx(loss.item(), abs=1e-6)


def test_a_batch_with_nothing_to_predict_is_an_error_not_a_zero():
    model = Stand()
    with pytest.raises(ValueError):
        model.lex_next_token_loss(torch.tensor([[5]]))


def test_validate_schedule_derives_cumulative_steps():
    stages = [{"name": "a", "sources": ["x"], "steps": 10},
              {"name": "b", "sources": ["x", "y"], "steps": 5}]
    validated = tr.validate_schedule(stages, ["x", "y"])
    assert [stage["cumulative_steps"] for stage in validated] == [10, 15]


@pytest.mark.parametrize("stages,available,message", [
    ([{"name": "a", "sources": ["x"], "steps": 1}], ["x", "y"], "teaches"),
    ([{"name": "a", "sources": ["z"], "steps": 1}], ["x"], "no training items"),
    ([{"name": "a", "sources": ["x"], "steps": 0}], ["x"], "non-positive"),
    ([{"name": "a", "sources": [], "steps": 1}], ["x"], "no source"),
    ([{"name": "a", "sources": ["x"], "steps": 1, "cumulative_steps": 99}], ["x"], "declares cumulative"),
])
def test_validate_schedule_rejects_malformed_ladders(stages, available, message):
    with pytest.raises(ValueError, match=message):
        tr.validate_schedule(stages, available)


def test_validate_schedule_rejects_a_duplicate_stage_name():
    stage = {"name": "a", "sources": ["x"], "steps": 1}
    with pytest.raises(ValueError, match="Duplicate stage"):
        tr.validate_schedule([stage, dict(stage)], ["x"])


def test_measure_next_token_fails_closed_on_no_rows():
    result = tr.measure_next_token(Stand(), [])
    assert result["status"] == "UNAVAILABLE"
    assert result["qualifies"] is False
    assert "mean_nll" not in result and "perplexity" not in result


def test_measure_next_token_fails_closed_when_every_batch_raises():
    class Broken:
        def lex_next_token_loss(self, ids):
            raise RuntimeError("deliberate")

    rows = [{"ids": [5, 6, 7], "source": "x"}]
    result = tr.measure_next_token(Broken(), rows)
    assert result["status"] == "UNAVAILABLE"
    assert result["counts"]["failures"] == 1
    assert result["qualifies"] is False


def test_measure_choice_fails_closed_on_no_rows():
    result = tr.measure_choice(Stand(), None, [], "cloze")
    assert result["status"] == "UNAVAILABLE"
    assert result["qualifies"] is False
    assert "accuracy" not in result


@needs_corpus
def test_measure_choice_reports_accuracy_against_the_declared_chance():
    torch.manual_seed(0)
    tokens = Tokens()
    model = Stand(size=len(tokens), width=8)
    rows = [{"id": f"q{index}", "family": "cloze", "source": "x", "chance": 0.125,
             "prefix": ["the", "quick", "brown"], "answer": "fox",
             "candidates": ["fox", "dog", "cat", "cow", "hen", "bat", "owl", "ant"]}
            for index in range(8)]
    result = tr.measure_choice(model, tokens, rows, "cloze")
    assert result["status"] == "MEASURED"
    assert result["questions"] == 8
    assert 0. <= result["accuracy"] <= 1.
    assert result["chance"] == 0.125


@needs_corpus
def test_source_stream_is_deterministic_and_resumable():
    from experiments.owner_language.corpus import load_items
    tokens = Tokens()
    items = [row for row in load_items(split="train") if row["source"] == "grammar"]
    first = tr.SourceStream("grammar", items, tokens, length=32)
    taken = first.take(5)
    cursor = first.cursor()
    second = tr.SourceStream("grammar", items, tokens, length=32)
    second.restore(cursor)
    third = tr.SourceStream("grammar", items, tokens, length=32)
    assert third.take(5) == taken
    assert second.take(3) == first.take(3)


@needs_corpus
def test_mixed_stream_draws_from_every_declared_source():
    from experiments.owner_language.corpus import load_items
    tokens = Tokens()
    items = load_items(split="train")
    sources = ["grammar", "plato", "calculus"]
    streams = {source: tr.SourceStream(source, [row for row in items if row["source"] == source],
                                       tokens, length=32) for source in sources}
    _, exposure = tr.MixedStream(streams, sources).batch(9)
    assert sorted(exposure) == sorted(sources)
    assert all(count == 3 for count in exposure.values())


def test_mixed_stream_reports_a_source_that_yields_nothing():
    class Empty:
        source = "empty"

        def take(self, count):
            return []

    with pytest.raises(ValueError, match="no progress"):
        tr.MixedStream({"empty": Empty()}, ["empty"]).batch(4)


def test_checkpoint_store_verifies_its_own_bytes(scratch):
    store = tr.CheckpointStore(scratch / "store")
    payload = {"language_state": {"a": torch.zeros(3)}, "optimizer": {}, "rng": {}}
    written = store.write(payload, {"revision": 0, "step": 0, "note": "first"})
    assert (scratch / "store/current.json").is_file()
    metadata, restored = store.read()
    assert metadata["revision"] == 0
    assert torch.equal(restored["language_state"]["a"], torch.zeros(3))
    weights = scratch / "store/revisions" / written["weights"]
    weights.write_bytes(weights.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="weights failed"):
        store.read()


def test_checkpoint_store_keeps_every_revision(scratch):
    store = tr.CheckpointStore(scratch / "store")
    for revision in range(3):
        store.write({"language_state": {}, "optimizer": {}, "rng": {}},
                    {"revision": revision, "step": revision * 10})
    assert store.history() == ["000000.json", "000001.json", "000002.json"]
    metadata, _ = store.read()
    assert metadata["revision"] == 2
    first = json.loads((scratch / "store/revisions/000000.json").read_text())
    assert first["step"] == 0
