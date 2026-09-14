import pytest
import torch

from sera.contracts import EvidenceKind, Experience, Observation, Provenance, StateOwner
from sera.data import make_batch
from sera.models import KINDS, ModelConfig, Session, StatefulModel
from sera.quantum import density_residuals
from sera.training import TrainConfig, load_model, train


def test_data_reproducible_disjoint_and_query_values_masked():
    a = make_batch(32, 12, 0, split="train")
    b = make_batch(32, 12, 0, split="train")
    test = make_batch(32, 12, 0, split="test")
    assert a.dataset_id == b.dataset_id != test.dataset_id
    assert torch.count_nonzero(a.inputs[:, -1, 4:8]) == 0
    for task in (1, 4):
        batch = make_batch(32, 12, 0, split="check", task=task)
        for x, target in zip(batch.inputs, batch.targets):
            keys = x[:-1, :4].argmax(-1)
            query = x[-1, :4].argmax()
            positions = (keys == query).nonzero().flatten()
            position = positions[-1 if task == 1 else 0]
            assert x[position, 4:8].argmax() == target


def test_observation_target_boundary():
    source = Provenance("simulator", "event-1", EvidenceKind.OBSERVATION)
    event = Observation("numeric", (1.0,), 0, source, units="meter")
    with pytest.raises(ValueError, match="units"):
        Observation("numeric", (1.0,), 0, source)
    with pytest.raises(ValueError, match="Learning"):
        Experience((event,), 1, Provenance("model", "guess", EvidenceKind.PREDICTION))
    assert Experience((event,), 1, Provenance("simulator", "truth", EvidenceKind.SYNTHETIC))


@pytest.mark.parametrize("kind", KINDS)
def test_streaming_batch_equivalence_gradients_and_state_reset(kind, tmp_path):
    torch.manual_seed(1)
    model = StatefulModel(ModelConfig(kind=kind, width=8, heads=2, memory_dim=4))
    batch = make_batch(4, 5, 0, split="test")
    logits = model(batch.inputs)
    logits.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    owner = StateOwner("session-a", "v1")
    session = Session(model, owner)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    for event in batch.inputs[0, :2]:
        session.observe(event)
    path = tmp_path / "state.pt"
    session.save(path)
    restored = Session(model, owner)
    restored.restore(path)
    for event in batch.inputs[0, 2:]:
        expected = session.observe(event)
        actual = restored.observe(event)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, logits[0].softmax(-1), atol=2e-6, rtol=2e-6)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(original[key], value)
    torch.testing.assert_close(model(batch.inputs), logits)
    with pytest.raises(ValueError, match="owner"):
        Session(model, StateOwner("other", "v1")).restore(path)
    if "density" in session.state:
        residuals = density_residuals(session.state["density"])
        assert residuals["trace_error"] < 1e-5
        assert residuals["minimum_eigenvalue"] > -1e-6


def test_training_resume_matches_uninterrupted_optimizer_trajectory(tmp_path):
    cfg = ModelConfig(width=8, memory_dim=4)
    common = dict(batch_size=8, validation_samples=8, validate_every=2)
    whole = train(cfg, TrainConfig(steps=4, **common), tmp_path / "whole")
    train(cfg, TrainConfig(steps=2, **common), tmp_path / "resumed")
    resumed = train(cfg, TrainConfig(steps=4, **common), tmp_path / "resumed", resume=True)
    for a, b in zip(whole.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    restored, _ = load_model(tmp_path / "resumed" / "checkpoint.pt")
    x = make_batch(2, 6, 0, split="restored").inputs
    torch.testing.assert_close(restored(x), resumed(x))
