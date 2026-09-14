import copy

import pytest
import torch

from sera.contracts import EvidenceKind, Provenance
from sera.environments import collect, make_world
from sera.experience import EvidenceReplay, Trajectory
from sera.r1 import RecurrentWorldModel, WorldSession, fit, tensors
from sera.r2 import ControlledInstrument, flatten_program


def test_learning_admission_masks_truth_and_roundtrips(tmp_path):
    records, truth = collect(make_world(11), seed=0, count=4, length=5, mask_rate=1)
    assert all(r.observations[1:] == (-1,) * 5 for r in records)
    assert (truth >= 0).all()
    replay = EvidenceReplay(records)
    replay.save(tmp_path / "replay.json")
    restored = EvidenceReplay.load(tmp_path / "replay.json")
    assert restored.identifiers == replay.identifiers
    row = records[0]
    predicted = Trajectory(row.world_id, row.observations, row.actions, row.rewards, row.goal,
                           Provenance("model", "prediction", EvidenceKind.PREDICTION), "support")
    with pytest.raises(ValueError, match="provenance"):
        replay.admit(predicted)
    query, _ = collect(make_world(11), seed=0, count=1, split="test-hidden")
    with pytest.raises(ValueError, match="evaluation"):
        replay.admit(query[0])


def test_r1_feedback_updates_encoder_memory_transition_and_reward_and_plans():
    torch.manual_seed(77)
    world = make_world(3)
    rows, _ = collect(world, seed=0, count=16, length=4, mask_rate=0.3)
    model = RecurrentWorldModel(width=16, memory_dim=4)
    before = copy.deepcopy(model.state_dict())
    fit(model, EvidenceReplay(rows), steps=2, batch_size=8)
    for prefix in ("memory.encoder", "memory.cores", "transition", "reward"):
        assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()
                   if name.startswith(prefix))
    session = WorldSession(model, world.identifier, 2, 0)
    state = copy.deepcopy(session.state)
    actions = session.plan(horizon=2, beam=2)
    assert 1 <= len(actions) <= 2
    for name in state:
        torch.testing.assert_close(session.state[name], state[name])
    outcome = world.sensor_transition(0, actions[0])
    session.observe(outcome, actions[0], float(outcome == 2))
    assert any(not torch.equal(value, state[name]) for name, value in session.state.items())


@pytest.mark.parametrize("complex_valued", [True, False])
def test_controlled_instrument_matches_unnormalized_joint_likelihood(complex_valued):
    torch.manual_seed(72)
    model = ControlledInstrument(complex_valued=complex_valued)
    operators = model.operators()
    colors, actions = torch.tensor([[0, 2, 1, 3]]), torch.tensor([[3, 1, 2]])
    initial, _ = model.observe(model.initial(1), colors[:, 0], operators)
    unnormalized = initial
    for action, color in zip(actions[0], colors[0, 1:]):
        k = operators[0][action]
        unnormalized = sum(operator[None] @ unnormalized @ operator.mH[None] for operator in k)
        event = operators[1][color]
        unnormalized = event[None] @ unnormalized @ event.mH[None]
    conditional, _, _ = model.sequence(colors, actions, torch.tensor([3]))
    probability = conditional.gather(-1, colors[:, 1:, None]).squeeze(-1).prod(-1)
    torch.testing.assert_close(probability, unnormalized.diagonal(dim1=-2, dim2=-1).sum(-1).real,
                               atol=1e-7, rtol=1e-5)
    (-probability.log().sum()).backward()
    assert torch.isfinite(model.action_raw.grad).all() and model.action_raw.grad.abs().max() > 0
    assert max(model.validity().values()) < 1e-5
    prior = model.control(initial, actions[:, 0], operators)
    missing, _ = model.observe(prior, torch.tensor([-1]), operators)
    torch.testing.assert_close(missing, model.branches(prior, operators).sum(1))


def test_program_composition_fuel_and_recursion():
    library = {"move": {"body": {"op": "repeat", "count": 2,
                                    "body": {"op": "action", "value": 1}}}}
    body = {"op": "sequence", "items": [{"op": "call", "skill": "move"},
                                            {"op": "action", "value": 2}]}
    assert flatten_program(body, library) == (1, 1, 2)
    with pytest.raises(RuntimeError, match="fuel"):
        flatten_program(body, library, fuel=2)
    with pytest.raises(ValueError, match="recursive"):
        flatten_program({"op": "call", "skill": "cycle"},
                        {"cycle": {"body": {"op": "call", "skill": "cycle"}}})


def test_reference_state_dimensions_and_lowrank_density_validity():
    torch.manual_seed(17)
    model = RecurrentWorldModel(width=256, heads=8, memory_dim=32, kind="reference")
    state = model.initial(1)
    assert sum(v.numel() * v.element_size() for v in state.values()) == 35840
    assert state["delta"].shape == (1, 8, 32, 32)
    assert state["rotor"].shape == (1, 128, 2)
    assert state["density"].shape == (1, 4, 16, 4)
    rows, _ = collect(make_world(1), seed=0, count=1, length=2)
    logits, reward = model(*tensors(rows))
    (logits.square().sum() + reward.square().sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    factor = state["density"]
    rho = factor @ factor.mH
    torch.testing.assert_close(rho.diagonal(dim1=-2, dim2=-1).sum(-1).real, torch.ones(1, 4))
    assert torch.linalg.eigvalsh(rho).min() > -1e-6
    assert 0 <= model.memory.density.last_discarded_mass <= 1
