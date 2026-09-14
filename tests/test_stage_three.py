import json

import pytest
import torch

from sera.contracts import EvidenceKind, Provenance
from sera.r1 import RecurrentWorldModel, WorldSession
from sera.storage import digest


@pytest.mark.parametrize("kind", ["delta", "lowrank_hybrid"])
def test_world_session_resume_preserves_future_behavior_and_provenance(tmp_path, kind):
    torch.manual_seed(19)
    model = RecurrentWorldModel(kind=kind)
    source = Provenance("executed-world", "trajectory-19", EvidenceKind.VERIFIED)
    original = WorldSession(model, "resume-world", 3, 0, owner="research-session", model_version="v7",
                            admitted_provenance=[source], memory_references=["replay:19"])
    original.observe(-1, 1, 0.0)
    original.observe(2, 0, 0.0)
    path = tmp_path / "session.json"
    original.save(path)
    restored = WorldSession.load(path, model, owner="research-session", model_version="v7")
    assert restored.admitted_provenance == original.admitted_provenance
    assert restored.memory_references == ["replay:19"]
    assert torch.equal(restored.hidden, original.hidden)
    for observation, action in [(-1, 2), (1, 3), (3, 0)]:
        original.observe(observation, action, float(observation == 3))
        restored.observe(observation, action, float(observation == 3))
        assert torch.equal(restored.hidden, original.hidden)
        assert restored.plan(horizon=2) == original.plan(horizon=2)
    assert restored.events_seen == original.events_seen


def test_world_session_rejects_owner_shape_and_model_mismatches(tmp_path):
    model = RecurrentWorldModel()
    session = WorldSession(model, "world", 3, 0, owner="owner-a")
    path = tmp_path / "session.json"
    session.save(path)
    with pytest.raises(ValueError, match="ownership"):
        WorldSession.load(path, model, owner="owner-b")
    with pytest.raises(ValueError, match="ownership"):
        WorldSession.load(path, model, owner="owner-a", encoder_version="wrong-encoder")
    envelope = json.loads(path.read_text())
    envelope["payload"]["hidden"]["shape"] = [1, 999]
    envelope["sha256"] = digest(envelope["payload"])
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="shape"):
        WorldSession.load(path, model, owner="owner-a")
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="model changed"):
        session.plan()


@pytest.mark.parametrize("complex_valued", [False, True])
def test_general_instrument_joint_likelihood_and_history(complex_valued):
    from sera.r2 import ControlledInstrument
    torch.manual_seed(21)
    model = ControlledInstrument(dimension=8, rank=2, event_kind="kraus", event_rank=2,
                                 complex_valued=complex_valued)
    operators = model.operators()
    observations, actions = torch.tensor([[0, 2, -1, 1]]), torch.tensor([[1, 3, 0]])
    initial, _ = model.observe(model.initial(1), observations[:, 0], operators)
    joint = initial
    for action, color in zip(actions[0], observations[0, 1:]):
        joint = sum(k[None] @ joint @ k.mH[None] for k in operators[0][action])
        events = operators[1].flatten(0, 1) if color < 0 else operators[1][color]
        joint = sum(k[None] @ joint @ k.mH[None] for k in events)
    likelihoods, _, posterior = model.sequence(observations, actions, torch.tensor([1]))
    visible = observations[:, 1:] >= 0
    selected = likelihoods.gather(-1, observations[:, 1:, None].clamp_min(0)).squeeze(-1)
    actual = selected[visible].prod()
    torch.testing.assert_close(actual, joint.diagonal(dim1=-2, dim2=-1).sum(-1).real[0])
    torch.testing.assert_close(posterior.diagonal(dim1=-2, dim2=-1).sum(-1).real, torch.ones(1))
    assert max(model.validity().values()) < 1e-5
    first = torch.zeros(1, 8, 8, dtype=model.action_raw.dtype)
    second = first.clone()
    first[0, 0, 0], second[0, 1, 1] = 1, 1
    a, _ = model.observe(first, torch.tensor([2]), operators)
    b, _ = model.observe(second, torch.tensor([2]), operators)
    assert float((a - b).abs().max().detach()) > .01
    (-actual.log()).backward()
    for parameter in (model.action_raw, model.event_raw):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().max() > 0


def test_program_search_composes_prior_skill_and_pins_dependencies():
    import copy

    from sera.environments import WorldSpec
    from sera.r2 import search_program, validate_library
    world = WorldSpec("composing-cycle", tuple(tuple([(state + 1) % 4] * 4) for state in range(4)))
    base = search_program(world, 0, 2, budget=20, max_length=2)
    assert base["success"] and base["record"]["failure_cases"]
    library = {base["skill_id"]: {"kind": "action_program", **base["record"]}}
    composed = search_program(world, 0, 3, library=library, budget=8, max_length=3)
    without = search_program(world, 0, 3, budget=8, max_length=3)
    assert composed["success"] and not without["success"]
    record = {"kind": "action_program", **composed["record"]}
    assert base["skill_id"] in record["dependencies"]
    assert len(record["actions"]) == 3
    library[composed["skill_id"]] = record
    validate_library(library, world.identifier)
    with pytest.raises(ValueError, match="world domain"):
        validate_library(library, "another-world")
    tampered = copy.deepcopy(library)
    tampered[base["skill_id"]]["examples"] = []
    with pytest.raises(ValueError, match="version changed"):
        validate_library(tampered, world.identifier)


def test_selective_router_skips_unselected_state_and_persists_rank_diagnostics(tmp_path):
    torch.manual_seed(8)
    model = RecurrentWorldModel(kind="lowrank_hybrid", routing="top1")
    with torch.no_grad():
        model.memory.router.weight.zero_()
        model.memory.router.bias.copy_(torch.tensor([-20.0, -20.0, 20.0]))
    initial = model.initial(1)
    session = WorldSession(model, "selective-world", 1, 0, owner="density-session")
    assert torch.equal(session.state["delta"], initial["delta"])
    assert torch.equal(session.state["rotor"], initial["rotor"])
    row = session.diagnostics[-1]
    assert row["branch_rows_executed"] == {"delta": 0, "rotor": 0, "density": 1}
    assert row["density"]["rank"] == 2
    assert all(0 <= mass <= 1 for mass in row["density"]["discarded_mass"][0])
    path = tmp_path / "selective.json"
    session.save(path)
    restored = WorldSession.load(path, model, owner="density-session")
    assert restored.diagnostics == session.diagnostics
    restored.observe(1, 0, 1.0)
    assert len(restored.diagnostics) == 2
def test_typed_admission_units_masks_and_saved_inference(tmp_path):
    import dataclasses
    import json

    from sera.cli import main
    from sera.contracts import EvidenceKind, Observation, Provenance
    from sera.models import ModelConfig, StatefulModel
    from sera.solver import Solver, SolverStore
    from sera.typed_learning import (
        TypedEncoder,
        TypedEvidence,
        TypedReasoner,
        fit_typed,
        typed_examples,
    )

    p = Provenance("test", "one", EvidenceKind.OBSERVATION)
    meter = TypedEncoder.features(Observation("numeric", (1.,), 0, p, units="m"))
    centimeter = TypedEncoder.features(Observation("numeric", (100.,), 0, p, units="cm"))
    assert meter[:32] == centimeter[:32]
    masked = TypedEncoder.features(Observation("numeric", (100.,), 0, p, units="m", available=(False,)))
    assert masked[0] == masked[16] == 0
    examples = typed_examples(seed=1, count=2)
    with pytest.raises(ValueError, match="Sealed"):
        TypedEvidence([dataclasses.replace(examples[0], split="test-final")])
    with pytest.raises(ValueError, match="admitted"):
        TypedEvidence([dataclasses.replace(examples[0], evidence=p)])
    model = TypedReasoner(width=16, memory_dim=4)
    fit_typed(model, TypedEvidence(examples), validation=typed_examples(seed=2, count=1, split="validation"), steps=2)
    assert all(adapter[0].weight.grad is not None for adapter in model.encoder.adapters.values())
    store = SolverStore(tmp_path / "solver")
    store.initialize(Solver(StatefulModel(ModelConfig()), components={"typed": model}))
    row = examples[0]
    expected = model.predict(row.observations, row.task)
    assert store.load().components["typed"].predict(row.observations, row.task) == expected
    request = tmp_path / "request.json"
    value = {"task": row.task, "observations": [{k: v for k, v in dataclasses.asdict(obs).items() if k != "provenance"}
                                                 for obs in row.observations]}
    request.write_text(json.dumps(value))
    assert main(["typed-solve", str(store.root), str(request)]) == 0


def test_adapter_migration_and_freeze_are_real():
    from sera.environments import collect, make_world
    from sera.experience import EvidenceReplay
    from sera.models import ModelConfig, StatefulModel
    from sera.mutations import Mutation, mutate
    from sera.r1 import RecurrentWorldModel, fit, tensors
    from sera.solver import Solver

    model = RecurrentWorldModel(width=16, memory_dim=4)
    rows, _ = collect(make_world(1), seed=1, count=8)
    parent = Solver(StatefulModel(ModelConfig()), components={"r1": model})
    child, record = mutate(parent, Mutation("insert_adapter", value=4))
    assert record["function_preserving_at_initialization"]
    adapted = child.components["r1"]
    with torch.no_grad():
        assert torch.equal(model(*tensors(rows))[0], adapted(*tensors(rows))[0])
    before = {k: v.clone() for k, v in adapted.state_dict().items()}
    fit(adapted, EvidenceReplay(rows), steps=3, update_mode="adapter")
    changed = {k for k, v in adapted.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and all(k.startswith("adapter.") for k in changed)
    assert parent.identity() == record["parent_solver"]


def test_actual_round_budget_replay_and_failure_history(tmp_path):
    from sera.connected import autonomous_round
    from sera.curriculum import ImprovementPolicy
    from sera.environments import make_world
    from sera.experience import EvidenceReplay
    from sera.models import ModelConfig, StatefulModel
    from sera.r1 import RecurrentWorldModel
    from sera.solver import Solver, SolverStore

    controller = ImprovementPolicy(features=16, methods=("none", "targeted"))
    for p in controller.parameters():
        p.data.zero_()
    controller.network[-1].bias.data[1] = 1
    store = SolverStore(tmp_path)
    store.initialize(Solver(StatefulModel(ModelConfig()), components={"r1": RecurrentWorldModel(width=16),
                                                                    "controller": controller}))
    replay, world = EvidenceReplay(), make_world(5)
    _, construction, _ = autonomous_round(store, world, [], replay, seed=1, samples=8, steps=5, total_update_budget=2)
    assert construction["update_steps"] == 2
    assert len(replay.records) == 96
    _, next_construction, _ = autonomous_round(store, world, [], replay, seed=2, samples=8, steps=5, total_update_budget=2)
    assert next_construction["method"] == "none"
    assert next_construction["update_steps"] == 0
    assert len(EvidenceReplay.load(tmp_path / "experience.json").records) == 128
    import json
    history = json.loads((tmp_path / "learning-history.json").read_text())
    assert history[-1]["diagnosis"]["previous_attempts"] == 1
    assert history[-1]["diagnosis"]["remaining_budget"] == 0


def test_costs_include_failure_and_policy_rejects_sealed_episodes():
    from sera.accounting import Costs
    from sera.curriculum import ImprovementPolicy, fit_policy
    costs = Costs()
    with pytest.raises(ValueError):
        with costs.phase("failed-proposal"):
            raise ValueError("expected")
    assert costs.record()["phases"][0]["status"] == "failed"
    assert costs.record()["process_peak_rss_bytes"] > 0
    with pytest.raises(ValueError, match="Sealed"):
        fit_policy(ImprovementPolicy(), [{"episode_id": "a", "split": "anchor"}],
                   [{"episode_id": "b"}], steps=1)


def test_arithmetic_program_is_induced_verified_and_reused():
    from sera.typed_learning import typed_examples
    from sera.typed_programs import execute_rule, induce, validate_program
    support = typed_examples(seed=5, count=48)
    validation = typed_examples(seed=6, count=16, split="validation")
    for task in ("modular_sum", "byte_sum"):
        result = induce(support, validation, task=task)
        assert result["accepted"]
        assert result["record"]["rule"] == {"operation": "sum", "modulus": 4}
        validate_program(result["record"])
        heldout = [row for row in typed_examples(seed=7, count=16, split="test", family="structure") if row.task == task]
        assert all(execute_rule(result["record"]["rule"], row.observations) == row.target for row in heldout)
        result["record"]["rule"]["modulus"] = 3
        with pytest.raises(ValueError, match="integrity"):
            validate_program(result["record"])


def test_general_proposal_scores_the_actual_final_event():
    import math

    from sera.r2 import ControlledInstrument, ranked_programs
    torch.manual_seed(84)
    model = ControlledInstrument(dimension=8, rank=2, event_kind="kraus")
    operators = model.operators()
    rho, _ = model.observe(model.initial(1), torch.tensor([0]), operators)
    expected = {}
    for action in range(4):
        log_prior = float(model.proposal_logits(rho, torch.tensor([2])).log_softmax(-1)[0, action].detach())
        next_state = model.control(rho, torch.tensor([action]), operators)
        probability = float(model.probabilities(next_state, operators)[0, 2].detach())
        expected[(action,)] = log_prior + math.log(probability) - .04
    for value, actions, _ in ranked_programs(model, 0, 2, max_length=1):
        assert value == pytest.approx(expected[actions], abs=1e-6)


def test_archive_uses_development_and_retrieves_immutable_specialist(tmp_path):
    from sera.archive import retrieve_specialist, useful_archive
    from sera.models import ModelConfig, StatefulModel
    from sera.solver import Solver, SolverStore
    store = SolverStore(tmp_path)
    store.initialize(Solver(StatefulModel(ModelConfig())))
    with pytest.raises(ValueError, match="development"):
        useful_archive(store, lambda s: {"one": .5}, dataset_id="heldout", split="test")
    archive = useful_archive(store, lambda s: {"one": .5}, dataset_id="development-draw")
    assert archive["useful_versions"] == ["v0"]
    assert retrieve_specialist(store, archive, "one").identity() == store.load().identity()


def test_all_missing_instrument_batch_does_not_invent_targets():
    from sera.contracts import EvidenceKind, Provenance
    from sera.experience import EvidenceReplay, Trajectory
    from sera.r2 import ControlledInstrument, fit_instrument
    from sera.solver import Work
    model = ControlledInstrument(dimension=8, rank=2, event_kind="kraus")
    row = Trajectory("masked", (0, -1, -1, -1), (0, 1, 2), (0., 0., 0.), 0,
                     Provenance("observed-execution", "masked-case", EvidenceKind.VERIFIED), "support")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    work = Work()
    result = fit_instrument(model, EvidenceReplay([row]), steps=2, work=work)
    assert result["history"] == [0., 0.]
    assert work.counts["instrument_visible_target_draws"] == 0
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    credit = Trajectory("masked", (0, 1), (0,), (1.,), 1,
                        Provenance("observed-execution", "verified-plan", EvidenceKind.VERIFIED), "program-support")
    result = fit_instrument(model, EvidenceReplay([row]), plans=[credit], steps=2)
    assert all(torch.isfinite(torch.tensor(result["history"])))
    assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items())
