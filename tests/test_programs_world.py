import numpy as np
import pytest
import torch

from sera.programs import (
    BudgetExhausted,
    SkillLibrary,
    TransitionProgram,
    discover,
    verify_program,
)
from sera.world import FiniteWorld, plan, world_experiment


def test_discovery_unseen_dynamics_transfer_and_library_round_trip(tmp_path):
    rng = np.random.default_rng(10)
    for n in (3, 4, 6):
        table = rng.integers(0, n, (n, 3))
        table[:, 0] = (np.arange(n) + 1) % n  # guarantee reachability
        world = FiniteWorld(table, environment_id=f"unseen-{n}")
        result = discover(world.query, environment_id=world.environment_id, actions=world.actions)
        assert result.oracle_queries == 1 + 2 * n * world.actions
        verification = verify_program(
            result.program, world.query, seed=500, samples=30, lengths=(12, 96)
        )
        assert verification["passed"]
        library = SkillLibrary(tmp_path)
        identifier = library.admit(result, verification)
        loaded = library.load(identifier)
        for actions in rng.integers(0, 3, (10, 200)):
            assert loaded.execute(actions) == world.query(actions)
        with pytest.raises(ValueError, match="domain"):
            loaded.execute([0], environment_id="different-world")


def test_budget_and_stochastic_failure_are_explicit():
    world = FiniteWorld()
    with pytest.raises(BudgetExhausted):
        discover(world.query, environment_id="budget", actions=4, max_queries=4)
    replies = iter([0, 1, 2])
    with pytest.raises(ValueError, match="deterministic"):
        discover(lambda _: next(replies), environment_id="stochastic", actions=1)
    program = TransitionProgram("bounded", 0, ((1,), (0,)), max_steps=2)
    with pytest.raises(BudgetExhausted):
        program.execute([0, 0, 0])
    with pytest.raises(ValueError):
        program.execute([-1])
    with pytest.raises(ValueError):
        TransitionProgram("bad", 0, ((999,),))


def test_planner_obeys_budget_and_verifies_action_conditioning():
    world = FiniteWorld()

    def distribution(s, a):
        return np.eye(world.states)[world.transition(s, a)]

    result = plan(distribution, start=0, goal=3, actions=4)
    assert result.reached and world.query(result.actions) == 3
    blocked = plan(distribution, start=0, goal=3, actions=4, max_expansions=1)
    assert not blocked.reached and blocked.expansions == 1
    torch.set_num_threads(1)
    _, learned = world_experiment(seed=0, steps=100)
    assert learned["transition_accuracy"] == 1
    assert learned["learned_success"] == learned["oracle_success"] == 1
