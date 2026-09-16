import copy

import torch

from experiments.continuing_control.core import InteractiveR1
from experiments.grounded_language.acquisition import lesson
from experiments.grounded_language.data import examples
from experiments.grounded_language.model import LanguageR1
from sera.generative import GenerativeSharedR1
from workbench.owner import LiveR1


def test_lesson_resume_matches_uninterrupted_parameters_and_draws():
    torch.set_num_threads(1)
    torch.manual_seed(170003)
    owner = LanguageR1.attach(LiveR1.attach(InteractiveR1.extend(
        GenerativeSharedR1(width=12, heads=2, memory_dim=4, kind="delta"), 32)), 170009)
    teaching = examples(170021, 24, "train", lesson=True)
    development = examples(170027, 24, "development", lesson=True)
    full, partial, resumed = copy.deepcopy(owner), copy.deepcopy(owner), copy.deepcopy(owner)
    saved = []
    first = lesson(full, teaching, development, kind="word", steps=40, seed=170033)
    lesson(partial, teaching, development, kind="word", steps=20, seed=170033,
           checkpoint=lambda state: saved.append(copy.deepcopy(state)))
    second = lesson(resumed, teaching, development, kind="word", steps=40, seed=170033, resume=saved[-1])
    assert first["history"] == second["history"]
    assert second["new_steps"] == 20
    assert all(torch.equal(value, resumed.state_dict()[name]) for name, value in full.state_dict().items())
