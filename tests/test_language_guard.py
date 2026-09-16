import copy

import torch

from experiments.grounded_language.acquisition import guarded_interpret, qualified_shapes, shape
from experiments.grounded_language.study import RELEASE, load_selected
from sera.session_state import model_identity


def test_unsupported_wording_never_reaches_model():
    class NeverCall:
        def interpret(self, texts):
            raise AssertionError("Unsupported task reached the neural interpreter")
    owner = NeverCall()
    for text in ("multiply x by -2 then subtract 3 to get 4 modulo eleven",
                 "multiply x by 2.5 then subtract 3 to get 4 modulo eleven",
                 "not multiply x by two then subtract three to get four modulo eleven",
                 "multiply x by two then subtract three to get four modulo twelve"):
        result = guarded_interpret(owner, text, threshold=0, shapes=qualified_shapes())
        assert result["status"] in ("WITHHELD", "NEEDS_LESSON")
    assert shape("multiply x by TWO then subtract 3 to get four modulo eleven") in qualified_shapes()


def test_selected_actual_owner_restores_without_changing_shared_aliases():
    import pytest
    if not (RELEASE/"L10-PILOT-INTERFACE/result.json").exists():
        pytest.skip("Local research checkpoint is required")
    torch.set_num_threads(1)
    learner, result = load_selected(RELEASE/"L10-PILOT-INTERFACE")
    owner = learner.session.owner
    identity = model_identity(owner)
    assert identity == result["selected_owner"]
    assert learner.solver.neural.owner is learner.solver.components["typed"].owner is owner
    text = "subtract three from x then multiply by two to get four modulo eleven"
    answer = guarded_interpret(owner, text, threshold=result["calibration"]["threshold"], shapes=qualified_shapes())
    assert answer["status"] == "ACCEPTED"
    assert answer["labels"] == [1, 2, 3, 4]
    assert answer["solutions"] == [5]
    copied = copy.deepcopy(owner)
    assert model_identity(copied) == identity
    assert model_identity(owner) == identity
