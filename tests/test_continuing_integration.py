import pytest
import torch

from experiments.continuing_control.core import InteractiveR1
from experiments.continuing_control.integration import validate_extension
from sera.generative import GenerativeSharedR1


def test_extension_migration_refuses_changed_legacy_tensors():
    parent = GenerativeSharedR1(width=16, heads=2, memory_dim=4)
    successor = InteractiveR1.extend(parent, 12, True)
    validate_extension(parent, successor)
    with torch.no_grad():
        successor.typed_numeric.bias[0].add_(.1)
    with pytest.raises(ValueError, match="tensors changed"):
        validate_extension(parent, successor)


def test_registered_workspace_cannot_hide_a_new_independent_trainable_model():
    parent = GenerativeSharedR1(width=16, heads=2, memory_dim=4)
    successor = InteractiveR1.extend(parent, 12, True)
    successor.register_parameter("interaction_unapproved", torch.nn.Parameter(torch.ones(2)))
    with pytest.raises(ValueError, match="Additional trainable"):
        validate_extension(parent, successor)
