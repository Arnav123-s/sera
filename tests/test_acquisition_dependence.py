import numpy as np
import pytest

from experiments.acquisition_dependence.study import (
    AcquisitionNotIdentified,
    assert_acquired_effect,
)


def test_an_invariant_representation_cannot_support_acquisition_attribution():
    with pytest.raises(AcquisitionNotIdentified, match="erases"):
        assert_acquired_effect(np.zeros((8, 2)), np.full((8, 2), 1e-15))


def test_detectable_effect_is_only_a_preflight():
    assert assert_acquired_effect([0, 1], [0, 1.2]) == pytest.approx(.2)
    with pytest.raises(ValueError, match="Matched finite"):
        assert_acquired_effect([0, 1], [0])
    with pytest.raises(ValueError, match="Matched finite"):
        assert_acquired_effect([np.nan], [0])
