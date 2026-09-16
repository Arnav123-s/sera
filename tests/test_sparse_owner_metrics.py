import numpy as np

from experiments.sparse_mechanisms.owner_weights import residual_weight_error


def test_weight_repair_zero_means_recovered_and_one_means_unchanged_fault():
    original = np.array([.17, -.21, .3])
    fault = np.array([0., 2., -3.])
    assert residual_weight_error(original, original, fault) == 0.
    np.testing.assert_allclose(residual_weight_error(original+fault, original, fault), 1.)
    np.testing.assert_allclose(residual_weight_error(original+fault/2, original, fault), .5)
