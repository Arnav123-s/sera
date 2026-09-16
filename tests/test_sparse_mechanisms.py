import copy

import numpy as np
import pytest

from experiments.generative_memory.core import Observations, design
from experiments.sparse_mechanisms.core import (
    dictionary,
    fit,
    guard,
    l1,
    matrix,
    observations,
    predict,
)


def test_common_dictionary_matches_existing_expression_interpreter():
    t = np.linspace(.02, .99, 27)
    terms = dictionary(17)
    a = matrix(terms, t)
    old = design({"kind": "expression", "terms": terms}, t)
    scales = np.array([1]+[np.sqrt(2)]*16)
    np.testing.assert_allclose(a/scales, old[::2, ::2])
    np.testing.assert_allclose(a/scales, old[1::2, 1::2])


def test_lp_dual_certificate_and_orthogonal_recovery():
    a = np.eye(7)
    y = np.zeros((7, 2))
    y[[1, 4]] = [[1, -.5], [.7, 1.2]]
    coefficients, cert = l1(a, y, 0)
    np.testing.assert_allclose(coefficients, y)
    assert all(c["objective_gap"] < 1e-9 and c["dual_violation"] < 1e-9 for c in cert)
    noisy, _ = l1(a, y+.01, .02)
    assert np.max(abs(a@noisy-(y+.01))) <= .02000001


def test_existing_roles_reject_imagination_and_overlap_and_restore_identity():
    t = np.arange(12)/12
    y = np.column_stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)])
    support = observations(t, y, "support")
    selection = observations(t+.011, y, "selection")
    with pytest.raises(ValueError, match="observed"):
        fit("ridge", Observations(t, y, "support", "imagined"), selection, dictionary(7), 0)
    with pytest.raises(ValueError, match="overlap"):
        fit("ridge", support, observations(t, y, "selection"), dictionary(7), 0)
    model = fit("ridge", support, selection, dictionary(7), 0)
    with pytest.raises(ValueError, match="overlaps"):
        guard(model, observations(t, y, "calibration"))
    restored = copy.deepcopy(model)
    np.testing.assert_array_equal(predict(model, t), predict(restored, t))
    restored["source"] = "0"*64
    with pytest.raises(ValueError, match="interpreter changed"):
        predict(restored, t)


def test_alias_is_withheld_even_with_zero_training_error():
    terms = dictionary(33)
    t = np.arange(12)/12
    y = np.column_stack([np.cos(2*np.pi*t), np.cos(2*np.pi*t)])
    selection_t = (np.arange(8)+.123)/8
    selection = observations(selection_t, np.column_stack([np.cos(2*np.pi*selection_t)]*2), "selection")
    model = fit("basis_pursuit", observations(t, y, "support"), selection, terms, 0)
    assert model["measurement_diagnostics"]["aliased_pairs"]
    calibration = observations(selection_t+.001, predict(model, selection_t+.001), "calibration")
    assert not guard(model, calibration)["accepted"]
