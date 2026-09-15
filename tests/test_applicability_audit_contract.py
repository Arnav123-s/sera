import copy

import numpy as np
import pytest

from experiments.generative_memory.acquisition import Action, Posterior, observed_event
from experiments.generative_memory.applicability_audit import batch, validate_observed_event


def test_independent_checker_accepts_actual_serialized_simulator_evidence():
    event = observed_event(Action(0, 0., "target", .0025), [1., 2.], 0, "checker/regression")
    validate_observed_event(event.record(), 0, [1., 2.])


def test_independent_coefficients_and_covariances_use_declared_storage_order():
    t = np.linspace(-1, 1, 7)
    y = np.stack([1+np.sin(np.pi*t), -2+.5*np.cos(np.pi*t)], axis=1)
    actions = [Action(i, float(value), "target", .0025) for i, value in enumerate(t)]
    posterior = Posterior(actions, max_observations=7)
    for i, action in enumerate(actions):
        posterior.observe(observed_event(action, y[i], i, f"checker/coefficients/{i}"))
    reconstructed = batch(t, y.ravel(), [0.3], .0025)[5]
    for (mean, covariance), original in zip(reconstructed, posterior.snapshot()["payload"]["models"]):
        np.testing.assert_allclose(mean, original["mean"], atol=2e-8, rtol=0)
        np.testing.assert_allclose(covariance, original["covariance"], atol=2e-8, rtol=0)


@pytest.mark.parametrize("change", ["prediction", "mask", "sequence", "role", "answer"])
def test_independent_checker_rejects_ineligible_or_changed_records(change):
    event = copy.deepcopy(observed_event(Action(0, 0., "target", .0025), [1., 2.], 0,
                                         "checker/regression").record())
    if change == "prediction":
        event["provenance"]["kind"] = "model_prediction"
    elif change == "mask":
        event["available"][-1] = False
    elif change == "sequence":
        event["sequence"] = 1
    elif change == "role":
        event["role"] = "target"
    else:
        event["values"][-1] = 8.
    with pytest.raises(ValueError, match="nonfactual"):
        validate_observed_event(event, 0, [1., 2.])
