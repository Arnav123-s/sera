import copy

import numpy as np
import pytest

from experiments.generative_memory.applicability import (
    LearnedGuard,
    extract_features,
    fit_guard,
    select_threshold,
    validate_teaching,
)
from experiments.generative_memory.applicability_audit import features_from_evidence, probability
from experiments.generative_memory.applicability_study import (
    QUERY_T,
    VARIANCE,
    assert_partition,
    evidence_episode,
    metrics,
    seed_for,
)
from experiments.generative_memory.core import digest


@pytest.fixture(scope="module")
def fixture():
    teaching, records = [], []
    for family in ("circle", "local_exception", "random_values"):
        for i in range(2):
            t, r = evidence_episode("timing", family, 20+i)
            t["split"] = "teach" if i == 0 else "probability_calibration"
            teaching.append(t)
            records.append(r)
    model = fit_guard([r for r in teaching if r["split"] == "teach"],
                      [r for r in teaching if r["split"] == "probability_calibration"],
                      optimizer_seed=390099, steps=25)
    return teaching, records, model


def test_protected_families_and_mechanism_instances():
    for split in ("teach", "probability_calibration", "threshold_selection"):
        for family in ("piecewise_drift", "chirp"):
            with pytest.raises(ValueError, match="Protected"):
                seed_for(split, family, 0)
    seeds = {seed_for(s, f, i) for s in ("teach", "probability_calibration", "threshold_selection", "final")
             for f in ("circle", "ellipse", "local_exception") for i in range(5)}
    assert len(seeds) == 60


def test_assessor_cannot_enter_training_and_splits_cannot_overlap(fixture):
    teaching, _, _ = fixture
    row = copy.deepcopy(teaching[0])
    row["family"] = "circle"
    with pytest.raises(ValueError, match="hidden"):
        validate_teaching([row], "teach")
    with pytest.raises(ValueError, match="duplicate"):
        validate_teaching([teaching[0], teaching[0]], "teach")
    cal = copy.deepcopy(teaching[0])
    cal["split"] = "probability_calibration"
    with pytest.raises(ValueError, match="reuses"):
        fit_guard([teaching[0]], [cal], optimizer_seed=1, steps=1)


def test_feature_algebra_independently_from_observations(fixture):
    _, rows, _ = fixture
    for row in rows:
        independent, mean, _, _ = features_from_evidence(
            row["posterior"]["payload"]["observed_events"], QUERY_T, VARIANCE)
        np.testing.assert_allclose(independent, row["features"], atol=2e-8, rtol=0)
        np.testing.assert_allclose(mean, row["predicted_means"], atol=2e-8, rtol=0)


def test_query_answers_do_not_affect_features(fixture):
    from experiments.generative_memory.acquisition import Posterior

    _, rows, _ = fixture
    row = copy.deepcopy(rows[0])
    row["assessor"]["family"] = "invented"
    row["assessor"]["query_observed"] = [[1e9, -1e9]]*len(QUERY_T)
    actual, _ = extract_features(Posterior.restore(row["posterior"]), QUERY_T, VARIANCE)
    assert actual.tolist() == row["features"]


def test_guard_restore_independent_numpy_and_external_mutation(fixture):
    _, rows, artifact = fixture
    caller_copy = copy.deepcopy(artifact)
    guard = LearnedGuard(caller_copy)
    x = np.array(rows[0]["features"])
    original = guard.predict(x)
    caller_copy["payload"]["calibration_offset"] = 900
    assert np.array_equal(original, guard.predict(x))
    np.testing.assert_allclose(original, probability(x, artifact["payload"]), atol=2e-14, rtol=0)
    assert LearnedGuard(guard.artifact()).artifact() == artifact


@pytest.mark.parametrize("field", ["feature_schema", "sources", "runtime"])
def test_stale_guard_rejected_even_with_recomputed_hash(fixture, field):
    artifact = copy.deepcopy(fixture[2])
    artifact["payload"]["contract"][field] = "changed"
    artifact["sha256"] = digest(artifact["payload"])
    with pytest.raises(ValueError, match="Stale"):
        LearnedGuard(artifact)


def test_empty_acceptance_has_no_risk_claim_and_thresholds_use_observed_errors():
    rejected = metrics([1, 0], [.001, 3], [False, False])
    assert rejected["coverage"] == 0 and rejected["selective_risk"] is None
    result = select_threshold([[.9, .8, .1], [.95, .7, .2]], [[1, 1, 0], [1, 0, 0]])
    assert result["selected"]["accepted"] == 3
    assert result["selected"]["selective_risk"] == 0
    result = select_threshold([[.1, .2]], [[0, 0]])
    assert result["selected"]["accepted"] == 0


def test_partition_duplicate_failure(fixture):
    row = copy.deepcopy(fixture[1][0])
    row["split"] = "teach"
    with pytest.raises(ValueError, match="reused"):
        assert_partition([row, row])


def test_ineligible_predictions_cannot_supply_features(fixture):
    from experiments.generative_memory.acquisition import Posterior

    artifact = copy.deepcopy(fixture[1][0]["posterior"])
    artifact["payload"]["observed_events"][0]["provenance"]["kind"] = "prediction"
    artifact["sha256"] = digest(artifact["payload"])
    with pytest.raises(ValueError):
        Posterior.restore(artifact)
