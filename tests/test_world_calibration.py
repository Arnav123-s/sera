import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.generative_memory.applicability import LearnedGuard
from experiments.generative_memory.core import digest
from experiments.world_calibration.audit import check_certificate, independent_upper
from experiments.world_calibration.risk import (
    PolicyContract,
    WorldCertificate,
    WorldOutcome,
    calibrate,
    upper_bound,
)
from experiments.world_calibration.study import GUARD, episode, feature_group


def contract():
    names = ("predictor", "features", "groups", "query_selector", "acquisition", "loss", "interpreter", "population")
    return PolicyContract(tuple((n, digest(n)) for n in names), ("all", "empty"), (.9, .99))


def rows():
    return [WorldOutcome(str(i), "all", .95, 0, digest(i)) for i in range(160)]


def test_bound_matches_closed_form_and_independent_binomial_inversion():
    assert upper_bound(0, 160, .001) == pytest.approx(1-.001**(1/160))
    for k, n in [(0, 0), (3, 3), (1, 100), (53, 160), (0, 160)]:
        assert upper_bound(k, n, .000625) == pytest.approx(independent_upper(k, n, .000625), abs=1e-11)


@pytest.mark.parametrize("kind", ["duplicate", "forbidden", "imagined", "final", "nan", "fractional_error"])
def test_ineligible_outcomes_cannot_certify(kind):
    r, forbidden = rows(), ()
    if kind == "duplicate":
        r.append(r[0])
    elif kind == "forbidden":
        forbidden = (r[0].world_id,)
    else:
        changes = {"imagined": {"origin": "model_prediction"}, "final": {"role": "assessment"},
                   "nan": {"score": float("nan")}, "fractional_error": {"error": .5}}
        r[0] = replace(r[0], **changes[kind])
    with pytest.raises(ValueError):
        calibrate(r, contract(), forbidden_worlds=forbidden)


def test_certificate_owns_state_abstains_and_rejects_all_changed_dependencies():
    c = contract()
    artifact = calibrate(rows(), c)
    cert = WorldCertificate(artifact, c)
    assert cert.decide("all", .96, c)["status"] == "CONDITIONAL"
    for group in ("all", "empty", "new"):
        assert not cert.decide(group, .2, c)["accepted"]
    artifact["payload"]["selected"]["all"] = 0.
    assert not cert.decide("all", .2, c)["accepted"]
    for name, _ in c.identities:
        stale = replace(c, identities=tuple((n, digest("new") if n == name else value) for n, value in c.identities))
        with pytest.raises(ValueError, match="dependency"):
            cert.decide("all", .99, stale)


def test_rehashing_a_forged_selection_does_not_make_it_valid():
    c = contract()
    artifact = calibrate(rows(), c)
    artifact["payload"]["selected"]["empty"] = .9
    artifact["sha256"] = digest(artifact["payload"])
    with pytest.raises(ValueError, match="witnesses"):
        WorldCertificate(artifact, c)


def test_independent_certificate_auditor_binds_outcomes():
    c = contract()
    r = rows()
    artifact = calibrate(r, c)
    evidence = [{"world_id": x.world_id, "group": x.group, "score": x.score,
                 "error": x.error, "witness_sha256": x.witness_sha256} for x in r]
    assert check_certificate(artifact, evidence, True) < 1e-11
    changed = copy.deepcopy(evidence)
    changed[0]["error"] = 1
    with pytest.raises(ValueError, match="witnesses"):
        check_certificate(artifact, changed, True)


def test_actual_guard_timing_world_is_label_independent_and_has_one_query():
    guard = LearnedGuard(json.loads(GUARD.read_text()))
    row = episode("timing", 0, guard)
    assert row["panel"] == "timing" and len(row["posterior"]["payload"]["observed_events"]) == 10
    assert row["group"] == feature_group(np.array(row["features"]))
    old = feature_group(np.array(row["features"]))
    row.update(error=1-row["error"], private_family="unknown")
    assert feature_group(np.array(row["features"])) == old
    assert guard.predict(np.array([row["features"]]))[0] == row["score"]
