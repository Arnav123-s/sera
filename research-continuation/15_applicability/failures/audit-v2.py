"""Independent GG-GUARD-001 audit. No learner or study imports.

Reconstruct batch Gaussian posteriors, prequential residuals, all features,
hand-evaluate learned weights, and rederive labels/threshold metrics from raw
observations. Absolute 2e-8 tolerance covers independent float64 algebra only;
serialized same-interpreter replay remains exact in the separate replay job.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

FAMILIES = ("circle", "ellipse", "line", "radial_orbit", "local_exception", "random_values",
            "piecewise_drift", "chirp")
BASES = {"teach": 410000, "probability_calibration": 510000,
         "threshold_selection": 610000, "final": 710000}


def validate_observed_event(event, sequence, expected_pair):
    """Match the actual serialized SERA evidence kind, without importing its enum."""
    if (event["sequence"] != sequence
            or event["provenance"]["kind"] != "simulator_ground_truth"
            or event["role"] != "observation" or event["available"] != [True]*5
            or event["values"][3:] != expected_pair):
        raise ValueError("Features include nonfactual or inconsistent observations")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def matrix(kind, t):
    t = np.array(t)
    sine, cosine = np.sin(np.pi*t), np.cos(np.pi*t)
    n = len(t)
    if kind == 0:
        a = np.zeros((n, 2, 4))
        a[:, 0, 0], a[:, 1, 1] = 1, 1
        a[:, 0, 2], a[:, 0, 3] = cosine, -sine
        a[:, 1, 2], a[:, 1, 3] = sine, cosine
    elif kind == 1:
        a = np.zeros((n, 2, 6))
        # core.expression orders x coefficients, then y coefficients.
        a[:, 0, :3] = np.stack([np.ones(n), sine, cosine], axis=1)
        a[:, 1, 3:] = np.stack([np.ones(n), sine, cosine], axis=1)
    elif kind == 2:
        a = np.zeros((n, 2, 4))
        a[:, 0, :2] = np.stack([np.ones(n), t], axis=1)
        a[:, 1, 2:] = np.stack([np.ones(n), t], axis=1)
    else:
        a = np.zeros((n, 2, 6))
        a[:, 0, 0], a[:, 1, 1] = 1, 1
        a[:, 0, 2], a[:, 0, 3] = cosine, -sine
        a[:, 1, 2], a[:, 1, 3] = sine, cosine
        a[:, 0, 4], a[:, 0, 5] = t*cosine, -t*sine
        a[:, 1, 4], a[:, 1, 5] = t*sine, t*cosine
    return a.reshape(2*n, -1)


def batch(t, y, query, variance):
    means, covs, evidences = [], [], []
    models = []
    for kind in range(4):
        a, b = matrix(kind, t), matrix(kind, query).reshape(len(query), 2, -1)
        # Observation-space Cholesky instead of sequential Joseph updates.
        observation_covariance = 4*a@a.T + variance*np.eye(len(y))
        chol = np.linalg.cholesky(observation_covariance)
        whitened_y = np.linalg.solve(chol, y)
        whitened_a = np.linalg.solve(chol, a)
        mean = 4*whitened_a.T@whitened_y
        cov = 4*np.eye(a.shape[1])-16*whitened_a.T@whitened_a
        cov = (cov+cov.T)/2
        models.append((mean, cov))
        means.append(b@mean)
        covs.append(b@cov@b.transpose(0, 2, 1)+variance*np.eye(2))
        evidences.append(-.5*(len(y)*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+whitened_y@whitened_y))
    weights = np.exp(evidences-logsumexp(evidences))
    means, covs = np.asarray(means), np.asarray(covs)
    mean = np.einsum("k,kqi->qi", weights, means)
    centered = means-mean
    covariance = (np.einsum("k,kqij->qij", weights, covs)
                  +np.einsum("k,kqi,kqj->qij", weights, centered, centered))
    return mean, covariance, means, covs, weights, models


def features_from_evidence(events, query, variance):
    t = np.array([r["values"][1] for r in events])
    y = np.array([r["values"][3:] for r in events])
    mean, covariance, _, covs, weights, models = batch(t, y.ravel(), query, variance)
    min_q, mix_q, late_t = [], [], []
    for index in range(4, len(t)):
        pm, pc, cm, cc, _, _ = batch(t[:index], y[:index].ravel(), [t[index]], variance)
        delta = y[index]-pm[0]
        mix_q.append(float(delta@np.linalg.solve(pc[0], delta)))
        qs = []
        for mu, cov in zip(cm[:, 0], cc[:, 0]):
            residual = y[index]-mu
            qs.append(float(residual@np.linalg.solve(cov, residual)))
        min_q.append(min(min(qs), -2*np.log(1e-300)))
        late_t.append(t[index])
    result = []
    for i, q in enumerate(query):
        total = np.trace(covariance[i])
        within = np.sum(weights*np.trace(covs[:, i], axis1=1, axis2=2))
        nearest = int(np.argmin(np.abs(np.array(late_t)-q)))
        result.append([q, abs(q), np.min(np.abs(t-q)), max(t.min()-q, q-t.max(), 0),
                       np.log1p(total), np.log1p(within), np.log1p(max(total-within, 0)),
                       -np.sum(weights*np.log(np.maximum(weights, 1e-300))), max(weights),
                       np.log1p(np.mean(min_q)), np.log1p(max(min_q)), np.log1p(np.mean(mix_q)),
                       np.log1p(min_q[nearest]), abs(q-late_t[nearest]), t.max()-t.min(), np.log1p(len(t))])
    return np.array(result), mean, models, weights


def probability(features, model):
    x = (features-np.array(model["feature_mean"]))/np.array(model["feature_scale"])
    params = model["parameters"]
    hidden = np.tanh(x@np.array(params["layers.0.weight"]).T+params["layers.0.bias"])
    logits = (hidden@np.array(params["layers.2.weight"]).T+params["layers.2.bias"]).ravel()
    return expit(model["calibration_slope"]*logits+model["calibration_offset"])


def audit(run, output):
    if output.exists():
        raise ValueError("Audit output must be fresh")
    started = time.perf_counter()
    protocol = json.loads((run/"protocol.json").read_text())
    config = protocol["payload"]
    if digest(config) != protocol["sha256"]:
        raise ValueError("Protocol integrity failure")
    model_record = json.loads((run/"guard-model.json").read_text())
    if digest(model_record["payload"]) != model_record["sha256"]:
        raise ValueError("Model integrity failure")
    model = model_record["payload"]
    summary = json.loads((run/"summary.json").read_text())
    raw = (run/"records.json.gz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != summary["records_sha256"]:
        raise ValueError("Evidence archive changed")
    data = json.loads(gzip.decompress(raw))
    expected = {(split, family, BASES[split]+1000*FAMILIES.index(family)+i)
                for split, n in config["worlds_per_family"].items()
                for family in (FAMILIES if split == "final" else FAMILIES[:6]) for i in range(n)}
    actual = {(r["split"], r["assessor"]["family"], r["assessor"]["environment_seed"]) for r in data["records"]}
    if actual != expected or len(actual) != len(data["records"]):
        raise ValueError("Episode count, partition or future-family coverage mismatch")
    if len({r["episode_id"] for r in data["records"]}) != len(actual):
        raise ValueError("Episode identities repeat")
    threshold_data = json.loads((run/"thresholds.json").read_text())
    before = json.loads((run/"before-final.json").read_text())
    if before != {"model_sha256": model_record["sha256"], "selections_sha256": digest(threshold_data["selections"])}:
        raise ValueError("Pre-final model or thresholds differ")
    max_error = {"features": 0., "prediction_means": 0., "guard_probability": 0., "coefficients": 0., "covariance": 0.}
    teaching_by_id = {r["episode_id"]: r for rows in data["teaching"].values() for r in rows}
    final_counts = {}
    for row in data["records"]:
        events = row["posterior"]["payload"]["observed_events"]
        if len(events) != 10 or [r["values"][0] for r in events] != config["support_indices"]:
            raise ValueError("Unequal or mismatched observation access")
        for index, event in enumerate(events):
            validate_observed_event(event, index,
                                    row["assessor"]["support_observed"][int(event["values"][0])])
        query = row["witness"]["query_t"]
        features, means, models, _ = features_from_evidence(events, query, .05**2)
        max_error["features"] = max(max_error["features"], float(np.max(np.abs(features-row["features"]))))
        max_error["prediction_means"] = max(max_error["prediction_means"], float(np.max(np.abs(means-row["predicted_means"]))))
        for (mu, cov), original in zip(models, row["posterior"]["payload"]["models"]):
            max_error["coefficients"] = max(max_error["coefficients"], float(np.max(np.abs(mu-original["mean"]))))
            max_error["covariance"] = max(max_error["covariance"], float(np.max(np.abs(cov-original["covariance"]))))
        valid = (np.sum((np.array(row["predicted_means"])-row["assessor"]["query_observed"])**2, axis=1) <= .2**2)
        clean_mse = np.mean((np.array(row["predicted_means"])-row["assessor"]["query_truth"])**2, axis=1)
        if valid.astype(int).tolist() != row["observed_valid"] or clean_mse.tolist() != row["clean_mse"]:
            raise ValueError("Outcome label or clean MSE mismatch")
        if (row["witness"]["query_observed"] != row["assessor"]["query_observed"]
                or row["witness"]["prediction_means_before_query_observation"] != row["predicted_means"]
                or row["witness"]["evidence_role"] != "synthetic_observed_outcome"):
            raise ValueError("Factual target witness differs")
        if row["split"] != "final":
            teaching = teaching_by_id[row["episode_id"]]
            if (teaching["features"] != row["features"] or teaching["observed_valid"] != row["observed_valid"]
                    or teaching["witness_sha256"] != digest(row["witness"]) or teaching["split"] != row["split"]):
                raise ValueError("Fitting data differ from witnessed evidence")
        else:
            predicted_probability = probability(np.array(row["features"]), model)
            max_error["guard_probability"] = max(max_error["guard_probability"],
                                                  float(np.max(np.abs(predicted_probability-row["scores"]["learned"]))))
            x = np.array(row["features"])
            expected_scores = {
                "always_predict": np.ones(len(valid)), "always_abstain": np.zeros(len(valid)),
                "support_distance": -x[:, 2],
                "residual_uncertainty": -np.log1p(np.maximum.reduce([
                    np.expm1(x[:, 9])/2, np.expm1(x[:, 10])/9.210340372, np.expm1(x[:, 4])/.04]))}
            if any(not np.array_equal(score, row["scores"][method])
                   for method, score in expected_scores.items()):
                raise ValueError("Independent fixed-control scores differ")
            for method, scores in row["scores"].items():
                threshold = .5 if method == "always_predict" else 1. if method == "always_abstain" else threshold_data["selections"][method]["selected"]["threshold"]
                accepted = np.array(scores) >= threshold
                m = row["methods"][method]
                if (m["accepted"] != int(accepted.sum()) or m["wrong_accepted"] != int((accepted & ~valid).sum())
                        or m["valid_rejected"] != int((~accepted & valid).sum())
                        or abs(m["utility"]-float(np.mean(accepted*np.where(valid, 1., -4.)))) > 1e-12):
                    raise ValueError("Guard decisions or risk metrics differ")
                key = (row["assessor"]["family"], method)
                counters = final_counts.setdefault(key, [0, 0, 0])
                counters[0] += len(valid)
                counters[1] += int(accepted.sum())
                counters[2] += int((accepted & ~valid).sum())
    for (family, method), (queries, accepted, wrong) in final_counts.items():
        report = summary["by_family"][family][method]
        if (queries, accepted, wrong) != (report["queries"], report["accepted"], report["wrong_accepted"]):
            raise ValueError("Published aggregate differs from per-world evidence")
    if max(max_error.values()) > 2e-8:
        raise ValueError(f"Independent algebra failed frozen tolerance: {max_error}")
    result = {"status": "PASS", "worlds_audited": len(actual), "query_feature_rows": len(actual)*33,
              "auditor_revision": "v2-provenance-enum-repair",
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "independent_batch_models_including_prequential": len(actual)*28,
              "maximum_absolute_errors": max_error, "absolute_tolerance": 2e-8,
              "exact_labels_and_decision_counts": True, "partition_and_future_family_checks": True,
              "independent_numpy_network": True, "wall_seconds": time.perf_counter()-started,
              "boundary": "Source-frozen independent algebra and data audit, not a proof of model adequacy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.run, arguments.output)
