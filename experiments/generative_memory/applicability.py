"""GG-GUARD-001 learner: observed evidence in, validity probability out.

The assessor lives in applicability_study.py. Neither this feature interface nor
the fitting interface accepts a mechanism name, seed, coefficient or clean target.
This is an experimental guard, not yet part of the shared SERA parameter owner.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .acquisition import Posterior, _prediction_record
from .core import canonical, digest

FEATURES = (
    "query_t", "abs_query_t", "nearest_support_distance", "outside_support_distance",
    "log_predictive_trace", "log_within_class_trace", "log_between_class_trace",
    "class_entropy", "largest_class_mass", "log_late_mean_min_component_q",
    "log_late_max_min_component_q", "log_late_mean_mixture_q",
    "log_local_prequential_q", "local_prequential_distance",
    "support_span", "log_observation_count",
)
FEATURE_SCHEMA = "sera.gg-guard.observed-features.v1"
METHODS = ("always_predict", "residual_uncertainty", "support_distance", "learned",
           "always_abstain")


def interpreter_contract():
    root = Path(__file__).resolve().parents[2]
    names = ["experiments/generative_memory/" + x for x in
             ("applicability.py", "acquisition.py", "core.py")]
    names += ["src/sera/" + x for x in ("contracts.py", "event_ir.py")]
    return {"feature_schema": FEATURE_SCHEMA, "features": list(FEATURES),
            "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in names},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__},
            "generator": "GG-ACT-001 four-class conjugate posterior; supplied phase"}


def extract_features(posterior, query_t, noise_variance):
    """Use only public coordinates and actually admitted observations.

    Rebuild prequential innovations from the evidence prefix, so no separately
    supplied diagnostic can forge a feature. Outcomes at these queries are absent.
    Early innovations (before four observations) are excluded from residual scores.
    """
    if type(posterior) is not Posterior or posterior.observations_used < 5:
        raise ValueError("Guard requires at least five admitted observations")
    query_t = np.asarray(query_t, dtype=float)
    if (query_t.ndim != 1 or not 1 <= len(query_t) <= 256
            or not np.isfinite(query_t).all() or (np.abs(query_t) > 4).any()
            or not np.isfinite(noise_variance) or noise_variance <= 0):
        raise ValueError("Invalid public query contract")
    record = posterior.snapshot()
    verified = Posterior.restore(record)
    contract = verified.contract
    prefix = Posterior(verified.actions, specs=contract["specs"],
                       prior_variance=contract["prior_variance"],
                       class_priors=contract["class_prior_masses"],
                       max_observations=contract["max_observations"],
                       alarm_alpha=contract["alarm_alpha"])
    from sera.event_ir import Event

    t_observed, q_component, q_mixture = [], [], []
    for row in record["payload"]["observed_events"]:
        event = Event.from_record(row)
        action = prefix.action(int(event.values[0]))
        if action.channel != "target":
            raise ValueError("GG-GUARD-001 feature contract uses target observations only")
        prediction = prefix.predict(action)
        delta = np.asarray(event.values[3:]) - prediction["mean"]
        q = float(delta @ np.linalg.solve(prediction["covariance"], delta))
        diagnostic = prefix.observe(event)
        if event.sequence >= 4:
            t_observed.append(action.t)
            tails = diagnostic["pre_observation_component_tail_probabilities_df2"]
            q_component.append(-2 * np.log(max(max(tails), 1e-300)))
            q_mixture.append(q)
    support = np.array([row["values"][1] for row in record["payload"]["observed_events"]])
    predicted = _prediction_record(verified, query_t, noise_variance)
    rows = []
    for t, p in zip(query_t, predicted):
        w = np.array(p["class_weights"])
        within = float(np.einsum("k,kii->", w, np.array(p["component_covariances"])))
        total = float(np.trace(p["covariance"]))
        nearest = int(np.argmin(np.abs(np.asarray(t_observed) - t)))
        rows.append([
            t, abs(t), np.min(np.abs(support-t)), max(support.min()-t, t-support.max(), 0),
            np.log1p(total), np.log1p(within), np.log1p(max(total-within, 0)),
            float(-np.sum(w * np.log(np.maximum(w, 1e-300)))), float(w.max()),
            np.log1p(np.mean(q_component)), np.log1p(np.max(q_component)),
            np.log1p(np.mean(q_mixture)), np.log1p(q_component[nearest]),
            abs(t-t_observed[nearest]), support.max()-support.min(), np.log1p(len(support)),
        ])
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (len(query_t), len(FEATURES)) or not np.isfinite(values).all():
        raise ValueError("Feature extraction produced invalid numeric state")
    return values, predicted


def validate_teaching(rows, expected_split):
    """The fitting boundary admits only distinct, observed episode witnesses."""
    if not rows:
        raise ValueError("Empty teaching cohort")
    seen = set()
    for row in rows:
        if set(row) != {"episode_id", "split", "features", "observed_valid", "witness_sha256"}:
            raise ValueError("Teaching interface contains hidden or undeclared fields")
        if row["split"] != expected_split or row["episode_id"] in seen:
            raise ValueError("Teaching split contamination or duplicate episode")
        seen.add(row["episode_id"])
        x, y = np.asarray(row["features"]), np.asarray(row["observed_valid"])
        if (x.ndim != 2 or x.shape != (len(y), len(FEATURES)) or len(y) == 0
                or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()
                or type(row["witness_sha256"]) is not str or len(row["witness_sha256"]) != 64):
            raise ValueError("Invalid teaching witnesses")
    if len({len(row["observed_valid"]) for row in rows}) != 1:
        raise ValueError("This protocol requires equal query count per episode")
    return seen


class GuardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURES), 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, value):
        return self.layers(value).squeeze(-1)


def fit_guard(training, probability_calibration, *, optimizer_seed, steps=500):
    train_ids = validate_teaching(training, "teach")
    cal_ids = validate_teaching(probability_calibration, "probability_calibration")
    if train_ids & cal_ids:
        raise ValueError("Calibration reuses teaching mechanisms")
    torch.set_num_threads(1)
    torch.manual_seed(optimizer_seed)
    x = np.concatenate([r["features"] for r in training])
    y = torch.tensor(np.concatenate([r["observed_valid"] for r in training]), dtype=torch.float64)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    x = torch.tensor((x-mean)/scale, dtype=torch.float64)
    network = GuardNetwork().double()
    optimizer = torch.optim.Adam(network.parameters(), lr=.01, weight_decay=.001)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(network(x), y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite guard training loss")
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps-1:
            losses.append({"step": step+1, "bce_before_update": float(loss.detach())})
    xc = np.concatenate([r["features"] for r in probability_calibration])
    yc = torch.tensor(np.concatenate([r["observed_valid"] for r in probability_calibration]),
                      dtype=torch.float64)
    with torch.no_grad():
        zc = network(torch.tensor((xc-mean)/scale, dtype=torch.float64)).detach()
    # Positive-slope Platt scaling preserves the ranking; fitted on separate worlds.
    log_slope = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    offset = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    calibration_optimizer = torch.optim.Adam([log_slope, offset], lr=.02)
    for _ in range(250):
        calibration_optimizer.zero_grad()
        calibration_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.exp(log_slope)*zc+offset, yc) + .001*(log_slope**2+offset**2)
        calibration_loss.backward()
        calibration_optimizer.step()
    payload = {"format": "sera.gg-guard.model.v1", "contract": interpreter_contract(),
               "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
               "parameters": {k: v.detach().tolist() for k, v in network.state_dict().items()},
               "calibration_slope": float(torch.exp(log_slope).detach()),
               "calibration_offset": float(offset.detach()),
               "training": {"optimizer_seed": optimizer_seed, "steps": steps,
                            "probability_calibration_steps": 250, "loss_trace": losses,
                            "train_episode_ids": sorted(train_ids),
                            "calibration_episode_ids": sorted(cal_ids),
                            "witnesses_sha256": digest(training+probability_calibration),
                            "training_valid_fraction": float(y.mean()),
                            "learned_parameters": sum(p.numel() for p in network.parameters())+2}}
    return {"payload": payload, "sha256": digest(payload)}


class LearnedGuard:
    """Immutable serialized weights, with complete feature/generator identity pinning."""
    def __init__(self, artifact):
        if (set(artifact) != {"payload", "sha256"}
                or digest(artifact["payload"]) != artifact["sha256"]):
            raise ValueError("Guard artifact integrity failure")
        payload = artifact["payload"]
        if payload["format"] != "sera.gg-guard.model.v1" or payload["contract"] != interpreter_contract():
            raise ValueError("Stale guard: feature, generator, dependency or runtime changed")
        # Own canonical bytes; caller mutation cannot silently change an accepted guard.
        self._serialized = canonical(artifact)
        self._mean = np.array(payload["feature_mean"])
        self._scale = np.array(payload["feature_scale"])
        self._network = GuardNetwork().double()
        self._network.load_state_dict({k: torch.tensor(v, dtype=torch.float64)
                                       for k, v in payload["parameters"].items()}, strict=True)
        self._network.eval()
        self._slope, self._offset = payload["calibration_slope"], payload["calibration_offset"]
        if (self._mean.shape != (len(FEATURES),) or self._scale.shape != self._mean.shape
                or not np.isfinite(self._mean).all() or not np.isfinite(self._scale).all()
                or (self._scale <= 0).any() or not np.isfinite([self._slope, self._offset]).all()
                or self._slope <= 0
                or any(not torch.isfinite(p).all() for p in self._network.parameters())):
            raise ValueError("Invalid learned guard numeric state")

    def artifact(self):
        return json.loads(self._serialized)

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
            raise ValueError("Invalid guard features")
        if self.artifact()["payload"]["contract"] != interpreter_contract():
            raise ValueError("Stale guard interpreter at execution")
        with torch.no_grad():
            z = self._network(torch.tensor((x-self._mean)/self._scale, dtype=torch.float64))
            return torch.sigmoid(self._slope*z+self._offset).numpy().copy()


def control_scores(features, guard):
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("Invalid control feature matrix")
    # A strong all-component residual/uncertainty control with the same calibration
    # exposure as the learned guard. Larger values consistently mean more supported.
    badness = np.maximum.reduce([np.expm1(x[:, 9])/2, np.expm1(x[:, 10])/9.210340372,
                                 np.expm1(x[:, 4])/.04])
    return {"always_predict": np.ones(len(x)), "always_abstain": np.zeros(len(x)),
            "support_distance": -x[:, 2], "residual_uncertainty": -np.log1p(badness),
            "learned": guard.predict(x)}


def select_threshold(scores, valid, *, target_risk=.10):
    """Empirical selection, NOT a conformal guarantee on conditional selective risk.

    The frozen candidate rule uses 101 calibration-score quantiles. All queries
    per world have equal weight. A completely rejected set has undefined risk.
    """
    scores, valid = np.asarray(scores), np.asarray(valid)
    if (scores.ndim != 2 or scores.shape != valid.shape or not scores.size
            or not np.isfinite(scores).all() or not np.isin(valid, [0, 1]).all()
            or not 0 < target_risk < 1):
        raise ValueError("Invalid episode-level threshold data")
    candidates = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 101)),
                                 np.nextafter(scores.max(), np.inf)])
    records = []
    for threshold in candidates:
        accepted = scores >= threshold
        count = int(accepted.sum())
        wrong = int((accepted & ~valid.astype(bool)).sum())
        risk = wrong/count if count else None
        records.append({"threshold": float(threshold), "accepted": count,
                        "coverage": count/scores.size, "selective_risk": risk})
    feasible = [r for r in records if r["accepted"] and r["selective_risk"] <= target_risk]
    chosen = (max(feasible, key=lambda r: (r["accepted"], r["threshold"]))
              if feasible else records[-1])
    return {"selected": chosen, "candidates": records, "target_empirical_risk": target_risk,
            "guarantee": "None for selective risk or unseen families; empirical calibration only."}
