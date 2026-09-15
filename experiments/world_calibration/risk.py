"""Independent-world selective risk with frozen policy and finite multiplicity.

Adapted from the supplied v3 risk_gate.py contract. Confidence is conditional on
IID worlds and the declared query selector, never arbitrary mechanism shift.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass

from scipy.stats import beta

from experiments.generative_memory.core import canonical, digest


def upper_bound(errors, count, alpha):
    if type(errors) is not int or type(count) is not int or not 0 <= errors <= count:
        raise ValueError("Integer error/acceptance counts required")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Invalid confidence allocation")
    if count == 0 or count == errors:
        return 1.
    return float(beta.ppf(1-alpha, errors+1, count-errors))


@dataclass(frozen=True)
class PolicyContract:
    identities: tuple[tuple[str, str], ...]
    groups: tuple[str, ...]
    thresholds: tuple[float, ...]
    alpha: float = .0125
    target_risk: float = .1
    minimum_accepts: int = 40

    def __post_init__(self):
        required = {"predictor", "features", "groups", "query_selector", "acquisition", "loss", "interpreter", "population"}
        if set(dict(self.identities)) != required or len(self.identities) != len(required):
            raise ValueError("Complete predictor, action, query and loss identities required")
        if any(type(v) is not str or re.fullmatch("[0-9a-f]{64}", v) is None for _, v in self.identities):
            raise ValueError("Identities must be SHA-256 fingerprints")
        if (type(self.groups) is not tuple or not self.groups or len(set(self.groups)) != len(self.groups)
                or any(type(g) is not str or not g for g in self.groups)):
            raise ValueError("Distinct fixed observable groups required")
        if (type(self.thresholds) is not tuple or not self.thresholds
                or self.thresholds != tuple(sorted(set(self.thresholds)))
                or any(not math.isfinite(t) or not 0 <= t <= 1 for t in self.thresholds)):
            raise ValueError("Fixed sorted probability threshold grid required")
        if (not math.isfinite(self.alpha) or not 0 < self.alpha < 1
                or not math.isfinite(self.target_risk) or not 0 < self.target_risk < 1
                or type(self.minimum_accepts) is not int or self.minimum_accepts < 1):
            raise ValueError("Invalid risk/confidence/count contract")

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class WorldOutcome:
    world_id: str
    group: str
    score: float
    error: int
    witness_sha256: str
    role: str = "calibration"
    origin: str = "simulator_observation"


def _validate_rows(rows, contract, forbidden):
    if not rows:
        raise ValueError("Empty independent-world calibration bank")
    seen = set()
    for row in rows:
        if (type(row) is not WorldOutcome or type(row.world_id) is not str or not row.world_id
                or row.world_id in seen or row.world_id in forbidden):
            raise ValueError("Repeated or forbidden calibration world")
        seen.add(row.world_id)
        if row.role != "calibration" or row.origin != "simulator_observation":
            raise ValueError("Only genuinely observed calibration outcomes qualify")
        if (row.group not in contract.groups or type(row.error) is not int or row.error not in (0, 1)
                or not math.isfinite(row.score) or not 0 <= row.score <= 1
                or type(row.witness_sha256) is not str or re.fullmatch("[0-9a-f]{64}", row.witness_sha256) is None):
            raise ValueError("Malformed world outcome")


def calibrate(rows, contract, *, forbidden_worlds=()):
    rows = tuple(rows)
    _validate_rows(rows, contract, set(forbidden_worlds))
    local_alpha = contract.alpha/(len(contract.groups)*len(contract.thresholds))
    table, selected = [], {}
    for group in contract.groups:
        group_rows = [row for row in rows if row.group == group]
        valid = []
        for threshold in contract.thresholds:
            accepted = [row for row in group_rows if row.score >= threshold]
            n, k = len(accepted), sum(row.error for row in accepted)
            bound = upper_bound(k, n, local_alpha)
            cell = {"group": group, "threshold": threshold, "accepted": n, "errors": k,
                    "upper_risk": bound, "alpha": local_alpha}
            table.append(cell)
            if n >= contract.minimum_accepts and bound <= contract.target_risk:
                valid.append(cell)
        selected[group] = (max(valid, key=lambda r: (r["accepted"], r["threshold"]))["threshold"]
                           if valid else None)
    payload = {"format": "sera.world-certificate.v1", "contract": asdict(contract),
               "contract_id": contract.identity, "selected": selected, "table": table,
               "rows": [asdict(row) for row in rows], "independent_worlds": len(rows),
               "scope": "Group selective risk under the frozen IID world/query distribution; "
                        "not every query, arbitrary shift, or factual truth of the supplied model class."}
    return {"payload": payload, "sha256": digest(payload)}


class WorldCertificate:
    """Restore by recomputing bounds/selection from retained outcome witnesses.

    This checks internal consistency, not cryptographic authenticity of a sensor.
    The independent experiment audit binds witnesses to actual recorded outcomes.
    """
    def __init__(self, artifact, contract):
        if (type(artifact) is not dict or set(artifact) != {"payload", "sha256"}
                or digest(artifact["payload"]) != artifact["sha256"]):
            raise ValueError("Certificate integrity failure")
        p = artifact["payload"]
        if p["contract_id"] != contract.identity or digest(p["contract"]) != digest(asdict(contract)):
            raise ValueError("Stale predictor, group, query, action, loss or population contract")
        rebuilt = calibrate([WorldOutcome(**row) for row in p["rows"]], contract)
        if digest(rebuilt) != digest(artifact):
            raise ValueError("Certificate table or selection does not match its witnesses")
        self._bytes = canonical(artifact)
        self._selected = tuple(p["selected"].items())
        self._scope, self._digest = p["scope"], artifact["sha256"]
        self.contract_id = contract.identity

    def record(self):
        return json.loads(self._bytes)

    def decide(self, group, score, current_contract):
        if current_contract.identity != self.contract_id:
            raise ValueError("Certificate invalidated by a changed policy dependency")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid score")
        threshold = dict(self._selected).get(group)
        accepted = threshold is not None and score >= threshold
        return {"accepted": accepted, "status": "CONDITIONAL" if accepted else "UNSUPPORTED",
                "group": group, "threshold": threshold, "score": score,
                "certificate": self._digest, "scope": self._scope}
