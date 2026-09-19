"""Replay scientific qualification separately from optimizer bookkeeping.

Frozen IU-001 sources and receipts remain unchanged. This additive adapter permits
only independently bounded refit differences with identical scientific decisions.
"""

import copy
import math
from contextlib import contextmanager

import numpy as np

from experiments import intervention_assess as original
from experiments.gap_inquiry import ROOT, digest, read, sha

STRICT = original.same_qualification
CONTRACT = ROOT / "research-continuation/47_intervention_understanding/publication/portability-contract.json"


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [item for k in a.keys() | b.keys() if k != "receipt"
                for item in differences(a.get(k), b.get(k), path+"/"+k)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [{"path": path, "saved": len(a), "replayed": len(b)}]
        return [item for i, (x, y) in enumerate(zip(a, b, strict=True)) for item in differences(x, y, path+"/"+str(i))]
    if a != b:
        return [{"path": path, "saved": a, "replayed": b}]
    return []


def compare(saved, replayed, emit=True):
    for value in (saved, replayed):
        if value.get("receipt") != digest({k: v for k, v in value.items() if k != "receipt"}):
            return False
    if STRICT(saved, replayed):
        return True
    # Everything outside the separate inverse-fit report retains IU-001's
    # original comparison, including model adequacy, source and admission.
    left, right = copy.deepcopy(saved), copy.deepcopy(replayed)
    old_fit, new_fit = left.pop("independent_fit"), right.pop("independent_fit")
    for value in (left, right):
        value["receipt"] = digest({k: v for k, v in value.items() if k != "receipt"})
    same_science = STRICT(left, right)
    controls = [g["applied"] for r in saved["selection"]+saved["audit"] for g in r["groups"]]
    old, new = np.asarray(old_fit["weights"]), np.asarray(new_fit["weights"])
    finite = old.shape == new.shape and old.ndim == 1 and len(old) in (2, 3) and np.isfinite(old).all() and np.isfinite(new).all()
    probability_difference = float(np.max(np.abs(original.predict(controls, old)-original.predict(controls, new)))) if finite else float("inf")
    likelihood_equal = math.isclose(old_fit["nll"], new_fit["nll"], rel_tol=0., abs_tol=1e-10)
    guard_unchanged = (type(old_fit["qualified"]) is bool and old_fit["qualified"] == new_fit["qualified"]
                       and bool(old_fit["max_prediction_difference"] <= .002) == old_fit["qualified"]
                       and bool(new_fit["max_prediction_difference"] <= .002) == new_fit["qualified"])
    accepted = bool(same_science and finite and probability_difference <= 2e-6 and likelihood_equal and guard_unchanged)
    if emit:
        delta = differences(saved, replayed)
        print({"qualification_portability": accepted, "subject": saved.get("subject"),
               "same_scientific_decisions": same_science, "independent_refit_probability_difference": probability_difference,
               "differing_fields": [r for r in delta if not isinstance(r["saved"], float) or not math.isclose(r["saved"], r["replayed"], rel_tol=1e-8, abs_tol=1e-10)][:20]}, flush=True)
    return accepted


def verify_contract():
    contract = read(CONTRACT)
    if sha(ROOT / contract["frozen_release"]) != contract["frozen_release_sha256"]:
        raise ValueError("Changed frozen scientific release")
    for path, expected in contract["files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Changed qualified portability adapter: "+path)


@contextmanager
def installed():
    verify_contract()
    previous = original.same_qualification
    if previous is not STRICT:
        raise ValueError("Unexpected qualification adapter already installed")
    original.same_qualification = compare
    try:
        yield
    finally:
        original.same_qualification = previous
