"""Independent world accounting, batch algebra and binomial-tail certificate audit.

Imports the previously verified independent batch reference, not the new study or
calibration implementation. Numeric tolerances are frozen before assessment.
"""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

from experiments.generative_memory.applicability_audit import (
    digest,
    features_from_evidence,
    probability,
    validate_observed_event,
)

ROOT = Path(__file__).resolve().parents[2]


def require(value, message):
    if not value:
        raise ValueError(message)


def independent_upper(k, n, alpha):
    return 1. if n == 0 or k == n else brentq(lambda p: binom.cdf(k, n, p)-alpha, 0., 1., xtol=1e-14)


def check_certificate(record, calibration, grouped):
    p = record["payload"]
    require(digest(p) == record["sha256"], "Certificate integrity")
    c = p["contract"]
    require(digest(c) == p["contract_id"], "Contract integrity")
    expected_rows = [{"world_id": r["world_id"], "group": r["group"] if grouped else "all",
                      "score": r["score"], "error": r["error"], "witness_sha256": r["witness_sha256"],
                      "role": "calibration", "origin": "simulator_observation"} for r in calibration]
    require(expected_rows == p["rows"], "Certificate witnesses do not bind calibration outcomes")
    expected_alpha = c["alpha"]/(len(c["groups"])*len(c["thresholds"]))
    selected, max_difference, cells = {}, 0., {}
    for cell in p["table"]:
        key = cell["group"], cell["threshold"]
        require(key not in cells, "Duplicate risk cell")
        cells[key] = cell
    require(set(cells) == {(g, t) for g in c["groups"] for t in c["thresholds"]}, "Incomplete risk family")
    for group in c["groups"]:
        valid = []
        for threshold in c["thresholds"]:
            rows = [r for r in expected_rows if r["group"] == group and r["score"] >= threshold]
            n, k = len(rows), sum(r["error"] for r in rows)
            u = independent_upper(k, n, expected_alpha)
            cell = cells[group, threshold]
            require(cell["accepted"] == n and cell["errors"] == k and cell["alpha"] == expected_alpha, "Risk counts/allocation")
            delta = abs(u-cell["upper_risk"])
            max_difference = max(max_difference, delta)
            require(delta < 1e-11, "Binomial-tail confidence inversion mismatch")
            if n >= c["minimum_accepts"] and u <= c["target_risk"]:
                valid.append((n, threshold))
        selected[group] = max(valid)[1] if valid else None
    require(selected == p["selected"], "Selected policy differs from independent risk calculation")
    return max_difference


def audit(run):
    start = time.perf_counter()
    protocol = json.loads((run/"protocol.json").read_text())
    cfg = protocol["payload"]
    require(digest(cfg) == protocol["sha256"], "Protocol integrity")
    require(all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha for name, sha in cfg["sources"].items()), "Frozen source changed")
    summary = json.loads((run/"summary.json").read_text())
    raw = (run/"worlds.json.gz").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == summary["worlds_sha256"], "World archive integrity")
    data = json.loads(gzip.decompress(raw))
    rows = data["calibration"]+data["assessment"]
    expected = {(panel, cfg["seed_bases"][panel]+i) for panel, n in cfg["counts"].items() for i in range(n)}
    require(len(rows) == len(expected) and {(r["panel"], r["seed"]) for r in rows} == expected, "Missing/duplicate world or split")
    require(len({r["world_id"] for r in rows}) == len(rows), "Repeated world")
    require(all(r["panel"] == "calibration" for r in data["calibration"]), "Final evidence in calibration")
    model = json.loads((ROOT/cfg["guard_path"]).read_text())
    require(hashlib.sha256((ROOT/cfg["guard_path"]).read_bytes()).hexdigest() == cfg["guard_file_sha256"], "Changed guard")
    maxima = {"features": 0., "means": 0., "probability": 0.}
    for row in rows:
        require(row["world_id"] == digest({"experiment": "GG-GUARD-002", "seed": row["seed"]}), "World identity")
        seed_bytes = hashlib.sha256(f"GG-GUARD-001/{row['seed']}/query-selector-v3".encode()).digest()
        # This selector is reconstructed independently from the public RNG contract.
        query_index = int(np.random.default_rng(np.frombuffer(seed_bytes[:16], dtype=np.uint32)).integers(33))
        require(row["query_index"] == query_index, "Query chosen after observing its answer")
        require(row["query_t"] == cfg["query_grid"][query_index], "Query coordinate mismatch")
        post = row["posterior"]
        require(digest(post["payload"]) == post["sha256"], "Posterior snapshot integrity")
        events = post["payload"]["observed_events"]
        require(len(events) == 10 and [e["values"][0] for e in events] == cfg["support_indices"], "Observation budget/order")
        for i, event in enumerate(events):
            validate_observed_event(event, i, event["values"][3:])
        x, mean, _, _ = features_from_evidence(events, [row["query_t"]], .05**2)
        maxima["features"] = max(maxima["features"], float(np.abs(x[0]-row["features"]).max()))
        maxima["means"] = max(maxima["means"], float(np.abs(mean[0]-row["witness"]["prediction_before_outcome"]).max()))
        # Evaluate network on the saved features so this separately isolates its algebra.
        maxima["probability"] = max(maxima["probability"], abs(float(probability(np.array([row["features"]]), model["payload"])[0])-row["score"]))
        exact = np.array(row["features"])
        group = ("consistent" if exact[9] <= np.log1p(4.) else "contradicted")+("_near" if exact[2] <= .18 and exact[3] == 0 else "_edge")
        require(group == row["group"], "Group used hidden mechanism/outcome")
        witness = row["witness"]
        require(digest(witness) == row["witness_sha256"] and witness["world_id"] == row["world_id"]
                and witness["prefix_sha256"] == post["sha256"] and witness["query_index"] == query_index
                and witness["origin"] == "simulator_observation", "Query witness linkage")
        prediction = np.array(witness["prediction_before_outcome"])
        require(int(np.linalg.norm(prediction-witness["observed_outcome"]) > .2) == row["error"], "Outcome label")
        require(float(np.mean((prediction-row["clean_truth"])**2)) == row["clean_mse"], "Clean assessor diagnostic")
    require(maxima["features"] < 2e-8 and maxima["means"] < 2e-8 and maxima["probability"] < 1e-12, "Independent algebra tolerance exceeded")
    certs = json.loads((run/"certificates.json").read_text())
    bound_delta = max(check_certificate(certs[key], data["calibration"], key == "group") for key in ("global", "group"))
    before = json.loads((run/"before-final.json").read_text())
    require(before == {"guard": model["sha256"], "certificates": {k: v["sha256"] for k, v in certs.items()}}, "Pre-final identity changed")
    old = json.loads((ROOT/"research-continuation/15_applicability/thresholds.json").read_text())["selections"]["learned"]["selected"]["threshold"]
    for row in data["assessment"]:
        x = row["features"]
        expected_decisions = {"always": True, "abstain": False, "old_empirical": row["score"] >= old,
                              "minimum_90": row["score"] >= .9,
                              "fixed_residual": bool(np.expm1(x[10]) <= 9.210340372 and np.expm1(x[4]) <= .04)}
        for name in ("global", "group"):
            threshold = certs[name]["payload"]["selected"].get(row["group"] if name == "group" else "all")
            expected_decisions[name+"_certificate"] = threshold is not None and row["score"] >= threshold
        require(expected_decisions == row["decisions"], "Frozen policy decisions")
    for panel, methods in summary["panels"].items():
        cohort = [r for r in data["assessment"] if r["panel"] == panel]
        for name, metrics in methods.items():
            accepted = [r for r in cohort if r["decisions"][name]]
            n, k = len(accepted), sum(r["error"] for r in accepted)
            require(metrics["worlds"] == len(cohort) and metrics["accepted"] == n and metrics["errors"] == k, "Reported risk denominator")
            require(metrics["coverage"] == n/len(cohort) and metrics["selective_risk"] == (k/n if n else None), "Reported risk/coverage")
    return {"status": "PASS", "independent_worlds": len(rows), "outcomes_per_world": 1,
            "independent_feature_and_prediction_max_differences": maxima,
            "independent_binomial_bound_max_difference": bound_delta, "wall_seconds": time.perf_counter()-start,
            "scope": "All saved observations, independent batch algebra, network, query selector, world/split coverage, witnesses, finite-grid binomial bounds, decisions and risk/coverage counts. Does not independently authenticate a simulator or prove arbitrary-shift validity."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
