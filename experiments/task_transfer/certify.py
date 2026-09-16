"""Supplementary primal/dual, estimator and selection checks on frozen results."""

import argparse
import gzip
import json
import math

import numpy as np

from experiments.sparse_mechanisms.arrays import ArrayReader

from .audit import design
from .study import RELEASE, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    folder = RELEASE/args.name
    sources = {row["seed"]: row for row in json.loads((folder/"sources-learned.json").read_text())}
    arrays = ArrayReader(folder/"arrays.zip")
    certificates = selections = candidates = 0
    maximum_primal = maximum_dual = maximum_gap = 0.
    seen = {}
    with gzip.open(folder/"records.jsonl.gz", "rt", encoding="utf-8") as f:
        for row in map(json.loads, f):
            if row["status"] != "FIT":
                continue
            digest = row["evidence"]["array_sha256"]
            if digest not in seen:
                seen[digest] = json.loads(arrays.get(row["evidence"]).tobytes())
            data = seen[digest]
            n = row["count"]+(32 if row["policy"] == "matched_label_scratch" else 0)
            a = design(data["fit"]["x"][:n])
            y = np.asarray(data["fit"]["y"][:n])
            basis = np.array(sources[row["seed"]]["wrong_basis" if row["policy"] == "wrong_transfer" else "basis"])
            allowed = ({"ridge", "omp", "sparse"} if row["policy"] in ("scratch", "matched_label_scratch") else
                       {"ridge", "omp", "sparse", "subspace", "innovation"} if row["policy"] in ("transfer", "wrong_transfer") else
                       {row["policy"]})
            viable = [c for c in row["candidate_trace"] if c["status"] == "OK" and c["method"] in allowed]
            selected = min(viable, key=lambda c: c["selection_mse"])
            assert selected == row["selected"]
            selections += 1
            for candidate in row["candidate_trace"]:
                if candidate["status"] != "OK":
                    continue
                candidates += 1
                w = np.array(candidate["coefficients"])
                method = candidate["method"]
                if method == "ridge":
                    alpha = candidate["setting"]
                    assert np.linalg.norm(a.T@(a@w-y)+alpha*w) < 1e-7*max(1., np.linalg.norm(y))
                if method == "subspace":
                    assert np.linalg.norm(w-basis@(basis.T@w)) < 1e-8
                    assert np.linalg.norm((a@basis).T@(a@w-y)) < 1e-7*max(1., np.linalg.norm(y))
                if method == "omp":
                    support = np.flatnonzero(abs(w) > 1e-12)
                    assert len(support) <= candidate["setting"]
                    assert np.linalg.norm(a[:, support].T@(a@w-y)) < 1e-7*max(1., np.linalg.norm(y))
                for certificate in candidate["certificate"]:
                    augmented = np.c_[a@basis, a] if method == "innovation" else a
                    penalty = np.r_[np.zeros(basis.shape[1]), np.ones(20)] if method == "innovation" else np.ones(20)
                    primal, dual = np.array(certificate["primal"]), np.array(certificate["dual"])
                    epsilon = certificate["epsilon"]
                    residual = float(np.maximum(abs(augmented@primal-y)-epsilon, 0).max())
                    if epsilon == 0:
                        violation = float(np.maximum(abs(augmented.T@dual)-penalty, 0).max())
                        objective = y@dual
                    else:
                        violation = max(float(np.maximum(abs(augmented.T@(dual[:n]-dual[n:]))-penalty, 0).max()), float(dual.max(initial=0)))
                        objective = np.r_[y+epsilon, -y+epsilon]@dual
                    gap = float(abs(penalty@abs(primal)-objective))
                    reconstructed = basis@primal[:basis.shape[1]]+primal[basis.shape[1]:] if method == "innovation" else primal
                    assert np.max(abs(reconstructed-w)) < 1e-9
                    assert residual < 1e-7 and violation < 1e-7 and gap < 1e-6
                    maximum_primal, maximum_dual, maximum_gap = max(maximum_primal, residual), max(maximum_dual, violation), max(maximum_gap, gap)
                    certificates += 1
    arrays.close()
    audit = json.loads((folder/"independent-audit.json").read_text())
    # Original handbook's bounded, independent-world rule; alpha=.025 is the first .05/2 allocation.
    conservative_lower = audit["paired_mean_improvement"]-math.sqrt(2*math.log(1/.025)/len(sources))
    result = {"status": "PASS", "selected_candidate_checks": selections, "candidate_checks_with_reuse": candidates,
              "lp_primal_dual_checks_with_reuse": certificates, "maximum_primal_violation": maximum_primal,
              "maximum_dual_violation": maximum_dual, "maximum_duality_gap": maximum_gap,
              "handbook_alpha": .025, "handbook_bounded_world_lower": conservative_lower,
              "handbook_lower_positive": conservative_lower > 0,
              "scope": "Additional conservative architecture audit and independent optimality/selection checks. No fitting, changed model or final-based tuning. Numerical outcome measurements remain per supplied task distribution, not general capability evidence."}
    write(folder/"mathematical-audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
