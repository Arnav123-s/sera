"""Independent execution and scoring of saved results; never refit a final trial."""

import argparse
import gzip
import json
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

from experiments.sparse_mechanisms.arrays import ArrayReader

from .study import RELEASE, sha, write


def design(x):
    x = np.asarray(x)
    terms = sorted(((i, j, k) for i in range(4) for j in range(4) for k in range(4)
                    if i+j+k <= 3), key=lambda p: (sum(p), p))
    columns = []
    for term in terms:
        product = np.ones(len(x))
        for axis, degree in enumerate(term):
            c = np.zeros(degree+1)
            c[-1] = 1
            product *= np.polynomial.legendre.legval(x[:, axis], c)*math.sqrt(2*degree+1)
        columns.append(product)
    return np.column_stack(columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    folder = RELEASE/args.name
    arrays = ArrayReader(folder/"arrays.zip")
    rows = []
    maximum = 0.
    certificates = 0
    shared_cache = {}
    with gzip.open(folder/"records.jsonl.gz", "rt", encoding="utf-8") as f:
        for row in map(json.loads, f):
            if row["status"] != "FIT":
                rows.append(row)
                continue
            key = row["evidence"]["array_sha256"]
            if key not in shared_cache:
                shared_cache[key] = json.loads(arrays.get(row["evidence"]).tobytes())
            data = shared_cache[key]
            query, expected = arrays.get(data["query"]), arrays.get(data["expected"])
            w = np.array(row["selected"]["coefficients"])
            predicted = design(query)@w
            maximum = max(maximum, float(abs(predicted-arrays.get(row["prediction"])).max()))
            error = abs(predicted-expected)
            assert abs(float(np.mean(error > .05))-row["failure_rate"]) < 1e-12
            assert abs(float(error.mean())-row["mae"]) < 1e-10
            assert abs(float(error.max())-row["max_error"]) < 1e-9
            probe = abs(design(arrays.get(data["probes"]))@w-arrays.get(data["probe_truth"]))
            assert abs(float(probe.max())-row["probe_max_error"]) < 1e-9
            residual = abs(design(data["calibration"]["x"])@w-data["calibration"]["y"])
            k, n = int(np.sum(residual > .05)), len(residual)
            upper = 1. if k == n else brentq(lambda p: binom.cdf(k, n, p)-.05, 0., 1.)
            assert k == row["gate"]["failures"]
            assert abs(upper-row["gate"]["risk_upper_95"]) < 1e-9
            assert (upper <= .05) == row["gate"]["accepted"]
            groups = [set(map(tuple, data[r]["x"])) for r in ("fit", "selection", "calibration")]
            assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
            selection_a = design(data["selection"]["x"])
            for candidate in row["candidate_trace"]:
                if candidate["status"] != "OK":
                    continue
                score = float(np.mean((selection_a@candidate["coefficients"]-data["selection"]["y"])**2))
                assert abs(score-candidate["selection_mse"]) < 1e-8*max(1, score)
                for certificate in candidate["certificate"]:
                    assert certificate["objective_gap"] <= 1e-6
                    assert certificate["dual_violation"] <= 1e-6
                    certificates += 1
            rows.append({k: v for k, v in row.items() if k not in ("candidate_trace", "evidence", "prediction", "selected")})
    arrays.close()
    assert maximum < 1e-9
    source = json.loads((folder/"sources-learned.json").read_text())
    for life in source:
        for key, basis_key in (("source_tasks", "basis"), ("wrong_tasks", "wrong_basis")):
            vectors = []
            for task in life[key]:
                assert task["labels"] == 128
                w = np.array(task["selected"]["coefficients"])
                calibration = task["batches"]["calibration"]
                err = abs(design(calibration["x"])@w-calibration["y"])
                assert np.sum(err > .05) == task["gate"]["failures"]
                if task["gate"]["accepted"]:
                    vectors.append(w)
            u, singular, _ = np.linalg.svd(np.array(vectors).T, full_matrices=False)
            rank = min(3, int(sum(singular > max(1e-10, singular[0]*1e-7))))
            b = np.array(life[basis_key])
            assert np.linalg.norm(b@b.T-u[:, :rank]@u[:, :rank].T) < 1e-8
    groups = {}
    for row in rows:
        key = (row["family"], row["count"], row["policy"])
        groups.setdefault(key, []).append(row)
    table = []
    for (family, count, policy), values in sorted(groups.items()):
        valid = [v for v in values if v["status"] == "FIT"]
        accepted = [v for v in valid if v["gate"]["accepted"]]
        table.append({"family": family, "count": count, "policy": policy, "trials": len(values),
                      "fit": len(valid), "accepted": len(accepted),
                      "fully_accurate": sum(v["failure_rate"] == 0 for v in valid),
                      "mean_failure_rate": float(np.mean([v["failure_rate"] for v in valid])) if valid else None,
                      "mean_mae": float(np.mean([v["mae"] for v in valid])) if valid else None,
                      "accepted_over_5_percent_future_risk": sum(v["failure_rate"] > .05 for v in accepted),
                      "accepted_with_wrong_targeted_probe": sum(v["probe_max_error"] > .05 for v in accepted)})
    paired = []
    for seed in sorted({r["seed"] for r in rows}):
        improvements = []
        for family in ("related", "innovation"):
            subset = {r["policy"]: r for r in rows if r["seed"] == seed and r["count"] == 12 and r["family"] == family}
            improvements.append(subset["scratch"]["failure_rate"]-subset["transfer"]["failure_rate"])
        paired.append(float(np.mean(improvements)))
    lower = float(np.mean(paired)-1.96*np.std(paired, ddof=1)/np.sqrt(len(paired)))
    dense = []
    for seed in sorted({r["seed"] for r in rows}):
        subset = {r["policy"]: r for r in rows if r["seed"] == seed and r["count"] == 12 and r["family"] == "dense"}
        dense.append(subset["transfer"]["failure_rate"]*subset["transfer"]["gate"]["accepted"]-
                     subset["scratch"]["failure_rate"]*subset["scratch"]["gate"]["accepted"])
    output = {"status": "PASS", "records": len(rows), "independent_task_families": len(source),
              "maximum_prediction_discrepancy": maximum, "lp_certificate_checks_with_reuse": certificates,
              "table": table, "paired_mean_improvement": float(np.mean(paired)),
              "paired_normal_95_lower": lower, "paired_by_family_seed": paired,
              "dense_mean_accepted_risk_degradation": float(np.mean(dense)),
              "numerical_promotion_condition": lower > 0 and float(np.mean(dense)) <= .01,
              "source_manifest": {p.name: sha(p) for p in (folder/"protocol.json", folder/"sources-learned.json", folder/"records.jsonl.gz", folder/"arrays.zip")},
              "boundary": "Audits frozen coefficients, input-role separation, predictions, selection scores, source subspaces and per-model risk calculations. Does not retrain, imply simultaneous confidence, or prove arbitrary pointwise accuracy."}
    write(folder/"independent-audit.json", output)
    print(json.dumps({k: v for k, v in output.items() if k not in ("table", "source_manifest", "paired_by_family_seed")}))
    print(json.dumps([v for v in table if v["count"] == 12 and v["policy"] in ("scratch", "transfer", "matched_label_scratch", "wrong_transfer")]))


if __name__ == "__main__":
    main()
