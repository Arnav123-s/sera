"""Independent saved-output arithmetic and LP-certificate audit, without refitting."""

import argparse
import gzip
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def independent_matrix(terms, coordinates):
    t = np.asarray(coordinates)
    columns = []
    for term in terms:
        if term == ["one"]:
            value = np.ones(len(t))
        elif term[0] == "cos":
            value = np.sqrt(2)*np.cos(float(term[1])*np.pi*t)
        elif term[0] == "sin":
            value = np.sqrt(2)*np.sin(float(term[1])*np.pi*t)
        else:
            raise ValueError("Unregistered term")
        columns.append(value)
    return np.array(columns).T


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b), 1e-12))


def independent_truth(spec, terms, coordinates):
    t = np.asarray(coordinates)
    value = independent_matrix(terms, t)@np.asarray(spec["weights"])
    if spec["family"] == "off_grid":
        signal = .7*np.sin(2*np.pi*spec["off_frequency"]*t)
        value[:, 0] += signal
        value[:, 1] -= .6*signal
    elif spec["family"] == "local_exception":
        signal = 2*np.exp(-((t-spec["exception_center"])/.012)**2/2)
        value[:, 0] += signal
        value[:, 1] -= signal
    elif spec["family"] == "chirp":
        signal = np.sin(np.pi*spec["off_frequency"]*(2*t+t**2))
        value[:, 0] += .8*signal
        value[:, 1] -= .5*signal
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    folder = ROOT/"research-continuation/22_sparse_mechanisms"/args.name
    protocol = json.loads((folder/"protocol.json").read_text())
    with zipfile.ZipFile(folder/"sources.zip") as archive:
        assert set(archive.namelist()) == set(protocol["sources"])
        for name, expected in protocol["sources"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    records, certificates, maximum_difference, failures = [], 0, 0., 0
    paired = {}
    with gzip.open(folder/"records.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["status"] != "COMPLETE":
                failures += 1
                records.append(row)
                continue
            artifact, spec = row["artifact"], row["world"]
            terms, weights = artifact["terms"], np.asarray(artifact["coefficients"])
            assert terms == protocol["dictionaries"][str(row["p"])]
            roles = [artifact["support"], artifact["selection"], row["calibration"], row["query"]]
            sets = [set(role["t"]) for role in roles]
            assert all(not sets[i] & sets[j] for i in range(4) for j in range(i))
            assert row["scalar_acquisition_cost"] == 2*(row["count"]+8+12)
            key = (row["p"], row["index"], row["family"], row["count"], row["sampler"])
            evidence_sha = hashlib.sha256(json.dumps(roles[:3], sort_keys=True).encode()).hexdigest()
            if key in paired:
                assert evidence_sha == paired[key]
            paired[key] = evidence_sha
            for role_index, role in enumerate(roles[:3]):
                actual = independent_truth(spec, terms, role["t"])
                if role_index == 0 and row["family"] == "corrupted":
                    actual[:2] += np.array([[.8, -.8], [-.8, .8]])
                assert np.max(abs(np.asarray(role["y"])-actual)) <= spec["nominal_noise_bound"]+1e-10
            actual = independent_truth(spec, terms, row["query"]["t"])
            prediction = independent_matrix(terms, row["query"]["t"])@weights
            difference = max(float(np.max(abs(actual-row["query"]["truth"]))),
                             float(np.max(abs(prediction-row["query"]["prediction"]))))
            maximum_difference = max(maximum_difference, difference)
            assert difference < 1e-9
            error = relative(prediction, actual)
            assert abs(error-row["prediction_error"]) < 1e-10 and row["capability"] == (error <= .05)
            if row["coefficient_error"] is not None:
                assert abs(relative(weights, np.asarray(spec["weights"]))-row["coefficient_error"]) < 1e-10
            calibration = row["calibration"]
            cal_error = relative(independent_matrix(terms, calibration["t"])@weights, np.asarray(calibration["y"]))
            assert abs(cal_error-row["guard"]["relative_calibration_error"]) < 1e-10
            cast = np.float32 if row["precision"] == "float32" else np.float64
            a = independent_matrix(terms, artifact["support"]["t"]).astype(cast).astype(float)
            y = np.array(artifact["support"]["y"]).astype(cast).astype(float)
            norm = np.linalg.norm(a, axis=0)
            nz = norm > 1e-10
            correlations = (a[:, nz]/norm[nz]).T@(a[:, nz]/norm[nz])
            np.fill_diagonal(correlations, 0)
            ambiguous = bool((~nz).any() or (np.abs(np.triu(correlations, 1)) > 1-1e-9).any())
            assert row["guard"]["accepted"] == (cal_error <= .05 and not ambiguous)
            for output, certificate in enumerate(artifact["certificates"]):
                estimate, dual = np.array(certificate["primal"]), np.array(certificate["dual"])
                epsilon = certificate["epsilon"]
                assert max(abs(a@estimate-y[:, output])) <= epsilon+1e-7
                if epsilon == 0:
                    objective = y[:, output]@dual
                    assert max(abs(a.T@dual)) <= 1+1e-7
                else:
                    constraints = np.r_[np.c_[a, -a], np.c_[-a, a]]
                    rhs = np.r_[y[:, output]+epsilon, -y[:, output]+epsilon]
                    objective = rhs@dual
                    assert max(dual) <= 1e-7 and max(constraints.T@dual) <= 1+1e-7
                assert abs(abs(estimate).sum()-objective) < 1e-6
                certificates += 1
            records.append({k: v for k, v in row.items() if k not in ("artifact", "query", "calibration", "world")} | {"world_seed": spec["seed"]})
    focus = [r for r in records if r["p"] == 33 and r["sampler"] == "random" and r["precision"] == "float64"]
    adequate = [r for r in focus if r["family"] in ("sparse", "noisy")]
    bp = [r for r in adequate if r["method"] == "basis_pursuit"]
    means = {method: float(np.mean([r["prediction_error"] for r in adequate if r["method"] == method])) for method in ("basis_pursuit", "omp", "ridge")}
    accepted = [r for r in focus if r["method"] == "basis_pursuit" and r.get("guard", {}).get("accepted")]
    risk = sum(not r["capability"] for r in accepted)/max(1, len(accepted))
    gain = 1-means["basis_pursuit"]/max(min(means["omp"], means["ridge"]), 1e-12)
    gates = {"restricted_sparse_success": sum(r["capability"] for r in bp)/len(bp) >= .9,
             "improves_strongest_peer": gain >= .1, "accepted_error_rate_all_families": risk <= .05,
             "adequate_family_coverage": sum(r["guard"]["accepted"] for r in bp)/len(bp) >= .5}
    grouped = defaultdict(list)
    for row in focus:
        grouped[f"{row['count']}/{row['family']}/{row['method']}"].append(row)
    table = {key: {"instances": len(rows), "solved": sum(r["capability"] for r in rows),
                   "mean_relative_error": float(np.mean([r["prediction_error"] for r in rows])),
                   "accepted": sum(r["guard"]["accepted"] for r in rows),
                   "accepted_wrong": sum(r["guard"]["accepted"] and not r["capability"] for r in rows)} for key, rows in grouped.items()}
    result = {"status": "PASS", "records": len(records), "failed_solves_preserved": failures,
              "paired_measurement_settings": len(paired), "lp_certificates_checked": certificates,
              "maximum_prediction_or_truth_discrepancy": maximum_difference, "focus_table": table,
              "qualification": "QUALIFIED_CONDITIONAL_COMPONENT" if all(gates.values()) else "REJECTED_FOR_INTEGRATION",
              "gates": gates, "adequate_family_mean_errors": means, "relative_improvement_over_strongest_peer": gain,
              "accepted_wrong": sum(not r["capability"] for r in accepted), "accepted": len(accepted), "accepted_error_rate": risk,
              "boundary": "Recomputes predictions, truth, declared errors, paired access, evidence separation and LP certificates from saved records. Does not refit solvers or certify global identifiability/adequacy."}
    (folder/"independent-audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "focus_table"}))


if __name__ == "__main__":
    main()
