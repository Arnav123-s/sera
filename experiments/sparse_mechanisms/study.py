"""Frozen common-access recovery study; final observations never enter selection."""

import argparse
import gzip
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy

from .core import dictionary, fit, guard, matrix, observations, predict, relative

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/"research-continuation/22_sparse_mechanisms"
FAMILIES = ("sparse", "noisy", "dense", "off_grid", "local_exception", "corrupted", "chirp")
COUNTS = {7: (4, 6), 17: (8, 12), 33: (12, 20)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    files = list(Path(__file__).parent.glob("*.py"))
    files += [ROOT/"experiments/generative_memory/core.py", ROOT/"scripts/run_sparse_bounded.py",
              ROOT/"scripts/run_v3_bounded.py", ROOT/"scripts/windows_job_v3.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def freeze(folder, config):
    folder.mkdir(parents=True, exist_ok=False)
    value = {"schema": "sera.sparse-study.1", "config": config, "sources": source_files(),
             "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
             "observation_contract": "Existing immutable Observations: two supplied coordinate values per phase, separate factual support/selection/calibration roles. Each phase costs two scalar values. No projected hidden trajectory.",
             "dictionaries": {str(p): dictionary(p) for p in COUNTS}, "support_counts": COUNTS,
             "selection_phases": 8, "calibration_phases": 12, "final_phases": 128,
             "methods": "Same normalized dictionary and observation arrays: ridge grid, OMP count grid, basis pursuit or bounded-infinity-noise LP. Existing fit_linear/subset criterion on seven terms only; no true support passed.",
             "precision": "Both float64 and float32 stored coefficients/solver inputs. Existing subset fitting uses its original float32 posterior means in both conditions; all such differences are disclosed.",
             "primary": "World-level relative future-curve error <=.05; coefficient recovery is a separate score. All paid selection/calibration observations and solver work count.",
             "gate": {"restricted_sparse_success": .9, "relative_error_improvement_over_strongest_peer": .1,
                      "accepted_error_rate_all_families_max": .05, "adequate_family_coverage_min": .5},
             "integration_rule": "All four empirical gates must pass on float64/random/33-term strata before shared-owner integration. No calibrated risk or learned eta claim even on a pass. If not, preserve/reject and diagnose with separate fresh measurement-policy study.",
             "no_novel_family_claim": "Fourier sparsity, dense controls and several adequacy challenges are known development concepts. Final seeds/mixtures are fresh, not new mathematical task families.",
             "scope": "Supplied phase, fixed dictionary and nominal sensor bound. Engineering algorithms and policies, not learned language, representations or learning procedures."}
    write(folder/"protocol.json", value)
    with zipfile.ZipFile(folder/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in value["sources"]:
            z.write(ROOT/name, name)
    return value


def world(p, family, seed):
    rng = np.random.default_rng(seed)
    weights = np.zeros((p, 2))
    ids = rng.choice(p, 2 if p == 7 else 3, replace=False)
    weights[ids] = rng.choice([-1., 1.], (len(ids), 2))*rng.uniform(.5, 1.5, (len(ids), 2))
    if family == "dense":
        weights = rng.normal(size=(p, 2))/np.sqrt(p)
    return {"p": p, "family": family, "seed": seed, "weights": weights.tolist(),
            "exception_center": float(rng.uniform(.1, .9)), "off_frequency": float((p-1)/2-.37),
            "nominal_noise_bound": .02 if family in ("noisy", "corrupted") else 0.}


def truth(spec, t):
    t = np.asarray(t)
    value = matrix(dictionary(spec["p"]), t)@np.asarray(spec["weights"])
    family = spec["family"]
    if family == "off_grid":
        value += (.7*np.sin(2*np.pi*spec["off_frequency"]*t))[:, None]*np.array([1., -.6])
    elif family == "local_exception":
        value += (2*np.exp(-.5*((t-spec["exception_center"])/.012)**2))[:, None]*np.array([1., -1.])
    elif family == "chirp":
        value += np.sin(2*np.pi*spec["off_frequency"]*(t+.5*t*t))[:, None]*np.array([.8, -.5])
    return value


def data(spec, count, sampler):
    rng = np.random.default_rng(spec["seed"]+400000)
    support_t = rng.uniform(0, 1, count) if sampler == "random" else np.arange(count)/count
    selection_t, calibration_t = rng.uniform(0, 1, 8), rng.uniform(0, 1, 12)
    rows = []
    for role, t in (("support", support_t), ("selection", selection_t), ("calibration", calibration_t)):
        y = truth(spec, t)+rng.uniform(-spec["nominal_noise_bound"], spec["nominal_noise_bound"], (len(t), 2))
        if role == "support" and spec["family"] == "corrupted":
            y[:2] += np.array([[.8, -.8], [-.8, .8]])
        rows.append(observations(t, y, role))
    return rows


def summarize(records):
    groups = {}
    for row in records:
        key = "/".join(str(row[k]) for k in ("p", "count", "family", "sampler", "precision", "method"))
        groups.setdefault(key, []).append(row)
    output = {}
    for key, rows in groups.items():
        accepted = [r for r in rows if r.get("guard", {}).get("accepted", False)]
        valid = [r for r in rows if r["status"] == "COMPLETE"]
        output[key] = {"instances": len(rows), "completed": len(valid), "solved": sum(r.get("capability", False) for r in rows),
                       "coefficient_recovered": sum(r.get("coefficient_error", 1) is not None and r.get("coefficient_error", 1) < 1e-6 for r in rows),
                       "mean_relative_error": None if not valid else float(np.mean([r["prediction_error"] for r in valid])),
                       "accepted": len(accepted), "accepted_wrong": sum(not r["capability"] for r in accepted),
                       "fit_seconds": sum(r.get("artifact", {}).get("work", {}).get("wall_seconds", 0) for r in rows)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    folder = RELEASE/args.name
    freeze(folder, vars(args))
    start, records = time.perf_counter(), []
    with gzip.open(folder/"records.jsonl.gz", "wt", encoding="utf-8") as stream:
        for p in COUNTS:
            for index in range(args.seeds):
                for family_index, family in enumerate(FAMILIES):
                    spec = world(p, family, args.start_seed+index*101+family_index*10007+p*100003)
                    for count in COUNTS[p]:
                        for sampler in ("random", "regular"):
                            support, selection, calibration = data(spec, count, sampler)
                            for precision in ("float64", "float32"):
                                for method in ("basis_pursuit", "omp", "ridge")+(("sera_subset",) if p == 7 else ()):
                                    row = {"p": p, "index": index, "family": family, "count": count, "sampler": sampler,
                                           "precision": precision, "method": method, "world": spec,
                                           "calibration": {"t": calibration.t.tolist(), "y": calibration.y.tolist()},
                                           "scalar_acquisition_cost": 2*(count+8+12)}
                                    try:
                                        artifact = fit(method, support, selection, dictionary(p), spec["nominal_noise_bound"], precision)
                                        checked = guard(artifact, calibration)
                                        # Final values materialize only after the model and acceptance decision.
                                        t = (np.arange(128)+.371)/128
                                        prediction, actual = predict(artifact, t), truth(spec, t)
                                        error = relative(prediction, actual)
                                        weights_error = None if family in ("off_grid", "local_exception", "chirp") else relative(artifact["coefficients"], np.asarray(spec["weights"]))
                                        row.update(status="COMPLETE", artifact=artifact, guard=checked, prediction_error=error,
                                                   coefficient_error=weights_error, capability=error <= .05,
                                                   query={"t": t.tolist(), "prediction": prediction.tolist(), "truth": actual.tolist()})
                                    except (ValueError, np.linalg.LinAlgError) as error:
                                        row.update(status="FAILED", error_type=type(error).__name__, error=str(error), capability=False)
                                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                                    # Keep only summary fields in memory; complete raw artifacts stay in the journal.
                                    records.append({k: v for k, v in row.items() if k not in ("query", "world", "calibration")} | {
                                        "artifact": {"work": row.get("artifact", {}).get("work", {})}})
                            stream.flush()
                            write(folder/"progress.json", {"completed_records": len(records), "last_setting": [p, index, family, count, sampler],
                                                          "state": "Completed records are immutable; each fit is deterministic from its declared seed. No optimizer state exists."})
    summary = {"status": "COMPLETE", "records": len(records), "groups": summarize(records),
               "failed_solves": sum(r["status"] != "COMPLETE" for r in records),
               "worker_seconds": time.perf_counter()-start, "protocol_sha256": sha(folder/"protocol.json")}
    write(folder/"summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "groups"}))


if __name__ == "__main__":
    main()
