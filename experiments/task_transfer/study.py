"""Prospective independent task families; evaluator truth never enters fitting."""

import argparse
import gzip
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from experiments.sparse_mechanisms.arrays import ArrayWriter

from .core import acquire, calibrate, candidate_models, choose, evidence, features, learned_basis

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/"research-continuation/23_task_transfer"
FAMILIES = ["related", "innovation", "noisy_innovation", "dense", "local_exception", "off_family"]
COUNTS = [8, 12, 20]
POLICIES = ["scratch", "transfer", "ridge", "sparse", "omp", "subspace", "innovation",
            "wrong_transfer", "matched_label_scratch"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sources():
    return {p.relative_to(ROOT).as_posix(): sha(p) for root in ("experiments", "workbench", "src/sera", "scripts")
            for p in (ROOT/root).rglob("*.py")}


def freeze(folder, protocol):
    folder.mkdir(exist_ok=False)
    protocol["sources"] = sources()
    write(folder/"protocol.json", protocol)
    with zipfile.ZipFile(folder/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in protocol["sources"]:
            z.write(ROOT/p, p)


def truth(x, coefficient, family):
    y = features(x)@coefficient
    if family == "local_exception":
        y += .75*((x[:, 0] > .8) & (x[:, 1] > .8))
    elif family == "off_family":
        y += .4*np.sin(6*x[:, 0])
    return y


def batch(rng, count, coefficient, family, role, noise=0.):
    x = rng.uniform(-1, 1, (count, 3))
    y = truth(x, coefficient, family)+rng.uniform(-noise, noise, count)
    return evidence(x, y, role)


def source_tasks(rng, hidden):
    records = []
    for _ in range(4):
        coefficient = hidden@rng.normal(size=3)
        records.append(acquire(batch(rng, 48, coefficient, "related", "fit"),
                               batch(rng, 16, coefficient, "related", "selection"),
                               batch(rng, 64, coefficient, "related", "calibration"),
                               policy="scratch", tolerance=.05))
    admitted = [r["selected"]["coefficients"] for r in records if r["gate"]["accepted"]]
    basis, singular = learned_basis(admitted)
    return records, basis, singular


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    args = parser.parse_args()
    folder = RELEASE/args.name
    freeze(folder, {"id": args.name, "config": vars(args), "families": FAMILIES,
                    "counts": COUNTS, "policies": POLICIES, "selection": 16, "calibration": 64,
                    "evaluation": 512, "targeted_exception_probes": 16, "tolerance": .05,
                    "source": "Four earlier tasks, each 48 fit + 16 selection + 64 independent calibration labels. Actual fitted source coefficients yield a rank-at-most-three SVD basis. Wrong prior is acquired from four different observed tasks, never from ground-truth vectors.",
                    "generators": "Independent random rank-three mixtures of 20 supplied orthonormal Legendre terms. New tasks use acquired span, span plus 3 sparse innovations, noisy innovations, unrelated dense coefficients, a 1%-mass local exception, or an out-of-family sinusoid.",
                    "claim": "Observed task-basis reuse in numerical readouts, not learned recurrent representation or eta. These numerical worlds are supplied synthetic mechanisms.",
                    "gate": "Per-model binomial 95% upper risk <=5%; calibration is never used to select or retry. Model risk is distributional, not pointwise.",
                    "prospective_decision": "Transfer proceeds as an optional empirical numerical route only if at count 12 its paired mean future failure-rate improvement over scratch has a positive normal 95% lower bound over independent task-family seeds in related+innovation cases; no greater than .01 mean degradation on dense cases after withholding; deployment must preserve parent tensors, check guards and pass persistence/drift tests. No global accuracy or lifetime-cost win is inferred.",
                    "cost_control": "For a declared 16-target lifetime, amortize 512 source labels as 32 additional fitting labels for the matched-label scratch arm. Charge selection/calibration to every arm, source and wrong-source construction separately. Evaluation and development are research costs."})
    arrays = ArrayWriter(folder/"arrays.zip")
    result, source_log = [], []
    started = time.perf_counter()
    with gzip.open(folder/"records.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index in range(args.seeds):
            seed = args.start_seed+1009*index
            rng = np.random.default_rng(seed)
            hidden = np.linalg.qr(rng.normal(size=(20, 3)))[0]
            wrong_hidden = np.linalg.qr(rng.normal(size=(20, 3)))[0]
            previous, basis, singular = source_tasks(rng, hidden)
            wrong_previous, wrong_basis, wrong_singular = source_tasks(rng, wrong_hidden)
            source_log.append({"seed": seed, "source_tasks": previous, "basis": basis.tolist(),
                               "singular_values": singular, "wrong_tasks": wrong_previous,
                               "wrong_basis": wrong_basis.tolist(), "wrong_singular": wrong_singular,
                               "evaluator_subspace_error": float(np.linalg.norm(basis@basis.T-hidden@hidden.T))})
            for family in FAMILIES:
                coefficient = hidden@rng.normal(size=3)
                if family in ("innovation", "noisy_innovation"):
                    coefficient[rng.choice(20, 3, replace=False)] += rng.choice([-1., 1.], 3)*rng.uniform(.2, .6, 3)
                elif family == "dense":
                    coefficient = rng.normal(size=20)/np.sqrt(20)
                noise = .005 if family == "noisy_innovation" else 0.
                full = batch(rng, 52, coefficient, family, "fit", noise)
                selection = batch(rng, 16, coefficient, family, "selection", noise)
                calibration = batch(rng, 64, coefficient, family, "calibration", noise)
                query = rng.uniform(-1, 1, (512, 3))
                expected = truth(query, coefficient, family)
                probes = rng.uniform(-1, 1, (16, 3))
                probes[:, :2] = rng.uniform(.8, 1, (16, 2))
                probe_truth = truth(probes, coefficient, family)
                shared = {"fit": full, "selection": selection, "calibration": calibration,
                          "query": arrays.put(query), "expected": arrays.put(expected),
                          "probes": arrays.put(probes), "probe_truth": arrays.put(probe_truth)}
                shared_ref = arrays.put(np.frombuffer(json.dumps(shared).encode(), dtype=np.uint8))
                for count in COUNTS:
                    fit = evidence(full["x"][:count], full["y"][:count], "fit")
                    candidates = candidate_models(fit, selection, basis, noise)
                    wrong_candidates = candidate_models(fit, selection, wrong_basis, noise)
                    extra = evidence(full["x"][:count+32], full["y"][:count+32], "fit")
                    extra_candidates = candidate_models(extra, selection, noise=noise)
                    for policy in POLICIES:
                        catalog = (wrong_candidates if policy == "wrong_transfer" else extra_candidates
                                   if policy == "matched_label_scratch" else candidates)
                        selector = ("transfer" if policy == "wrong_transfer" else "scratch"
                                    if policy == "matched_label_scratch" else policy)
                        selected = choose(catalog, selector)
                        if selected is None:
                            row = {"status": "NO_FIT", "seed": seed, "family": family,
                                   "count": count, "policy": policy, "evidence": shared_ref,
                                   "candidate_failures": catalog}
                        else:
                            w = np.array(selected["coefficients"])
                            predicted = features(query)@w
                            errors = abs(predicted-expected)
                            gate = calibrate(w, calibration, .05)
                            row = {"status": "FIT", "seed": seed, "family": family, "count": count,
                                   "policy": policy, "selected": selected, "gate": gate,
                                   "evidence": shared_ref, "prediction": arrays.put(predicted),
                                   "failure_rate": float(np.mean(errors > .05)),
                                   "mae": float(errors.mean()), "max_error": float(errors.max()),
                                   "probe_max_error": float(abs(features(probes)@w-probe_truth).max()),
                                   "coefficient_error": float(np.linalg.norm(w-coefficient)),
                                   "fresh_labels": count+80+(32 if policy == "matched_label_scratch" else 0),
                                   "prior_labels": 512 if policy in ("transfer", "subspace", "innovation", "wrong_transfer") else 0,
                                   "candidate_trace": catalog}
                        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                        result.append({k: v for k, v in row.items() if k not in ("candidate_trace", "candidate_failures", "evidence", "prediction", "selected")})
                stream.flush()
            write(folder/"progress.json", {"completed_seeds": index+1, "last_seed": seed,
                                            "records": len(result)})
    arrays.close()
    write(folder/"sources-learned.json", source_log)
    write(folder/"summary.json", {"status": "COMPLETE", "rows": result, "records": len(result),
                                  "seconds": time.perf_counter()-started})
    print(json.dumps({"status": "COMPLETE", "records": len(result), "seconds": time.perf_counter()-started}))


if __name__ == "__main__":
    main()
