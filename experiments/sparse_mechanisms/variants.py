"""Sparse sketches, corruption repair and real readout-weight adaptation.

Controlled synthetic studies, not improvements to the live SERA owner. Truth is
kept in the evaluator; recovery methods receive only observations and metadata.
"""

import argparse
import gzip
import json
import time
import zipfile

import numpy as np

from .arrays import ArrayWriter
from .core import l1, omp, relative
from .study import RELEASE, ROOT, source_files, write


def sketch_operator(seed, count, dimension):
    return np.random.default_rng(seed).normal(size=(count, dimension))/np.sqrt(count)


def repair_dense(a, received):
    """Least absolute deviations: unpenalized payload, sparse error penalty."""
    joined = np.c_[a, np.eye(len(a))]
    solution, certificates = l1(joined, np.asarray(received)[:, None], 0,
                                prior=list(range(a.shape[1])))
    return solution[:a.shape[1], 0], solution[a.shape[1]:, 0], certificates


def memory_cases(seed):
    rows = []
    for family in ("sparse4", "sparse12", "growing", "dense", "noisy4"):
        rng = np.random.default_rng(seed)
        size = 128
        ids = rng.choice(size, 12, replace=False)
        values = rng.normal(size=12)
        initial = np.zeros(size)
        active = 128 if family == "dense" else (12 if family == "sparse12" else 4)
        if active == 128:
            initial = rng.normal(size=size)
        else:
            initial[ids[:active]] = values[:active]
        updates = []
        for step in range(8):
            coordinate = int(ids[4+step] if family == "growing" else ids[step % min(active, 12)])
            updates.append((coordinate, float(rng.normal(scale=.2))))
        target = initial.copy()
        for coordinate, change in updates:
            target[coordinate] += change
        for count in (16, 32, 48):
            operator_seed = seed+300001
            a = sketch_operator(operator_seed, count, size)
            encoded = a@initial
            for coordinate, change in updates:
                encoded += a[:, coordinate]*change
            # The sketch can be updated without decompressing its previous state.
            assert np.max(abs(encoded-a@target)) < 1e-12
            for damage in ("none", "erased_quarter"):
                retained = np.arange(count) if damage == "none" else np.arange(count)[np.arange(count) % 4 != 0]
                noise_bound = .005 if family == "noisy4" else 0.
                observed = encoded[retained]+rng.uniform(-noise_bound, noise_bound, len(retained))
                observed_a = a[retained]
                for method in ("basis_pursuit", "minimum_norm", "direct_sparse"):
                    started = time.perf_counter()
                    certificates = []
                    if method == "basis_pursuit":
                        recovered, certificates = l1(observed_a, observed[:, None], noise_bound)
                        recovered = recovered[:, 0]
                    elif method == "minimum_norm":
                        recovered = np.linalg.lstsq(observed_a, observed, rcond=None)[0]
                    else:
                        # Strong storage baseline for a vector already known at encoding.
                        # It is not exposed to the sketch's row-erasure fault model.
                        sparse_ids = np.flatnonzero(target)
                        recovered = np.zeros(size)
                        recovered[sparse_ids] = target[sparse_ids]
                    elapsed = time.perf_counter()-started
                    nonzero = int(np.count_nonzero(target))
                    rows.append({"domain": "memory", "seed": seed, "family": family, "count": count,
                                 "damage": damage, "method": method, "operator_seed": operator_seed,
                                 "initial": initial.tolist(), "updates": updates, "target": target.tolist(),
                                 "retained": retained.tolist(), "observed": observed.tolist(),
                                 "noise_bound": noise_bound, "recovered": recovered.tolist(),
                                 "relative_error": relative(recovered, target), "certificates": certificates,
                                 "stored_numeric_bytes": 12*nonzero+8 if method == "direct_sparse" else 8*count+16,
                                 "dense_numeric_bytes": 8*size, "direct_sparse_numeric_bytes": 12*nonzero+8,
                                 "recreated_operator_bytes": 8*count*size,
                                 "encoding_multiply_adds": count*size, "update_multiply_adds": 8*count,
                                 "elapsed_seconds": elapsed,
                                 "boundary": "Numerical payload and seed/dimension metadata only; Python/container overhead excluded. Direct sparse baseline retains its own values; no equivalent corruption guarantee."})
    return rows


def corruption_cases(seed):
    rng = np.random.default_rng(seed+800001)
    a = sketch_operator(seed+900001, 64, 32)
    payload = rng.normal(size=32)
    clean = a@payload
    rows = []
    for corruption_count in (0, 4, 12, 24, 64):
        corrupted = rng.choice(64, corruption_count, replace=False)
        received = clean.copy()
        received[corrupted] += rng.normal(scale=2., size=corruption_count)
        for method in ("sparse_error", "least_squares"):
            started = time.perf_counter()
            if method == "sparse_error":
                decoded, error, cert = repair_dense(a, received)
            else:
                decoded = np.linalg.lstsq(a, received, rcond=None)[0]
                error, cert = received-a@decoded, []
            rows.append({"domain": "corruption", "seed": seed, "operator_seed": seed+900001,
                         "corruption_count": corruption_count, "method": method,
                         "payload": payload.tolist(), "received": received.tolist(),
                         "corrupted_indices": corrupted.tolist(), "decoded": decoded.tolist(),
                         "estimated_corruption": error.tolist(), "certificates": cert,
                         "relative_error": relative(decoded, payload),
                         "original_numeric_bytes": 256, "redundant_numeric_bytes": 528,
                         "elapsed_seconds": time.perf_counter()-started})
    return rows


def features(x, hidden, geometry):
    return np.asarray(x) if geometry == "linear" else np.tanh(np.asarray(x)@hidden)


def forward(x, hidden, weights, geometry):
    """Executable single-output linear / two-layer tanh network."""
    return features(x, hidden, geometry)@np.asarray(weights)


def weight_cases(seed):
    rows = []
    for geometry in ("linear", "tanh", "correlated_tanh"):
        rng = np.random.default_rng(seed+100003)
        hidden = rng.normal(size=(16, 64))/4
        if geometry == "correlated_tanh":
            hidden[:, 1::2] = hidden[:, ::2]+rng.normal(scale=.005, size=(16, 32))
        dimension = 64 if geometry == "linear" else 16
        x = rng.normal(size=(48+16+256, dimension))
        phi = features(x, hidden, geometry)
        parent = rng.normal(scale=.1, size=64)
        for family in ("sparse", "noisy", "dense", "outside_readout"):
            delta = np.zeros(64)
            ids = rng.choice(64, 4, replace=False)
            delta[ids] = rng.choice([-1., 1.], 4)*rng.uniform(.5, 1.5, 4)
            if family == "dense":
                delta = rng.normal(size=64)/4
            correction = phi@delta
            if family == "outside_readout":
                correction += np.sin(3*x[:, 0]*x[:, 1])
            noise_bound = .01 if family == "noisy" else 0.
            observed = correction+rng.uniform(-noise_bound, noise_bound, len(x))
            for count in (16, 32, 48):
                a, y = phi[:count], observed[:count]
                selected_a, selected_y = phi[48:64], observed[48:64]
                for method in ("basis_pursuit", "omp", "ridge", "frozen_parent"):
                    started = time.perf_counter()
                    certificates, candidates = [], []
                    if method == "basis_pursuit":
                        fitted, certificates = l1(a, y[:, None], noise_bound)
                        fitted = fitted[:, 0]
                        work = 1
                    elif method == "omp":
                        for sparsity in range(1, 9):
                            estimate = omp(a, y[:, None], sparsity)[:, 0]
                            candidates.append((float(np.mean((selected_a@estimate-selected_y)**2)), sparsity, estimate))
                        fitted = min(candidates, key=lambda z: z[0])[2]
                        work = sum(range(1, 9))
                    elif method == "ridge":
                        for alpha in (0., .0001, .01, .1, 1.):
                            estimate = np.linalg.lstsq(a, y, rcond=None)[0] if alpha == 0 else a.T@np.linalg.solve(a@a.T+alpha*np.eye(count), y)
                            candidates.append((float(np.mean((selected_a@estimate-selected_y)**2)), alpha, estimate))
                        fitted = min(candidates, key=lambda z: z[0])[2]
                        work = 5
                    else:
                        fitted, work = np.zeros(64), 0
                    elapsed = time.perf_counter()-started
                    # Materialize the changed network weights, then execute that network.
                    updated = parent+fitted
                    predicted = forward(x[64:], hidden, updated, geometry)
                    old_predictions = forward(x[64:], hidden, parent, geometry)
                    truth = old_predictions+correction[64:]
                    rows.append({"domain": "weights", "seed": seed, "geometry": geometry, "family": family,
                                 "count": count, "method": method, "noise_bound": noise_bound,
                                 "hidden": hidden.tolist(), "parent_weights": parent.tolist(), "true_delta": delta.tolist(),
                                 "fitted_delta": fitted.tolist(), "updated_weights": updated.tolist(),
                                 "support_x": x[:count].tolist(), "support_y": y.tolist(),
                                 "selection_x": x[48:64].tolist(), "selection_y": selected_y.tolist(),
                                 "query_x": x[64:].tolist(), "query_prediction": predicted.tolist(), "query_truth": truth.tolist(),
                                 "prediction_error": relative(predicted-old_predictions, correction[64:]),
                                 "coefficient_error": relative(fitted, delta),
                                 "old_task_change": relative(predicted, old_predictions),
                                 "certificates": certificates, "candidates": [[score, parameter] for score, parameter, _ in candidates],
                                 "paid_labels": count+16, "linear_solves": work, "elapsed_seconds": elapsed,
                                 "boundary": "Fixed hidden network and supplied sparse-delta hypothesis. Only output weights learn; the parent is a seeded network, not a pretrained expert. No shared-owner or learned-eta claim."})
    return rows


def run(args):
    folder = RELEASE/args.name
    folder.mkdir(exist_ok=False)
    protocol = {"id": args.name, "config": vars(args), "sources": source_files(),
                "domains": ["memory", "corruption", "weights"],
                "memory": "128 coordinates, 16/32/48 Gaussian sketch values, eight recorded online additions, sparse/growing/dense/noisy families, 25% erasure. Compare BP, min-norm and direct sparse storage.",
                "corruption": "32 dense values redundantly encoded as 64; unknown 0/4/12/24/64 corruptions. Compare sparse error correction with least squares. Expansion, not compression.",
                "weights": "64 output weights; fixed linear/tanh/highly-correlated-tanh geometry. Four sparse changes versus dense/noisy/outside-readout targets. 16/32/48 fit and 16 selection labels, 256 evaluation inputs inaccessible to fitting. BP, OMP, ridge, unchanged parent.",
                "success": "Memory/error repair relative error <=1e-6; network future correction error <=.05. Report every stratum; no pooled claim hiding negative cases.",
                "promotion": "Exploratory cross-domain evidence only; a live-owner change requires a separate downstream integration gate. A repair utility may return data only after independent payload digest verification.",
                "cost": "One worker. Record encoding, update and decode cost; seeds/regenerated operator/hidden network are not free stored knowledge. Paid labels include selection for every weight arm.",
                "multiple_comparisons": "Descriptive multi-domain screening. No significance or universal superiority claim; no final-driven hyperparameter edits."}
    write(folder/"protocol.json", protocol)
    with zipfile.ZipFile(folder/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["sources"]:
            archive.write(ROOT/name, name)
    rows, started = [], time.perf_counter()
    arrays = ArrayWriter(folder/"arrays.zip")
    with gzip.open(folder/"records.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index in range(args.seeds):
            seed = args.start_seed+index*101
            for producer in (memory_cases, corruption_cases, weight_cases):
                records = producer(seed)
                for row in records:
                    for key in ("hidden", "support_x", "selection_x", "query_x"):
                        if key in row:
                            row[key] = arrays.put(row[key])
                    stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                    rows.append({k: v for k, v in row.items() if k in ("domain", "seed", "family", "geometry", "count", "damage", "method", "corruption_count", "relative_error", "prediction_error", "coefficient_error", "old_task_change", "elapsed_seconds", "stored_numeric_bytes", "direct_sparse_numeric_bytes")})
                stream.flush()
                write(folder/"progress.json", {"records": len(rows), "last_seed": seed})
    arrays.close()
    write(folder/"summary.json", {"status": "COMPLETE", "records": len(rows), "worker_seconds": time.perf_counter()-started, "rows": rows})
    print(json.dumps({"status": "COMPLETE", "records": len(rows), "worker_seconds": time.perf_counter()-started}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
