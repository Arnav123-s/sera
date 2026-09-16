"""Check stored cross-domain results independently, without training again."""

import argparse
import gzip
import hashlib
import json
import zipfile
from collections import defaultdict

import numpy as np

from .arrays import ArrayReader
from .study import RELEASE, write


def error(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b), 1e-12))


def source_check(folder):
    protocol = json.loads((folder/"protocol.json").read_text())
    with zipfile.ZipFile(folder/"sources.zip") as archive:
        assert set(archive.namelist()) == set(protocol["sources"])
        for name, digest in protocol["sources"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    return protocol


def check_lp(a, y, certificates, unpenalized=()):
    weights = np.ones(a.shape[1])
    weights[list(unpenalized)] = 0
    for index, certificate in enumerate(certificates):
        primal = np.array(certificate["primal"])
        dual = np.array(certificate["dual"])
        epsilon = certificate["epsilon"]
        target = y[:, index]
        assert np.max(abs(a@primal-target)) <= epsilon+1e-7
        if epsilon == 0:
            assert np.max(abs(a.T@dual)-weights) <= 1e-7
            objective = target@dual
        else:
            constraints = np.r_[np.c_[a, -a], np.c_[-a, a]]
            rhs = np.r_[target+epsilon, -target+epsilon]
            assert np.max(constraints.T@dual-np.r_[weights, weights]) <= 1e-7 and dual.max() <= 1e-7
            objective = rhs@dual
        assert abs(weights@abs(primal)-objective) < 1e-5
    return len(certificates)


def independent_haar(n):
    rows = [np.ones(n)/np.sqrt(n)]
    width = n
    while width >= 2:
        for start in range(0, n, width):
            row = np.zeros(n)
            row[start:start+width//2] = 1/np.sqrt(width)
            row[start+width//2:start+width] = -1/np.sqrt(width)
            rows.append(row)
        width //= 2
    return np.array(rows)


def summarize(grouped):
    result = {}
    for key, records in sorted(grouped.items()):
        values = np.array([row["error"] for row in records])
        result[key] = {"instances": len(records), "mean_relative_error": float(values.mean()),
                       "median_relative_error": float(np.median(values)), "worst_relative_error": float(values.max()),
                       "exact_at_1e_6": int((values <= 1e-6).sum()), "within_5_percent": int((values <= .05).sum()),
                       "decoder_seconds": float(sum(r["seconds"] for r in records))}
        for name in ("roi_absolute_error", "contrast_absolute_error", "true_contrast", "predicted_contrast", "old_task_change", "stored_numeric_bytes", "direct_sparse_numeric_bytes", "coefficient_error", "stationarity"):
            items = [r[name] for r in records if r.get(name) is not None]
            if items:
                result[key]["mean_"+name] = float(np.mean(items))
    return result


def audit(folder):
    protocol = source_check(folder)
    arrays = ArrayReader(folder/"arrays.zip")
    grouped, pairs = defaultdict(list), {}
    certificates, count = 0, 0
    diagnostics = []
    with gzip.open(folder/"records.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            domain = row.get("domain", "imaging")
            metrics = {}
            if domain == "memory":
                a = np.random.default_rng(row["operator_seed"]).normal(size=(row["count"], 128))/np.sqrt(row["count"])
                target = np.array(row["initial"])
                for coordinate, value in row["updates"]:
                    target[coordinate] += value
                np.testing.assert_allclose(target, row["target"], atol=1e-13, rtol=0)
                observed_a = a[row["retained"]]
                observed = np.asarray(row["observed"])
                assert np.max(abs(observed-observed_a@target)) <= row["noise_bound"]+1e-12
                measured_error = error(row["recovered"], target)
                certificates += check_lp(observed_a, observed[:, None], row["certificates"])
                key = f"memory/{row['family']}/{row['count']}/{row['damage']}/{row['method']}"
                for field in ("stored_numeric_bytes", "direct_sparse_numeric_bytes"):
                    metrics[field] = row[field]
                pair_key = (domain, row["seed"], row["family"], row["count"], row["damage"])
                pair_data = [row["retained"], row["observed"]]
            elif domain == "corruption":
                a = np.random.default_rng(row["operator_seed"]).normal(size=(64, 32))/8
                payload, received = np.array(row["payload"]), np.array(row["received"])
                difference = received-a@payload
                assert np.count_nonzero(abs(difference) > 1e-12) == row["corruption_count"]
                np.testing.assert_allclose(a@row["decoded"]+row["estimated_corruption"], received, atol=1e-7, rtol=0)
                certificates += check_lp(np.c_[a, np.eye(64)], received[:, None], row["certificates"], range(32))
                measured_error = error(row["decoded"], payload)
                key = f"corruption/{row['corruption_count']}/{row['method']}"
                pair_key = (domain, row["seed"], row["corruption_count"])
                pair_data = row["received"]
            elif domain == "weights":
                hidden = arrays.get(row["hidden"])
                x, select_x, query_x = [arrays.get(row[name]) for name in ("support_x", "selection_x", "query_x")]
                transform = (lambda z: z) if row["geometry"] == "linear" else (lambda z: np.tanh(z@hidden))
                a, select_a, query_a = transform(x), transform(select_x), transform(query_x)
                parent, delta, fitted = [np.array(row[name]) for name in ("parent_weights", "true_delta", "fitted_delta")]
                np.testing.assert_allclose(parent+fitted, row["updated_weights"], atol=1e-13, rtol=0)
                truth_correction = query_a@delta
                if row["family"] == "outside_readout":
                    truth_correction += np.sin(3*query_x[:, 0]*query_x[:, 1])
                np.testing.assert_allclose(query_a@parent+truth_correction, row["query_truth"], atol=1e-11, rtol=0)
                np.testing.assert_allclose(query_a@(parent+fitted), row["query_prediction"], atol=1e-11, rtol=0)
                for design, features, observed in ((a, x, row["support_y"]), (select_a, select_x, row["selection_y"])):
                    expected = design@delta
                    if row["family"] == "outside_readout":
                        expected += np.sin(3*features[:, 0]*features[:, 1])
                    assert np.max(abs(np.array(observed)-expected)) <= row["noise_bound"]+1e-12
                assert not set(map(tuple, x)) & set(map(tuple, query_x))
                assert not set(map(tuple, select_x)) & set(map(tuple, query_x))
                measured_error = error(query_a@fitted, truth_correction)
                assert abs(error(fitted, delta)-row["coefficient_error"]) < 1e-10
                certificates += check_lp(a, np.asarray(row["support_y"])[:, None], row["certificates"])
                assert row["paid_labels"] == len(x)+len(select_x)
                key = f"weights/{row['geometry']}/{row['family']}/{row['count']}/{row['method']}"
                metrics = {"coefficient_error": row["coefficient_error"], "old_task_change": error(query_a@(parent+fitted), query_a@parent)}
                assert abs(metrics["old_task_change"]-row["old_task_change"]) < 1e-10
                pair_key = (domain, row["seed"], row["geometry"], row["family"], row["count"])
                pair_data = [row[name] for name in ("support_x", "support_y", "selection_x", "selection_y", "query_x")]
            else:
                original, restored, mask, measured, roi = [arrays.get(row[name]) for name in ("image", "reconstructed", "mask", "measured", "roi")]
                assert mask.dtype == bool and mask.sum() == row["count"]
                assert row["observed_scalar_cost"] == 2*int(mask.sum())
                if row["geometry"].startswith("cartesian"):
                    assert np.all((mask.sum(axis=1) == 0) | (mask.sum(axis=1) == 32))
                rng = np.random.default_rng(row["seed"]+700001)
                noise = row["noise_sigma"]*(rng.normal(size=(32, 32))+1j*rng.normal(size=(32, 32)))
                np.testing.assert_allclose(mask*(np.fft.fft2(original, norm="ortho")+noise), measured, atol=1e-12, rtol=0)
                measured_error = error(restored, original)
                if row["method"] == "zero_filled":
                    np.testing.assert_allclose(np.fft.ifft2(measured, norm="ortho").real, restored, atol=1e-13, rtol=0)
                else:
                    h = independent_haar(32)
                    transform = (lambda z: h@z@h.T) if row["method"] == "haar" else (lambda z: z)
                    coefficients = transform(restored)
                    residual = mask*np.fft.fft2(restored, norm="ortho")-measured
                    gradient = transform(np.fft.ifft2(residual, norm="ortho").real)
                    penalty = row["work"]["penalty"]
                    stepped = coefficients-gradient
                    projected = np.sign(stepped)*np.maximum(abs(stepped)-penalty, 0)
                    stationarity = float(np.linalg.norm(coefficients-projected)/max(np.linalg.norm(coefficients), 1e-12))
                    assert abs(stationarity-row["work"]["proximal_gradient_relative_norm"]) < 1e-10
                    objective = .5*np.linalg.norm(residual)**2+penalty*abs(coefficients).sum()
                    assert abs(objective-row["work"]["objective_trace"][-1][1]) < 1e-10
                    metrics["stationarity"] = stationarity
                if roi.any():
                    roi_error = float(np.mean(abs(restored[roi]-original[roi])))
                    assert abs(roi_error-row["roi_absolute_error"]) < 1e-12
                    positions = np.argwhere(roi)
                    low, high = positions.min(axis=0)-1, positions.max(axis=0)+2
                    ring = np.zeros_like(roi)
                    ring[low[0]:high[0], low[1]:high[1]] = True
                    ring &= ~roi
                    actual_contrast = float(original[roi].mean()-original[ring].mean())
                    predicted_contrast = float(restored[roi].mean()-restored[ring].mean())
                    metrics.update(roi_absolute_error=roi_error, true_contrast=actual_contrast,
                                   predicted_contrast=predicted_contrast, contrast_absolute_error=abs(predicted_contrast-actual_contrast))
                key = f"imaging/{row['family']}/{row['count']}/{row['geometry']}/{row['noise_sigma']}/{row['method']}"
                pair_key = (domain, row["seed"], row["count"], row["geometry"], row["noise_sigma"])
                pair_data = [row["mask"], row["measured"]]
            expected_error = row["prediction_error"] if domain == "weights" else row["relative_error"]
            assert abs(measured_error-expected_error) < 1e-10
            pair_digest = hashlib.sha256(json.dumps(pair_data, sort_keys=True).encode()).hexdigest()
            if pair_key in pairs:
                assert pairs[pair_key] == pair_digest
            pairs[pair_key] = pair_digest
            metrics.update(error=measured_error, seconds=row.get("elapsed_seconds", row.get("work", {}).get("worker_seconds", 0)))
            grouped[key].append(metrics)
            diagnostics.append({"record": count, "key": key, "seed": row["seed"], **metrics})
    arrays.close()
    expected = protocol["config"]["seeds"]*(192 if "image_shape" in protocol else 244)
    assert count == expected, (count, expected)
    result = {"status": "PASS", "records": count, "paired_settings": len(pairs), "lp_certificates": certificates,
              "table": summarize(grouped),
              "boundary": "Independent saved-output equations, measured access, numerical payload hashes and optimization certificates; no repeated training and no universal recovery certificate."}
    write(folder/"independent-audit.json", result)
    with gzip.open(folder/"audit-metrics.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in diagnostics:
            stream.write(json.dumps(row)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "table"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    audit(RELEASE/parser.parse_args().name)


if __name__ == "__main__":
    main()
