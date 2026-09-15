"""Fresh-process replay, evidence regeneration and independent score checks."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output is immutable; choose a fresh path")
    start, cpu = time.perf_counter(), time.process_time()
    state = json.loads((args.run / "state.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    assert state["status"] == "COMPLETED"
    for name, expected in state["source_files"].items():
        assert sha(args.run / "source" / name) == expected
    core = load_frozen(args.run / "source/core.py", "gg_frozen_core")
    data = load_frozen(args.run / "source/data.py", "gg_frozen_data")
    assert core.digest(protocol) == state["contract"]["protocol_sha256"]
    assert core.digest(state["source_files"]) == state["contract"]["source_sha256"]
    expected_cases = {(seed, family, noise, n) for seed in protocol["seeds"]
                      for family in protocol["families"] for noise in protocol["noise"] for n in protocol["support"]}
    seen = set()
    models = partitions = posterior_checks = 0
    maximum_prediction_difference = maximum_metric_difference = 0.
    for completed in state["completed"]:
        path = args.run / completed["path"]
        assert sha(path) == completed["sha256"]
        record = json.loads(path.read_text())
        for name, expected in record["files"].items():
            assert sha(path.parent / name) == expected
        case = record["case"]
        identity = tuple(case[key] for key in ("seed", "family", "noise", "support"))
        assert identity not in seen
        seen.add(identity)
        seed, family, noise, n = identity
        with np.load(path.parent / "observations.npz", allow_pickle=False) as observations:
            for role, count in (("support", max(protocol["support"])), ("selection", protocol["selection"]),
                                ("calibration", protocol["calibration"])):
                fresh = data.bank(seed, family, noise, role, count)
                for key, values in fresh.items():
                    np.testing.assert_array_equal(observations[f"{role}_{key}"], values[:n] if role == "support" else values)
            support_t, support_y = observations["support_t"], observations["support_y"]
            support_truth = observations["support_truth"]
            role_sets = [set(observations[f"{role}_t"]) for role in ("support", "selection", "calibration")]
            assert not any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i))
        assert [r["method"] for r in record["records"]] == protocol["methods"]
        with np.load(path.parent / "queries.npz", allow_pickle=False) as queries:
            for role in ("interpolation", "withheld_arc", "extrapolation"):
                fresh = data.bank(seed, family, noise, role, protocol["query_count"])
                for key, values in fresh.items():
                    np.testing.assert_array_equal(queries[f"{role}_{key}"], values)
                assert all(set(queries[f"{role}_t"]).isdisjoint(s) for s in role_sets)
            np.testing.assert_array_equal(queries["recall_t"], support_t)
            np.testing.assert_array_equal(queries["recall_y"], support_y)
            np.testing.assert_array_equal(queries["recall_truth"], support_truth)
            for r in record["records"]:
                artifact_path = path.parent / r["artifact"]
                assert sha(artifact_path) == r["artifact_sha256"]
                assert artifact_path.stat().st_size == r["artifact_bytes"]
                artifact = json.loads(artifact_path.read_text())
                assert artifact["decoder_sha256"] == state["source_files"]["core.py"]
                assert artifact["support_digest"] == core.digest({"t": support_t.tolist(), "y": support_y.tolist()})
                for model in artifact.get("models", []):
                    a = core.design(model["spec"], support_t)
                    # Independently solve an augmented least-squares system rather than normal equations.
                    augmented = np.vstack((a / noise, np.eye(a.shape[1]) / 2))
                    targets = np.concatenate((support_y.ravel() / noise, np.zeros(a.shape[1])))
                    reference = np.linalg.lstsq(augmented, targets, rcond=None)[0]
                    np.testing.assert_allclose(model["mean"], reference, rtol=3e-6, atol=2e-7)
                    covariance = np.asarray(model["covariance"])
                    assert np.linalg.eigvalsh(covariance).min() >= -1e-6 * max(1., np.linalg.norm(covariance))
                    posterior_checks += 1
                with np.load(path.parent / f"{r['method']}-predictions.npz", allow_pickle=False) as raw:
                    assert set(r["evaluation"]) == {"interpolation", "withheld_arc", "extrapolation", "recall"}
                    for role in r["evaluation"]:
                        prediction = core.predict(artifact, queries[f"{role}_t"])
                        for key, value in prediction.items():
                            saved = raw[f"{role}_{key}"]
                            if value.dtype == bool:
                                np.testing.assert_array_equal(value, saved)
                            else:
                                difference = float(np.max(abs(value - saved)))
                                maximum_prediction_difference = max(maximum_prediction_difference, difference)
                                np.testing.assert_allclose(value, saved, rtol=1e-12, atol=1e-12)
                        score = core.metrics(prediction, queries[f"{role}_truth"], queries[f"{role}_y"])
                        for key, value in score.items():
                            maximum_metric_difference = max(maximum_metric_difference, abs(value-r["evaluation"][role][key]))
                            assert np.isclose(value, r["evaluation"][role][key], rtol=1e-12, atol=1e-12)
                        errors = prediction["mean"].ravel() - queries[f"{role}_truth"].ravel()
                        independent_mse = float(np.dot(errors, errors) / errors.size)
                        assert np.isclose(independent_mse, score["mse"], rtol=1e-12, atol=1e-12)
                        independent_coverage = np.count_nonzero(
                            np.square(prediction["mean"] - queries[f"{role}_y"])
                            <= 3.8414588206941254 * prediction["variance"]) / prediction["mean"].size
                        assert independent_coverage == score["coverage95"]
                        partitions += 1
                assert sha(artifact_path) == r["artifact_sha256"]
                models += 1
    assert seen == expected_cases
    report = {"status": "PASS", "cases": len(seen), "models": models, "query_partitions": partitions,
              "independent_augmented_least_squares_checks": posterior_checks,
              "maximum_prediction_difference": maximum_prediction_difference,
              "maximum_metric_difference": maximum_metric_difference,
              "source_sha256": state["contract"]["source_sha256"],
              "audit_sha256": sha(Path(__file__)), "numeric_atol": 1e-12, "numeric_rtol": 1e-12,
              "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - start,
              "checks": ["complete frozen case/method coverage", "all source/artifact/raw hashes",
                         "all split banks regenerated", "role separation and nested support",
                         "fresh decoded prediction vectors", "independent MSE and coverage",
                         "independent least-squares posterior means", "no artifact mutation"],
              "boundary": "Same-source prediction replay with independent algebraic/metric checks, not an independent implementation of every generator and simulator."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
