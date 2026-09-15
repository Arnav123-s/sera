"""Independent completeness and batch-Gaussian audit of a frozen ACT cohort.

No acquisition, posterior, decoder or evaluator implementation is imported. This
complements exact procedural replay with supplied-formula linear algebra and
explicit expected cohort coverage. It is not an independent implementation of EIG.
"""

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def matrix(kind, rows):
    result = []
    for row in rows:
        t = row["action"]["t"]
        s, c = math.sin(math.pi*t), math.cos(math.pi*t)
        if kind == 0:
            a = [[1, 0, c, -s], [0, 1, s, c]]
        elif kind == 1:  # ACT ellipse parameter order: interleaved 1, sin, cos
            a = [[1, 0, s, 0, c, 0], [0, 1, 0, s, 0, c]]
        elif kind == 2:
            a = [[1, 0, t, 0], [0, 1, 0, t]]
        else:
            a = [[1, 0, c, -s, t*c, -t*s], [0, 1, s, c, t*s, t*c]]
        a = np.asarray(a, dtype=float)
        result.extend(a if row["action"]["channel"] == "target" else np.zeros_like(a))
    return np.asarray(result)


def audit(root):
    started, cpu = time.perf_counter(), time.process_time()
    manifest, protocol = read(root / "manifest.json"), read(root / "protocol.json")
    phase = manifest["phase"]
    config = protocol[phase]
    assert manifest["status"] == "complete"
    expected = {f"cases/{phase}-{seed}-{family}/{policy}.json"
                for seed, family, policy in itertools.product(config["environment_seeds"],
                                                             config["families"], protocol["policies"])}
    names = [r["path"] for r in manifest["records"]]
    assert len(names) == len(set(names)) and set(names) == expected and expected
    artifacts = manifest["artifact_files"]
    assert expected <= set(artifacts)
    for name, info in artifacts.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve()) and path.is_file()
        assert path.stat().st_size == info["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
    peaks = {"mean": 0., "covariance": 0., "log_class_weight": 0.}
    models_checked, steps_checked = 0, 0
    for item in manifest["records"]:
        record = read(root / item["path"])
        assert record["status"] == "complete" and item["status"] == "complete"
        assert item["observations"] == record["observation_count"] == config["observation_budget"]
        assert len(record["trace"]) == config["observation_budget"]
        assert record["case"] == Path(item["path"]).parent.name
        assert record["policy"] == Path(item["path"]).stem
        assert record["phase"] == phase
        prefix = record["trace"][:len(config["initial_indices"])]
        assert [s["action"]["index"] for s in prefix] == config["initial_indices"]
        assert len({s["action"]["index"] for s in record["trace"]}) == config["observation_budget"]
        world = read((root / item["path"]).with_name("assessor-only.json"))
        for index, step in enumerate(record["trace"]):
            rows = record["trace"][:index+1]
            observations = [s["observed_event"] for s in rows]
            assert [e["sequence"] for e in observations] == list(range(index+1))
            assert step["observed_event"]["values"][3:] == world["potential_observation_pairs"][str(step["action"]["index"])]
            assert step["hypothetical_before_observation"]["learning_eligible"] is False
            y = np.asarray([e["values"][3:] for e in observations]).ravel()
            noise_variances = np.repeat([r["action"]["variance"] for r in rows], 2)
            whitened_y = y / np.sqrt(noise_variances)
            log_evidence = []
            actual = step["posterior"]["payload"]
            for kind, model in enumerate(actual["models"]):
                a = matrix(kind, rows) / np.sqrt(noise_variances[:, None])
                dimension = a.shape[1]
                prior = protocol["prior_variance"]
                augmented = np.vstack((a, np.eye(dimension) / math.sqrt(prior)))
                target = np.concatenate((whitened_y, np.zeros(dimension)))
                mean = np.linalg.lstsq(augmented, target, rcond=None)[0]
                covariance = np.linalg.solve(augmented.T @ augmented, np.eye(dimension))
                # Stable observation-space eigensystem, including its orthogonal complement.
                u, singular, _ = np.linalg.svd(a, full_matrices=False)
                projected = u.T @ whitened_y
                residual = whitened_y - u @ projected
                quadratic = residual @ residual + np.sum(projected**2 / (1+prior*singular**2))
                logdet = np.log(noise_variances).sum() + np.log1p(prior*singular**2).sum()
                log_evidence.append(-.5*(len(y)*math.log(2*math.pi) + logdet + quadratic))
                for name, expected_value, atol in (("mean", mean, 1e-8), ("covariance", covariance, 1e-9)):
                    peaks[name] = max(peaks[name], float(np.max(np.abs(np.asarray(model[name])-expected_value))))
                    np.testing.assert_allclose(model[name], expected_value, atol=atol, rtol=1e-9)
                models_checked += 1
            logw = np.asarray(log_evidence) + np.log(protocol["model"]["class_prior_masses"])
            logw -= np.max(logw) + math.log(np.exp(logw-np.max(logw)).sum())
            peaks["log_class_weight"] = max(peaks["log_class_weight"],
                                             float(np.max(np.abs(np.asarray(actual["log_class_weights"])-logw))))
            np.testing.assert_allclose(actual["log_class_weights"], logw, atol=1e-7, rtol=1e-9)
            steps_checked += 1
    assert steps_checked == len(expected) * config["observation_budget"]
    return {"status": "PASS", "complete_policy_records": len(expected), "observation_steps": steps_checked,
            "independent_batch_gaussians": models_checked, "artifact_files_verified": len(artifacts),
            "max_absolute_differences": peaks, "tolerances": {"mean_atol": 1e-8, "covariance_atol": 1e-9,
                                                               "log_class_weight_atol": 1e-7, "rtol": 1e-9},
            "protocol_sha256": manifest["protocol_sha256"], "manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
            "boundary": "Independent expected cohort coverage, archived file bytes/hashes, event ordering/budgets, "
                        "augmented least squares coefficient means, posterior covariances and SVD class evidence. "
                        "Exact procedural replay separately verifies EIG scores and query metrics. "
                        "This audit does not establish an empirical advantage or learned applicability."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new audit output path")
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
