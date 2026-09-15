"""Independent saved-data/checkpoint audit; no training or final-bank resampling."""

import argparse
import gzip
import json

import numpy as np
import torch

from sera.contracts import EvidenceKind, Observation, Provenance
from sera.shared_archive import load_shared_checkpoint

from .study import CHECKPOINT, PARENT_SHA, RELEASE, ROOT, check_contract, parent, read, sha, write


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def observations(row):
    result = []
    for event in row["observations"]:
        provenance = event["provenance"]
        result.append(Observation(event["modality"], tuple(event["values"]), event["position"],
                                  Provenance(provenance["source"], provenance["record_id"], EvidenceKind(provenance["kind"])),
                                  units=event["units"], scale=event["scale"],
                                  available=None if event["available"] is None else tuple(event["available"])))
    return tuple(result)


def geometric_target(row):
    points = np.asarray([o["values"] for o in row["observations"]])
    # Solve the circumcenter, then rotate the last displacement by the observed
    # angle. This independently checks the circle recurrence, not generator code.
    center = np.linalg.solve(2*(points[1:]-points[0]), (points[1:]**2).sum(-1)-(points[0]**2).sum())
    u, v = points[0]-center, points[1]-center
    cosine = float(u@v/(u@u))
    sine = float((u[0]*v[1]-u[1]*v[0])/(u@u))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return points[2] + rotation@(points[2]-points[1])


@torch.no_grad()
def checkpoint_predictions(model, rows):
    result = []
    for start in range(0, len(rows), 64):
        batch = rows[start:start+64]
        result.extend(model.forward_typed([observations(r) for r in batch], ["motion"]*len(batch))["numeric"].tolist())
    return np.asarray(result)


def audit(name):
    directory = RELEASE/name
    cfg = check_contract(directory)
    reference = parent().components["r1"]
    source_state = reference.state_dict()
    results = {arm: [] for arm in cfg["arms"]}
    differences = {"prediction": 0.0, "summary_mse": 0.0, "circle_label": 0.0,
                   "retention": 0.0, "paired_training_state": 0}
    cases_checked = 0
    for seed in cfg["seeds"]:
        seed_dir = directory/f"seed-{seed}"
        bank = unpack(seed_dir/"bank.json.gz")
        baseline = unpack(seed_dir/"parent-retention.json.gz")
        for family in ("final", "extent"):
            for row in bank[family]:
                error = float(np.max(np.abs(geometric_target(row)-row["target"])))
                differences["circle_label"] = max(differences["circle_label"], error)
                if error > 1e-10:
                    raise ValueError("Saved final target violates declared circular motion")
        fitted = {}
        for arm in cfg["arms"]:
            path = seed_dir/arm
            result = read(path/"result.json")
            if result["status"] != "COMPLETE" or result["pilot"]:
                raise ValueError("Incomplete or nonconfirmatory result")
            if result["raw_sha256"] != sha(path/"raw.json.gz") or result["bank_sha256"] != sha(seed_dir/"bank.json.gz"):
                raise ValueError("Saved evidence bytes changed")
            selected = ROOT/result["selected_checkpoint"]["path"]
            if sha(selected) != result["selected_checkpoint"]["sha256"]:
                raise ValueError("Selected model bytes changed")
            for checkpoint in result["resumable_checkpoints"]:
                if sha(ROOT/checkpoint["path"]) != checkpoint["sha256"]:
                    raise ValueError("Resumable state bytes changed")
            model = load_shared_checkpoint(selected)
            changed = [n for n, x in model.state_dict().items() if not torch.equal(x, source_state[n])]
            if sorted(changed) != sorted(result["training"]["changed_tensors"]) or not changed:
                raise ValueError("Claimed numerical learning is inaccurate")
            if cfg["config"]["scope"] == "readout" and any(not n.startswith("typed_numeric.") for n in changed):
                raise ValueError("Protected shared state changed")
            if any(n.startswith("generator_") for n in changed):
                raise ValueError("Acquired generator state changed")
            raw = unpack(path/"raw.json.gz")
            for family in ("final", "extent"):
                prediction = checkpoint_predictions(model, bank[family])
                saved = np.asarray(raw["motion"][family]["predictions"])
                prediction_error = float(np.max(np.abs(prediction-saved)))
                differences["prediction"] = max(differences["prediction"], prediction_error)
                if prediction_error != 0:
                    raise ValueError("Restored checkpoint predictions changed")
                mse = np.square(saved-np.asarray([r["target"] for r in bank[family]])).mean(-1)
                if not np.allclose(mse, raw["motion"][family]["mse"], rtol=0, atol=1e-15):
                    raise ValueError("Per-world errors do not match saved answers")
                error = abs(float(mse.mean())-result["metrics"][family]["mse"])
                differences["summary_mse"] = max(differences["summary_mse"], error)
                if error > 1e-14:
                    raise ValueError("Motion summary does not match raw evidence")
                cases_checked += len(mse)
            retained = raw["retention"]
            if len(retained["groups"]) != 61:
                raise ValueError("Incomplete retained capabilities")
            for group, values in retained["scores"]["capabilities"].items():
                expected = float(np.mean(values))
                if abs(expected-retained["groups"][group]) > 1e-14:
                    raise ValueError("Retained score mean differs from raw answers")
            for group, value in retained["groups"].items():
                drop = baseline["groups"][group]-value
                error = abs(drop-result["retention_drops"][group])
                differences["retention"] = max(differences["retention"], error)
                if error > 1e-14:
                    raise ValueError("Retention drop differs from saved controls")
            fitted[arm] = result
            results[arm].append(result)
        # Exact-gradient controls must agree even though their owner topology differs.
        for left, right in (("integrated", "detached"), ("observed_only", "extra_capacity")):
            if fitted[left]["training"]["state_sha256"] != fitted[right]["training"]["state_sha256"]:
                raise ValueError("Matched detached/capacity learning diverged")
            if fitted[left]["training"]["work"] != fitted[right]["training"]["work"]:
                raise ValueError("Matched computation paths differ")
        for arm in cfg["arms"]:
            expected = arm not in {"detached", "extra_capacity"}
            changed_owner = fitted[arm]["original_owner_sha256"] != fitted[arm]["evaluated_owner_sha256"]
            if expected != changed_owner:
                raise ValueError("Shared ownership claim does not match result")
    if sha(CHECKPOINT) != PARENT_SHA:
        raise ValueError("Original parent checkpoint changed")
    aggregate = {}
    for arm, values in results.items():
        mse = [r["metrics"]["final"]["mse"] for r in values]
        aggregate[arm] = {"mean_mse": float(np.mean(mse)), "rmse": float(np.sqrt(np.mean(mse))),
                          "per_stream_mse": mse,
                          "extent_mean_mse": float(np.mean([r["metrics"]["extent"]["mse"] for r in values])),
                          "maximum_retention_drop": max(r["maximum_retention_drop"] for r in values),
                          "training_seconds": sum(r["training"]["training_seconds"] for r in values),
                          "worker_seconds": sum(r["total_worker_seconds"] for r in values),
                          "checkpoint_bytes": sum(r["checkpoint_bytes"] for r in values),
                          "trainable_parameters": values[0]["training"]["trainable_parameters"],
                          "extra_owner_parameters": values[0]["training"]["extra_owner_parameters"]}
    integrated = np.asarray(aggregate["integrated"]["per_stream_mse"])
    observed = np.asarray(aggregate["observed_only"]["per_stream_mse"])
    gains = observed-integrated
    rng = np.random.default_rng(17060915)
    bootstrap = gains[rng.integers(0, len(gains), (20000, len(gains)))].mean(-1)
    interval = np.quantile(bootstrap, [.0125, .9875]).tolist()
    relative = float(gains.mean()/observed.mean())
    gate = {"all_streams_improve": bool((gains > 0).all()), "relative_gain_at_least_10_percent": relative >= .10,
            "paired_stream_interval_positive": interval[0] > 0,
            "retention": aggregate["integrated"]["maximum_retention_drop"] <= .02,
            "matched_detached_prediction": max(abs(integrated-np.asarray(aggregate["detached"]["per_stream_mse"]))) <= 1e-7}
    gate["conditional_component_pass"] = all(gate.values())
    summary = {"status": "PASS", "experiment": name, "aggregate": aggregate,
               "relative_mse_gain_vs_observed": relative, "paired_stream_mse_gain": gains.tolist(),
               "paired_stream_bootstrap_97_5_percent": interval, "gate": gate,
               "differences": differences, "checkpoint_prediction_cases_checked": cases_checked,
               "parent_unchanged": True, "final_training_streams": len(cfg["seeds"]),
               "inference_limit": "Five independent teaching/optimizer streams from one pretrained parent. Bootstrap is descriptive, not a universal or finite-sample guarantee.",
               "scope": "Conditional motion-route transfer only. Supplied grammar, no eta learning, no arbitrary applicability or project completion.",
               "operational_promotion": False}
    output = directory/"audit-summary.json"
    if output.exists():
        raise FileExistsError("Completed audit cannot be silently replaced")
    write(output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    audit(parser.parse_args().name)
