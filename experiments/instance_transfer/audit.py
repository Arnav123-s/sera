"""Audit saved instance-transfer predictions and the two planned K contrasts."""

import json

import numpy as np
import torch

from experiments.cross_route_transfer.audit import checkpoint_predictions, geometric_target, unpack
from experiments.cross_route_transfer.study import ROOT, read, sha, write
from sera.shared_archive import load_shared_checkpoint

from .study import ARMS, RELEASE, SEEDS, contract


def run():
    protocol = contract()
    results = {arm: [] for arm in ARMS}
    max_target_error, predictions_checked = 0.0, 0
    for seed in SEEDS:
        folder = RELEASE/f"seed-{seed}"
        bank = unpack(folder/"bank.json.gz")
        baseline = unpack(folder/"parent-retention.json.gz")
        for partition in ("final", "extent"):
            for row in bank[partition]:
                error = float(np.max(np.abs(geometric_target(row)-row["target"])))
                max_target_error = max(max_target_error, error)
                if error > 1e-10:
                    raise ValueError("Raw instance target violates the geometric recurrence")
        for arm in ARMS:
            cell = folder/arm
            result = read(cell/"result.json")
            if result["protocol_sha256"] != protocol["sha256"] or result["status"] != "COMPLETE" or not result["parent_unchanged"]:
                raise ValueError("Incomplete or unbound result")
            if result["source_intervention_effect"] <= 1e-10 or result["bank_sha256"] != sha(folder/"bank.json.gz"):
                raise ValueError("Lost acquired-state intervention or changed data")
            if result["raw_sha256"] != sha(cell/"raw.json.gz"):
                raise ValueError("Saved outcomes changed")
            for record in result["checkpoints"]+[result["selected_checkpoint"]]:
                if sha(ROOT/record["path"]) != record["sha256"]:
                    raise ValueError("Checkpoint changed")
            model = load_shared_checkpoint(ROOT/result["selected_checkpoint"]["path"])
            raw = unpack(cell/"raw.json.gz")
            for partition in ("final", "extent"):
                computed = checkpoint_predictions(model, bank[partition])
                recorded = np.asarray(raw["motion"][partition]["predictions"])
                if not np.array_equal(computed, recorded):
                    raise ValueError("Restored instance model changed its answers")
                mse = np.square(recorded-np.asarray([r["target"] for r in bank[partition]])).mean(-1)
                if abs(float(mse.mean())-result["metrics"][partition]["mse"]) > 1e-14:
                    raise ValueError("Instance summary differs from raw predictions")
                if not np.allclose(mse, raw["motion"][partition]["mse"], rtol=0, atol=1e-15):
                    raise ValueError("Instance per-case errors are incorrect")
                predictions_checked += len(mse)
            if len(result["retention_drops"]) != 61:
                raise ValueError("Incomplete retention coverage")
            for name, value in raw["retention"]["groups"].items():
                if abs(baseline["groups"][name]-value-result["retention_drops"][name]) > 1e-14:
                    raise ValueError("Instance retention summary is incorrect")
            if any(not n.startswith("typed_numeric.") for n in result["training"]["changed_tensors"]):
                raise ValueError("Instance training changed a protected representation")
            results[arm].append(result)
    aggregate = {}
    for arm, rows in results.items():
        values = [r["metrics"]["final"]["mse"] for r in rows]
        aggregate[arm] = {"per_stream_mse": values, "mse": float(np.mean(values)), "rmse": float(np.sqrt(np.mean(values))),
                          "extent_mse": float(np.mean([r["metrics"]["extent"]["mse"] for r in rows])),
                          "maximum_retention_drop": max(r["maximum_retention_drop"] for r in rows),
                          "training_seconds": sum(r["training"]["training_seconds"] for r in rows),
                          "worker_seconds": sum(r["worker_seconds"] for r in rows),
                          "checkpoint_bytes": sum(sum(c["bytes"] for c in r["checkpoints"])+r["selected_checkpoint"]["bytes"] for r in rows)}
    corrected = np.asarray(aggregate["corrected_teacher"]["per_stream_mse"])
    contrasts = {}
    for control in ("observed_only", "uncorrected_teacher"):
        comparator = np.asarray(aggregate[control]["per_stream_mse"])
        gains = comparator-corrected
        rng = np.random.default_rng(18060915)
        boot = gains[rng.integers(0, len(gains), (20000, len(gains)))].mean(-1)
        interval = np.quantile(boot, [.0125, .9875]).tolist()
        relative = float(gains.mean()/comparator.mean())
        contrasts[control] = {"relative_mse_gain": relative, "per_stream_mse_gain": gains.tolist(),
                              "paired_bootstrap_97_5_percent": interval,
                              "passes": bool(relative >= .10 and interval[0] > 0 and (gains > 0).all())}
    passed = all(c["passes"] for c in contrasts.values()) and aggregate["corrected_teacher"]["maximum_retention_drop"] <= .02
    summary = {"status": "PASS", "experiment": "A06-INSTANCE-001", "aggregate": aggregate,
               "contrasts": contrasts, "gate": {"causal_component_pass": passed,
                                                  "retention": aggregate["corrected_teacher"]["maximum_retention_drop"] <= .02},
               "checkpoint_predictions_checked": predictions_checked,
               "maximum_independent_geometric_target_error": max_target_error,
               "source_intervention_preserved": True,
               "scope": "One acquired physical instance; five new trajectory/optimizer streams; existing readout only. Shared representations and eta are unchanged.",
               "operational_promotion": False}
    target = RELEASE/"audit-summary.json"
    if target.exists():
        raise FileExistsError("Completed audit is immutable")
    write(target, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()
