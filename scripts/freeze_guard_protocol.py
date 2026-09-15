"""Freeze GG-GUARD-001 only after the disjoint timing fixture, before final data."""

import gzip
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from experiments.generative_memory.applicability import FEATURES
from experiments.generative_memory.applicability_study import (
    FUTURE_FAMILIES,
    PARTITION_BASES,
    QUERY_T,
    SUPPORT_INDICES,
    SUPPORT_T,
    TRAIN_FAMILIES,
    runtime,
    source_inventory,
    write,
)
from experiments.generative_memory.core import canonical, digest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "research-continuation/15_applicability"


def main():
    if (RELEASE/"protocol.json").exists():
        raise ValueError("Protocol already frozen; retain it and declare a new cohort for amendments")
    if any((ROOT/"runs").glob("GG-GUARD-001-final*")):
        raise ValueError("Final-run directory exists before protocol freeze")
    timing = json.loads((ROOT/"runs/GG-GUARD-001-timing/timing.json").read_text())
    supervisor = json.loads((ROOT/"runs/GG-GUARD-001-timing-supervisor/state.json").read_text())
    if timing["status"] != "PASS" or supervisor["status"] != "PASS":
        raise ValueError("Timing fixture failed")
    sources = source_inventory()
    if timing["sources"] != sources:
        raise ValueError("Experiment sources changed after timing; preserve and retime the new version")
    payload = {
        "id": "GG-GUARD-001", "status": "frozen_before_final_observations", "date": "2026-09-15",
        "parent_commit": subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                                  "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "runtime": runtime(), "sources": sources,
        "train_families": list(TRAIN_FAMILIES), "future_families": list(FUTURE_FAMILIES),
        "partition_seed_bases": PARTITION_BASES,
        "worlds_per_family": {"teach": 96, "probability_calibration": 32, "threshold_selection": 32, "final": 48},
        "mechanism_instance_unit": "One distinct environment seed per episode; all its queries stay in the same partition.",
        "features": list(FEATURES), "support_grid": SUPPORT_T.tolist(),
        "support_indices": list(SUPPORT_INDICES), "query_t": QUERY_T.tolist(),
        "noise_std": .05, "correctness": "Euclidean distance of prediction to separately observed noisy outcome <= 0.2.",
        "clean_mse": "Secondary assessor-only squared coordinate error to latent mechanism; never a guard label or input.",
        "optimizer_seed": 920001, "training_steps": 500, "hidden_units": 16,
        "training": "Float64 full-batch Adam, lr 0.01, weight decay 0.001, BCE; equal queries per world.",
        "probability_calibration": "Positive-slope Platt scaling; separate worlds; 250 Adam steps, lr 0.02, penalty 0.001.",
        "threshold_selection": "Separate worlds, 101 empirical score quantiles plus reject-all; greatest coverage with <= 10% observed selective risk.",
        "controls": ["always_predict", "residual_uncertainty", "support_distance", "learned", "always_abstain"],
        "fairness": "Identical ten factual observations per world for all controls. Fixed controls see the same threshold outcomes. Extra learned-guard teaching/calibration is separately charged.",
        "primary_score": "Per-world mean(correct accepted - 4 * wrong accepted); abstentions score zero.",
        "uncertainty": "2000 world-stratified paired bootstrap replicates, seed 990001; fixed family mixture, not uncertainty over all possible families.",
        "gate": ["Each in-menu family coverage >= 0.8", "Each family with coverage >= 0.05 has observed selective risk <= 0.1",
                 "Pooled protected-family coverage >= 0.1", "Paired utility lower 95% bootstrap bound > 0 against both fixed guards"],
        "guarantees": "No conformal selective-risk guarantee; no exchangeability claim between teaching and protected families. No M2 or common-owner integration from this component alone.",
        "alias_audit": "One separate circle prefix paired with a local bump wholly outside observed support; no labels or tuning from this diagnostic.",
        "bounded_wall_allowances_seconds": {"final": 300, "replay_and_refit": 300, "independent_audit": 180},
        "timing_fixture": {"worlds": 12, "wall_seconds": timing["wall_seconds"], "supervised_wall_seconds": supervisor["charged_seconds"],
                           "peak_job_committed_bytes": supervisor["peak_job_committed_bytes"],
                           "source_sha256": hashlib.sha256((ROOT/"runs/GG-GUARD-001-timing/timing.json").read_bytes()).hexdigest(),
                           "selection_reason": "1344 worlds is 112 times the 12-world fixture. A 300s wall cap exceeds conservative linear scaling of the 1.77s fixture; includes startup and reporting margin."},
        "independent_algebra_absolute_tolerance": 2e-8,
        "same_interpreter_replay": "Exact features, means, learned probabilities, refitted model bytes, thresholds and result summaries.",
    }
    RELEASE.mkdir(parents=True, exist_ok=True)
    write(RELEASE/"protocol.json", {"payload": payload, "sha256": digest(payload)})
    with zipfile.ZipFile(RELEASE/"frozen-sources.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT/name, name)
    # Preserve all substantive prior local model/result cohorts without copying or deleting them.
    previous = []
    for folder in ("runs/SHARED-GG-001", "runs/GG-ACT-001-final", "runs/generative-memory-GG-P0-001",
                   "runs/parameter-cloud-main", "runs/sera-0.5-current"):
        if not (ROOT/folder).is_dir():
            raise ValueError(f"Missing prior cohort: {folder}")
        for path in sorted((ROOT/folder).rglob("*")):
            if path.is_file():
                raw = path.read_bytes()
                previous.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
                                 "sha256": hashlib.sha256(raw).hexdigest()})
    preserved = {"parent_commit": payload["parent_commit"], "files": previous,
                 "bytes": sum(r["bytes"] for r in previous), "file_count": len(previous)}
    with (RELEASE/"predecessor-inventory.json.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(canonical(preserved).encode())
    write(RELEASE/"timing.json", {k: timing[k] for k in ("status", "worlds", "wall_seconds", "cpu_seconds")})
    write(RELEASE/"timing-supervisor.json", supervisor)
    print(json.dumps({"protocol_sha256": digest(payload), "frozen_source_files": len(sources),
                      "preserved_files": len(previous), "preserved_bytes": preserved["bytes"]}))


if __name__ == "__main__":
    main()
