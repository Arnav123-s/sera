"""Seal completed transfer evidence and local checkpoint inventories once."""

import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/18_instance_transfer"
NORMALIZED = ROOT/"research-continuation/17_transfer"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def row(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2)+"\n")


def archive(path, paths):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for source in sorted(paths):
            item = zipfile.ZipInfo(source.relative_to(ROOT).as_posix(), (2026, 9, 15, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(item, source.read_bytes())


def main():
    started = time.perf_counter()
    outputs = ["preservation.json", "local-artifact-inventory.json", "final-costs.json",
               "worker-budget.json", "worker-logs.zip", "source-manifest.json",
               "current-sources.zip", "release-manifest.json"]
    if any((RELEASE/name).exists() for name in outputs):
        raise ValueError("Release already or partially sealed; inspect existing evidence first")
    if (ROOT/"runs/v3-batch-001/active.lock").exists():
        raise ValueError("Owned worker active; preserve its files and wait for reconciliation")

    predecessor = read(ROOT/"research-continuation/16_v3/preservation.json")
    for name, sha in predecessor["unchanged_shared_sources"].items():
        if row(ROOT/name)["sha256"] != sha:
            raise ValueError(f"Original shared source changed: {name}")
    for expected in predecessor["predecessor_files"]:
        if row(ROOT/expected["path"]) != expected:
            raise ValueError(f"Predecessor artifact changed: {expected['path']}")
    old_release = read(ROOT/"research-continuation/16_v3/release-manifest.json")
    for expected in old_release["files"]:
        if row(ROOT/expected["path"]) != expected:
            raise ValueError(f"Previous release changed: {expected['path']}")
    kavi = ROOT/"research-continuation/00_sources/kavi-pinned"
    kavi_head = subprocess.check_output(
        ["git", "-c", f"safe.directory={kavi.as_posix()}", "-C", str(kavi), "rev-parse", "HEAD"],
        text=True).strip()
    if kavi_head != "50f743cc44794b67bc1d30b927e25991197edce2":
        raise ValueError("Kavi comparison pin changed")
    write(RELEASE/"preservation.json", {
        "status": "PASS", "starting_head": "0c5d018f46225d3261c6f6b35c53c014014edeae",
        "kavi_head": kavi_head, "unchanged_shared_sources": predecessor["unchanged_shared_sources"],
        "predecessor_files": predecessor["predecessor_files"],
        "previous_release_files_verified": len(old_release["files"]),
        "previous_manifest": row(ROOT/"research-continuation/16_v3/release-manifest.json"),
        "completed_experiments_restarted": 0,
        "scope": "Current original files compared with their previously sealed bytes; no refitting or replay of old cohorts."})

    local_files = sorted(p for name in ("A06-transfer", "A06-instance-transfer", "A06-integrated-001")
                         for p in (ROOT/"runs"/name).rglob("*") if p.is_file())
    inventory = [row(p) for p in local_files]
    write(RELEASE/"local-artifact-inventory.json", {
        "schema": "sera.transfer-local-artifacts.1", "files": inventory,
        "bytes": sum(r["bytes"] for r in inventory),
        "storage": "Preserved local ignored runs; this inventory does not embed trained weights in Git.",
        "continuation": "Immutable parent plus deltas, optimizer/RNG/best state; completed training cells cannot restart.",
        "shared_parent": row(ROOT/"runs/A06-transfer/parent.pt")})

    budget = read(ROOT/"runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in budget["jobs"].items() if "/A06-" in k}
    states = {k: read(ROOT/k/"state.json") for k in jobs}
    logs = []
    for key, state in states.items():
        log = ROOT/key/"process.log"
        if row(log)["sha256"] != state["log_sha256"]:
            raise ValueError(f"Changed resource log: {key}")
        if state["status"] != jobs[key]["status"] or state["charged_seconds"] != jobs[key]["charged_seconds"]:
            raise ValueError(f"Resource ledger differs: {key}")
        logs.extend([ROOT/key/"state.json", log])
    archive(RELEASE/"worker-logs.zip", logs)
    write(RELEASE/"worker-budget.json", budget)
    write(RELEASE/"final-costs.json", {
        "schema": "sera.transfer-complete-worker-costs.1", "jobs": states,
        "worker_job_count": len(jobs), "charged_worker_seconds": sum(v["charged_seconds"] for v in jobs.values()),
        "cumulative_prior_and_current_worker_seconds": budget["initial_seconds"]-budget["remaining_seconds"],
        "remaining_allowance_seconds": budget["remaining_seconds"],
        "peak_job_committed_bytes": max(v["peak_job_committed_bytes"] for v in states.values()),
        "failed_execution_jobs": [k for k, v in jobs.items() if v["status"] != "PASS"],
        "scientific_rejections": ["A06-PILOT-001", "A06-PILOT-002", "A06-TRANSFER-001 retention", "normalized acquired-coefficient attribution"],
        "historical_source_acquisition": "Original 16 observations plus 1 paid sensor recheck retained in SHARED-GG-001; its historical costs are not zero or included twice here.",
        "local_artifact_bytes": sum(r["bytes"] for r in inventory),
        "boundary": "Supervised wall time includes imports, all failures, pilots, training, model/source saves, evaluation, independent audits, integration, 210 regressions and three read-only CLI checks. Interactive research, engineering, file reads and lint are outside worker time; release-verification receipt records its own elapsed time. Exact CPU/FLOPs and interactive time are not measured. No paid compute.",
        "log_archive": row(RELEASE/"worker-logs.zip"),
        "prior_snapshot": "../17_transfer/costs.json preserves the earlier report snapshot; this adds all three CLI jobs.",
        "sealing_elapsed_seconds_before_source_archive": time.perf_counter()-started})

    sources = {ROOT/name for name in read(RELEASE/"protocol.json")["payload"]["source_sha256"]}
    for folder in ("acquisition_dependence", "audit_compatibility", "cross_route_transfer", "instance_transfer", "transfer_integration"):
        sources.update((ROOT/"experiments"/folder).glob("*.py"))
    sources.add(ROOT/"experiments/transfer_postflight.py")
    scripts = ["finalize_instance_report.py", "finalize_transfer_report.py", "resume_transfer.py",
               "run_audit_compat_bounded.py", "run_dependence_bounded.py", "run_instance_batch.ps1",
               "run_instance_bounded.py", "run_integration_bounded.py", "run_postflight_bounded.py",
               "run_transfer_batch.ps1", "run_transfer_bounded.py", "seal_transfer_release.py",
               "verify_transfer_release.py"]
    sources.update(ROOT/"scripts"/name for name in scripts)
    sources.update(ROOT/"tests"/name for name in ("conftest.py", "test_acquisition_dependence.py",
                   "test_audit_compatibility.py", "test_cross_route_transfer.py",
                   "test_instance_transfer.py", "test_transfer_integration.py"))
    write(RELEASE/"source-manifest.json", {p.relative_to(ROOT).as_posix(): row(p)["sha256"] for p in sorted(sources)})
    archive(RELEASE/"current-sources.zip", sources)
    files = set(sources)
    for folder in (RELEASE, NORMALIZED):
        files.update(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    write(RELEASE/"release-manifest.json", {
        "schema": "sera.transfer-release.1", "files": [row(p) for p in sorted(files)],
        "scope": "Frozen numerical protocols, failures, raw compact results, public source and local checkpoint inventory. Manifest excludes itself and the subsequent read-only verification receipt; trained weights remain local."})
    print(json.dumps({"status": "SEALED", "release_files": len(files), "local_artifacts": len(inventory),
                      "worker_seconds": sum(v["charged_seconds"] for v in jobs.values()),
                      "remaining_worker_seconds": budget["remaining_seconds"],
                      "sealing_elapsed_seconds": time.perf_counter()-started}))


if __name__ == "__main__":
    main()
