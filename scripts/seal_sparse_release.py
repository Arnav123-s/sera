"""Freeze existing evidence and costs; never executes a numerical experiment."""

import hashlib
import importlib.metadata
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/22_sparse_mechanisms"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8", newline="\n")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024*1024), b""):
            result.update(piece)
    return result.hexdigest()


def descriptor(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}


def main():
    if (RELEASE/"release-manifest.json").exists():
        raise FileExistsError("Release already sealed; do not overwrite it")
    if (ROOT/"runs/v3-batch-001/active.lock").exists():
        raise ValueError("A numerical job is active; seal after its charge is recorded")
    before, after = read(RELEASE/"budget-before.json"), read(ROOT/"runs/v3-batch-001/budget.json")
    new_jobs = {key: value for key, value in after["jobs"].items() if key not in before["jobs"]}
    own_jobs = {key: value for key, value in new_jobs.items() if key.startswith("runs/CS-")}
    total = sum(value["charged_seconds"] for value in own_jobs.values())
    receipts, maximum_memory = [], 0
    with zipfile.ZipFile(RELEASE/"supervision.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for job, ledger in sorted(own_jobs.items()):
            receipt = read(ROOT/job/"state.json")
            assert receipt["status"] == ledger["status"]
            assert abs(receipt["charged_seconds"]-ledger["charged_seconds"]) < 1e-8
            assert receipt["worker_threads"] == 1 and receipt["memory_limit_bytes"] == 2*1024**3
            assert sha(ROOT/job/"process.log") == receipt["log_sha256"]
            assert receipt["status"] == "PASS" or job == "runs/CS-repair-cli-expected-reject-001"
            maximum_memory = max(maximum_memory, receipt["peak_job_committed_bytes"])
            receipts.append({"job": job, **ledger, "state_sha256": sha(ROOT/job/"state.json"),
                             "expected_rejection": job == "runs/CS-repair-cli-expected-reject-001"})
            for filename in ("state.json", "process.log"):
                archive.write(ROOT/job/filename, job+"/"+filename)
    assert abs(before["remaining_seconds"]-after["remaining_seconds"]-sum(v["charged_seconds"] for v in new_jobs.values())) < 1e-7
    costs = {"status": "RECONCILED", "opening_remaining_seconds": before["remaining_seconds"],
             "cycle_charged_seconds": total, "cycle_charged_minutes": total/60,
             "closing_remaining_seconds": after["remaining_seconds"], "closing_remaining_minutes": after["remaining_seconds"]/60,
             "owned_jobs": receipts, "other_concurrent_charges": {k: v for k, v in new_jobs.items() if k not in own_jobs},
             "peak_job_committed_bytes": maximum_memory, "memory_cap_bytes": 2*1024**3, "worker_threads": 1,
             "boundary": "Supervised job wall time includes model work, imports, tests, audits, failure/refusal, serialization and report rendering. Source reading, web research, documentation, static checks and file hashing are outside this numerical allowance; this is not total project labor or CPU-instruction accounting."}
    write(RELEASE/"costs.json", costs)
    write(RELEASE/"budget-after.json", after)
    tests, occurrences = set(), 0
    reports = []
    for path in sorted(RELEASE.glob("*.xml")):
        root = ET.parse(path).getroot()
        cases = list(root.iter("testcase"))
        assert cases and not list(root.iter("failure")) and not list(root.iter("error"))
        occurrences += len(cases)
        tests.update((case.get("classname"), case.get("name")) for case in cases)
        reports.append({**descriptor(path), "test_executions": len(cases)})
    write(RELEASE/"test-summary.json", {"status": "PASS", "distinct_targeted_tests": len(tests),
                                       "test_executions": occurrences, "reports": reports,
                                       "scope": "New affected contracts only. Predecessor evidence checked separately without rerunning completed training or its regression suite."})
    runtime = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "matplotlib", "pytest", "ruff")}
    write(RELEASE/"environment.json", runtime)
    paths = []
    for folder in ("src/sera", "experiments", "workbench", "scripts", "tests"):
        paths.extend((ROOT/folder).rglob("*.py"))
    paths += [ROOT/"pyproject.toml"]
    paths = sorted(set(paths))
    source_records = [descriptor(path) for path in paths]
    with zipfile.ZipFile(RELEASE/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    write(RELEASE/"source-manifest.json", {"sources": source_records,
                                         "scope": "Exact repository Python source plus dependency declaration at release; earlier per-experiment source snapshots remain authoritative for those outcomes."})
    pointer = ROOT/"runs/sera-workbench/current.json"
    current = read(pointer)
    revision = ROOT/"runs/sera-workbench/revisions"/current["revision"]
    saved = read(revision)
    assert sha(revision) == current["sha256"] and saved["owner_sha256"] == read(RELEASE/"results.json")["actual_owner"]
    application = {"owner_sha256": saved["owner_sha256"], "pointer": descriptor(pointer), "revision": descriptor(revision),
                   "language_checkpoint": saved["language"]["checkpoint"],
                   "numerical_contexts": len(saved["contexts"]), "revision_number": saved["revision"],
                   "transactions": len(saved["transactions"]), "live_mutation_by_this_cycle": False,
                   "predecessor_release": "research-continuation/21_grounded_language/release-manifest.json"}
    checkpoint = ROOT/saved["language"]["checkpoint"]["path"]
    assert sha(checkpoint) == saved["language"]["checkpoint"]["sha256"]
    write(RELEASE/"application-preservation.json", application)
    source_head = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    artifacts = [descriptor(path) for path in sorted(RELEASE.rglob("*")) if path.is_file() and path.name not in {"release-manifest.json", "verification.json"}]
    manifest = {"schema": "sera.sparse-release.1", "source_head_before_release_commit": source_head,
                "artifact_files": artifacts, "preserved_parent": "4625d8bb70520ca40be73bc0db610d0e9553b826",
                "result": read(RELEASE/"results.json"), "cost_receipt": descriptor(RELEASE/"costs.json"),
                "no_unfinished_numerical_jobs": True,
                "resume": "Completed direct solves have coefficients/arrays and frozen sources; no unfinished optimizer or experiment is claimed. Live owner resumes through release 21's existing checkpoint chain."}
    write(RELEASE/"release-manifest.json", manifest)
    print(json.dumps({"status": "SEALED", "artifact_files": len(artifacts), "source_files": len(paths),
                      "charged_seconds": total, "remaining_seconds": after["remaining_seconds"], "tests": len(tests)}))


if __name__ == "__main__":
    main()
