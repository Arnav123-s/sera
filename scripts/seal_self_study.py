"""Package completed local evidence without rerunning learning or replacing archives."""
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/27_self_study"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def archive(name, files):
    files = sorted(set(files))
    destination = OUT / name
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in files:
            zipped.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(destination) as zipped:
        assert zipped.testzip() is None
        for path in files:
            assert hashlib.sha256(zipped.read(path.relative_to(ROOT).as_posix())).hexdigest() == sha(path)
    return {**record(destination), "members": [record(path) for path in files]}


def main():
    if (ROOT / "runs/v3-batch-001/active.lock").exists():
        raise ValueError("Wait for the owned numerical worker to settle before sealing costs")
    if (OUT / "archive-manifest.json").exists():
        raise ValueError("Completed release already sealed; preserve it and use a new revision")
    now = datetime.now(timezone.utc).isoformat()
    budget = read(ROOT / "runs/v3-batch-001/budget.json")
    before = read(OUT / "budget-before.json")
    jobs, logs = {}, []
    for key in budget["jobs"]:
        path = ROOT / key / "state.json"
        if not path.exists():
            continue
        state = read(path)
        ours = key.startswith("runs/SS-") or (key not in before["jobs"] and
                any("experiments.self_study." in arg for arg in state.get("argv", [])))
        if ours:
            assert state["status"] not in {"RUNNING", "RESERVED"}
            jobs[key] = state
            logs.extend((path, path.with_name("process.log")))
    costs = {"schema": "sera.self-study.costs.1", "sealed_utc": now, "jobs": jobs,
             "job_count": len(jobs), "charged_seconds": sum(j["charged_seconds"] for j in jobs.values()),
             "failed_seconds": sum(j["charged_seconds"] for j in jobs.values() if j["status"] != "PASS"),
             "peak_job_committed_bytes": max(j["peak_job_committed_bytes"] for j in jobs.values()),
             "remaining_seconds_at_seal": budget["remaining_seconds"], "cpu_threads": 1,
             "worker_memory_cap_bytes": 2147483648, "paid_services": False,
             "source_acquisitions": [json.loads(line) for line in
                                     (ROOT / "runs/SS-sources/acquisitions.jsonl").read_text().splitlines()],
             "language_preparation_seconds": read(OUT / "language-sources.json")["seconds"],
             "unmetered": "Static engineering, Git, filesystem auditing and source-review time was not numerically supervised; not counted as zero.",
             "boundary": "Numerical costs are supervised elapsed wall time; peak memory is process-tree committed memory, not RSS. Failures remain charged."}
    write(OUT / "costs.json", costs)
    write(OUT / "resource-budget-at-seal.json", budget)
    write(OUT / "v11-intake.json", read(ROOT / "research/intake/v11-20260917/intake.json"))

    preserved = {}
    for name, expected in read(OUT / "reconciliation.json")["preserved"].items():
        if name == "runs/v3-batch-001/budget.json":
            continue
        actual = record(ROOT / name)
        assert actual["sha256"] == expected["sha256"], name
        preserved[name] = actual
    preceding = read(ROOT / "research-continuation/26_stream_curriculum/release-manifest.json")
    prior_matches, prior_differences = [], []
    for expected in preceding["artifacts"]:
        path = ROOT / expected["path"]
        target = prior_matches if path.exists() and sha(path) == expected["sha256"] else prior_differences
        target.append(expected["path"])
    # Some phase26 documentation was already updated after its original seal.
    # Verify any such tracked bytes against this continuation's reconciled HEAD.
    reconciled = read(OUT / "reconciliation.json")["published_head"]
    baseline_checks = {}
    for name in prior_differences:
        original = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                   "show", f"{reconciled}:{name}"], cwd=ROOT, capture_output=True, check=True).stdout
        baseline_checks[name] = hashlib.sha256(original).hexdigest() == sha(ROOT / name)
        assert baseline_checks[name], name
    write(OUT / "preservation.json", {"utc": now, "protected_live_files": preserved,
          "prior_release_original_seal_matches": len(prior_matches),
          "prior_release_already_newer_at_reconciliation": baseline_checks,
          "live_descendant": "runs/sera-study-live", "deleted_files": [],
          "reset_repositories": [], "other_jobs_stopped": [], "paid_services": False})

    runs = [ROOT / "runs" / name for name in ("SS-lang-replay-2711", "SS-lang-replay-2711-r1",
            "SS-lang-naive-2711", "SS-lang-replay-2712", "SS-lang-naive-2712", "SS-rehearsal-2721")]
    local_files = [path for directory in runs for path in directory.rglob("*") if path.is_file()]
    local_files.extend(path for path in (ROOT / "runs/sera-study-live").rglob("*") if path.is_file())
    local_files.extend(path for path in (ROOT / "runs/SS-sources").glob("*") if path.is_file())
    write(OUT / "local-state-manifest.json", {"utc": now, "files": [record(p) for p in sorted(local_files)],
          "note": "All intermediate checkpoints and cached sources remain local; terminal states and source receipts have separate public archives."})
    checkpoints = [ROOT / "runs/SS-lang-replay-2711/step-000100.pt"]
    research_files = []
    for directory in runs:
        research_files.extend(directory.glob("*.json"))
        result = directory / "result.json"
        if result.exists():
            item = read(result)["checkpoint"]
            path = ROOT / item["path"]
            assert sha(path) == item["sha256"]
            checkpoints.append(path)
    source_files = list((ROOT / "experiments/self_study").glob("*.py"))
    source_files.extend(ROOT / name for name in ("scripts/run_self_study_bounded.py", "scripts/sera_study.py",
                         "scripts/sera_sources.py", "scripts/seal_self_study.py", "tests/test_self_study.py",
                         "tests/test_self_study_lifecycle.py", "pyproject.toml"))
    source_files.extend(OUT / name for name in ("protocol.md", "correction-protocol.md", "retention-repair-protocol.md"))
    data_files = list((ROOT / "runs/SS-language-data").glob("*.jsonl"))
    data_files.extend(ROOT / "runs/SS-language-data" / name for name in ("LICENSE.txt", "manifest.json"))
    live_files = [path for path in (ROOT / "runs/sera-study-live").rglob("*") if path.is_file()]
    receipt_files = list((ROOT / "runs/SS-sources").glob("*.json"))
    receipt_files.append(ROOT / "runs/SS-sources/acquisitions.jsonl")
    archive_records = [archive("checkpoints.zip", checkpoints + research_files), archive("data.zip", data_files),
                       archive("sources.zip", source_files), archive("supervision.zip", logs + receipt_files),
                       archive("live-state.zip", live_files)]
    write(OUT / "archive-manifest.json", {"utc": now, "archives": archive_records})
    migration = read(OUT / "metadata-migration/migration.json")
    write(OUT / "RESEARCH_STATE.json", {"schema": "sera.self-study.state.1", "utc": now,
          "status": "INTEGRATED_AND_VERIFIED", "completed": ["v11-replay", "SS-001", "SS-002", "SS-003", "SS-004"],
          "owner": migration["snapshot"]["owner"], "runtime_source": migration["current_source"],
          "checkpoint": migration["snapshot"]["request_update"], "live_store": "runs/sera-study-live",
          "open_parent_goals": ["RH"], "final_partitions": "Closed to further tuning; saved predictions may be independently recounted.",
          "background_jobs": [], "next_action": ".venv/Scripts/python.exe scripts/sera_study.py motion --coefficients 2,3,1 --time 3 --position 5 --velocity=-1",
          "next_research": "Freeze a fresh multilingual request-to-local-task transition curriculum; preserve the applicability gate and compare verified correction/replay controls."})
    write(OUT / "EVIDENCE_LEDGER.json", {"schema": "sera.self-study.evidence.1", "claims": [
          {"id": "SS-001", "claim": "Learned source operators:256/256 new polynomial and64/64 motion results",
           "evidence": "evaluation-001/summary.json", "controls": ["read_only", "unchecked_control", "installed_symbolic_reference"]},
          {"id": "SS-002", "claim": "Two paired sequential multilingual lifetimes and retained failures",
           "evidence": "language-evaluation/summary.json", "source_scope": "3600 new-language examples; parallel semantic IDs can overlap earlier English learning"},
          {"id": "SS-004", "claim": "Prospective larger rehearsal restored English development to126/160 while admitting new-language learning",
           "evidence": "language-evaluation/selection.json", "boundary": "Larger rehearsal resource; not storage-matched to SS-002"},
          {"id": "SS-003", "claim": "Unique correction among96 candidates, then128/128 new transfer results",
           "evidence": "correction-final/result.json", "boundary": "Supplied bounded proposer/checkers, retained operator update learned"},
          {"id": "integration", "claim": "Same owner, source-free reuse, exact restore, protected tensors and unchanged parent statement",
           "evidence": "integration/integration.json", "metadata_correction": "metadata-migration/migration.json"},
          {"id": "audit", "claim": "Independent recount of2400 language and1088 mathematical outcomes",
           "evidence": "independent-audit/recount.json"}]})
    print(json.dumps({"archives": [(p["path"], p["bytes"]) for p in archive_records],
                      "jobs": len(jobs), "seconds": costs["charged_seconds"],
                      "remaining_seconds": costs["remaining_seconds_at_seal"],
                      "prior_files_verified": len(prior_matches) + len(prior_differences)}, indent=2))


if __name__ == "__main__":
    main()
