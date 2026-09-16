"""One-time static manifest/cost packaging after completed numerical work."""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/23_task_transfer"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8", newline="\n")


def main():
    if (RELEASE/"release-manifest.json").exists():
        raise ValueError("Release is already sealed; preserve it and create a new version")
    before = read(RELEASE/"budget-before.json")
    after_path = ROOT/"runs/v3-batch-001/budget.json"
    after = read(after_path)
    (RELEASE/"budget-after.json").write_bytes(after_path.read_bytes())
    new_jobs = set(after["jobs"])-set(before["jobs"])
    owned = sorted(k for k in new_jobs if k.startswith(("runs/TT-", "runs/task-command-")))
    records = []
    with zipfile.ZipFile(RELEASE/"supervision.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for job in owned:
            state_path = ROOT/job/"state.json"
            state = read(state_path)
            assert state["status"] == after["jobs"][job]["status"] == "PASS"
            assert sha(ROOT/job/"process.log") == state["log_sha256"]
            records.append({"job": job, "status": state["status"], "charged_seconds": state["charged_seconds"],
                            "state_sha256": sha(state_path), "peak_committed_bytes": state["peak_job_committed_bytes"]})
            for name in ("state.json", "process.log"):
                z.write(ROOT/job/name, job+"/"+name)
    charged = sum(r["charged_seconds"] for r in records)
    other = sum(after["jobs"][k]["charged_seconds"] for k in new_jobs-set(owned))
    assert abs(before["remaining_seconds"]-after["remaining_seconds"]-charged-other) < 1e-7
    costs = {"status": "RECONCILED", "opening_remaining_seconds": before["remaining_seconds"],
             "cycle_charged_seconds": charged, "cycle_charged_minutes": charged/60,
             "closing_remaining_seconds": after["remaining_seconds"], "closing_remaining_minutes": after["remaining_seconds"]/60,
             "other_concurrent_job_seconds": other, "owned_jobs": records,
             "maximum_job_committed_bytes": max(r["peak_committed_bytes"] for r in records),
             "scope": "Every numerical study, failed candidate inside a study, audit, test, migration, figure render and actual task command is included. Literature reading, implementation, static inspection and hashing require additional elapsed work outside this worker allowance. No new grant, paid compute or remote operation."}
    write(RELEASE/"costs.json", costs)
    reports = sorted(RELEASE.glob("*.xml"))
    unique, executions = set(), 0
    for path in reports:
        tree = ET.parse(path)
        assert not list(tree.iter("failure")) and not list(tree.iter("error"))
        for case in tree.iter("testcase"):
            unique.add((case.get("classname"), case.get("name")))
            executions += 1
    write(RELEASE/"test-summary.json", {"status": "PASS", "distinct_tests": len(unique), "executions": executions,
                                       "reports": [p.relative_to(ROOT).as_posix() for p in reports]})
    state_path = ROOT/"research-continuation/RESEARCH_STATE.json"
    (RELEASE/"research-state-before.json").write_bytes(state_path.read_bytes())
    state = read(state_path)
    state["active_direction"] = "Persistent task-time numerical acquisition with observed task-basis reuse, sparse innovations, calibrated empirical predictions, drift withdrawal and fresh correction. New readouts live on an explicitly preserved descendant; the existing workbench stays intact. No recurrent representation transfer or learned eta claim."
    state["stages"].append({"id": "23-task-transfer", "status": "bounded_empirical_route_complete_with_cost_and_exception_limits",
                            "evidence": "23_task_transfer/report.md", "detail": "3888 final outcomes across 24 independent task families; acquired K helps scarce-label adaptation, but amortized extra-label scratch is stronger. Actual-owner persistence/correction/retention pass. Cross-name calibration reuse repaired through an explicit preserving migration."})
    state["completed_local_evidence"]["task_transfer"] = {"report": "23_task_transfer/report.md", "manifest": "23_task_transfer/release-manifest.json",
        "final_outcomes": 3888, "actual_owner": read(RELEASE/"TT-MIGRATION-001/result.json")["owner"],
        "workspace": "../runs/sera-task-transfer", "remaining_seconds_at_release": after["remaining_seconds"],
        "distinct_targeted_tests": len(unique), "learned_eta": False, "general_task_learning": False}
    state["latest_continuation"] = "23_task_transfer/report.md"
    state["next_action"] = "Reconcile the saved descendant, live service and current allowance. Use the numerical task runner on fresh examples. The next research gap is acquiring representations or task materials beyond the supplied polynomial vocabulary, with independently held-out tasks and full acquisition/guard cost. Preserve the negative total-cost and rare-exception results; do not rerun or tune against the exposed finals."
    state["source_pins_scope"] += " Task-transfer release 23 begins at 1848b80; its source/receipt manifests and explicit calibration-scope migration bind the new evidence."
    write(state_path, state)
    (RELEASE/"research-state-at-release.json").write_bytes(state_path.read_bytes())
    write(RELEASE/"environment.json", {"python": sys.version, "packages": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "scipy", "matplotlib", "pytest", "ruff")}})
    sources = []
    for directory in ("experiments", "workbench", "src/sera", "scripts", "tests"):
        sources += list((ROOT/directory).rglob("*.py"))
    sources += [ROOT/"pyproject.toml", ROOT/"README.md", state_path]
    source_rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(set(sources))]
    write(RELEASE/"source-manifest.json", {"sources": source_rows})
    with zipfile.ZipFile(RELEASE/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for row in source_rows:
            z.write(ROOT/row["path"], row["path"])
    # Earlier artifacts and executable sources are checked, never rewritten.
    previous = ROOT/"research-continuation/22_sparse_mechanisms"
    old_manifest = read(previous/"release-manifest.json")
    for item in old_manifest["artifact_files"]:
        assert sha(ROOT/item["path"]) == item["sha256"]
    old_sources = read(previous/"source-manifest.json")["sources"]
    unchanged = 0
    for item in old_sources:
        path = ROOT/item["path"]
        if path.suffix == ".py":
            assert sha(path) == item["sha256"]
            unchanged += 1
    write(RELEASE/"predecessor-preservation.json", {"status": "PASS", "release_22_artifacts_unchanged": len(old_manifest["artifact_files"]),
                                                  "prior_python_files_unchanged": unchanged,
                                                  "scope": "Read-only byte checks after additive work, no previous experiments replayed"})
    resume = f"""# Resume without repeating completed experiments

This cycle is complete. No numerical worker or optimizer remains to resume. The final cohort is TT-FINAL-001: 3,888 outcomes across 24 independent task families, with separate numerical and mathematical audits. Development is TT-PILOT-001, 648 outcomes. Do not refit either completed final or use its exposed outcomes for new tuning.

The usable descendant is `runs/sera-task-transfer`, revision ten, owner `{read(RELEASE/'TT-MIGRATION-001/result.json')['owner']}`. `TT-OWNER-001` preserves the original nine revisions and source; `TT-MIGRATION-001` preserves the explicit repaired revision. Its learned tensors did not change during migration. The original trained neural checkpoint chain in the parent session remains required locally. Run `python scripts/sera_tasks.py status` or follow [the usage guide](README.md). Every numerical command is supervised and charged.

The original live service remains at http://127.0.0.1:8765/?panel=reason with owner 64b48d5f8192176ff6a4092b629822dfda6741d8f017b9cfbce295baf0eb7589 and pointer 000014-4ae564d91bce.json at release time. Do not restart it or overwrite a newer user revision. Reconcile both owners and the ledger before continuing.

This cycle used **{charged:.7f} numerical seconds ({charged/60:.2f} minutes)** across {len(records)} supervised jobs. **{after['remaining_seconds']:.7f} seconds ({after['remaining_seconds']/60:.2f} minutes)** remained at release. Limits remain one numerical thread and 2 GiB committed worker-tree memory. This is a saved balance, not a new grant or a promise of unattended work.

Read-only verification: `python scripts/verify_task_transfer_release.py`; add `--local-state` only when checking whether local owners still match this release. A later legitimate user task can advance those pointers without invalidating historical evidence. Exact frozen sources and receipts are archived. Source mismatches require explicit migration.

The useful result is acquired numerical task structure that reduces new-task data needs. Prior acquisition costs 512 labels; the extra-label scratch control performs better under the declared amortized cost comparison. Rare exceptions still invalidate pointwise claims. All three facts must remain together.

The next research step should acquire a representation or grounded task materials beyond the supplied polynomial vocabulary, using new task families, independent calibration and full acquisition/guard costs. The separate persistent-owner extension does not close recurrent representation transfer, general language, autonomous current-paper understanding or learned eta. Earlier failed and qualified branches remain intact.
"""
    (RELEASE/"resume.md").write_text(resume, encoding="utf-8", newline="\n")
    checklist = RELEASE/"PLAN.md"
    checklist.write_text(checklist.read_text().replace("- [ ]", "- [x]"), encoding="utf-8", newline="\n")
    head = subprocess.check_output(["git", "-c", "safe.directory="+ROOT.as_posix(), "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    artifacts = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(RELEASE.rglob("*")) if p.is_file() and p.name not in ("release-manifest.json", "verification.json")]
    write(RELEASE/"release-manifest.json", {"schema": "sera.task-transfer-release.1", "source_head_before_commit": head,
                                           "artifacts": artifacts, "empirical_route": True,
                                           "formal_answer_promotion": False, "learned_eta": False})
    print(json.dumps({"status": "SEALED", "artifacts": len(artifacts), "source_files": len(source_rows),
                      "charged_seconds": charged, "remaining_seconds": after["remaining_seconds"]}))


if __name__ == "__main__":
    main()
