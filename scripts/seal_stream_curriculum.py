"""Preserve this completed cycle, update navigation, and seal immutable evidence."""

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/26_stream_curriculum"
PREVIOUS = ROOT / "research-continuation/25_constraint_inquiry"
NAV = ["README.md", "research-continuation/RESEARCH_STATE.json", "research-continuation/NEXT_ACTIONS.md",
       "research-continuation/NEXT_EXPERIMENT_PROTOCOL.md", "research-continuation/DECISION_LOG.md"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row(path):
    checksum = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(block)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": checksum.hexdigest()}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def archive(name, paths):
    paths = sorted(set(paths))
    with zipfile.ZipFile(OUT / name, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in paths:
            output.write(path, path.relative_to(ROOT).as_posix())
    return [row(path) for path in paths]


def append(name, value):
    with (ROOT / "research-continuation" / name).open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(value, allow_nan=False) + "\n")


def main():
    if (OUT / "release-manifest.json").exists():
        raise FileExistsError("A sealed cycle must remain immutable")
    budget_path = ROOT / "runs/v3-batch-001/budget.json"
    if (budget_path.parent / "active.lock").exists():
        raise ValueError("Finish the owned numerical job before sealing")
    budget, baseline = read(budget_path), read(PREVIOUS / "resource-budget-at-seal.json")
    head = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == "8ae0b3ef55e2a4eda26d1dfff2c01fbcb505099f"
    prior_artifacts = read(PREVIOUS / "release-manifest.json")["artifacts"]
    for item in prior_artifacts:
        assert row(ROOT / item["path"])["sha256"] == item["sha256"], item["path"]
    unchanged = []
    for item in read(PREVIOUS / "source-manifest.json")["sources"]:
        if item["path"].endswith(".py"):
            assert row(ROOT / item["path"])["sha256"] == item["sha256"], item["path"]
            unchanged.append(item["path"])
    old_local = read(PREVIOUS / "local-state-manifest.json")
    for item in old_local["immutable_artifacts"]:
        assert row(ROOT / item["path"])["sha256"] == item["sha256"]
    assert read(ROOT / "runs/sera-constraints/current.json") == old_local["release_pointer"]
    old_pointers = read(PREVIOUS / "preservation.json")["old_pointers_unchanged"]
    for name, saved in old_pointers.items():
        assert row(ROOT / "runs" / name / "current.json")["sha256"] == saved["sha256"]
    receipts, receipt_files = [], []
    for job in sorted(set(budget["jobs"]) - set(baseline["jobs"])):
        directory = ROOT / job
        state = read(directory / "state.json")
        assert state["status"] != "RUNNING" and state["peak_job_committed_bytes"] <= 2147483648
        assert row(directory / "process.log")["sha256"] == state["log_sha256"]
        receipts.append({"job": job, "status": state["status"], "charged_seconds": state["charged_seconds"],
                         "peak_job_committed_bytes": state["peak_job_committed_bytes"]})
        receipt_files.extend(directory / name for name in ("state.json", "process.log"))
    charged = sum(item["charged_seconds"] for item in receipts)
    assert abs(baseline["remaining_seconds"] + 3600 - charged - budget["remaining_seconds"]) < 1e-7
    tree = ET.parse(OUT / "full-regression.xml")
    tests = len(list(tree.iter("testcase")))
    assert tests >= 331 and not list(tree.iter("failure")) and not list(tree.iter("error"))
    assert read(OUT / "independent-audit.json")["status"] == "PASS"
    integration = read(OUT / "integration.json")
    assert integration["status"] == "PASS"
    archive("navigation-before.zip", [ROOT / name for name in NAV])
    costs = {"cycle_charged_seconds": charged, "initial_remaining_seconds": baseline["remaining_seconds"],
             "explicit_grant_seconds": 3600, "remaining_seconds_at_seal": budget["remaining_seconds"],
             "jobs": receipts, "threads_per_worker": 1, "memory_limit_bytes": 2147483648,
             "peak_job_committed_bytes": max(item["peak_job_committed_bytes"] for item in receipts),
             "unmeasured": ["Research/engineering wall time", "Static collection, hashing and archiving", "FLOPs", "Energy", "Peak RSS"],
             "source_compressed_bytes_read": 40251390,
             "additional_exposure": "Exact optimizer continuation; completed prefixes were not repeated",
             "orchestration_rejection": "One attempted dispatch while a prior job owned active.lock was refused before worker creation; no numerical charge"}
    write(OUT / "costs.json", costs)
    write(OUT / "resource-budget-at-seal.json", budget)
    write(OUT / "preservation.json", {"parent_head": head, "prior_artifacts_verified": len(prior_artifacts),
                                      "prior_local_artifacts_verified": len(old_local["immutable_artifacts"]),
                                      "prior_python_sources_unchanged": unchanged,
                                      "old_pointers_unchanged": old_pointers,
                                      "constraint_pointer_unchanged": old_local["release_pointer"]})
    archive("supervision.zip", receipt_files)
    folders = [path for path in (ROOT / "runs").glob("SC-*") if path.is_dir()]
    folders += [ROOT / "runs/sera-requests", ROOT / "runs/sera-requests-live"]
    local_paths = sorted({path for folder in folders for path in folder.rglob("*") if path.is_file()})
    local_rows = [row(path) for path in local_paths]
    write(OUT / "local-state-manifest.json", {"artifacts": local_rows,
          "preservation": "All intermediate checkpoints remain at these local paths. Published checkpoint archives contain terminal models and exact resumable state, not duplicate every intermediate.",
          "total_local_bytes": sum(item["bytes"] for item in local_rows)})
    raw_paths = [path for path in local_paths if path.suffix in (".json", ".jsonl", ".txt") and "-view-" not in path.as_posix() and "SC-data-" not in path.as_posix()]
    write(OUT / "raw-records-manifest.json", archive("raw-records.zip", raw_paths))
    checkpoint_rows = []
    for prefix in ("SC-full-", "SC-mixed-", "SC-exposure-", "SC-sequential-"):
        paths = []
        for folder in folders:
            if folder.name.startswith(prefix) and (folder / "summary.json").exists():
                summary = read(folder / "summary.json")
                if "checkpoint" in summary:
                    paths.append(ROOT / summary["checkpoint"])
        name = "checkpoints-" + prefix[3:-1] + ".zip"
        checkpoint_rows.extend({**item, "archive": name} for item in archive(name, paths))
    pilot = [ROOT / "runs/SC-pilot-001/step-000080.pt", ROOT / "runs/SC-audit-002/interrupted.pt", ROOT / "runs/SC-audit-002/resumed.pt"]
    checkpoint_rows.extend({**item, "archive": "checkpoints-pilot.zip"} for item in archive("checkpoints-pilot.zip", pilot))
    write(OUT / "checkpoint-archives.json", checkpoint_rows)
    data_paths = [p for p in (ROOT / "research/intake/massive-1.1-en-US-20260916").rglob("*") if p.is_file()]
    data_paths += [p for p in local_paths if "-view-" in p.as_posix() or "SC-data-" in p.as_posix()]
    write(OUT / "data-manifest.json", archive("data.zip", data_paths))
    final = read(ROOT / "runs/SC-final-001/summary.json")
    selected = next(item for item in final["scores"] if item["checkpoint"]["path"] == final["selected"]["path"])
    state = read(ROOT / NAV[1])
    state["date"] = datetime.now(timezone.utc).isoformat()
    state["active_direction"] = "Actual-owner acquisition from real requests, exact optimizer continuation, measured successive-domain replay and useful request-file annotation; next connect corrected request entities to persistent task execution."
    state["latest_continuation"] = "26_stream_curriculum/report.md"
    state["latest_user_steering"]["capability_reporting"] = "Teach and evaluate before conclusions; report demonstrated acquisition and investigate finite errors without universal inability or unmatched LLM ranking."
    state["next_action"] = "Use scripts/sera_annotate.py or scripts/sera_requests.py on the saved learner. For research, read 26_stream_curriculum/next-cycle.md and freeze a fresh correction-to-task stream; preserve the now-exposed MASSIVE test. Read live resource balance first."
    state["completed_local_evidence"]["stream_curriculum"] = {
        "report": "26_stream_curriculum/report.md", "manifest": "26_stream_curriculum/release-manifest.json",
        "full_suite_tests": tests, "source_training_requests": 11514, "domains": 18, "intents": 60,
        "test_requests": final["eligible_rows"], "selected": final["selected"],
        "selected_test_intent_accuracy": selected["intent_accuracy"], "selected_test_slot_f1": selected["slot_span_f1"],
        "selected_test_frame_accuracy": selected["frame_accuracy"], "workspace": "../runs/sera-requests-live",
        "owner": integration["owner"], "remaining_seconds_at_release": budget["remaining_seconds"]}
    state["stages"].append({"id": "26-stream-curriculum", "status": "completed_finite_training_and_usable_annotation",
                            "evidence": "26_stream_curriculum/report.md", "detail": "Five finite cycles, 16 terminal cohort checkpoints, four successive-domain lifetimes, one frozen official evaluation and exact selected-owner persistence."})
    (ROOT / NAV[1]).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8", newline="\n")
    notice = "Latest: [real-request acquisition and replay](26_stream_curriculum/report.md). The saved learner annotates local request files; [commands](26_stream_curriculum/README.md), [next research cycle](26_stream_curriculum/next-cycle.md). Earlier checklists below remain historical.\n\n"
    for name in NAV[2:4]:
        path = ROOT / name
        first, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + notice + rest.lstrip("\n"), encoding="utf-8", newline="\n")
    with (ROOT / NAV[4]).open("a", encoding="utf-8", newline="\n") as file:
        file.write("\n\n## Real-request acquisition and successive-domain replay\n\nI trained the actual owner's request interface from 11,514 MASSIVE requests, repaired exposure order through a frozen comparison, measured two replay lifetimes per arm, resumed longer fits exactly, and evaluated a fixed cohort once on the official test. I selected the annotation checkpoint on development exact frames and preserved all alternatives and costs. See release 26 for the complete records and source attribution.\n")
    with (ROOT / "README.md").open("a", encoding="utf-8", newline="\n") as file:
        file.write("\n\nThe [real-request learner](research-continuation/26_stream_curriculum/README.md) now labels task requests and extracts entities from local text/JSONL files. It was taught 11,514 real requests across 18 domains. [Held-out results, replay and costs](research-continuation/26_stream_curriculum/report.md) and the [architecture audit](research-continuation/26_stream_curriculum/architecture-audit.md) preserve the complete experiment. Run `.venv/Scripts/python.exe scripts/sera_requests.py ask --id my-request --text \"set an alarm for nine am\"` in the prepared workspace.\n")
    for number, protocol in enumerate(("protocol.md", "full-training-protocol.md", "mixed-protocol.md", "sequential-protocol.md", "exposure-protocol.md"), 1):
        append("EXPERIMENT_REGISTRY.jsonl", {"id": f"SC-{number:03d}", "status": "complete", "protocol": "26_stream_curriculum/" + protocol, "report": "26_stream_curriculum/report.md"})
    append("EVIDENCE_LEDGER.jsonl", {"id": "SC-OWNER-001", "class": "real_request_acquisition_and_replay", "claim": "Fixed public-source training with measured request annotation, exact continuation and two matched successive-domain replay comparisons", "evidence": "26_stream_curriculum/release-manifest.json", "scope": "Supplied intent/slot labels and fixed optimizer/replay; matched internal controls; one held-out official cohort"})
    append("FAILURE_LEDGER.jsonl", {"id": "SC-preserved-diagnoses", "status": "preserved_and_addressed_with_finite_comparisons", "detail": "Original source-order forgetting, slot errors, a preparation timeout, guarded-fact attachment error, metadata correction and any check failures remain in raw records and full costs; mixed exposure and replay were tested prospectively on development", "evidence": "26_stream_curriculum/report.md"})
    append("LITERATURE_LEDGER.jsonl", {"id": "SC-primary-review", "evidence": "26_stream_curriculum/literature.md", "scope": "MASSIVE/SLURP provenance and tiny episodic replay methods; Dark Experience Replay abstract as a future competitor"})
    with (OUT / "report.md").open("a", encoding="utf-8", newline="\n") as file:
        file.write(f"\nAt sealing, the cycle charged **{charged:.3f} supervised seconds** across **{len(receipts)} jobs**, including failed attempts. Peak process-tree committed memory was **{costs['peak_job_committed_bytes']:,} bytes**. The approved allowance retained **{budget['remaining_seconds']:.3f} seconds**. Source acquisition read **40,251,390 compressed bytes**. Research and static engineering costs are listed separately as unmeasured, not zero.\n")
    source_paths = {p for folder in ("src", "tests", "scripts", "experiments", "workbench") for p in (ROOT / folder).rglob("*")
                    if p.is_file() and p.suffix in (".py", ".html", ".js", ".css")}
    source_paths.update(ROOT / name for name in [*NAV, "pyproject.toml", ".gitignore", ".gitattributes"])
    source_paths.update(ROOT / "research-continuation" / name for name in (
        "EXPERIMENT_REGISTRY.jsonl", "EVIDENCE_LEDGER.jsonl", "FAILURE_LEDGER.jsonl", "LITERATURE_LEDGER.jsonl"))
    write(OUT / "source-manifest.json", {"sources": archive("sources.zip", source_paths)})
    write(OUT / "test-summary.json", {"full_suite_cases": tests, "failures": 0, "errors": 0,
                                     "skipped": len(list(tree.iter("skipped"))), "source": "full-regression.xml"})
    checklist = OUT / "checklist.md"
    checklist.write_text(checklist.read_text(encoding="utf-8").replace(
        "- [ ] Verify prior artifact/source/store preservation and archive exact resumable endpoints.",
        "- [x] Verify prior artifact/source/store preservation and archive exact resumable endpoints.").replace(
        "- [ ] Seal source/cost manifests, update research navigation, and prepare the complete local release for commit.",
        "- [x] Seal source/cost manifests, update research navigation, and prepare the complete local release for commit."), encoding="utf-8", newline="\n")
    artifacts = [row(p) for p in sorted(OUT.rglob("*")) if p.is_file()]
    write(OUT / "release-manifest.json", {"schema": "sera.stream-curriculum.release.1",
                                          "time_utc": datetime.now(timezone.utc).isoformat(), "parent_head": head,
                                          "artifacts": artifacts, "local_checkpoint_policy": "All intermediates retained locally; terminal and pilot resumable states archived with source identities"})
    print(json.dumps({"artifacts": len(artifacts), "sources": len(source_paths), "local_files": len(local_rows),
                      "charged_seconds": charged, "remaining_seconds": budget["remaining_seconds"]}))


if __name__ == "__main__":
    main()
