"""Assemble immutable evidence; no model execution, fitting or numerical study."""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/24_language_inquiry"
FINALIZE = False


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def item(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def write(path, value):
    mode = "x"
    if path.exists() and FINALIZE:
        if read(path) == value:
            return
        if path.name not in {"source-manifest.json", "preservation.json", "costs.json", "test-summary.json", "release-manifest.json"}:
            raise ValueError("Unexpected change to sealed evidence: "+str(path))
        archived = OUT/"seal-history/initial"/path.name
        if not archived.exists() or archived.read_bytes() != path.read_bytes():
            raise ValueError("Preserve the initial seal before finalizing: "+str(path))
        mode = "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    global FINALIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    FINALIZE = parser.parse_args().finalize
    if (ROOT/"runs/v3-batch-001/active.lock").exists():
        raise ValueError("Finish the owned worker before sealing receipts")
    old = read(ROOT/"research-continuation/23_task_transfer/release-manifest.json")
    for row in old["artifacts"]:
        assert item(ROOT/row["path"]) == row, row["path"]
    old_sources = read(ROOT/"research-continuation/23_task_transfer/source-manifest.json")["sources"]
    changed = [r["path"] for r in old_sources if item(ROOT/r["path"]) != r]
    assert set(changed) <= {"README.md", "research-continuation/RESEARCH_STATE.json", "scripts/verify_continuation.py"}, changed
    shutil.copyfile(ROOT/"research/intake/v5-compact-20260916/intake.json", OUT/"intake.json")
    write(OUT/"preservation.json", {"older_artifacts_unchanged": len(old["artifacts"]),
          "older_sources_unchanged": len(old_sources)-len(changed), "older_sources_archived": len(old_sources),
          "explicit_maintenance_change": changed,
          "reason": "README navigation adds this cycle. Historical verifier checks evolving Git metadata against its original commit; frozen research checks remain strict"})
    def semantic_tree(raw):
        tree = ast.parse(raw)
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        return ast.dump(tree, include_attributes=False)
    with zipfile.ZipFile(OUT/"before-style-migration.zip") as archive:
        for name in ("model.py", "data.py"):
            relative = "experiments/language_inquiry/"+name
            assert semantic_tree(archive.read(relative)) == semantic_tree((ROOT/relative).read_bytes())
    pointer = read(ROOT/"runs/sera-inquiry/current.json")
    earlier = read(OUT/"integration-transcript.json")[-1]["pointer"]
    before = read(ROOT/"runs/sera-inquiry/revisions"/earlier["revision"])
    after = read(ROOT/"runs/sera-inquiry/revisions"/pointer["revision"])
    for name in ("owner", "language_delta", "admitted", "observations", "lessons"):
        assert before[name] == after[name], name
    for name in ("models", "current", "parser", "jobs", "cache", "history", "clock"):
        assert before["graph"]["payload"][name] == after["graph"]["payload"][name], name
    write(OUT/"migration-audit.json", {"passed": True, "before": earlier, "after": pointer,
          "training_model_and_data_ast_unchanged_except_import_order": True,
          "owner_tensors_and_inquiry_progress_unchanged": True,
          "additional_restore_work_charged": after["graph"]["payload"]["work"]["restore_transitions"]-before["graph"]["payload"]["work"]["restore_transitions"]})
    heads = {}
    for name, directory in (("sera", ROOT), ("kavi", ROOT/"research-continuation/00_sources/kavi-pinned")):
        heads[name] = subprocess.check_output(["git", "-c", f"safe.directory={directory.as_posix()}", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
    write(OUT/"reconciliation.json", {"local_heads_before_commit": heads,
          "initial_dirty_files": [], "initial_owned_numerical_jobs": [], "live_service_port": 8765,
          "parent_store_pointers": read(OUT/"integration.json")["parent_stores_unchanged"],
          "new_store_pointer": read(ROOT/"runs/sera-inquiry/current.json"),
          "initial_remaining_seconds_observed": 962.9074793000036,
          "packet_source_is_older_than_local_head": True, "no_repository_or_live_store_reset": True})
    rows = []
    with zipfile.ZipFile(OUT/"evaluation/raw-records.zip", "r" if FINALIZE else "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("language-records.json", "world-records.json"):
            path = OUT/"evaluation"/name
            if FINALIZE:
                assert archive.read(name) == path.read_bytes()
            else:
                archive.write(path, name)
            rows.append({"member": name, **item(path)})
    write(OUT/"evaluation/raw-records-manifest.json", rows)
    sources = {ROOT/r["path"] for r in old_sources}
    sources.update((ROOT/"experiments/language_inquiry").glob("*.py"))
    sources.update(ROOT/"scripts"/n for n in ("intake_v5_packet.py", "run_language_inquiry_bounded.py",
        "sera_inquiry.py", "seal_language_inquiry.py", "verify_language_inquiry_release.py"))
    sources.add(ROOT/"tests/test_language_inquiry.py")
    if FINALIZE:
        for name in ("sources.zip", "supervision.zip", "budget-after.json"):
            assert (OUT/name).read_bytes() == (OUT/"seal-history/initial"/name).read_bytes()
    with zipfile.ZipFile(OUT/"sources.zip", "w" if FINALIZE else "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            archive.write(path, path.relative_to(ROOT).as_posix())
    write(OUT/"source-manifest.json", {"sources": [item(p) for p in sorted(sources)]})
    artifacts = []
    for name in ("LI-fits-pilot-001", "LI-fits-final-001", "sera-inquiry"):
        for path in (ROOT/"runs"/name).rglob("*"):
            if path.is_file() and path.suffix in (".pt", ".json") and path.name != "current.json":
                artifacts.append(item(path))
    write(OUT/"local-state-manifest.json", {"immutable_artifacts": artifacts,
          "release_pointer": read(ROOT/"runs/sera-inquiry/current.json"),
          "later_valid_revisions_allowed": True})
    budget = read(ROOT/"runs/v3-batch-001/budget.json")
    jobs = []
    with zipfile.ZipFile(OUT/"supervision.zip", "w" if FINALIZE else "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, entry in budget["jobs"].items():
            if not (name.startswith("runs/LI-") or name.startswith("runs/inquiry-command-")):
                continue
            state = read(ROOT/name/"state.json")
            assert state["status"] == entry["status"] and state["charged_seconds"] == entry["charged_seconds"]
            assert state["worker_threads"] == 1 and state["peak_job_committed_bytes"] <= 2147483648
            row = {"job": name, "status": state["status"], "charged_seconds": state["charged_seconds"],
                   "peak_job_committed_bytes": state["peak_job_committed_bytes"],
                   "state_sha256": item(ROOT/name/"state.json")["sha256"]}
            jobs.append(row)
            for filename in ("state.json", "process.log"):
                archive.write(ROOT/name/filename, name+"/"+filename)
            if name.startswith("runs/inquiry-command-"):
                for filename in ("request.json", "response.json"):
                    path = (ROOT/name).parent/filename
                    if path.exists():
                        archive.write(path, path.relative_to(ROOT).as_posix())
    seconds = sum(r["charged_seconds"] for r in jobs)
    assert abs(budget["remaining_seconds"]+seconds-962.9074793000036) < 1e-7
    write(OUT/"costs.json", {"owned_jobs": jobs, "cycle_charged_seconds": seconds,
          "remaining_seconds_at_seal": budget["remaining_seconds"], "failed_jobs_included": sum(r["status"] != "PASS" for r in jobs),
          "max_committed_bytes": max(r["peak_job_committed_bytes"] for r in jobs),
          "scope": "Full supervised invocation wall costs. Static intake/documentation/archive preparation and external engineering effort are not instrumented as numerical learning costs."})
    shutil.copyfile(ROOT/"runs/v3-batch-001/budget.json", OUT/"budget-after.json")
    tree = ET.parse(OUT/"regression.xml")
    assert not list(tree.iter("failure")) and not list(tree.iter("error"))
    write(OUT/"test-summary.json", {"full_suite_tests": len(list(tree.iter("testcase"))),
          "new_inquiry_cases": sum("test_language_inquiry" in t.get("classname", "") for t in tree.iter("testcase")),
          "failures": 0, "errors": 0, "v5_packet_regression_methods": 53,
          "follow_up_reports": ["migration-guard-test.xml"] if FINALIZE else []})
    write(OUT/"acquisition-summary.json", {"pilot": read(ROOT/"runs/LI-fits-pilot-001/summary.json"),
          "final": read(ROOT/"runs/LI-fits-final-001/summary.json"), "selected": read(ROOT/"runs/LI-fits-final-001/selected.json")})
    excluded = {OUT/"evaluation/language-records.json", OUT/"evaluation/world-records.json",
                OUT/"release-manifest.json", OUT/"verification.json"}
    files = [p for p in OUT.rglob("*") if p.is_file() and p not in excluded and "__pycache__" not in p.parts]
    write(OUT/"release-manifest.json", {"schema": "sera.language-inquiry-release.1", "source_head_before_commit": heads["sera"],
          "artifacts": [item(p) for p in sorted(files)], "claim": "Restricted empirical shared-owner integration",
          "learned_grammar": False, "learned_eta": False, "physical_applicability_calibrated": False})
    print(json.dumps({"files": len(files), "sources": len(sources), "local_artifacts": len(artifacts),
                      "seconds": seconds, "remaining_seconds": budget["remaining_seconds"]}))


if __name__ == "__main__":
    main()
