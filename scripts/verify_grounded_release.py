"""Read-only checks of the frozen language release; never launches an experiment."""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/21_grounded_language"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_files(records):
    for row in records:
        path = (ROOT/row["path"]).resolve()
        require(path.is_relative_to(ROOT), "Artifact escapes repository")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Changed artifact: "+row["path"])


def check_zip(path, records):
    expected = {row["path"]: row for row in records}
    require(len(expected) == len(records), "Duplicate archive manifest entries")
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == set(expected) and len(archive.namelist()) == len(expected), "Archive entries differ")
        for name, row in expected.items():
            raw = archive.read(name)
            require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "Changed archived bytes: "+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-checkpoints", action="store_true", help="Also hash local tensor checkpoints; no model is loaded")
    args = parser.parse_args()
    manifest = read(RELEASE/"release-manifest.json")
    check_files(manifest["artifacts"])
    sources = read(RELEASE/"source-manifest.json")
    check_zip(RELEASE/"sources.zip", sources["files"])
    check_zip(RELEASE/"resources.zip", read(RELEASE/"costs.json")["archived_files"])
    check_zip(RELEASE/"application-history.zip", read(RELEASE/"application-history-manifest.json")["files"])
    # The current tree may evolve later. Frozen sources remain the release contract.
    qualification = read(RELEASE/"qualification.json")
    require(qualification["status"] == "QUALIFIED_RESTRICTED_INTERFACE" and all(qualification["gates"].values()), "Qualification differs")
    require(qualification["audit_sha256"] == sha(RELEASE/"independent-audit.json"), "Qualification audit changed")
    require(qualification["cohort_protocol_sha256"] == sha(RELEASE/"cohort-protocol.json"), "Cohort protocol changed")
    audit = read(RELEASE/"independent-audit.json")
    require(audit["status"] == "PASS" and audit["independently_parsed_and_checked_predictions"] == 41472 and audit["checkpoint_hashes_verified"] == 297, "Research audit differs")
    for seed in (151001, 151009, 151027):
        result = read(RELEASE/f"L10-FINAL-{seed}-interface/result.json")
        retention = read(RELEASE/f"L10-FINAL-{seed}-interface-assessment/result.json")["retention"]
        require(result["metrics"]["known"]["exact_translation"] == 1, "Familiar-form outcome differs")
        require(retention["groups"] == 61 and retention["exact_scores_equal"] and retention["maximum_drop"] == 0, "Retained-core outcome differs")
    continuing = read(RELEASE/"L11-ONREQUEST-001/result.json")
    require(continuing["status"] == "PASS" and continuing["requests"] == 6 and continuing["verified_revisions"] == 7, "Continuing study differs")
    require(all(r["independently_correct"] and r["exact_restored_owner"] and r["original_contexts_preserved"] for r in continuing["records"]), "Continuing evidence differs")
    require(continuing["records"][-1]["lesson_count"] == 5 and continuing["records"][-1]["cumulative_teaching_examples"] == 816, "Lesson costs differ")
    contract = read(RELEASE/"CONTRACT-001/result.json")
    require(contract["previously_accepted_errors"] == 5 and contract["accepted_errors"] == 0 and contract["accepted"] == 1270, "Corrective contract result differs")
    require(contract["checker_sha256"] == next(r["sha256"] for r in sources["files"] if r["path"] == "workbench/math_contract.py"), "Corrective checker source differs")
    counts, cases = {}, set()
    for name in ("tests.xml", "program-reuse-tests.xml"):
        suites = ET.parse(RELEASE/name).getroot().findall("testsuite")
        require(sum(int(s.attrib.get(k, 0)) for s in suites for k in ("errors", "failures", "skipped")) == 0, "Test suite did not pass")
        counts[name] = sum(int(s.attrib["tests"]) for s in suites)
        cases.update((c.attrib["classname"], c.attrib["name"]) for s in suites for c in s.findall("testcase"))
    require(counts == {"tests.xml": 241, "program-reuse-tests.xml": 3} and len(cases) == 243, "Test coverage accounting differs")
    migration = read(RELEASE/"program-reuse-migration.json")["result"]
    require(migration["old_owner"] == migration["new_owner"] and migration["checkpoint_preserved"], "Migration changed trained state")
    snapshot = read(RELEASE/"application-snapshot.json")
    require(len(snapshot["interaction"]["events"]) == 85 and len(snapshot["contexts"]) == 3, "Application lost original evidence")
    last = snapshot["language"]["attempts"][-1]
    require(last["solutions"] == [5] and not last["learned_on_request"] and last["translation_verification"], "Final task was not reused and checked")
    require(last["execution"]["work"]["indexed_lookup"] == 1 and last["same_owner"], "Acquired program was not executed")
    browser = read(RELEASE/"browser-checks.json")
    require(browser["mobile"]["viewport_width"] == 390 and browser["mobile"]["scroll_width"] <= 390 and browser["reload_preserved_result"], "Browser verification differs")
    costs = read(RELEASE/"costs.json")
    require(math.isclose(costs["starting_remaining_seconds"]-costs["charged_seconds"], costs["remaining_seconds_at_release"], abs_tol=1e-8), "Allowance accounting differs")
    require(costs["worker_threads"] == 1 and costs["peak_job_committed_bytes"] <= costs["memory_limit_bytes"] == 2147483648, "Resource limits differ")
    checkpoints = read(RELEASE/"checkpoint-manifest.json")
    if args.local_checkpoints:
        check_files(checkpoints["files"])
    print(json.dumps({"status": "PASS", "artifact_files": len(manifest["artifacts"]),
                      "frozen_source_files": len(sources["files"]), "resource_jobs": len(costs["jobs"]),
                      "local_checkpoints_hashed": len(checkpoints["files"]) if args.local_checkpoints else 0,
                      "full_suite_tests": counts["tests.xml"], "distinct_tests_across_runs": len(cases),
                      "new_experiments": 0, "limitations": "Checks preserved bytes and reported gates; does not retrain or independently reproduce neural logits."}))


if __name__ == "__main__":
    main()
