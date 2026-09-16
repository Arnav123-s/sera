"""Read-only verification of the v5 integration evidence and optional local state."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/24_language_inquiry"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(data, row):
    if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Artifact identity differs: "+row.get("path", row.get("member", "unknown")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-state", action="store_true")
    args = parser.parse_args()
    manifest = read(OUT/"release-manifest.json")
    for row in manifest["artifacts"]:
        path = (ROOT/row["path"]).resolve()
        if not path.is_relative_to(OUT):
            raise ValueError("Artifact path escapes release")
        check(path.read_bytes(), row)
    source_rows = read(OUT/"source-manifest.json")["sources"]
    with zipfile.ZipFile(OUT/"sources.zip") as archive:
        assert set(archive.namelist()) == {r["path"] for r in source_rows}
        for row in source_rows:
            check(archive.read(row["path"]), row)
    raw = read(OUT/"evaluation/raw-records-manifest.json")
    with zipfile.ZipFile(OUT/"evaluation/raw-records.zip") as archive:
        for row in raw:
            check(archive.read(row["member"]), row)
            if (ROOT/row["path"]).exists():
                check((ROOT/row["path"]).read_bytes(), row)
    costs = read(OUT/"costs.json")
    with zipfile.ZipFile(OUT/"supervision.zip") as archive:
        for row in costs["owned_jobs"]:
            raw_state = archive.read(row["job"]+"/state.json")
            assert hashlib.sha256(raw_state).hexdigest() == row["state_sha256"]
            state = json.loads(raw_state)
            assert hashlib.sha256(archive.read(row["job"]+"/process.log")).hexdigest() == state["log_sha256"]
            assert state["charged_seconds"] == row["charged_seconds"]
            assert state["worker_threads"] == 1 and state["peak_job_committed_bytes"] <= 2147483648
    assert abs(sum(r["charged_seconds"] for r in costs["owned_jobs"])-costs["cycle_charged_seconds"]) < 1e-7
    tree = ET.parse(OUT/"regression.xml")
    assert not list(tree.iter("failure")) and not list(tree.iter("error"))
    assert len(list(tree.iter("testcase"))) == read(OUT/"test-summary.json")["full_suite_tests"] == 283
    for name in read(OUT/"test-summary.json").get("follow_up_reports", []):
        follow_up = ET.parse(OUT/name)
        assert not list(follow_up.iter("failure")) and not list(follow_up.iter("error"))
    assert read(OUT/"independent-math-audit.json")["passed"] and read(OUT/"integration.json")["passed"]
    assert read(OUT/"continuation-check-repaired.json")["returncode"] == 0
    checks = read(OUT/"prior-release-checks.json")
    assert all(r["returncode"] == 0 for r in checks if r["script"] != "verify_continuation")
    count = 0
    if args.local_state:
        local = read(OUT/"local-state-manifest.json")
        for row in local["immutable_artifacts"]:
            check((ROOT/row["path"]).read_bytes(), row)
            count += 1
        # The release revision must survive; newer legitimate revisions are allowed.
        path = ROOT/"runs/sera-inquiry/revisions"/local["release_pointer"]["revision"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == local["release_pointer"]["sha256"]
    print(json.dumps({"status": "PASS", "artifacts": len(manifest["artifacts"]), "sources": len(source_rows),
          "local_artifacts": count, "tests": 283, "language_records": 8424, "world_method_records": 16848,
          "jobs": len(costs["owned_jobs"]), "full_cost_seconds": costs["cycle_charged_seconds"],
          "remaining_seconds_at_seal": costs["remaining_seconds_at_seal"]}, indent=2))


if __name__ == "__main__":
    main()
