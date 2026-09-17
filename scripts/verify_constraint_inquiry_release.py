"""Read-only artifact, checkpoint, receipt and lineage identity verification."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/25_constraint_inquiry"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(data, row):
    if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Changed artifact: " + row["path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-state", action="store_true")
    args = parser.parse_args()
    release = read(OUT / "release-manifest.json")
    for row in release["artifacts"]:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(OUT):
            raise ValueError("Invalid release path")
        check(path.read_bytes(), row)
    source_rows = read(OUT / "source-manifest.json")["sources"]
    with zipfile.ZipFile(OUT / "sources.zip") as archive:
        assert set(archive.namelist()) == {r["path"] for r in source_rows}
        for row in source_rows:
            check(archive.read(row["path"]), row)
    with zipfile.ZipFile(OUT / "raw-records.zip") as archive:
        for row in read(OUT / "raw-records-manifest.json"):
            check(archive.read(row["path"]), row)
            if (ROOT / row["path"]).exists():
                check((ROOT / row["path"]).read_bytes(), row)
    local = read(OUT / "local-state-manifest.json")
    with zipfile.ZipFile(OUT / "checkpoints.zip") as archive:
        for row in local["immutable_artifacts"]:
            check(archive.read(row["archive_member"]), row)
            if args.local_state:
                check((ROOT / row["path"]).read_bytes(), row)
    costs = read(OUT / "costs.json")
    with zipfile.ZipFile(OUT / "supervision.zip") as archive:
        for receipt in costs["owned_jobs"]:
            raw = archive.read(receipt["job"] + "/state.json")
            assert hashlib.sha256(raw).hexdigest() == receipt["state_sha256"]
            state = json.loads(raw)
            assert hashlib.sha256(archive.read(receipt["job"] + "/process.log")).hexdigest() == state["log_sha256"]
            assert state["charged_seconds"] == receipt["charged_seconds"] and state["worker_threads"] == 1
            assert state["peak_job_committed_bytes"] <= 2147483648
    assert abs(sum(r["charged_seconds"] for r in costs["owned_jobs"]) - costs["cycle_charged_seconds"]) < 1e-7
    xml = ET.parse(OUT / "full-regression.xml")
    assert len(list(xml.iter("testcase"))) == 302 and not list(xml.iter("failure")) and not list(xml.iter("error"))
    assert read(OUT / "independent-audit.json")["max_absolute_difference"] < 1e-12
    if args.local_state:
        path = ROOT / "runs/sera-constraints/revisions" / local["release_pointer"]["revision"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == local["release_pointer"]["sha256"]
    print(json.dumps({"status": "PASS", "artifacts": len(release["artifacts"]), "sources": len(source_rows),
                      "local_artifacts": len(local["immutable_artifacts"]) if args.local_state else None,
                      "tests": 302, "independent_numeric_records": 7680, "remaining_seconds_at_seal": costs["remaining_seconds_at_seal"]}))


if __name__ == "__main__":
    main()
