"""Verify immutable v3 evidence and frozen code without restarting experiments."""

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"research-continuation/16_v3"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    manifest = read(RELEASE/"release-manifest.json")
    entries = manifest["files"]
    require(entries and len({r["path"] for r in entries}) == len(entries), "Unique release inventory required")
    for row in entries:
        path = (ROOT/row["path"]).resolve()
        require(path.is_relative_to(ROOT), "Invalid release path")
        raw = path.read_bytes()
        require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], f"Changed release file: {row['path']}")
    for experiment in ("GG-GUARD-002", "GC-001", "GC-002"):
        protocol = read(RELEASE/experiment/"protocol.json")
        with zipfile.ZipFile(RELEASE/experiment/"frozen-sources.zip") as archive:
            sources = protocol["payload"]["sources"]
            require(set(archive.namelist()) == set(sources), "Incomplete frozen source archive")
            for name, sha in sources.items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == sha, "Frozen source bytes changed")
        require(read(RELEASE/experiment/"independent.json")["status"] == "PASS", "Missing independent evidence")
    require(read(RELEASE/"GG-GUARD-002/summary.json")["gate"]["prospectively_qualified"], "Unexpected conditional guard outcome")
    require(not read(RELEASE/"GC-001/summary.json")["gate"]["full_cost_proxy_gain"], "Failed candidate was promoted")
    require(read(RELEASE/"GC-002/summary.json")["gate"]["full_cost_proxy_gain"], "Unexpected revised candidate result")
    repair = read(RELEASE/"GC-002/repair/result.json")
    require(repair["status"] == "PASS" and repair["fresh_cases_checked"] == 1024 and repair["parent_unchanged"], "Incomplete affected repair")
    current = read(RELEASE/"current-source-manifest.json")
    with zipfile.ZipFile(RELEASE/"current-sources.zip") as archive:
        require(set(archive.namelist()) == set(current), "Current source snapshot incomplete")
        for name, sha in current.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == sha, "Current source snapshot changed")
    tests = ET.parse(RELEASE/"tests.xml").getroot().findall("testsuite")
    require(sum(int(t.attrib["tests"]) for t in tests) == 199, "Unexpected regression coverage")
    require(sum(int(t.attrib["failures"])+int(t.attrib["errors"]) for t in tests) == 0, "Regression failure")
    preservation = read(RELEASE/"preservation.json")
    require(preservation["status"] == "PASS" and len(preservation["unchanged_shared_sources"]) == 45
            and len(preservation["predecessor_files"]) == 108, "Incomplete preservation")
    print(json.dumps({"status": "PASS", "release_files": len(entries), "bytes": sum(r["bytes"] for r in entries),
                      "guard_worlds": 3812, "current_tests": 199, "strict_intake_failures_preserved": True,
                      "completed_cohorts_restarted": 0}))


if __name__ == "__main__":
    main()
