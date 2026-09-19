"""Publish compact, hash-linked evidence from a local attempt into the tracked stage.

Large artefacts (full case records, the owner tensor inventory, checkpoints) stay
in the ignored runs directory.  What is published is a summary plus the SHA-256
of every retained local artefact, so a reader can ask for the exact file and
check it.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAB_ROOT = Path(__file__).resolve().parents[1]
STAGE = LAB_ROOT / "research-continuation/48_owner_language_acquisition"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def retained(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file()}


def compact_cases(path):
    cases = json.loads(Path(path).read_text())
    compact = []
    for case in cases:
        row = {"id": case.get("id"), "kind": case.get("kind"), "status": case.get("status")}
        if "error" in case:
            row["error"] = case["error"]
        compact.append(row)
    return compact


def publish_baseline(attempt, name="baseline-verification.json"):
    attempt = Path(attempt).resolve()
    record = json.loads((attempt / "baseline-verification.json").read_text())
    published = {"schema": "sera.owner-language.published-baseline.1",
                 "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "attempt": attempt.relative_to(LAB_ROOT).as_posix(),
                 "supervisor_state": json.loads((attempt / "state.json").read_text()),
                 "process": record["process"],
                 "restore": record["restore"],
                 "restore_seconds": record["restore_seconds"],
                 "owner_inventory_summary": record["owner_inventory_summary"],
                 "owner_config_digest": hashlib.sha256(
                     json.dumps(record["owner_config"], sort_keys=True, default=str).encode()).hexdigest(),
                 "retention": record["retention"],
                 "practical_examples": record["practical_examples"],
                 "retention_cases": compact_cases(attempt / "retention-cases.json"),
                 "practical_example_cases": compact_cases(attempt / "practical-examples-cases.json"),
                 "passed": record["passed"],
                 "retained_local_artefacts": retained(attempt)}
    published["supervisor_state"].pop("source_sha256", None)
    STAGE.mkdir(parents=True, exist_ok=True)
    out = STAGE / name
    out.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    return out


def publish_lineage(name="lineage-summary.json"):
    manifest = json.loads((LAB_ROOT / "runs/owner-learning-001/lineage-manifest.json").read_text())
    summary = {"schema": "sera.owner-language.lineage-summary.1",
               "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "reference_root": manifest["reference_root"],
               "reference_is_read_only": True,
               "owner_identity": manifest["owner_identity"],
               "baseline": {"current": manifest["baseline"]["current"],
                            "entries": manifest["baseline"]["entries"]},
               "manifest_sha256": sha256(LAB_ROOT / "runs/owner-learning-001/lineage-manifest.json"),
               "copied_stores": [{k: store[k] for k in ("store", "files", "bytes", "current")}
                                 for store in manifest["copied_stores"]],
               "copied_files": manifest["copied_files"],
               "total_bytes": manifest["total_bytes"]}
    STAGE.mkdir(parents=True, exist_ok=True)
    out = STAGE / name
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-attempt", type=Path)
    parser.add_argument("--lineage", action="store_true")
    options = parser.parse_args()
    written = []
    if options.lineage:
        written.append(publish_lineage())
    if options.baseline_attempt:
        written.append(publish_baseline(options.baseline_attempt))
    print(json.dumps([p.relative_to(LAB_ROOT).as_posix() for p in written], indent=2))


if __name__ == "__main__":
    main()
