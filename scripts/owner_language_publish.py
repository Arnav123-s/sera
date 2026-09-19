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


def publish_corpus(name="corpus-manifest.json"):
    """The reproducibility manifest only: hashes, counts and splits, never the text."""
    runs = LAB_ROOT / "runs/owner-learning-001/corpus"
    manifest = json.loads((runs / "manifest.json").read_text())
    evaluation = json.loads((runs / "evaluation/manifest.json").read_text())
    for record in manifest["sources"].values():
        record.pop("laboratory_path", None)
    published = {"schema": "sera.owner-language.published-corpus.1",
                 "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "note": "Raw book bytes and the cut items stay in the ignored runs directory; "
                         "this manifest is what a reader needs to rebuild them exactly.",
                 **{key: value for key, value in manifest.items() if key != "items_path"},
                 "evaluation": {"vocabulary_digest": evaluation["vocabulary_digest"],
                                "chance_accuracy": evaluation["chance_accuracy"],
                                "sets": {split: {key: value[key] for key in
                                                 ("sha256", "counts", "available", "skipped", "groups", "sources")}
                                         for split, value in evaluation["sets"].items()}}}
    STAGE.mkdir(parents=True, exist_ok=True)
    out = STAGE / name
    out.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    return out


def publish_attempt(attempt, source_name, name):
    """Copy one attempt record plus its supervisor state into the stage."""
    attempt = Path(attempt).resolve()
    record = json.loads((attempt / source_name).read_text())
    record["supervisor_state"] = json.loads((attempt / "state.json").read_text())
    record["retained_local_artefacts"] = retained(attempt)
    record["attempt"] = attempt.relative_to(LAB_ROOT).as_posix()
    STAGE.mkdir(parents=True, exist_ok=True)
    out = STAGE / name
    out.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-attempt", type=Path)
    parser.add_argument("--lineage", action="store_true")
    parser.add_argument("--corpus", action="store_true")
    parser.add_argument("--gate-attempt", type=Path)
    parser.add_argument("--calibration-attempt", type=Path)
    options = parser.parse_args()
    written = []
    if options.lineage:
        written.append(publish_lineage())
    if options.corpus:
        written.append(publish_corpus())
    if options.baseline_attempt:
        written.append(publish_baseline(options.baseline_attempt))
    if options.calibration_attempt:
        written.append(publish_attempt(options.calibration_attempt, "calibration.json", "calibration.json"))
    if options.gate_attempt:
        written.append(publish_attempt(options.gate_attempt, "pre-training-gate.json", "pre-training-gate.json"))
    print(json.dumps([p.relative_to(LAB_ROOT).as_posix() for p in written], indent=2))


if __name__ == "__main__":
    main()
