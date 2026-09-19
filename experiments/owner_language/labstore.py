"""Materialise an explicit laboratory copy of the immutable owner lineage.

The restore chain of the current owner walks a series of append-only
``workbench.storage.Store`` directories.  Only the final intervention revision
was copied into this checkout, and it was copied flat, so no loader could read
it.  This module rebuilds a real store layout inside the laboratory and records
the exact bytes it copied, so that a later restore is demonstrably reading lab
files rather than silently selecting production.

Nothing here writes to the reference root.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
LAB_RUNS = LAB_ROOT / "runs"
BASELINE = LAB_RUNS / "owner-learning-001/baseline"
MANIFEST = LAB_RUNS / "owner-learning-001/lineage-manifest.json"
REFERENCE = Path("D:/ai/projects/sera")

OWNER_STORE = "runs/sera-intervention-live"
BASELINE_REVISION = "000000-b29d8cccea6a.json"
BASELINE_SHA256 = "b29d8cccea6a7b8b9e0f0bbc28432d146a6388bca7716a828c446b3b42036d95"
OWNER_IDENTITY = "62ea82cff82b88d1eb3848f7246494979c63894d869db603007e3d06111063ec"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def store_entries(directory):
    """Every file of an append-only store, with its hash."""
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return {p.relative_to(directory).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size} for p in files}


def verify_pointer(directory):
    pointer = Path(directory) / "current.json"
    if not pointer.exists():
        raise FileNotFoundError(f"{directory} has no current.json pointer")
    reference = json.loads(pointer.read_text())
    revision = Path(directory) / "revisions" / reference["revision"]
    actual = digest(revision)
    if actual != reference["sha256"]:
        raise ValueError(f"{directory}: pointer sha256 {reference['sha256']} != revision {actual}")
    return {"revision": reference["revision"], "sha256": actual, "bytes": revision.stat().st_size}


def repair_baseline_store():
    """Give the flat copied baseline the layout its loader actually requires."""
    flat = BASELINE / BASELINE_REVISION
    target = BASELINE / "revisions" / BASELINE_REVISION
    if flat.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(flat, target)
    if digest(target) != BASELINE_SHA256:
        raise ValueError("The copied baseline revision does not match the recorded SHA-256")
    return verify_pointer(BASELINE)


def copy_store(relative, reference=REFERENCE, runs=LAB_RUNS):
    """Copy one immutable store from the read-only reference into the laboratory."""
    source = Path(reference) / relative
    target = Path(runs) / Path(relative).relative_to("runs")
    if not source.is_dir():
        raise FileNotFoundError(f"Missing reference store {source}")
    source_pointer = verify_pointer(source)
    before = store_entries(source)
    if target.exists():
        after = store_entries(target)
        if after != before:
            raise ValueError(f"Existing laboratory copy of {relative} differs from the reference; preserve and reconcile")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        after = store_entries(target)
        if after != before:
            raise ValueError(f"Copy of {relative} did not reproduce the reference bytes")
    target_pointer = verify_pointer(target)
    if target_pointer != source_pointer:
        raise ValueError(f"Copied pointer for {relative} differs from the reference")
    return {"store": relative, "reference": source.as_posix(), "laboratory": target.as_posix(),
            "files": len(after), "bytes": sum(v["bytes"] for v in after.values()),
            "current": target_pointer, "entries": after}


def copy_file(relative, reference=REFERENCE, root=LAB_ROOT):
    """Copy one immutable auxiliary artefact the restore chain reads."""
    source = Path(reference) / relative
    target = Path(root) / relative
    if not source.is_file():
        raise FileNotFoundError(f"Missing reference artefact {source}")
    expected = digest(source)
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"Existing laboratory copy of {relative} differs; preserve and reconcile")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected:
            raise ValueError(f"Copy of {relative} did not reproduce the reference bytes")
    return {"file": relative, "sha256": expected, "bytes": source.stat().st_size}


def materialise(stores, files=(), reference=REFERENCE):
    record = {"schema": "sera.owner-language.lineage-manifest.1",
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "reference_root": Path(reference).as_posix(),
              "reference_is_read_only": True,
              "laboratory_root": LAB_ROOT.as_posix(),
              "owner_identity": OWNER_IDENTITY,
              "baseline": {"store": OWNER_STORE, "source": "copied into this checkout before the assignment",
                           "current": repair_baseline_store(),
                           "entries": store_entries(BASELINE)},
              "copied_stores": [copy_store(name, reference) for name in stores],
              "copied_files": [copy_file(name, reference) for name in files]}
    record["total_bytes"] = sum(s["bytes"] for s in record["copied_stores"])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
