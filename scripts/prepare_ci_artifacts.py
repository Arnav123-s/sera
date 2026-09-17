"""Bind traced prerequisites to existing archives; package only missing witnesses."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/28_concept_refinement/publication-hardening"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = OUT / "fixtures.json"
    if manifest.exists():
        raise FileExistsError("Preserve the completed fixture inventory")
    trace = json.loads((OUT / "fixture-reads.json").read_text(encoding="utf-8"))
    if trace["pytest_returncode"]:
        raise ValueError("Trace must come from passing integration tests")
    index = {}
    for path in sorted((ROOT / "research-continuation").rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                index.setdefault(name, []).append(path)
    entries, missing, archives = [], [], {}
    for row in trace["files"]:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Traced artifact changed")
        candidates = index.get(row["path"], [])
        found = None
        for path in candidates:
            with zipfile.ZipFile(path) as archive:
                raw = archive.read(row["path"])
                if hashlib.sha256(raw).hexdigest() == row["sha256"]:
                    found = path
                    break
        if found is None:
            missing.append(row)
        else:
            name = found.relative_to(ROOT).as_posix()
            archives.setdefault(name, sha(found))
            entries.append({**row, "archive": name, "member": row["path"]})
    additional = OUT / "additional-fixtures.zip"
    with zipfile.ZipFile(additional, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in missing:
            archive.write(ROOT / row["path"], row["path"])
            entries.append(
                {**row, "archive": additional.relative_to(ROOT).as_posix(), "member": row["path"]}
            )
    archives[additional.relative_to(ROOT).as_posix()] = sha(additional)
    value = {
        "schema": "sera.test-artifact-prerequisites.1",
        "files": sorted(entries, key=lambda r: r["path"]),
        "archives": archives,
        "trace_sha256": sha(OUT / "fixture-reads.json"),
        "purpose": "Restore existing verified checkpoints/data; no retraining, downloads or final scoring.",
    }
    manifest.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "files": len(entries),
                "reused_archives": len(archives) - 1,
                "additional_files": len(missing),
                "additional_bytes": additional.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
