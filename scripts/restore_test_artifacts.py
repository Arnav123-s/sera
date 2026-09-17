"""Restore exact published test prerequisites, never overwrite divergent local work."""

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research-continuation/28_concept_refinement/publication-hardening/fixtures.json"


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verified = set()
    restored = existing = 0
    for entry in manifest["files"]:
        name = entry["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
            raise ValueError("Unsafe artifact destination")
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not name.startswith(("runs/", "research/intake/")):
            raise ValueError("Artifact destination is outside declared local-data roots")
        if path.exists():
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"Preserve divergent local work: {name}")
            existing += 1
            continue
        archive = (ROOT / entry["archive"]).resolve()
        if not archive.is_relative_to(ROOT / "research-continuation"):
            raise ValueError("Artifact archive is outside published research")
        if archive not in verified:
            if (
                hashlib.sha256(archive.read_bytes()).hexdigest()
                != manifest["archives"][entry["archive"]]
            ):
                raise ValueError("Published archive identity changed")
            verified.add(archive)
        with zipfile.ZipFile(archive) as bundle:
            raw = bundle.read(entry["member"])
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Artifact identity changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        restored += 1
    print(
        json.dumps(
            {"restored": restored, "already_identical": existing, "experiments_restarted": 0}
        )
    )


if __name__ == "__main__":
    main()
