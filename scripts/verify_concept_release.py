"""Verify published C01/C02 bytes and archives without opening numerical evaluations."""

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/28_concept_refinement"


def main():
    manifest = json.loads((OUT / "release-manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Release path escapes repository")
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed release artifact: {entry['path']}")
    entries = 0
    for name, expected in manifest["archives"].items():
        with zipfile.ZipFile(OUT / name) as archive:
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
                raise ValueError("Archive member set changed")
            for member in archive.namelist():
                path = PurePosixPath(member)
                if path.is_absolute() or ".." in path.parts or ":" in member or "\\" in member:
                    raise ValueError("Unsafe archive member")
                if hashlib.sha256(archive.read(member)).hexdigest() != expected[member]:
                    raise ValueError(f"Changed archive bytes: {member}")
                entries += 1
    print(
        json.dumps(
            {"status": "PASS", "artifacts": len(manifest["artifacts"]), "archive_members": entries}
        )
    )


if __name__ == "__main__":
    main()
