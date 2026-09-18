"""Verify and restore exact grounded-book artifacts without replacing local changes."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/30_grounded_books"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((OUT / "release-manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != item["sha256"]:
            raise ValueError("Published book evidence changed: "+item["path"])
    restored = 0
    with zipfile.ZipFile(OUT / "runtime.zip") as archive:
        expected = [item["path"] for item in manifest["runtime"]]
        if sorted(archive.namelist()) != sorted(expected):
            raise ValueError("Unexpected or duplicate grounded runtime archive members")
        for item in manifest["runtime"]:
            name = item["path"]
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name or not name.startswith("runs/"):
                raise ValueError("Unsafe grounded artifact path")
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Changed grounded artifact")
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT / "runs"):
                raise ValueError("Artifact escaped its data root")
            if path.exists():
                if sha(path) != item["sha256"]:
                    raise ValueError("Preserve divergent local artifact: "+name)
            elif args.restore:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(raw)
                restored += 1
    print(json.dumps({"verified_files": len(manifest["files"]), "runtime_members": len(manifest["runtime"]),
                      "restored": restored, "experiments_restarted": 0}))


if __name__ == "__main__":
    main()
