"""Verify or restore the compact, exact human-reading runtime; never replace local work."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/29_human_reading"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((OUT / "release-manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != item["sha256"]:
            raise ValueError("Published evidence changed: "+item["path"])
    archive = OUT / "runtime.zip"
    restored = 0
    with zipfile.ZipFile(archive) as bundle:
        for item in manifest["runtime"]:
            name = item["path"]
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name or not name.startswith("runs/"):
                raise ValueError("Unsafe runtime artifact path")
            raw = bundle.read(name)
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Changed runtime archive member")
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
