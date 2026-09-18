"""Restore exact historical language probes required by the continuing owner."""

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research-continuation/33_capability_portfolio/publication/retention-fixtures.json"


def restore(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    restored, existing = 0, 0
    for archive in manifest["archives"]:
        path = (root / archive["path"]).resolve()
        if not path.is_relative_to(root / "research-continuation"):
            raise ValueError("Archive outside the published research root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != archive["sha256"]:
            raise ValueError("Historical data archive changed")
        with zipfile.ZipFile(path) as bundle:
            for entry in archive["files"]:
                name = entry["path"]
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name or not name.startswith("runs/"):
                    raise ValueError("Unsafe retention artifact destination")
                raw = bundle.read(name)
                if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError("Historical retention input changed")
                target = (root / name).resolve()
                if not target.is_relative_to(root / "runs"):
                    raise ValueError("Retention artifact escaped runs")
                if target.exists():
                    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
                        raise ValueError("Preserve divergent local work: "+name)
                    existing += 1
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(raw)
                    restored += 1
    return {"restored": restored, "already_identical": existing, "models_changed": 0, "experiments_restarted": 0}


if __name__ == "__main__":
    print(json.dumps(restore()))
