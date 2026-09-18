"""Verify the completion release and restore exact artifacts without replacing local work."""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research-continuation/32_verified_completion"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal():
    """Seal this finite release once; subsequent receipts are separately versioned."""
    manifest_path = OUT / "release-manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Preserve the published completion seal")
    selected = json.loads((OUT / "selection.json").read_text())["checkpoint"]
    runtime = [ROOT / selected]
    research = []
    for directory in ("runs/VC-study-001", "runs/VC-owner-audit-001", "runs/VC-packet-check-001",
                      "runs/VC-parent-001", "runs/VC-live-snapshot-001", "runs/ST-sources-001"):
        research.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and p not in runtime)
    manifest = {"schema": "sera.verified-completion.release.1", "files": [], "archives": {}}
    for key, paths in (("runtime", runtime), ("research", research)):
        archive_path = OUT / f"{key}.zip"
        if archive_path.exists():
            raise FileExistsError("Preserve existing archive: "+str(archive_path))
        records = []
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(set(paths)):
                relative = path.relative_to(ROOT).as_posix()
                archive.write(path, relative)
                records.append({"path": relative, "sha256": sha(path), "bytes": path.stat().st_size})
        manifest[key] = records
        manifest["archives"][key] = sha(archive_path)
    files = []
    for directory in ("experiments/verified_completion", "experiments/stem_learning",
                      "research-continuation/31_stem_acquisition", "research-continuation/32_verified_completion"):
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".zip", ".pyc"}
                     and p.name not in {"checklist.md", "NEXT_ACTION.md", "publication.json"})
    files.extend(ROOT / p for p in ("scripts/completion_artifacts.py", "scripts/run_completion_bounded.py",
                                    "scripts/run_stem_bounded.py", "tests/test_verified_completion.py"))
    for path in sorted(set(files)):
        manifest["files"].append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"sealed_files": len(files), "runtime_members": len(runtime), "research_members": len(research)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="Restore the selected runtime checkpoint")
    parser.add_argument("--research", action="store_true", help="Also restore preserved numerical study and audit records")
    parser.add_argument("--seal", action="store_true", help="Seal the completed finite release once")
    parser.add_argument("--predecessors", action="store_true", help="Verify the preserved predecessor releases")
    args = parser.parse_args()
    if args.predecessors:
        for script in ("verify_release.py", "verify_continuation.py", "verify_applicability.py", "verify_v3_release.py",
                       "verify_transfer_release.py", "verify_continuing_release.py", "verify_concept_release.py",
                       "reading_artifacts.py", "books_artifacts.py"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
        return
    if args.seal:
        seal()
        return
    manifest = json.loads((OUT / "release-manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != item["sha256"]:
            raise ValueError("Published completion evidence changed: "+item["path"])
    restored = 0
    for key in ("runtime", "research"):
        archive_path = OUT / f"{key}.zip"
        if sha(archive_path) != manifest["archives"][key]:
            raise ValueError("Completion archive identity changed")
        with zipfile.ZipFile(archive_path) as archive:
            if sorted(archive.namelist()) != sorted(item["path"] for item in manifest[key]):
                raise ValueError("Unexpected or duplicate completion archive members")
            for item in manifest[key]:
                name = item["path"]
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name or not name.startswith("runs/"):
                    raise ValueError("Unsafe completion artifact path")
                raw = archive.read(name)
                if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                    raise ValueError("Changed completion artifact")
                if key == "research" and not args.research:
                    continue
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
                      "research_members": len(manifest["research"]), "restored": restored, "experiments_restarted": 0}))


if __name__ == "__main__":
    main()
