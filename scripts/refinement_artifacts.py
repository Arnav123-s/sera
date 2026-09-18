"""Compact exact semantic checkpoints with deduplicated tensor blobs, plus raw evidence."""

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

from experiments.verified_completion.common import ROOT, read, sha, write

STAGES = {38: ("38_continuing_growth", "CG-study-001", "CG-"),
          39: ("39_sustained_refinement", "SR-study-001", "SR-"),
          40: ("40_step_resolution", "RS-study-001", "RS-"),
          41: ("41_self_chosen_discovery", "SD-study-001", "SD-"),
          42: ("42_composed_discovery", "CF-study-001", "CF-")}
RELEASE_BASE = "https://github.com/Arnav123-s/sera/releases/download/research-2026-09-18-discovery/"


class SplitFile(io.RawIOBase):
    def __init__(self, paths):
        self.paths, self.position = paths, 0
        self.sizes = [p.stat().st_size for p in paths]
        self.total = sum(self.sizes)

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.total)
        if self.position < 0:
            raise ValueError("Negative archive offset")
        return self.position

    def read(self, size=-1):
        size = self.total-self.position if size < 0 else min(size, self.total-self.position)
        if size <= 0:
            return b""
        chunks, offset = [], self.position
        for path, length in zip(self.paths, self.sizes, strict=True):
            if offset >= length:
                offset -= length
                continue
            with path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(min(size, length-offset))
            chunks.append(data)
            size -= len(data)
            self.position += len(data)
            offset = 0
            if size <= 0:
                break
        return b"".join(chunks)


def archive_file(out, manifest):
    return SplitFile(part_paths(out, manifest))


def part_paths(out, manifest):
    paths = []
    for row in manifest["parts"]:
        path = out / row["name"]
        if not path.exists():
            path = ROOT / "runs/artifact-cache" / out.name / row["name"]
            if not path.exists():
                name = row["asset"]
                if "/" in name or "\\" in name or ".." in name:
                    raise ValueError("Invalid repository release asset name")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix+".download")
                request = urllib.request.Request(RELEASE_BASE+name, headers={"User-Agent": "SERA-artifact-verifier"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                    received = 0
                    while block := response.read(1024*1024):
                        received += len(block)
                        if received > row["bytes"]:
                            raise ValueError("Archive asset exceeded its signed manifest size")
                        stream.write(block)
                if temporary.stat().st_size != row["bytes"] or sha(temporary) != row["sha256"]:
                    raise ValueError("Downloaded artifact failed its content hash")
                temporary.replace(path)
        paths.append(path)
    return paths


def split_archive(path, out):
    parts = []
    with path.open("rb") as stream:
        while raw := stream.read(40*1024*1024):
            name = f"research.zip.part{len(parts)+1:03d}"
            target = out / name
            with target.open("xb") as target_stream:
                target_stream.write(raw)
            parts.append({"name": name, "asset": out.name+"-"+name,
                          "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return parts


def repackage(stage, archive_path, old_manifest):
    """Preserve an already roundtrip-checked archive; only change transport format."""
    out = ROOT / "research-continuation" / STAGES[stage][0]
    if (out / "release-manifest.json").exists():
        raise FileExistsError("Preserve the completed segmented release")
    previous = read(old_manifest)
    if not archive_path.resolve().is_relative_to(ROOT / "runs") or sha(archive_path) != previous["archive"]:
        raise ValueError("Use an exact preserved owned archive")
    costs(stage)
    shutil.copyfile(old_manifest.with_name("checkpoint-index.json"), out / "checkpoint-index.json")
    parts = split_archive(archive_path, out)
    files = [{"path": r["path"], "sha256": sha(ROOT / r["path"])} for r in previous["files"]]
    write(out / "release-manifest.json", {"archive": previous["archive"], "parts": parts,
                                          "members": previous["members"], "files": files,
                                          "transport_migration": {"original_manifest": sha(old_manifest),
                                                                  "scientific_archive_bytes_unchanged": True}})


def packed(value, archive, names):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        raw = buffer.getvalue()
        name = "tensors/"+hashlib.sha256(raw).hexdigest()+".npy"
        if name not in names:
            archive.writestr(name, raw)
            names.add(name)
        return {"kind": "tensor", "blob": name}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[k, packed(v, archive, names)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [packed(v, archive, names) for v in value]}
    return value


def unpacked(value, archive):
    if not isinstance(value, dict):
        return value
    if value["kind"] == "tensor":
        raw = archive.read(value["blob"])
        if hashlib.sha256(raw).hexdigest() != Path(value["blob"]).stem:
            raise ValueError("Changed shared tensor blob")
        return torch.from_numpy(np.load(io.BytesIO(raw), allow_pickle=False).copy())
    if value["kind"] == "dict":
        return {k: unpacked(v, archive) for k, v in value["items"]}
    if value["kind"] in {"list", "tuple"}:
        return (list if value["kind"] == "list" else tuple)(unpacked(v, archive) for v in value["items"])
    raise ValueError("Unknown safe checkpoint component")


def costs(stage):
    name, _, prefix = STAGES[stage]
    out = ROOT / "research-continuation" / name
    target = out / "publication" if (out / "release-manifest.json").exists() else out
    ledger = read(ROOT / "runs/v3-batch-001/budget.json")
    jobs = {k: v for k, v in ledger["jobs"].items() if Path(k).name.startswith(prefix) and v["status"] != "RESERVED"}
    for key in jobs:
        for filename in ("state.json", "process.log"):
            dest = target / "checks" / (Path(key).name+"-"+filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / key / filename, dest)
    write(target / "costs.json", {"jobs": jobs, "charged_seconds": sum(v["charged_seconds"] for v in jobs.values()),
                                 "remaining_seconds_snapshot": ledger["remaining_seconds"],
                                 "cpu_threads": 1, "memory_limit_bytes": 2147483648})


def seal(stage):
    name, run_name, _ = STAGES[stage]
    out, run = ROOT / "research-continuation" / name, ROOT / "runs" / run_name
    if (out / "release-manifest.json").exists():
        raise FileExistsError("Preserve an existing scientific archive")
    costs(stage)
    names, originals, members = set(), [], []
    from scripts.refinement_audit import equal
    raw_archive = ROOT / "runs" / ("portable-"+run_name+".zip")
    with zipfile.ZipFile(raw_archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(run.rglob("*")):
            if not path.is_file() or path.suffix not in {".pt", ".json", ".npz", ".csv"}:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix == ".pt":
                saved = torch.load(path, weights_only=True)
                raw = json.dumps(packed(saved, archive, names), separators=(",", ":"), allow_nan=False).encode()
                member = rel+".json"
                archive.writestr(member, raw)
                originals.append({"original_path": rel, "original_sha256": sha(path), "portable": member})
            else:
                raw, member = path.read_bytes(), rel
                archive.writestr(member, raw)
            members.append({"path": member, "sha256": hashlib.sha256(raw).hexdigest()})
    # Verify semantic byte-exact tensor/optimizer/RNG/event recovery against
    # every original local checkpoint before publishing the compact archive.
    with zipfile.ZipFile(raw_archive) as archive:
        for record in originals:
            restored = unpacked(json.loads(archive.read(record["portable"])), archive)
            original = torch.load(ROOT / record["original_path"], weights_only=True)
            if not equal(restored, original):
                raise AssertionError("Portable checkpoint differs from original")
        for name in sorted(names):
            members.append({"path": name, "sha256": hashlib.sha256(archive.read(name)).hexdigest()})
    write(out / "checkpoint-index.json", {"originals": originals, "semantic_roundtrip_exact": True,
                                          "shared_tensor_blobs": len(names)})
    sources = [ROOT / "scripts/refinement_artifacts.py", ROOT / "scripts/refinement_audit.py",
               ROOT / "scripts/continuing_use.py"]
    if stage == 38:
        sources += sorted((ROOT / "experiments/continuing_growth").glob("*.py"))
        sources += [ROOT / p for p in ("scripts/run_growth_bounded.py", "scripts/growth_artifacts.py",
                                      "tests/test_continuing_growth.py")]
    elif stage == 39:
        sources += [ROOT / p for p in ("experiments/sustained_refinement.py", "scripts/run_refinement_bounded.py",
                                      "tests/test_sustained_refinement.py")]
    elif stage == 40:
        sources += [ROOT / p for p in ("experiments/refinement_resolution.py", "scripts/run_resolution_bounded.py",
                                      "scripts/resolution_audit.py", "tests/test_refinement_resolution.py")]
    elif stage == 41:
        sources += sorted((ROOT / "experiments/self_chosen").glob("*.py"))
        sources += [ROOT / p for p in ("experiments/discovery_observation.py", "scripts/self_chosen_audit.py",
                                      "scripts/self_chosen_artifacts.py", "scripts/self_chosen_use.py",
                                      "scripts/run_self_chosen_bounded.py", "scripts/self_chosen_report.py",
                                      "tests/test_self_chosen.py", "tests/test_discovery_observation.py")]
    else:
        sources += [ROOT / p for p in ("experiments/discovery_frontier.py", "scripts/frontier_audit.py",
                                      "scripts/frontier_artifacts.py", "scripts/frontier_report.py",
                                      "scripts/run_frontier_bounded.py", "tests/test_discovery_frontier.py")]
    sources += [p for p in out.rglob("*") if p.is_file() and p.suffix != ".zip"
                and p.name not in {"release-manifest.json", "checklist.md"} and "publication" not in p.parts]
    parts = split_archive(raw_archive, out)
    write(out / "release-manifest.json", {"archive": sha(raw_archive), "parts": parts, "members": members,
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(sources)]})


def verify(stage):
    out = ROOT / "research-continuation" / STAGES[stage][0]
    manifest = read(out / "release-manifest.json")
    for row in manifest["files"]:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Changed scientific source: "+row["path"])
    whole = hashlib.sha256()
    for part, path in zip(manifest["parts"], part_paths(out, manifest), strict=True):
        if path.stat().st_size != part["bytes"] or sha(path) != part["sha256"]:
            raise ValueError("Changed compact archive part")
        with path.open("rb") as stream:
            while block := stream.read(1024*1024):
                whole.update(block)
    if whole.hexdigest() != manifest["archive"]:
        raise ValueError("Changed compact checkpoint archive")
    with zipfile.ZipFile(archive_file(out, manifest)) as archive:
        if set(archive.namelist()) != {r["path"] for r in manifest["members"]}:
            raise ValueError("Changed archive membership")
        for row in manifest["members"]:
            if hashlib.sha256(archive.read(row["path"])).hexdigest() != row["sha256"]:
                raise ValueError("Changed checkpoint component")
    print(json.dumps({"stage": stage, "verified_files": len(manifest["files"]),
                      "archive_members": len(manifest["members"])}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=STAGES, required=True)
    p.add_argument("--seal", action="store_true")
    p.add_argument("--costs", action="store_true")
    p.add_argument("--from-archive", type=Path)
    p.add_argument("--from-manifest", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.from_archive:
        if args.from_manifest is None:
            raise ValueError("The preserved source manifest is required")
        repackage(args.stage, args.from_archive, args.from_manifest)
    elif args.seal:
        seal(args.stage)
    if args.costs:
        costs(args.stage)
    else:
        verify(args.stage)


if __name__ == "__main__":
    main()
