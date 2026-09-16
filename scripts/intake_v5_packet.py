"""Inspect compact transport paths, then run the reviewed, supplied UNPACK.py."""

import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Path("D:/SERA_v5_Compact.zip")
    base = ROOT/"research/intake/v5-compact-20260916"
    if base.exists():
        raise ValueError("Preserve an existing intake; use its verified record")
    with zipfile.ZipFile(source) as z:
        expected = {"PAYLOAD.tar.xz", "PAYLOAD_SHA256.json", "UNPACK.py", "READ_ME_FIRST.md"}
        if set(z.namelist()) != expected or len(z.infolist()) != 4 or sum(i.file_size for i in z.infolist()) > 16_000_000:
            raise ValueError("Unexpected compact transport")
        base.mkdir(parents=True)
        transport = base/"transport"
        transport.mkdir()
        for name in expected:
            with (transport/name).open("xb") as f:
                f.write(z.read(name))
    info = json.loads((transport/"PAYLOAD_SHA256.json").read_text())
    destination = base/"expanded"
    names, total = set(), 0
    with tarfile.open(transport/"PAYLOAD.tar.xz", "r:xz") as archive:
        members = archive.getmembers()
        if len(members) > 1600:
            raise ValueError("Excessive member count")
        for member in members:
            name = PurePosixPath(member.name)
            target = destination.joinpath(*name.parts).resolve()
            if ("\\" in member.name or ":" in member.name or ".." in name.parts or
                    not name.parts or name.parts[0] != "SERA_Research_v5" or name.is_absolute() or
                    not target.is_relative_to(destination) or not (member.isfile() or member.isdir())):
                raise ValueError("Unsafe platform-specific path or archive member")
            key = str(target).casefold()
            if key in names:
                raise ValueError("Case-insensitive path collision")
            names.add(key)
            total += member.size
        if total > 400_000_000:
            raise ValueError("Archive exceeds the reviewed size limit")
    completed = subprocess.run([sys.executable, "-X", "utf8", str(transport/"UNPACK.py"),
                                "--output", str(destination)], check=True, text=True,
                               encoding="utf-8", capture_output=True)
    record = {"archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "unpacker_sha256": hashlib.sha256((transport/"UNPACK.py").read_bytes()).hexdigest(),
              "payload_sha256": info["sha256"], "members": len(names), "unpacked_bytes": total,
              "output": str(destination), "unpacker_output": completed.stdout,
              "scope": "Inspected supplied unpacker, additional Windows-path preflight, fresh-directory extraction and supplied manifest check. No research experiment executed."}
    (base/"intake.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
