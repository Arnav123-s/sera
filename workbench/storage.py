"""Append-only, checksummed application transactions; old research stays immutable."""

import hashlib
import json
import os
from pathlib import Path


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))+"\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


class Store:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def read(self):
        pointer = self.directory/"current.json"
        if not pointer.exists():
            return None
        ref = json.loads(pointer.read_text())
        name = ref["revision"]
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Invalid saved revision")
        raw = (self.directory/"revisions"/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("Saved session integrity check failed")
        return json.loads(raw)

    def commit(self, value, previous):
        self.directory.mkdir(parents=True, exist_ok=True)
        revisions = self.directory/"revisions"
        revisions.mkdir(exist_ok=True)
        value = {**value, "previous_sha256": None if previous is None else digest(previous),
                 "revision": 0 if previous is None else previous["revision"]+1}
        raw = encoded(value)
        name = f"{value['revision']:06d}-{hashlib.sha256(raw).hexdigest()[:12]}.json"
        with (revisions/name).open("xb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        temporary = self.directory/"current.tmp"
        temporary.write_bytes(encoded({"revision": name, "sha256": hashlib.sha256(raw).hexdigest()}))
        temporary.replace(self.directory/"current.json")
        return value

    def verify_history(self):
        records = [json.loads(p.read_text()) for p in sorted((self.directory/"revisions").glob("*.json"))]
        previous = None
        for index, value in enumerate(records):
            if value["revision"] != index or value["previous_sha256"] != (None if previous is None else digest(previous)):
                raise ValueError("Broken session lineage")
            previous = value
        if previous != self.read():
            raise ValueError("Current pointer differs from the preserved history")
        return len(records)
