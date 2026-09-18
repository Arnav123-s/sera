"""Preserve continuing-growth checkpoints with exact deduplicated tensor storage."""

import argparse
import json
import zipfile
from pathlib import Path

import torch

from experiments.continuing_growth.common import OUT, SKILLS, read
from experiments.continuing_growth.runtime import validate_events
from scripts.refinement_artifacts import archive_file, costs, seal, unpacked, verify


def verify_growth():
    verify(38)
    manifest = read(OUT / "release-manifest.json")
    decisions, checkpoints = 0, 0
    with zipfile.ZipFile(archive_file(OUT, manifest)) as archive:
        for row in manifest["members"]:
            path = row["path"]
            if path.endswith("1536.pt.json") or Path(path).name.startswith("future-") and path.endswith(".pt.json"):
                saved = unpacked(json.loads(archive.read(path)), archive)
                validate_events(saved["events"], saved["highwater"])
                checkpoints += 1
                if len(saved["events"]) == 1536:
                    decisions += len(saved["events"])
                    for start in range(0, 1536, 128):
                        block = saved["events"][start:start+128]
                        counts = [sum(e["skill"] == s for e in block) for s in SKILLS]
                        if min(counts) < 2 or max(counts) > 32:
                            raise ValueError("Bounded broad exploration contract changed")
    if decisions != 6144:
        raise ValueError("Changed main continuing decision count")
    print(json.dumps({"exact_checkpoints": checkpoints, "main_decisions": decisions}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seal", action="store_true")
    p.add_argument("--costs", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.seal:
        seal(38)
    if args.costs:
        costs(38)
    else:
        verify_growth()


if __name__ == "__main__":
    main()
