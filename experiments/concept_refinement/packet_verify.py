"""Once-only supplied-evidence replay; actual parent profiling stays distinct."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from experiments.self_study.runtime import StudySession
from sera.session_state import model_identity
from workbench.storage import Store

ROOT = Path(__file__).resolve().parents[2]
INTAKE = ROOT / "research/intake/v13-continuation-20260917"
OUT = ROOT / "research-continuation/28_concept_refinement"


def main():
    torch.set_num_threads(1)
    folder = OUT / "packet-verification"
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    for version, name, args in (
        (13, "manifest", ["code/verify_manifest.py"]),
        (13, "tests", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
        (13, "science", ["-m", "conceptlab.verify", "--output", str(folder / "v13-science.json")]),
        (12, "replay", ["VERIFY.py", "--output", str(folder / "v12")]),
    ):
        packet = INTAKE / f"SERA_v{version}"
        environment = {
            **os.environ,
            "PYTHONPATH": str(packet / "code"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        start = time.perf_counter()
        with (folder / f"v{version}-{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", *args],
                cwd=packet,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=160,
            )
        records.append(
            {
                "version": version,
                "job": name,
                "returncode": result.returncode,
                "wall_seconds": time.perf_counter() - start,
            }
        )
        (folder / "receipts.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(records[-1]), flush=True)
        if result.returncode:
            raise RuntimeError(
                "Preserve and inspect the failed verification; do not silently change tolerances"
            )
    parent = Store(ROOT / "runs/sera-study-live").read()
    session = StudySession(parent["parent"], parent)
    record = {
        "owner": model_identity(session.owner),
        "config": session.owner.export_config(),
        "state_shapes": {k: list(v.shape) for k, v in session.owner.initial(1).items()},
        "state_bytes": session.owner.core_state_bytes(),
        "memory_parameters": {n: list(p.shape) for n, p in session.owner.memory.named_parameters()},
        "parent_tensor_count": len(session.owner.state_dict()),
        "exact_motion": session.motion([2, 3, 1], "3", "5", "-1"),
    }
    (OUT / "parent-profile.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "owner": record["owner"],
                "config": record["config"],
                "state_shapes": record["state_shapes"],
            }
        )
    )


if __name__ == "__main__":
    main()
