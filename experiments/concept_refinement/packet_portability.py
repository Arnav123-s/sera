"""Finish unexecuted checks and quantify a preserved cross-runtime replay failure."""

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from .packet_verify import INTAKE, OUT


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(1)
    folder = OUT / "packet-portability"
    folder.mkdir(exist_ok=False)
    from experiments.self_study.runtime import StudySession
    from sera.session_state import model_identity

    parent = json.loads((OUT / "parent-study.json").read_text())
    session = StudySession(parent["parent"], parent)
    write(
        OUT / "parent-profile.json",
        {
            "owner": model_identity(session.owner),
            "config": session.owner.export_config(),
            "state_bytes": session.owner.core_state_bytes(),
            "state_shapes": {k: list(v.shape) for k, v in session.owner.initial(1).items()},
            "memory_parameters": {
                n: list(p.shape) for n, p in session.owner.memory.named_parameters()
            },
            "parent_tensor_count": len(session.owner.state_dict()),
            "exact_motion": session.motion([2, 3, 1], "3", "5", "-1"),
        },
    )
    sys.path.insert(0, str(INTAKE / "SERA_v13/code"))
    from conceptlab import diagnosis, physics, production, reservoir
    from conceptlab.common import read_json

    packet = INTAKE / "SERA_v13"
    records = []
    for directory in sorted((packet / "results/neural").glob("*_*/")):
        if not (directory / "model.pt").exists():
            continue
        saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        net = reservoir.MemoryNet(saved["kind"])
        net.load_state_dict(saved["state"])
        net.eval()
        original = np.load(directory / "predictions.npz")
        fit = read_json(directory / "fit.json")
        for split in ("iid", "tail", "shifted"):
            x, y, _ = reservoir.make_data(saved["family"], split)
            with torch.no_grad():
                actual = net(torch.from_numpy(x)).numpy()
            independent = reservoir.independent(net, x)
            records.append(
                {
                    "model": directory.name,
                    "split": split,
                    "max_torch_archive_difference": float(np.max(np.abs(actual - original[split]))),
                    "strict_original_torch_tolerance_pass": bool(
                        np.allclose(actual, original[split], atol=2e-6, rtol=2e-6)
                    ),
                    "max_independent_torch_difference": float(np.max(np.abs(independent - actual))),
                    "original_independent_tolerance_pass": bool(
                        np.allclose(actual, independent, atol=2e-5, rtol=2e-5)
                    ),
                    "max_independent_archive_difference": float(
                        np.max(np.abs(independent - original[split]))
                    ),
                    "mse_difference": abs(
                        float(np.mean((actual - y) ** 2)) - fit["metrics"][split]
                    ),
                }
            )
    write(
        folder / "neural-differences.json",
        {
            "records": records,
            "runtime": {"torch": torch.__version__, "numpy": np.__version__},
            "interpretation": "Strict original thresholds retained. Differences quantified; no retraining or blanket tolerance waiver.",
        },
    )
    remaining = {"diagnosis": diagnosis.execute(packet / "results/diagnosis", True)}
    for name, module in (("physics", physics), ("production", production)):
        actual = module.execute(save=False)
        assert actual == read_json(packet / "results" / f"{name}.json")
        remaining[name] = "exact replay passed"
    write(folder / "v13-remaining.json", remaining)
    packet = INTAKE / "SERA_v12"
    jobs = [
        ("manifest-data", ["VERIFY.py", "--output", str(folder / "v12-data"), "--max-jobs", "1"]),
        ("body", ["code/verify_science.py", "body", "--output", str(folder / "v12-body.json")]),
        ("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
    ]
    receipts = []
    for name, args in jobs:
        started = time.perf_counter()
        with (folder / f"v12-{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", *args],
                cwd=packet,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=90,
            )
        receipts.append(
            {"job": name, "returncode": result.returncode, "seconds": time.perf_counter() - started}
        )
    write(folder / "v12-receipts.json", receipts)
    print(
        json.dumps(
            {
                "neural_arrays": len(records),
                "strict_replay_failures": sum(
                    not r["strict_original_torch_tolerance_pass"] for r in records
                ),
                "independent_tolerance_failures": sum(
                    not r["original_independent_tolerance_pass"] for r in records
                ),
                "max_torch_difference": max(r["max_torch_archive_difference"] for r in records),
                "v12": receipts,
            }
        )
    )
    if any(r["returncode"] for r in receipts):
        raise RuntimeError("Preserved v12 check failure needs review")


if __name__ == "__main__":
    main()
