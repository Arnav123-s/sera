"""Reproduce the reviewed, hash-pinned source benchmark without invoking its main scripts."""

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from sera.accounting import Costs
from sera.storage import write_json

ARCHIVE = "7e9268f8a0281190551ecfdadf8270eab989190e62e4621c03b93590c3932cce"
REVIEWED = {"experiment.py": "894fac1ddc244fbeea08a3171c1573c496f7fac2fff7d0202d4710416fa6c57b",
            "verify_and_extend.py": "993f003b03850d3b5978f95309787d465cccff5a2cb486c13b3c3b3de2f0fff2"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--retrain", action="store_true")
    args = p.parse_args()
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != ARCHIVE:
        raise ValueError("The reviewed source archive identity differs")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh reproduction directory")
    args.output.mkdir(parents=True, exist_ok=True)
    cost = Costs()
    torch.set_num_threads(1)
    results = {"archive_sha256": ARCHIVE, "reviewed_modules": REVIEWED, "checkpoint_checks": [], "retraining": []}
    try:
        with zipfile.ZipFile(args.archive) as archive:
            prefix = "Quantum_AGI_Research_Package/"
            for name, expected in REVIEWED.items():
                data = archive.read(prefix + "code/" + name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("Reviewed code bytes differ")
                (args.output / name).write_bytes(data)
            experiment = load_module("experiment", args.output / "experiment.py")
            verification = load_module("source_verification", args.output / "verify_and_extend.py")
            experiment.ROOT = verification.ROOT = args.output.resolve()
            (args.output / "results").mkdir()
            with cost.phase("original-mathematics-and-gradients"):
                results["mathematics"] = verification.checks()
                if not all(row["passed"] for row in results["mathematics"]):
                    raise ValueError("Original numerical checks did not reproduce")
            for kind in experiment.KINDS:
                for seed in range(3):
                    expected = json.loads(archive.read(prefix + f"results/{kind}_{seed}.json"))
                    with cost.phase(f"checkpoint-recheck/{kind}/{seed}"):
                        import io
                        raw = archive.read(prefix + f"results/checkpoints/{kind}_{seed}.pt")
                        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                        model = experiment.Model(kind)
                        model.load_state_dict(payload["state_dict"])
                        observed = experiment.evaluate(model, 12, 20000000 + seed)
                        error = max(abs(observed[task]["accuracy"] - expected["ID"][task]["accuracy"]) for task in expected["ID"])
                        results["checkpoint_checks"].append({"model": kind, "seed": seed,
                                                              "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
                                                              "max_accuracy_difference": error, "passed": error <= 1e-8})
                    if args.retrain:
                        with cost.phase(f"from-scratch/{kind}/{seed}"):
                            experiment.run_job(kind, seed, expected["steps"])
                            fresh = json.loads((args.output / f"results/{kind}_{seed}.json").read_text())
                            fresh["source_id_macro"] = float(np.mean([value["accuracy"] for value in expected["ID"].values()]))
                            fresh["fresh_id_macro"] = float(np.mean([value["accuracy"] for value in fresh["ID"].values()]))
                            results["retraining"].append(fresh)
                    write_json(args.output / "reproduction.json", results)
                    print(f"{kind}/{seed}: checkpoint difference {error:.6f}; retrained={args.retrain}", flush=True)
    finally:
        results["costs"] = cost.record()
        results["scope"] = "Source mechanism benchmark reproduced separately from SERA 0.3. Reviewed source modules are pinned and their main blocks are never called; ROOT redirects all generated outputs to this run. Checkpoints load with weights_only=True. Fresh training begins from the original seeds and never initializes from supplied weights. Original manuscript titles and results remain historical provenance."
        write_json(args.output / "reproduction.json", results)


if __name__ == "__main__":
    main()
