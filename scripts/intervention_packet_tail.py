"""Verify remaining archived results; distinguish refit portability from prediction replay."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/intake/v18-understanding/SERA_v18"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKET / "src"))
from scripts.intervention_packet_portable import equal_values  # noqa: E402 -- isolated packet path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def save(name, result):
        with (out / (name+".json")).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(name, result, flush=True)

    gap, gate_counts = 0., {}
    records = read(PACKET / "results/expansion/records.json")
    for row in records:
        losses = {}
        for arm, width in (("base", 2), ("extended", 3)):
            weights = np.array(row["fits"][arm]["w"])
            def predict(actions):
                return np.array([.5+.5*np.exp(-sum(w*v for w, v in zip(weights, [a["duration"], abs(a["area"]), a["flips"]][:width]))) for a in actions])
            p = predict(row["test_actions"])
            difference = float(np.max(abs(p-row["fits"][arm]["prediction"])))
            gap = max(gap, difference)
            if difference >= 1e-12 or abs(np.mean((p-row["truth"])**2)-row["fits"][arm]["mse"]) >= 1e-12:
                raise ValueError("Archived expansion prediction or score failed independent replay")
            p = np.clip(predict(row["valid_actions"]), 1e-12, 1-1e-12)
            f = np.asarray(row["valid_counts"]) / row["shots"]
            losses[arm] = float(-(f*np.log(p)+(1-f)*np.log1p(-p)).mean())
            if abs(losses[arm]-row["fits"][arm]["validation_nll"]) >= 1e-12:
                raise ValueError("Archived validation likelihood changed")
        selected = "extended" if losses["extended"]+.001 < losses["base"] else "base"
        if selected != row["selected"]:
            raise ValueError("Archived admission decision changed")
        gate_counts.setdefault(row["family"], 0)
        gate_counts[row["family"]] += selected == "extended"
    save("expansion_archived", {"status": "PASS", "worlds": len(records), "fitted_models_replayed": 2*len(records),
                               "max_prediction_difference": gap, "extension_admissions": gate_counts,
                               "refit_scope": "The supplied 1e-8 coefficient-refit assertion failed separately; archived outcome replay is a different check"})

    from classical_control import fit
    from core import FAMILIES, SEEDS
    classical = []
    for family in FAMILIES:
        for seed in SEEDS:
            with np.load(PACKET / f"results/operators/{family}-{seed}-psd/data.npz") as data:
                covariance, _ = fit(data["train_x"], data["train_y"])
            with np.load(PACKET / f"results/classical/{family}-{seed}.npz") as archived:
                difference = float(np.max(abs(covariance-archived["covariance"])))
            classical.append({"family": family, "seed": seed, "max_abs_difference": difference, "original_tolerance_met": difference < 1e-12})
    save("classical", {"records": classical, "all_original_tolerances_met": all(r["original_tolerance_met"] for r in classical)})

    import physics_checks
    with tempfile.TemporaryDirectory() as temporary:
        previous = physics_checks.ROOT
        physics_checks.ROOT = Path(temporary)
        try:
            physics_checks.run()
        finally:
            physics_checks.ROOT = previous
        old, new = read(PACKET / "results/physics.json"), read(Path(temporary) / "results/physics.json")
        write_raw = out / "physics_recomputed.json"
        write_raw.write_text(json.dumps(new, indent=2)+"\n", encoding="utf-8", newline="\n")
        comparisons = {}
        for name in old:
            try:
                difference = equal_values(old[name], new[name], 1e-12)
                comparisons[name] = {"within_1e_minus_12": True, "max_difference": difference}
            except ValueError as exc:
                comparisons[name] = {"within_1e_minus_12": False, "error": str(exc), "original": old[name], "new": new[name]}
        save("physics", comparisons)

    prior = ROOT / "research/intake/v17-understanding/SERA_v17"
    manifest = read(prior / "MANIFEST.json")
    for item in manifest["files"]:
        path = (prior / item["path"]).resolve()
        if not path.is_relative_to(prior) or path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("v17 manifest discrepancy")
    process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=prior,
                             env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    (out / "v17-tests.log").write_text(process.stdout+process.stderr, encoding="utf-8", newline="\n")
    save("v17", {"manifest_entries": len(manifest["files"]), "unit_tests_returncode": process.returncode,
                 "scope": "Original source integrity and regression contracts, including retained obligations; no neural retraining"})
    if process.returncode:
        raise ValueError("Preserved v17 contract test failure")


if __name__ == "__main__":
    main()
