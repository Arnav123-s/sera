"""Separate numerical portability audit; leave the packet and strict failure intact."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, xlog1py, xlogy

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/intake/v18-understanding/SERA_v18"
sys.path.insert(0, str(PACKET / "src"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def equal_values(old, new, tolerance=1e-11):
    if type(old) is not type(new):
        raise ValueError("Changed replay value type")
    if isinstance(old, dict):
        if old.keys() != new.keys():
            raise ValueError("Changed replay fields")
        return max((equal_values(old[k], new[k], tolerance) for k in old), default=0.)
    if isinstance(old, list):
        if len(old) != len(new):
            raise ValueError("Changed replay length")
        return max((equal_values(a, b, tolerance) for a, b in zip(old, new)), default=0.)
    if type(old) is float:
        gap = abs(old - new)
        if not np.isfinite(new) or gap > tolerance:
            raise ValueError(f"Replay difference {gap} exceeds {tolerance}")
        return gap
    if old != new:
        raise ValueError("Changed decision, count, string or discrete replay value")
    return 0.


def audit_datasets(verifier):
    original = verifier.dataset
    cached, changed, count, maximum = {}, 0, 0, 0.
    outcome_differences = []
    for path in sorted((PACKET / "results/operators").glob("*/result.json")):
        record = read(path)
        key = record["family"], record["seed"]
        generated, world = original(*key)
        with np.load(path.parent / "data.npz", allow_pickle=False) as archive:
            archived = {k: archive[k].copy() for k in archive.files}
        if generated.keys() != archived.keys():
            raise ValueError("Dataset keys changed")
        for name, value in generated.items():
            saved = archived[name]
            if value.shape != saved.shape or value.dtype != saved.dtype:
                raise ValueError("Dataset shape or dtype changed")
            gap = float(np.max(np.abs(value - saved)))
            changed += int(np.count_nonzero(value != saved))
            count += 1
            if name in {"train_y", "valid_y"}:
                if np.any((saved < 0) | (saved > 1)) or not np.array_equal(saved * 512, np.rint(saved * 512)):
                    raise ValueError("Archived outcomes do not represent the declared binomial counts")
                if not np.array_equal(value, saved):
                    outcome_differences.append({"archive": path.parent.relative_to(PACKET).as_posix(), "array": name,
                                                "differing_elements": int(np.count_nonzero(value != saved)), "max_abs_difference": gap})
            elif value.dtype.kind in "biu":
                if not np.array_equal(value, saved):
                    raise ValueError("Generated random outcomes or controls changed")
            elif not np.allclose(value, saved, atol=1e-14, rtol=1e-14):
                raise ValueError("Dataset discrepancy exceeds the separate portability bound")
            else:
                maximum = max(maximum, gap)
        if key in cached and any(not np.array_equal(cached[key][0][k], v) for k, v in archived.items()):
            raise ValueError("Matched model arms did not share exactly the same archived data")
        cached[key] = archived, world
    verifier.dataset = lambda family, seed: cached[(family, seed)]
    return {"arrays": count, "differing_elements_across_archives": changed, "max_abs_difference": maximum,
            "sampled_outcomes_reproduce": not outcome_differences, "outcome_differences": outcome_differences,
            "archived_outcomes_on_declared_512_shot_grid": True, "control_and_probability_atol": 1e-14, "rtol": 1e-14,
            "operator_replay_inputs": "Original archived arrays; original prediction/metric tolerances unchanged"}


def interventions():
    from interventions import candidates, episode, probs
    count, gap, posterior_gap = 0, 0., 0.
    with gzip.open(PACKET / "results/interventions/episodes.jsonl.gz", "rt") as stream:
        for line in stream:
            old = json.loads(line)
            new = episode(old["family"], old["seed"], old["policy"])
            gap = max(gap, equal_values(old, new))
            parameters, _ = candidates()
            log = np.full(len(parameters), -np.log(len(parameters)))
            for row in old["initial"] + old["path"]:
                p = np.clip(probs(parameters, [row["action"]])[:, 0], 1e-14, 1-1e-14)
                log += xlogy(row["count"], p) + xlog1py(row["shots"]-row["count"], -p)
            posterior_gap = max(posterior_gap, float(np.max(np.abs(np.exp(log-logsumexp(log)) - old["path"][-1]["posterior"]))))
            if posterior_gap >= 1e-11:
                raise ValueError("Independent posterior replay exceeds original tolerance")
            count += 1
    if count != 1280:
        raise ValueError("Incomplete intervention cohort")
    return {"episodes": count, "discrete_decisions_and_counts_exact": True,
            "max_record_float_difference": gap, "independent_posterior_max_abs": posterior_gap}


def physics(verifier):
    import physics_checks
    from classical_control import fit
    from core import FAMILIES, SEEDS
    gap = 0.
    for family in FAMILIES:
        for seed in SEEDS:
            data, _ = verifier.dataset(family, seed)
            covariance, _ = fit(data["train_x"], data["train_y"])
            with np.load(PACKET / "results/classical" / f"{family}-{seed}.npz") as saved:
                gap = max(gap, float(np.max(np.abs(covariance-saved["covariance"]))))
            if gap >= 1e-12:
                raise ValueError("Classical refit exceeds original coefficient tolerance")
    with tempfile.TemporaryDirectory() as temporary:
        previous = physics_checks.ROOT
        physics_checks.ROOT = Path(temporary)
        try:
            physics_checks.run()
        finally:
            physics_checks.ROOT = previous
        other = equal_values(read(PACKET / "results/physics.json"), read(Path(temporary) / "results/physics.json"), 1e-12)
    return {"classical_refits": 12, "covariance_max_abs": gap, "physics_record_max_abs": other}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Use a fresh owned output directory")
    output.mkdir(parents=True, exist_ok=True)
    specification = importlib.util.spec_from_file_location("v18_verifier", PACKET / "verify.py")
    verifier = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(verifier)
    source_paths = [Path(__file__), PACKET / "verify.py", PACKET / "MANIFEST.json", *(PACKET / "src").glob("*.py")]
    identity = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    jobs = {"dataset_portability": lambda: audit_datasets(verifier), "operators": verifier.operators,
            "refinement": verifier.refinements, "interventions": interventions,
            "expansion": verifier.expansion_replay, "physics_and_classical": lambda: physics(verifier)}
    for name, function in jobs.items():
        path = output / (name + ".json")
        if path.exists():
            old = read(path)
            if old["identity"] != identity or old["status"] != "PASS":
                raise ValueError("Preserve the mismatched replay attempt")
            if name == "dataset_portability":
                # Restore the process-local archived-input binding without repeating numerical experiments.
                cache = {}
                for p in (PACKET / "results/operators").glob("*/result.json"):
                    row = read(p)
                    key = row["family"], row["seed"]
                    if key not in cache:
                        with np.load(p.parent / "data.npz") as z:
                            cache[key] = {k: z[k].copy() for k in z.files}, None
                verifier.dataset = lambda family, seed: cache[(family, seed)]
            continue
        started = time.perf_counter()
        try:
            result = function()
        except Exception as exc:
            write(output / (name + "-failure.json"), {"error": repr(exc), "seconds": time.perf_counter()-started, "identity": identity})
            raise
        write(path, {"status": "PASS", "result": result, "identity": identity, "seconds": time.perf_counter()-started})
        print(name, result, flush=True)


if __name__ == "__main__":
    main()
