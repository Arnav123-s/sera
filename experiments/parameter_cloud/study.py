"""Prospective R7 comparison; run as python -m experiments.parameter_cloud.study."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from sera.accounting import Costs

from .core import CLOUD_METHODS, Evidence, fit, score

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("well_specified", "correlated", "misspecified")
SUITES = ("joint_two", "factor_six")
SUPPORTS = (8, 32)
TUNED = ("schrodinger", "dephased", "imaginary", "diffusion", "dirac", "unitary")
PROTOCOL = {
    "schema": "sera-parameter-cloud/1", "grid_size": 65, "grid_limit": 2.0,
    "noise": 0.2, "phase": 1.0, "steps": 16, "sweeps": 24,
    "main_seeds": list(range(5200, 5210)), "development_seeds": [9100, 9101],
    "support_sizes": list(SUPPORTS), "validation_samples": 64, "query_samples_per_partition": 256,
    "families": list(FAMILIES), "suites": list(SUITES),
    "mixing_candidates": [0.0003, 0.003, 0.03], "dirac_mixing_candidates": [0.01, 0.1, 1.0],
    "selection": "mean development validation NLL; one setting per suite/method; earliest tie",
    "test_policy": "all models in a case frozen to checkpoints before any query scores",
    "primary": "fresh independent-design prediction MSE and observation NLL, paired by seed",
    "intervals": "95% Gaussian moment bands, not exact finite-mixture credible intervals",
    "claim_boundary": "classical simulation of finite predictive-weight clouds; no upstream solver promotion",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash():
    result = hashlib.sha256()
    for name in ("core.py", "study.py"):
        result.update(name.encode())
        result.update((Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n"))
    return result.hexdigest()


def features(values, suite):
    if suite == "joint_two":
        return values[:, :2]
    x, v, u = values.T
    return np.column_stack((np.ones_like(x), x, v, u, x**3, v * np.abs(v)))


def problem(suite, family, seed, support, query_samples=256):
    """Evaluator generates truth; fit receives only the admitted design and targets."""
    rng = np.random.default_rng(seed)
    if suite == "joint_two":
        theta = np.array([-rng.uniform(.4, 1.2), -rng.uniform(.1, .5)])
    else:
        theta = np.array([rng.uniform(-.3, .3), -rng.uniform(.4, 1.2), -rng.uniform(.1, .5),
                          rng.uniform(.6, 1.4), -rng.uniform(.25, .8), -rng.uniform(.15, .6)])
    result = {"theta": theta}
    partitions = (("support", support, 1.), ("validation", 64, 1.),
                  ("familiar", query_samples, 1.), ("independent", query_samples, 1.),
                  ("extrapolation", query_samples, 1.5))
    for index, (name, count, extent) in enumerate(partitions):
        # Named RNG streams keep query identity independent of the support budget.
        local = np.random.default_rng(seed * 100 + index + 2000000)
        draw_count = max(SUPPORTS) if name == "support" else count
        values = local.uniform(-extent, extent, (draw_count, 3))
        if family == "correlated" and name in {"support", "validation", "familiar"}:
            values[:, 1] = values[:, 0] + local.normal(0, .02, draw_count)
        design = features(values, suite)
        truth = design @ theta
        if family == "misspecified":
            truth = truth + .8 * np.sin(4 * values[:, 0])
        observed = truth + PROTOCOL["noise"] * local.normal(size=draw_count)
        result[name] = tuple(torch.from_numpy(np.asarray(a[:count], dtype=np.float64)) for a in (design, truth, observed))
    return result


def model_methods(suite):
    return ("gaussian", "gaussian_diagonal", "gradient_map", *CLOUD_METHODS,
            *(("marginal_product",) if suite == "joint_two" else ()))


def fit_one(case, suite, method, mixing, *, grid_size=65, steps=16, sweeps=24):
    design, _, observed = case["support"]
    clock, cpu = time.perf_counter(), time.process_time()
    model = fit(Evidence(design, observed), method, noise=PROTOCOL["noise"],
                grid_size=grid_size, mixing=mixing, phase=PROTOCOL["phase"],
                steps=steps, sweeps=sweeps, joint=suite == "joint_two")
    measured = {"wall_seconds": time.perf_counter() - clock, "process_cpu_seconds": time.process_time() - cpu}
    return model, measured


def score_one(model, partition):
    result = score(model, *partition, noise=PROTOCOL["noise"])
    summary = {key: value for key, value in result.items() if not isinstance(value, torch.Tensor)}
    vectors = {key: value.numpy() for key, value in result.items() if isinstance(value, torch.Tensor)}
    return summary, vectors


def save_model(path, model):
    tensors = {key: value.numpy() for key, value in model.items() if isinstance(value, torch.Tensor) and key != "points"}
    np.savez_compressed(path, **tensors)
    metadata = {key: value for key, value in model.items() if not isinstance(value, torch.Tensor)}
    # Mean/covariance are readout caches. The probability matrix is an audit cache.
    retained = model["state"].numel() * model["state"].element_size()
    if "grid" in model:
        retained += model["grid"].numel() * model["grid"].element_size()
    else:
        retained = model["mean"].numel() * 8 + model["covariance"].numel() * 8
    metadata.update(checkpoint_sha256=digest(path), checkpoint_bytes=path.stat().st_size,
                    retained_state_bytes=retained,
                    all_checkpoint_array_bytes=sum(a.nbytes for a in tensors.values()))
    return metadata


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    if "probability" in result:
        result["points"] = torch.cartesian_prod(result["grid"], result["grid"])
    return result


def tune(output, *, smoke=False):
    trials = []
    selected = {}
    seeds = PROTOCOL["development_seeds"] if not smoke else [9199]
    families = FAMILIES if not smoke else FAMILIES[:1]
    grid_size, steps, sweeps = (65, 16, 24) if not smoke else (17, 4, 3)
    for suite, method in itertools.product(SUITES, TUNED):
        candidates = PROTOCOL["dirac_mixing_candidates"] if method == "dirac" else PROTOCOL["mixing_candidates"]
        choices = []
        for mixing in candidates:
            rows = []
            for seed, family in itertools.product(seeds, families):
                case = problem(suite, family, seed, 32, query_samples=8)
                model, costs = fit_one(case, suite, method, mixing, grid_size=grid_size, steps=steps, sweeps=sweeps)
                validation, _ = score_one(model, case["validation"])
                name = f"{suite}-{method}-{mixing}-{seed}-{family}"
                metadata = save_model(output / (name + ".npz"), model)
                row = {"suite": suite, "method": method, "mixing": mixing, "seed": seed,
                       "family": family, "validation": validation, "fit_costs": costs, "model": metadata}
                write_json(output / (name + ".json"), row)
                trials.append(row)
                rows.append(validation["nll"])
            choices.append({"mixing": mixing, "mean_validation_nll": float(np.mean(rows))})
        best = min(choices, key=lambda row: row["mean_validation_nll"])
        selected.setdefault(suite, {})[method] = best["mixing"]
        print(json.dumps({"selected": suite + "/" + method, **best}), flush=True)
    return {"selected": selected, "trials": trials, "smoke": smoke}


def main_study(output, tuning, seeds, *, smoke=False):
    if tuning["smoke"] != smoke:
        raise ValueError("Development protocol and study mode differ")
    trials = []
    families = FAMILIES if not smoke else FAMILIES[:1]
    supports = SUPPORTS if not smoke else SUPPORTS[:1]
    samples = 256 if not smoke else 8
    grid_size, steps, sweeps = (65, 16, 24) if not smoke else (17, 4, 3)
    for suite, family, seed, support in itertools.product(SUITES, families, seeds, supports):
        case = problem(suite, family, seed, support, query_samples=samples)
        name = f"{suite}-{family}-{seed}-{support}"
        folder = output / name
        folder.mkdir()
        tensors = {partition + "_" + key: value.numpy() for partition in ("support", "validation", "familiar", "independent", "extrapolation")
                   for key, value in zip(("design", "truth", "observed"), case[partition])}
        # Original observations, including withheld query vectors, remain auditable.
        np.savez_compressed(folder / "evidence.npz", **tensors)
        pending = []
        for method in model_methods(suite):
            mixing = tuning["selected"][suite].get(method, 0.0)
            model, costs = fit_one(case, suite, method, mixing, grid_size=grid_size, steps=steps, sweeps=sweeps)
            metadata = save_model(folder / (method + ".npz"), model)
            pending.append({"suite": suite, "family": family, "seed": seed, "support": support,
                            "method": method, "mixing": mixing, "model": metadata, "fit_costs": costs,
                            "folder": name, "evidence_sha256": digest(folder / "evidence.npz")})
        # No final query score is computed until every candidate in this case is saved.
        for row in pending:
            method = row["method"]
            model = load_model(folder / (method + ".npz"))
            clock, cpu = time.perf_counter(), time.process_time()
            metrics, raw = {}, {}
            for partition in ("familiar", "independent", "extrapolation"):
                metrics[partition], vectors = score_one(model, case[partition])
                raw.update({partition + "_" + key: value for key, value in vectors.items()})
            np.savez_compressed(folder / (method + "-scores.npz"), **raw)
            row.update(metrics=metrics, evaluation_costs={"wall_seconds": time.perf_counter() - clock,
                       "process_cpu_seconds": time.process_time() - cpu},
                       score_sha256=digest(folder / (method + "-scores.npz")))
            write_json(folder / (method + ".json"), row)
            trials.append(row)
        print(json.dumps({"completed_case": name, "models": len(pending)}), flush=True)
    return {"trials": trials, "smoke": smoke}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tune", "study"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source"
    snapshot.mkdir()
    for name in ("core.py", "study.py"):
        (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
    torch.set_num_threads(1)
    costs = Costs()
    result = {"status": "failed", "source_sha256": source_hash(), "protocol": PROTOCOL,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    result["source_git_commit"] = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(args.output / "protocol.json", PROTOCOL)
    try:
        if args.mode == "tune":
            result.update(tune(args.output, smoke=args.smoke))
        else:
            if args.tuning is None:
                raise ValueError("Study requires frozen development choices")
            tuning = json.loads(args.tuning.read_text(encoding="utf-8"))
            if tuning["source_sha256"] != source_hash():
                raise ValueError("Study code differs from development code")
            if tuning["status"] != "completed":
                raise ValueError("Development did not complete")
            result["tuning_sha256"] = digest(args.tuning)
            result.update(main_study(args.output, tuning, args.seeds or PROTOCOL["main_seeds"], smoke=args.smoke))
        result["status"] = "completed"
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["costs"] = costs.record()
        write_json(args.output / "results.json", result)


if __name__ == "__main__":
    main()
