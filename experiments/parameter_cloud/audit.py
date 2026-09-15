"""Reconstruct Born probabilities and replay every saved R7 prediction."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from sera.accounting import Costs

from .core import fit
from .study import (
    FAMILIES,
    PROTOCOL,
    SUITES,
    SUPPORTS,
    digest,
    fit_one,
    load_model,
    model_methods,
    problem,
    score_one,
    source_hash,
    write_json,
)


def reconstruct(model, method, suite):
    state = model["state"]
    if "grid" not in model:
        return 0.0
    size = len(model["grid"])
    if method == "marginal_product":
        probability = state[0, :, None] * state[1, None, :]
    elif method in {"filter", "diffusion"}:
        probability = state
    elif method == "dirac":
        probability = state.abs().square()
        probability = (probability.reshape(size, 2, size, 2).sum((1, 3)) if suite == "joint_two"
                       else probability.reshape(len(model["mean"]), size, 2).sum(2))
    else:
        probability = state.abs().square()
    expected = model["probability"] if suite == "joint_two" else model["marginals"]
    difference = float((probability - expected).abs().max())
    if suite == "joint_two":
        assert abs(float(probability.sum()) - 1) < 1e-12
        points = torch.cartesian_prod(model["grid"], model["grid"])
        mean = probability.flatten() @ points
        centered = points - mean
        covariance = centered.T @ (probability.flatten()[:, None] * centered)
    else:
        torch.testing.assert_close(probability.sum(1), torch.ones(len(probability), dtype=torch.float64), atol=1e-12, rtol=0)
        mean = probability @ model["grid"]
        covariance = torch.diag((probability * (model["grid"][None, :] - mean[:, None]).square()).sum(1))
    torch.testing.assert_close(mean, model["mean"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(covariance, model["covariance"], atol=1e-12, rtol=1e-12)
    assert difference < 1e-12
    return difference


def audit(root):
    recorded = json.loads((root / "results.json").read_text(encoding="utf-8"))
    assert recorded["status"] == "completed" and not recorded["smoke"]
    assert recorded["source_sha256"] == source_hash()
    for name in ("core.py", "study.py"):
        assert (root / "source" / name).read_bytes() == (Path(__file__).parent / name).read_bytes()
    expected = {(suite, family, seed, support, method)
                for suite, family, seed, support in itertools.product(SUITES, FAMILIES, PROTOCOL["main_seeds"], SUPPORTS)
                for method in model_methods(suite)}
    actual = {(r["suite"], r["family"], r["seed"], r["support"], r["method"]) for r in recorded["trials"]}
    assert actual == expected and len(actual) == len(recorded["trials"])
    checked_cases, checks = {}, []
    for index, row in enumerate(recorded["trials"]):
        folder, method = root / row["folder"], row["method"]
        path = folder / (method + ".npz")
        assert digest(path) == row["model"]["checkpoint_sha256"]
        assert digest(folder / (method + "-scores.npz")) == row["score_sha256"]
        assert digest(folder / "evidence.npz") == row["evidence_sha256"]
        if row["folder"] not in checked_cases:
            identities, overlap = set(), 0
            case = problem(row["suite"], row["family"], row["seed"], row["support"])
            with np.load(folder / "evidence.npz", allow_pickle=False) as data:
                for partition in ("support", "validation", "familiar", "independent", "extrapolation"):
                    x = data[partition + "_design"]
                    current = {a.tobytes() for a in x}
                    overlap += len(identities.intersection(current))
                    identities.update(current)
                    for key, generated in zip(("design", "truth", "observed"), case[partition]):
                        np.testing.assert_array_equal(data[partition + "_" + key], generated.numpy())
            assert overlap == 0
            checked_cases[row["folder"]] = {"cross_partition_overlap": overlap, "unique_designs": len(identities)}
        model = load_model(path)
        born_error = reconstruct(model, method, row["suite"])
        maximum = 0.0
        with np.load(folder / "evidence.npz", allow_pickle=False) as data, np.load(folder / (method + "-scores.npz"), allow_pickle=False) as raw:
            for partition in ("familiar", "independent", "extrapolation"):
                values = tuple(torch.from_numpy(data[partition + "_" + key].copy()) for key in ("design", "truth", "observed"))
                summary, vectors = score_one(model, values)
                for name, value in summary.items():
                    if isinstance(value, str):
                        assert value == row["metrics"][partition][name]
                    else:
                        assert abs(value - row["metrics"][partition][name]) < 1e-11
                for name, value in vectors.items():
                    old = raw[partition + "_" + name]
                    error = float(np.max(np.abs(value.astype(float) - old.astype(float))))
                    maximum = max(maximum, error)
                    assert error < 1e-11
                # Independent NumPy readout check, in addition to Torch replay.
                prediction = data[partition + "_design"] @ model["mean"].numpy()
                np.testing.assert_allclose(prediction, vectors["predictions"], atol=1e-12, rtol=1e-12)
                mse = np.mean((prediction - data[partition + "_truth"])**2)
                assert abs(mse - summary["mse"]) < 1e-11
        checks.append({"case": row["folder"], "method": method, "passed": True,
                       "maximum_score_difference": maximum, "born_reconstruction_error": born_error})
        if (index + 1) % 100 == 0:
            print(f"Replayed {index + 1} models", flush=True)
    return {"passed": True, "models_replayed": len(checks), "query_partitions_replayed": 3 * len(checks),
            "source_sha256": source_hash(), "source_file_checks": 2,
            "cases": checked_cases, "checks": checks}


def diagnostics(tuning):
    """Declared numerical diagnostics on development cases; no settings change."""
    records = []
    for suite, family, method in itertools.product(SUITES, ("well_specified", "correlated"), ("schrodinger", "dirac", "imaginary", "diffusion")):
        case = problem(suite, family, 9100, 32)
        reference = None
        for steps in (16, 32, 64):
            model, costs = fit_one(case, suite, method, tuning["selected"][suite][method], steps=steps)
            validation, _ = score_one(model, case["validation"])
            probability = model.get("probability", model.get("marginals"))
            change = None if reference is None else float((probability - reference).abs().sum())
            records.append({"suite": suite, "family": family, "method": method, "steps": steps,
                            "validation": validation, "probability_l1_change_from_previous": change,
                            "fit_costs": costs, "norm_error": model["max_unitary_norm_error"]})
            reference = probability
    # Independent NumPy solve of the same declared Gaussian objective.
    case = problem("factor_six", "well_specified", 9100, 32)
    design, _, observed = case["support"]
    from .core import Evidence
    result = fit(Evidence(design, observed), "gaussian")
    x, y = design.numpy(), observed.numpy()
    covariance = np.linalg.solve(np.eye(6) + x.T @ x / .2**2, np.eye(6))
    mean = covariance @ (x.T @ y / .2**2)
    np.testing.assert_allclose(result["mean"].numpy(), mean, atol=1e-12, rtol=1e-12)
    return {"records": records, "independent_numpy_gaussian_parity": True,
            "settings_changed_after_diagnostics": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    costs = Costs()
    result = {"passed": False}
    try:
        result.update(audit(args.root))
        tuning = json.loads(args.tuning.read_text(encoding="utf-8"))
        result["numerical_diagnostics"] = diagnostics(tuning)
    except Exception as error:
        result.update(passed=False, failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["costs"] = costs.record()
        result["audit_script_sha256"] = digest(Path(__file__))
        write_json(args.output / "verification.json", result)


if __name__ == "__main__":
    main()
