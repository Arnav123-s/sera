"""One frozen held-out pass, with assessor-only changed-law outcomes kept separate."""

import argparse
import copy
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.language_inquiry.study import ROOT, read, sha, write

from .data import corpus, encode
from .model import ConstraintR1, apply, source
from .settling import context, endpoints, initialize, proposal_features, step, summarize
from .study import OUT, parent, restore


def language(folder):
    owner = parent().owner
    records, summaries = [], []
    for result in read(folder / "summary.json"):
        saved = torch.load(ROOT / result["checkpoint"], weights_only=True, map_location="cpu")
        if saved["source"] != source():
            raise ValueError("Model source changed")
        model = copy.deepcopy(owner)
        ConstraintR1.attach(model, saved["seed"], saved["kind"])
        apply(model, saved["delta"])
        for partition in ("pairs", "surface"):
            rows = corpus(partition)
            x = torch.tensor(encode([r["text"] for r in rows]))
            with torch.no_grad():
                logits = model.binding(x)
                predictions = torch.stack([p.argmax(-1) for p in logits], 1).tolist()
                # Independently housed equal-parameter copy; no extra training claim.
                detached = copy.deepcopy(model)
                copied = detached.binding(x)
                copy_error = max(float((a - b).abs().max()) for a, b in zip(logits, copied))
            for row, predicted in zip(rows, predictions):
                records.append({"seed": saved["seed"], "kind": saved["kind"], "steps": saved["step"],
                                "partition": partition, **row, "prediction": predicted,
                                "exact": predicted == row["target"]})
            chosen = records[-len(rows):]
            summaries.append({"seed": saved["seed"], "kind": saved["kind"], "steps": saved["step"],
                              "partition": partition, "correct": sum(r["exact"] for r in chosen),
                              "total": len(rows), "independent_copy_max_logit_difference": copy_error})
    return records, summaries


def assess(model, control, torque):
    angle, velocity = model["angle"], model["velocity"]
    a, gain, bias = model["mean"]
    for u in control:
        velocity = float(np.clip(a * velocity + gain * u + bias + torque * math.sin(2 * angle), -.65, .65))
        angle += velocity
    xy = np.array(model["center"]) + model["radius"] * np.array([math.cos(angle), math.sin(angle)])
    return float(np.linalg.norm(xy - np.array(model["target"])))


def cases(owner):
    rng = np.random.default_rng(25200)
    rows = []
    for family in ("ordinary", "narrow", "unreachable", "omitted_torque"):
        for i in range(256):
            model = context(owner, [0., 0.])
            model["angle"] = float(rng.uniform(-math.pi, math.pi))
            model["velocity"] = float(rng.uniform(-.25, .25))
            control = rng.uniform(-.8, .8, 2)
            if family == "narrow":
                control = rng.choice([-1., 1.]) * rng.uniform(.97, 1., 2)
            with torch.no_grad():
                target = endpoints(model, torch.tensor([control.tolist()], dtype=torch.float64))[0, 0].numpy()
            if family == "unreachable":
                target = np.array(model["center"]) + (target - np.array(model["center"])) * (1 + .4 / model["radius"])
            model["target"] = target.tolist()
            rows.append({"case": len(rows), "family": family, "within_family": i, "model": model,
                         "assessor_torque": .08 if family == "omitted_torque" else 0.})
    return rows


def numerical(owner, rows):
    results = []
    grid = torch.tensor(list(itertools.product(np.linspace(-1, 1, 33), repeat=2)), dtype=torch.float64)
    for row in rows:
        model = row["model"]
        for method in ("uniform", "uniform_refine", "amortized", "amortized_refine", "grid", "analytic"):
            started = time.perf_counter()
            gradients = 0
            if method == "grid":
                points = grid
            elif method == "analytic":
                feature = proposal_features(model)[0].double()
                velocity, a, gain, bias, offset = feature.tolist()
                vector = torch.tensor([gain * (1 + a), gain], dtype=torch.float64)
                residual = offset - (a + a * a) * velocity - (2 + a) * bias
                points = (vector * residual / vector.square().sum()).clamp(-1, 1)[None]
            else:
                points = initialize(owner, model, "amortized" if method.startswith("amortized") else "uniform", row["case"])
                if method.endswith("refine"):
                    for _ in range(12):
                        points = step(model, points)
                        gradients += 1
            answer = summarize(model, points)
            elapsed = time.perf_counter() - started
            error = assess(model, answer["controls"], row["assessor_torque"])
            results.append({"case": row["case"], "family": row["family"], "method": method,
                            **answer, "assessed_distance_m": error, "assessed_success": error <= .03,
                            "gradient_batches": gradients,
                            "transition_points": len(points) * 7 * 2 * (gradients + 2),
                            "initializer_forwards": int(method.startswith("amortized")), "seconds": elapsed})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    folder = ROOT / "runs/CI-fits-final-001"
    write(args.output / "freeze.json", {"source": source(), "evaluator": sha(Path(__file__)),
                                        "protocol": sha(OUT / "protocol.md"), "selection": read(folder / "selected.json")})
    records, metrics = language(folder)
    write(args.output / "language-records.json", records)
    write(args.output / "language-summary.json", metrics)
    runtime, _ = restore(ROOT / read(folder / "selected.json")["checkpoint"])
    rows = cases(runtime.owner)
    write(args.output / "cases.json", rows)
    results = numerical(runtime.owner, rows)
    write(args.output / "numerical-records.json", results)
    summary = []
    for family in sorted({r["family"] for r in results}):
        for method in sorted({r["method"] for r in results}):
            selected = [r for r in results if r["family"] == family and r["method"] == method]
            summary.append({"family": family, "method": method, "cases": len(selected),
                            "nominal_witnesses": sum(r["status"] == "CONDITIONAL_WITNESS" for r in selected),
                            "assessed_successes": sum(r["assessed_success"] for r in selected),
                            "mean_error_m": float(np.mean([r["assessed_distance_m"] for r in selected])),
                            "seconds": sum(r["seconds"] for r in selected),
                            "gradient_batches": sum(r["gradient_batches"] for r in selected),
                            "transition_points": sum(r["transition_points"] for r in selected)})
    write(args.output / "numerical-summary.json", summary)
    print({"language_records": len(records), "case_method_records": len(results), "cases": len(rows)})


if __name__ == "__main__":
    main()
