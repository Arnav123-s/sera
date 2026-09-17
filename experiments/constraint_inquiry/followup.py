"""Fresh exploratory cases for efficient, explicitly fixed residual stopping."""

import copy
import math
import time

import numpy as np
import torch

from experiments.language_inquiry.study import ROOT, read, sha, write

from .evaluate import assess
from .settling import context, endpoints, initialize, proposal_features, step, summarize
from .study import OUT, restore


def main():
    output = OUT / "followup"
    output.mkdir(exist_ok=False)
    runtime, _ = restore(ROOT / read(ROOT / "runs/CI-fits-final-001/selected.json")["checkpoint"])
    owner = runtime.owner
    write(output / "freeze.json", {"protocol": sha(OUT / "followup-protocol.md"), "source": sha(__file__)})
    rng = np.random.default_rng(25400)
    cases, records = [], []
    for family in ("ordinary", "narrow", "unreachable", "omitted_torque"):
        for _ in range(128):
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
            index = len(cases)
            cases.append({"case": index, "family": family, "model": copy.deepcopy(model)})
            for method in ("uniform", "amortized", "analytic"):
                started = time.perf_counter()
                steps = 0
                if method == "analytic":
                    v, a, g, b, d = proposal_features(model)[0].double().tolist()
                    vector = torch.tensor([g * (1 + a), g], dtype=torch.float64)
                    points = (vector * (d - (a + a * a) * v - (2 + a) * b) / vector.square().sum()).clamp(-1, 1)[None]
                else:
                    points = initialize(owner, model, method, index)
                answer = summarize(model, points)
                while method != "analytic" and answer["status"] != "CONDITIONAL_WITNESS" and steps < 12:
                    points = step(model, points)
                    steps += 1
                    answer = summarize(model, points)
                elapsed = time.perf_counter() - started
                distance = assess(model, answer["controls"], .08 if family == "omitted_torque" else 0.)
                records.append({"case": index, "family": family, "method": method, "steps": steps,
                                "seconds": elapsed, "answer": answer, "assessed_distance_m": distance,
                                "assessed_success": distance <= .03,
                                "transition_points": len(points) * 7 * 2 * (2 + 3 * steps)})
    write(output / "cases.json", cases)
    write(output / "records.json", records)
    summary = []
    for family in sorted({r["family"] for r in records}):
        for method in ("uniform", "amortized", "analytic"):
            selected = [r for r in records if r["family"] == family and r["method"] == method]
            summary.append({"family": family, "method": method, "cases": len(selected),
                            "nominal_witnesses": sum(r["answer"]["status"] == "CONDITIONAL_WITNESS" for r in selected),
                            "assessed_successes": sum(r["assessed_success"] for r in selected),
                            "gradient_steps": sum(r["steps"] for r in selected),
                            "seconds": sum(r["seconds"] for r in selected),
                            "transition_points": sum(r["transition_points"] for r in selected)})
    write(output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
