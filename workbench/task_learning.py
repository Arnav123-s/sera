"""Task-time fitting from retrieved or user-supplied examples with a held-out gate."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


def retrieve(task, examples):
    if not isinstance(task, str) or not 1 <= len(task) <= 200:
        raise ValueError("Describe the numerical transformation in 1–200 characters")
    if examples:
        return {"title": task, "examples": examples, "source": "User-supplied paired examples",
                "evidence": "User-provided values; physical provenance not independently verified."}
    query = " ".join(task.lower().replace("_", " ").replace("→", " to ").split())
    bank = json.loads((Path(__file__).parent/"materials/numerical.json").read_text())
    found = [m for m in bank if m["phrase"] in query]
    if len(found) != 1:
        raise ValueError("No unambiguous examples found in the local material library. Supply x,y examples for this task.")
    return found[0]


def learn(task, queries, examples="", tolerance=.01):
    material = retrieve(task, examples)
    text = material["examples"]
    if not isinstance(text, str) or len(text) > 50_000:
        raise ValueError("Examples must be CSV text smaller than 50 KB")
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if reader.fieldnames != ["x", "y"]:
        raise ValueError("Paired-example CSV header must be x,y")
    pairs = []
    for row in reader:
        if None in row:
            raise ValueError("Each example needs exactly x and y")
        pair = [float(row["x"]), float(row["y"])]
        if not all(math.isfinite(v) and abs(v) <= 1e9 for v in pair):
            raise ValueError("Examples must contain bounded finite numbers")
        pairs.append(pair)
    if not 18 <= len(pairs) <= 256 or len({x for x, _ in pairs}) != len(pairs):
        raise ValueError("Supply 18–256 examples with distinct x values")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 100 or any(type(x) not in (float, int) or not math.isfinite(x) or abs(x) > 1e9 for x in queries):
        raise ValueError("Provide 1–100 finite numerical inputs")
    if type(tolerance) not in (float, int) or not math.isfinite(tolerance) or not 1e-10 <= tolerance <= 1e6:
        raise ValueError("Absolute validation tolerance must be between 1e-10 and 1e6")
    data = np.array(pairs)
    order = np.random.default_rng(140003).permutation(len(data))
    n = len(data)//5
    test, validation, train = order[:n], order[n:2*n], order[2*n:]
    location, scale = float(data[train, 0].mean()), max(float(data[train, 0].std()), 1e-9)
    z = (data[:, 0]-location)/scale
    candidates = []
    for degree in (1, 2):
        matrix = np.vander(z, degree+1)
        weights, _, rank, _ = np.linalg.lstsq(matrix[train], data[train, 1], rcond=None)
        if rank != degree+1:
            raise ValueError("Examples do not identify a unique fit")
        predictions_ = matrix@weights
        candidates.append({"degree": degree, "weights": weights,
                           "validation_mae": float(np.mean(abs(predictions_[validation]-data[validation, 1]))),
                           "test_max_error": float(np.max(abs(predictions_[test]-data[test, 1]))),
                           "predictions": predictions_})
    choice = 0 if candidates[0]["validation_mae"] <= candidates[1]["validation_mae"]+1e-9 else 1
    chosen = candidates[choice]
    normalized = np.pad(chosen["weights"], (3-len(chosen["weights"]), 0))
    a, b, c = normalized
    coefficients = [float(a/scale**2), float(b/scale-2*a*location/scale**2),
                    float(c-b*location/scale+a*location**2/scale**2)]
    domain = [float(data[:, 0].min()), float(data[:, 0].max())]
    in_domain = all(domain[0] <= x <= domain[1] for x in queries)
    passes = chosen["test_max_error"] <= tolerance and in_domain
    return {"status": "ACCEPTED" if passes else "WITHHELD", "task": task, "material": material,
            "material_sha256": hashlib.sha256(text.encode()).hexdigest(), "examples": len(pairs),
            "split": {"train": train.tolist(), "selection": validation.tolist(), "test": test.tolist()},
            "degree": chosen["degree"], "coefficients": coefficients, "domain": domain,
            "validation_mae": chosen["validation_mae"], "held_out_max_error": chosen["test_max_error"],
            "tolerance": tolerance, "queries_in_example_domain": in_domain, "queries": queries,
            "held_out": [{"x": float(data[i, 0]), "expected": float(data[i, 1]),
                          "prediction": float(chosen["predictions"][i])} for i in test],
            "outputs": [], "trained_parameters": chosen["degree"]+1,
            "scope": "A supplied affine/quadratic family fitted on request. Held-out examples are finite checks, not a guarantee on arbitrary functions or new domains. Material retrieval is from a local, source-labeled library or provided examples; no autonomous web comprehension."}
