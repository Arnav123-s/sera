"""Finite, explicitly supplied generator languages and serializable prediction.

The controls learn constants, expression choices, network weights and residuals.
They do not discover a general geometric ontology. No simulator is imported here.
"""

import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def f32(value):
    return np.asarray(value, dtype=np.float32).tolist()


@dataclass(frozen=True)
class Observations:
    t: np.ndarray
    y: np.ndarray
    role: str
    origin: str = "observed"

    def __post_init__(self):
        t, y = np.array(self.t, dtype=float, copy=True), np.array(self.y, dtype=float, copy=True)
        if t.ndim != 1 or y.shape != (len(t), 2) or not len(t):
            raise ValueError("Expected scalar coordinates and two observed outputs")
        if not np.isfinite(t).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite evidence")
        if self.role not in {"support", "selection", "calibration", "query"}:
            raise ValueError("Unknown evidence role")
        if self.origin not in {"observed", "imagined", "prediction"}:
            raise ValueError("Unknown evidence origin")
        t.flags.writeable = y.flags.writeable = False
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)

    def require(self, role):
        if self.role != role or self.origin != "observed":
            raise ValueError("Only observed evidence with the required role is admissible")


def term_value(term, t):
    """Interpret a bounded expression AST; no eval or arbitrary callable code."""
    op = term[0]
    if op == "one":
        return np.ones_like(t)
    if op == "t":
        return t
    if op == "power":
        return t ** int(term[1])
    if op == "sin":
        return np.sin(float(term[1]) * np.pi * t)
    if op == "cos":
        return np.cos(float(term[1]) * np.pi * t)
    if op == "mul":
        return term_value(term[1], t) * term_value(term[2], t)
    if op == "hinge":
        return np.maximum(t - float(term[1]), 0)
    raise ValueError("Unknown expression node")


def design(spec, t):
    """Joint design in interleaved (x_i,y_i) order."""
    t = np.asarray(t, dtype=float)
    if spec["kind"] == "circle":
        c, s = np.cos(np.pi * t), np.sin(np.pi * t)
        a = np.zeros((len(t), 2, 4))
        a[:, 0, 0], a[:, 1, 1] = 1, 1
        a[:, 0, 2], a[:, 0, 3] = c, -s
        a[:, 1, 2], a[:, 1, 3] = s, c
    elif spec["kind"] == "radial_orbit":
        c, s = np.cos(spec["frequency"] * np.pi * t), np.sin(spec["frequency"] * np.pi * t)
        a = np.zeros((len(t), 2, 6))
        a[:, 0, 0], a[:, 1, 1] = 1, 1
        a[:, 0, 2], a[:, 0, 3], a[:, 0, 4], a[:, 0, 5] = c, -s, t*c, -t*s
        a[:, 1, 2], a[:, 1, 3], a[:, 1, 4], a[:, 1, 5] = s, c, t*s, t*c
    elif spec["kind"] == "expression":
        basis = np.column_stack([term_value(term, t) for term in spec["terms"]])
        a = np.zeros((len(t), 2, 2 * basis.shape[1]))
        a[:, 0, 0::2], a[:, 1, 1::2] = basis, basis
    else:
        raise ValueError("Unknown generator kind")
    return a.reshape(2 * len(t), -1)


def expression(name, terms):
    return {"kind": "expression", "name": name, "terms": [["one"], *terms]}


CIRCLE = {"kind": "circle", "name": "circle"}
ELLIPSE = expression("ellipse", [["sin", 1], ["cos", 1]])
LINE = expression("line", [["t"]])


def menu(method):
    if method == "C_circle":
        return [CIRCLE]
    if method == "D_family":
        return [CIRCLE, ELLIPSE, LINE]
    if method == "E_symbolic":
        atoms = [["t"], ["power", 2], ["power", 3], ["sin", 1], ["cos", 1],
                 ["sin", 2], ["cos", 2], ["mul", ["t"], ["sin", 1]],
                 ["mul", ["t"], ["cos", 1]]]
        return [expression(f"expression-{i}", list(terms)) for i, terms in enumerate(
            itertools.chain.from_iterable(itertools.combinations(atoms, k) for k in (0, 1, 2, 3)))]
    if method in {"G_program", "H_residual"}:
        return [CIRCLE, ELLIPSE, LINE,
                {"kind": "radial_orbit", "name": "linear-radius-orbit", "frequency": 1},
                expression("second-harmonic", [["sin", 1], ["cos", 1], ["sin", 2], ["cos", 2]]),
                expression("piecewise-line", [["t"], ["hinge", -.25], ["hinge", .25]])]
    raise ValueError("Method has no symbolic menu")


def fit_linear(spec, evidence, noise):
    evidence.require("support")
    a, y = design(spec, evidence.t), evidence.y.ravel()
    variance, prior_variance = noise * noise, 4.0
    precision = a.T @ a / variance + np.eye(a.shape[1]) / prior_variance
    b = a.T @ y / variance
    covariance = np.linalg.solve(precision, np.eye(a.shape[1]))
    mean = covariance @ b
    logdet = np.linalg.slogdet(precision * prior_variance)[1]
    evidence_logp = -.5 * (len(y) * math.log(2 * math.pi * variance) + logdet
                            + y @ y / variance - b @ mean)
    return {"spec": spec, "mean": f32(mean), "covariance": f32(covariance),
            "noise_variance": variance, "log_evidence": float(evidence_logp)}


def linear_predict(model, t):
    a = design(model["spec"], t)
    mean = a @ np.asarray(model["mean"])
    variance = np.einsum("ij,jk,ik->i", a, np.asarray(model["covariance"]), a)
    return mean.reshape(-1, 2), np.maximum(variance, 0).reshape(-1, 2) + model["noise_variance"]


def network():
    return torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(),
                               torch.nn.Linear(32, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))


def network_predict(weights, t):
    x = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    for layer in ("0", "2", "4"):
        x = x @ np.asarray(weights[layer + ".weight"]).T + weights[layer + ".bias"]
        if layer != "4":
            x = np.tanh(x)
    return x


METHODS = ("A_episodic", "B_interpolation", "C_circle", "D_family", "E_symbolic",
           "F_neural", "G_program", "H_residual")


def predict(artifact, t):
    t = np.asarray(t, dtype=float)
    method = artifact["method"]
    if t.ndim != 1 or not np.isfinite(t).all():
        raise ValueError("Finite scalar query coordinates required")
    availability = np.ones(len(t), dtype=bool)
    if method in {"A_episodic", "B_interpolation"}:
        x, y = np.asarray(artifact["coordinates"]), np.asarray(artifact["outputs"])
        if method == "A_episodic":
            # The supplied coordinate codec is float32 for both storage and lookup.
            match = np.asarray(t, np.float32)[:, None] == x[None, :]
            availability = match.any(axis=1)
            mean = np.where(availability[:, None], y[match.argmax(axis=1)], 0)
            variance = np.full_like(mean, artifact["noise_variance"])
            variance[~availability] = 2.0
        else:
            mean = np.column_stack([np.interp(t, x, y[:, j]) for j in range(2)])
            variance = np.full_like(mean, artifact["noise_variance"])
        components, weights = [(mean, variance)], np.ones(1)
    elif method == "F_neural":
        mean = network_predict(artifact["weights"], t)
        components, weights = [(mean, np.full_like(mean, artifact["noise_variance"]))], np.ones(1)
    else:
        components = [linear_predict(model, t) for model in artifact["models"]]
        weights = np.asarray(artifact["class_weights"])
        if method == "H_residual":
            mean, variance = components[0]
            x, residual = np.asarray(artifact["coordinates"]), np.asarray(artifact["residuals"])
            distance = (t[:, None] - x[None, :]) / .08
            kernel = np.exp(-.5 * distance ** 2)
            correction = kernel @ residual / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-30)
            permitted = (t >= x[0]) & (t <= x[-1])
            components = [(mean + correction * permitted[:, None], variance)]
    means, variances = np.array([p[0] for p in components]), np.array([p[1] for p in components])
    mean = np.einsum("k,kij->ij", weights, means)
    variance = np.einsum("k,kij->ij", weights, variances + means ** 2) - mean ** 2
    inflation = artifact.get("variance_inflation", 0.)
    return {"mean": mean, "variance": np.maximum(variance + inflation, 1e-12),
            "component_means": means, "component_variances": variances + inflation,
            "class_weights": weights, "available": availability}


def fit(method, support, selection, calibration, noise, initialization_seed=42, steps=160):
    for rows, role in ((support, "support"), (selection, "selection"), (calibration, "calibration")):
        rows.require(role)
    sets = [set(rows.t.tolist()) for rows in (support, selection, calibration)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Observation roles overlap")
    if noise <= 0 or method not in METHODS:
        raise ValueError("Invalid method or noise")
    start, cpu = time.perf_counter(), time.process_time()
    artifact = {"schema": "sera.generator.v1", "method": method,
                "precision": {"learned_arrays": "float32 values serialized as JSON numbers",
                              "calibration_class_weights_and_metadata": "float64 JSON numbers",
                              "inference_arithmetic": "float64 NumPy"},
                "coordinate_schema": "supplied-phase-pi-t/two-coordinates/dimensionless",
                "noise_variance": noise * noise, "dependencies": [],
                "support_digest": digest({"t": support.t.tolist(), "y": support.y.tolist()}),
                "guard": {"type": "empirical-observed-range", "low": float(min(support.t)),
                          "high": float(max(support.t)), "learned_semantics": False}}
    work = {"support_observations": len(support.t), "selection_observations": len(selection.t),
            "calibration_observations": len(calibration.t), "candidate_fits": 0,
            "gradient_steps": 0, "candidate_trace": []}
    order = np.argsort(support.t)
    if method in {"A_episodic", "B_interpolation"}:
        artifact.update(coordinates=f32(support.t[order]), outputs=f32(support.y[order]))
    elif method == "F_neural":
        torch.manual_seed(initialization_seed)
        model = network()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        x, y = torch.tensor(support.t[:, None], dtype=torch.float32), torch.tensor(support.y, dtype=torch.float32)
        vx, vy = torch.tensor(selection.t[:, None], dtype=torch.float32), torch.tensor(selection.y, dtype=torch.float32)
        best, saved = float("inf"), None
        for step in range(steps):
            optimizer.zero_grad()
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0 or step + 1 == steps:
                with torch.no_grad():
                    score = float(torch.mean((model(vx) - vy) ** 2))
                work["candidate_trace"].append({"step": step + 1, "selection_mse": score})
                if score < best:
                    best = score
                    saved = {name: f32(value.detach().numpy()) for name, value in model.state_dict().items()}
        if saved is None:
            raise ValueError("Neural budget must contain at least one update")
        artifact["weights"] = saved
        work.update(gradient_steps=steps, candidate_fits=1)
    else:
        candidates = [fit_linear(spec, support, noise) for spec in menu(method)]
        work["candidate_fits"] = len(candidates)
        for candidate in candidates:
            mu, _ = linear_predict(candidate, selection.t)
            score = float(np.mean((mu - selection.y) ** 2))
            work["candidate_trace"].append({"name": candidate["spec"]["name"],
                "selection_mse": score, "log_evidence": candidate["log_evidence"],
                "parameters": len(candidate["mean"])})
        if method in {"E_symbolic", "G_program", "H_residual"}:
            # Frozen selection penalty is a heuristic, not a model posterior.
            chosen = min(range(len(candidates)), key=lambda i:
                         work["candidate_trace"][i]["selection_mse"] + 1e-4 * len(candidates[i]["mean"]))
            artifact.update(models=[candidates[chosen]], class_weights=[1.0],
                            selection="selection-MSE-plus-1e-4-per-parameter")
        else:
            logw = np.asarray([m["log_evidence"] for m in candidates])
            weights = np.exp(logw - logw.max())
            weights /= weights.sum()
            artifact.update(models=candidates, class_weights=weights.tolist(),
                            selection="equal-class-prior-exact-Gaussian-marginal-likelihood")
        if method == "H_residual":
            base, _ = linear_predict(artifact["models"][0], support.t)
            artifact.update(coordinates=f32(support.t[order]), residuals=f32((support.y - base)[order]))
    # Frozen calibration data adjust marginal dispersion, never coefficients or class choices.
    prediction = predict(artifact, calibration.t)
    error = (prediction["mean"] - calibration.y) ** 2
    artifact["variance_inflation"] = float(max(0, np.mean(error - prediction["variance"])))
    artifact["calibration"] = {
        "observations": len(calibration.t), "mse": float(error.mean()),
        "menu_inadequacy_alarm": bool(error.mean() > 9 * noise * noise),
        "interpretation": "Empirical in-region residual alarm; not a guarantee outside that regime",
    }
    artifact["uncertainty"] = "marginal model-conditional distribution plus calibration variance"
    artifact = json.loads(canonical(artifact))
    work.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - start,
                artifact_bytes=len(canonical(artifact).encode()))
    return artifact, work


def metrics(prediction, truth, observed):
    means, variances = prediction["component_means"], prediction["component_variances"]
    logp = -.5 * (np.log(2 * np.pi * variances) + (observed[None] - means) ** 2 / variances)
    logp += np.log(np.maximum(prediction["class_weights"], 1e-300))[:, None, None]
    largest = logp.max(axis=0)
    mixture_logp = largest + np.log(np.exp(logp - largest).sum(axis=0))
    radius = 1.959963984540054 * np.sqrt(prediction["variance"])
    return {"mse": float(np.mean((prediction["mean"] - truth) ** 2)),
            "marginal_nll": float(-mixture_logp.mean()),
            "coverage95": float(np.mean(abs(prediction["mean"] - observed) <= radius)),
            "width95": float(2 * radius.mean()), "availability": float(prediction["available"].mean())}
