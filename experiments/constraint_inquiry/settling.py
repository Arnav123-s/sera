"""Continuous conditional search; supplied objective, acquired numerical operator."""

import math

import numpy as np
import torch

from experiments.language_inquiry.graph import situation


def context(owner, target, avoid=False):
    model = situation(owner)
    mean = np.array(model["mean"])
    chol = np.linalg.cholesky(np.array(model["covariance"]) + np.eye(3) * 1e-15)
    model["parameters"] = np.vstack([mean, *(mean + 2 * chol[:, j] for j in range(3)),
                                     *(mean - 2 * chol[:, j] for j in range(3))]).tolist()
    model.update(target=list(target), avoid=bool(avoid))
    return model


def endpoints(model, controls):
    params = controls.new_tensor(model["parameters"])
    angle = controls.new_full((len(controls), len(params)), model["angle"])
    velocity = torch.full_like(angle, model["velocity"])
    for t in range(2):
        velocity = (params[:, 0] * velocity + params[:, 1] * controls[:, t, None] + params[:, 2]).clamp(-.65, .65)
        angle = angle + velocity
    return controls.new_tensor(model["center"]) + model["radius"] * torch.stack((angle.cos(), angle.sin()), -1)


def energy(model, controls):
    distances = ((endpoints(model, controls) - controls.new_tensor(model["target"])) ** 2).sum(-1)
    if model["avoid"]:
        distances = (0.12 - distances.clamp_min(1e-16).sqrt()).clamp_min(0) ** 2
    return distances @ controls.new_tensor([.25] + [.125] * 6)


def proposal_features(model):
    x, y = np.array(model["target"]) - np.array(model["center"])
    offset = (math.atan2(y, x) - model["angle"] + math.pi) % (2 * math.pi) - math.pi
    return torch.tensor([[model["velocity"], *model["mean"], offset]], dtype=torch.float32)


def initialize(owner, model, method="amortized", seed=0):
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand((8, 2), generator=generator, dtype=torch.float64) * 2 - 1
    if method == "amortized":
        with torch.no_grad():
            point = owner.cloud_proposal(proposal_features(model))[0].double()
        points[:4] = (point + torch.tensor([[0., 0.], [.1, -.1], [-.1, .1], [.15, .15]])).clamp(-1, 1)
    elif method != "uniform":
        raise ValueError("Unknown initializer")
    return points


def step(model, points):
    points = points.detach().requires_grad_(True)
    grad, = torch.autograd.grad(energy(model, points).sum(), points)
    # Scale by the supplied local operator sensitivity, with finite clipping.
    a, gain, _ = model["mean"]
    scale = max(2 * model["radius"] ** 2 * gain ** 2 * ((1 + a) ** 2 + 1), 1e-5)
    return (points - .8 * grad / scale).clamp(-1, 1).detach()


def summarize(model, points):
    with torch.no_grad():
        scores = energy(model, points)
        index = int(scores.argmin())
        positions = endpoints(model, points)[index]
        errors = ((positions - points.new_tensor(model["target"])) ** 2).sum(-1).sqrt()
    nominal = float(errors[0])
    witness = nominal >= .12 if model["avoid"] else nominal <= .03
    return {"status": "CONDITIONAL_WITNESS" if witness else "UNRESOLVED_SEARCH",
            "controls": points[index].tolist(), "parameter_endpoints": positions.tolist(),
            "nominal_distance_m": nominal, "worst_parameter_distance_m": float(errors.max()),
            "energy": float(scores[index]), "occurrence": "NOT_ESTABLISHED",
            "applicability": "UNCALIBRATED; GG-GUARD-002 does not cover this route",
            "assumptions": ["Acquired circle and linear action-response family", "Continuous control interpolation",
                            "Static parameters across the imagined branch", "Declared target location and modality"],
            "uncertainty": "Finite parameter quadrature; neither exhaustive modes nor model-adequacy probability"}
