"""Assessor-only environment and split generation. Never imported by core.py."""

import hashlib

import numpy as np

FAMILIES = ("circle", "ellipse", "exception", "random")


def rng_for(seed, namespace):
    value = int.from_bytes(hashlib.sha256(f"{seed}/{namespace}".encode()).digest()[:8], "little")
    return np.random.default_rng(value)


def environment(seed, family):
    if family not in FAMILIES:
        raise ValueError("Unknown assessor task family")
    rng = rng_for(seed, "environment/" + family)
    return {"family": family, "center": rng.uniform(-.5, .5, 2), "radius": rng.uniform(.4, 1.2),
            "phase": rng.uniform(-np.pi, np.pi), "axis_ratio": rng.uniform(.35, .85),
            "rotation": rng.uniform(-np.pi, np.pi), "exception_center": rng.uniform(-.35, .35)}


def truth(env, t, rng):
    if env["family"] == "random":
        return rng.normal(0, 1, (len(t), 2))
    angle = np.pi * t + env["phase"]
    radius = env["radius"]
    if env["family"] == "exception":
        radius = radius * (1 + .5 * np.exp(-.5 * ((t - env["exception_center"]) / .085) ** 2))
    points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    if env["family"] == "ellipse":
        points[:, 1] *= env["axis_ratio"]
        c, s = np.cos(env["rotation"]), np.sin(env["rotation"])
        points = points @ np.array([[c, s], [-s, c]])
    return points + env["center"]


def bank(seed, family, noise, role, count):
    rng = rng_for(seed, f"{family}/{noise}/{role}")
    if role == "withheld_arc":
        t = rng.uniform(.65, 1., count) * rng.choice([-1, 1], count)
    elif role == "extrapolation":
        t = rng.uniform(1.05, 1.6, count) * rng.choice([-1, 1], count)
    else:
        t = rng.uniform(-.6, .6, count)
    clean = truth(environment(seed, family), t, rng)
    return {"t": t, "truth": clean, "y": clean + rng.normal(0, noise, clean.shape)}
