"""Supplied observation curriculum. Assessor variables never enter model inputs."""
import hashlib
import json

import numpy as np

DT = .05
STEPS = 80
HORIZON = 12
FAMILIES = ("simple", "delayed", "omitted")
CONDITIONS = ("clean", "noisy", "missing")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parameters(rng, family):
    return dict(b=float(rng.uniform(.7, 1.5)), w=float(rng.uniform(-.15, .15)),
                d=float(rng.uniform(.05, .35)), c=0. if family == "simple" else float(rng.uniform(.25, 1.3)),
                tau=float(rng.uniform(.15, .9)), q=float(rng.uniform(.3, .6)) if family == "omitted" else 0.)


def step(s, u, p):
    h, v, f = s
    rho = np.exp(-DT / p["tau"])
    a = -1 + p["b"] * u + p["w"] - p["d"] * v + f - p["q"] * v * abs(v)
    nv = v + DT * a
    return np.array([h + DT * nv, nv, rho * f - (1 - rho) * p["c"] * v])


def trajectory(p, rng):
    t = np.arange(STEPS + 1) * DT
    v = np.full_like(t, -.3)
    for frequency in (.45, 1.3, 2.8):
        phase = rng.uniform(-np.pi, np.pi)
        v += rng.uniform(.1, .4) * (np.sin(frequency * t + phase) - np.sin(frequency * t[-1] + phase))
    states = np.zeros((STEPS + 1, 3))
    initial_force = rng.uniform(-.3, .3)
    states[0] = [10 - DT * v[1:].sum(), v[0], initial_force if p["c"] != 0 else 0.]
    controls = []
    for i in range(STEPS):
        _, vel, f = states[i]
        u = ((v[i+1] - vel) / DT + 1 - p["w"] + p["d"] * vel - f + p["q"] * vel * abs(vel)) / p["b"]
        controls.append(float(u))
        states[i+1] = step(states[i], u, p)
    np.testing.assert_allclose(states[-1, :2], [10, -.3], atol=2e-13, rtol=0)
    # Endpoint equality is a supplied excitation construction, not a latent label.
    states[-1, :2] = [10, -.3]
    future_u = .8 + .25 * np.sin(np.arange(HORIZON) * .7)
    future, s = [], states[-1].copy()
    for u in future_u:
        s = step(s, u, p)
        future.append(s[:2].copy())
    return states, np.array(controls), future_u, np.array(future)


def observed(states, condition, rng):
    values, mask = states[:, :2].copy(), np.ones((len(states), 2), bool)
    if condition != "clean":
        values += rng.normal(size=values.shape) * [.002, .01]
    if condition == "missing":
        mask = rng.uniform(size=values.shape) >= .1
    mask[[0, -1]] = True
    # A common clean current sensor reading isolates history, not present noise.
    values[-1] = states[-1, :2]
    for t in range(1, len(values)):
        values[t] = np.where(mask[t], values[t], values[t-1])
    return values, mask


def features(values, controls, masks):
    return np.column_stack((values[:, 0] / 10, values[:, 1], controls,
                            masks.astype(float), np.ones(len(values)))).astype("float32")


def bank(split):
    if split not in ("train", "dev", "final"):
        raise ValueError("Unknown sealed split")
    seeds = dict(train=28111, dev=28222, final=28333)
    rng = np.random.default_rng(seeds[split])
    count = 64 if split == "train" else (8 if split == "dev" else 12)
    result = []
    for family in (("simple", "delayed") if split == "train" else FAMILIES):
        for world in range(count if split != "train" else count // 2):
            p = parameters(rng, family)
            for history in range(4 if split == "train" else 2):
                states, u, future_u, truth = trajectory(p, rng)
                conditions = ("clean",) if split == "train" else CONDITIONS
                for condition in conditions:
                    obs, mask = observed(states, condition, rng)
                    result.append(dict(id=f"{split}/{family}/{world}/{history}/{condition}", family=family,
                        world=world, history=history, condition=condition, assessor_parameters=p,
                        assessor_hidden=states[:, 2].tolist(), values=obs.tolist(), masks=mask.tolist(),
                        controls=u.tolist(), future_controls=future_u.tolist(), truth=truth.tolist()))
    return result


def training_arrays(rows):
    x = np.stack([features(np.array(r["values"][:-1]), np.array(r["controls"]), np.array(r["masks"][:-1])) for r in rows])
    y = np.stack([np.diff(np.array(r["values"])[:, 1]) / DT for r in rows]).astype("float32")
    return x, y
