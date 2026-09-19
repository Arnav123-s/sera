"""Independent NumPy qualification and fresh simulated assessment for field views."""

import numpy as np
from scipy.integrate import solve_ivp

from experiments.gap_inquiry import digest


def unpack(weights, kind):
    w = np.asarray(weights, dtype=float)
    if w.shape != ((4,) if kind == "radial" else (6,)) or not np.isfinite(w).all():
        raise ValueError("Wrong independent coefficient contract")
    if kind == "radial":
        return np.diag([w[0], w[0]]), *w[1:]
    if kind == "directional":
        return np.array([[w[0], w[2]], [w[2], w[1]]]), *w[3:]
    raise ValueError("Unknown physical field")


def prediction(x, weights, kind):
    x = np.asarray(x, dtype=float)
    q, v, u = x[..., :2], x[..., 2:4], x[..., 4:]
    k, c, gamma, b = unpack(weights, kind)
    jv = np.stack((-v[..., 1], v[..., 0]), -1)
    return -(q @ k.T) - c * (q*q).sum(-1, keepdims=True)*q - gamma*v + b*jv + u


def independent_fit(rows, kind):
    """Build coefficients by probing the independent force, then augmented QR/SVD."""
    x, y = np.array([r["x"] for r in rows]), np.array([r["y"] for r in rows])
    n = 4 if kind == "radial" else 6
    a = np.stack([prediction(x, np.eye(n)[i], kind) - x[:, 4:] for i in range(n)], -1).reshape(-1, n)
    return np.linalg.lstsq(np.vstack((a, np.eye(n)*.001)), np.r_[(y-x[:, 4:]).reshape(-1), np.zeros(n)], rcond=None)[0]


def qualification(subject, record, weights, selection, audit):
    from experiments.structural_field import observation
    if len(selection) != 16 or len(audit) != 32 or len(record["observations"]) < 12:
        raise ValueError("A complete independent selection and adequacy bank is required")
    checked = []
    used = {r["id"] for r in record["observations"]}
    if len(used) != len(record["observations"]):
        raise ValueError("Duplicate acquired evidence")
    for rows, purpose in ((selection, "selection"), (audit, "adequacy")):
        for row in rows:
            r = observation(row, subject, purpose)
            if r["id"] in used or r["source"] != record["source"]:
                raise ValueError("Assessment overlaps acquisition or changed source")
            used.add(r["id"])
            checked.append(r)
    errors, replay = {}, {}
    sx, sy = np.asarray([r["x"] for r in selection]), np.asarray([r["y"] for r in selection])
    for kind, values in weights.items():
        duplicate = independent_fit(record["observations"], kind)
        replay[kind] = float(np.max(np.abs(duplicate - values)))
        if replay[kind] > 1e-7:
            raise ValueError("Owner weights do not replay from their acquired evidence")
        errors[kind] = float(np.mean((prediction(sx, values, kind) - sy)**2))
    selected = "directional" if errors["directional"] < .8*errors["radial"] and errors["radial"]-errors["directional"] > .0001 else "radial"
    ax, ay = np.asarray([r["x"] for r in audit]), np.asarray([r["y"] for r in audit])
    residual = prediction(ax, weights[selected], selected) - ay
    rmse, tail = float(np.sqrt(np.mean(residual**2))), float(np.quantile(np.abs(residual), .95))
    k, c, gamma, _ = unpack(weights[selected], selected)
    passive = bool(np.linalg.eigvalsh(k).min() >= 0 and c >= 0 and gamma >= 0)
    accepted = rmse <= .04 and tail <= .09 and passive
    body = {"subject": subject, "goal": record["goal"], "source": record["source"], "revision": record["revision"],
            "acquisition": digest(record["observations"]), "weights": digest(weights), "selection": selection, "audit": audit,
            "selection_mse": errors, "selected": selected, "rmse": rmse, "q95_absolute_error": tail,
            "passive": passive, "accepted": accepted, "independent_refit_max_difference": replay,
            "next_action": "Answer the original conditional question" if accepted else "Preserve alternatives; acquire or construct a missing mechanism",
            "grade": "EMPIRICAL_SCOPE_QUALIFICATION", "causal_identification": "candidate preference under tested assumptions"}
    return body | {"id": digest(body)}


def same_qualification(saved, replay):
    """Keep source/evidence/decision exact; tolerate only numerical replay noise."""
    numerical = {"selection_mse", "rmse", "q95_absolute_error", "independent_refit_max_difference"}
    for receipt in (saved, replay):
        if receipt.get("id") != digest({k: v for k, v in receipt.items() if k != "id"}):
            return False
    if {k: v for k, v in saved.items() if k not in numerical | {"id"}} != {
            k: v for k, v in replay.items() if k not in numerical | {"id"}}:
        return False
    for name in numerical:
        a, b = saved[name], replay[name]
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                return False
            a, b = [a[k] for k in sorted(a)], [b[k] for k in sorted(b)]
        if not np.allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=False):
            return False
    return True


def states(rng, n, narrow=False, wide=False):
    x = rng.uniform(-1.5, 1.5, (n, 6))
    x[:, 4:] /= 3
    if narrow:
        x[:, 1] *= .015
    if wide:
        x[:, :4] *= 1.8
    return x


class Source:
    """Assessor-owned numerical environment, with explicit simulated provenance."""
    def __init__(self, family, seed, subject):
        if family not in ("radial", "directional", "omitted"):
            raise ValueError("Unknown prospective family")
        self.family, self.seed, self.subject = family, seed, subject
        r = np.random.default_rng(seed + 130013)
        k = float(r.uniform(.7, 1.3))
        self.weights = [k, k, 0., float(r.uniform(.08, .22)), float(r.uniform(.05, .15)), float(r.uniform(-.2, .2))]
        if family != "radial":
            self.weights[1], self.weights[2] = k*2.3, .37
        self.identity = digest({"family": family, "seed": seed, "weights": self.weights, "contract": "SF-001 simulated force; independent teacher"})

    def truth(self, x):
        x = np.asarray(x, dtype=float)
        q, v = x[..., :2], x[..., 2:4]
        kx, ky, cross, c, gamma, b = self.weights
        radius = (q*q).sum(-1)
        y = np.stack((-kx*q[..., 0]-cross*q[..., 1]-c*radius*q[..., 0]-gamma*v[..., 0]-b*v[..., 1],
                      -cross*q[..., 0]-ky*q[..., 1]-c*radius*q[..., 1]-gamma*v[..., 1]+b*v[..., 0]), -1) + x[..., 4:]
        if self.family == "omitted":
            y[..., 0] += .6*np.sin(2.1*q[..., 1])
            y[..., 1] += .4*np.tanh(2.7*q[..., 0]*v[..., 0])
        return y

    def observe(self, x, index, purpose):
        key = digest({"source": self.identity, "x": list(x), "index": str(index), "purpose": purpose})
        noise = np.random.default_rng(int(key[:16], 16)).normal(0, .01, 2)
        return {"id": key, "subject": self.subject, "source": self.identity, "kind": "SIMULATED_OBSERVATION",
                "purpose": purpose, "x": list(map(float, x)), "y": (self.truth(x)+noise).tolist(),
                "units": ["m", "m/s", "m/s^2", "m/s^2"]}

    def bank(self, purpose, n, *, narrow=False, wide=False):
        offsets = {"initial": 33001, "candidates": 43001, "selection": 53001, "adequacy": 63001,
                   "final": 73001, "wide": 83001, "trajectory": 93001}
        return states(np.random.default_rng(self.seed+offsets[purpose]), n, narrow, wide)

    def trajectory(self, initial, control, steps=20, dt=.05):
        def derivative(_, z):
            return np.r_[z[2:], self.truth(np.r_[z, control])]
        result = solve_ivp(derivative, (0, steps*dt), initial, t_eval=np.linspace(0, steps*dt, steps+1), rtol=1e-10, atol=1e-11)
        if not result.success:
            raise ValueError("Independent teacher integration failed")
        return result.y.T
