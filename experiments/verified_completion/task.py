"""Finite conditional practice; hidden outcomes never enter policy features."""

import numpy as np

from .common import OUT, digest, read

AUDIT_VARIANCE = .01
STOP = 8


def phi(uv):
    uv = np.asarray(uv, dtype=np.float64)
    return np.concatenate((uv[..., :1], -uv[..., 1:2], np.ones_like(uv[..., :1])), axis=-1)


def update(mean, covariance, design, outcome, variance):
    cross = np.einsum("bij,bj->bi", covariance, design)
    denominator = np.einsum("bi,bi->b", design, cross)+variance
    gain = cross/denominator[:, None]
    after = mean+gain*(outcome-np.einsum("bi,bi->b", design, mean))[:, None]
    cov = covariance-np.einsum("bi,bj->bij", gain, cross)
    return after, (cov+cov.transpose(0, 2, 1))/2


def make_bank(n, seed, family="matched"):
    rng = np.random.default_rng(seed)
    prior = read(OUT / "prior.json")
    center = np.array(prior["mean"])
    spread = np.array(prior["covariance"])
    theta = rng.multivariate_normal(center+(np.array([.65, -.35, .55]) if family == "shifted" else 0),
                                    spread*(1.5 if family == "shifted" else 1), n)

    def response(uv):
        value = np.einsum("b...i,bi->b...", phi(uv), theta)
        return value+(.75*np.sin(2.4*uv[..., 0])*uv[..., 1] if family == "omitted" else 0)

    support_uv = rng.uniform(-1, 1, (n, 2, 2))
    support_uv[:, 1] = support_uv[:, 0]+rng.normal(0, .08, (n, 2))
    support_var = rng.uniform(.008, .08, (n, 2))
    support_y = response(support_uv)+rng.normal(size=(n, 2))*np.sqrt(support_var)
    mean = np.broadcast_to(center, (n, 3)).copy()
    cov = np.broadcast_to(spread, (n, 3, 3)).copy()
    for t in range(2):
        mean, cov = update(mean, cov, phi(support_uv[:, t]), support_y[:, t], support_var[:, t])
    target_uv = rng.uniform(-1.35, 1.35, (n, 2))
    uv = rng.uniform(-1.25, 1.25, (n, 8, 2))
    uv[:, 0] = target_uv+rng.normal(0, .18, (n, 2))
    uv[:, 6] = support_uv[:, 0]
    design = phi(uv)
    design[:, 7] = 0
    variance = np.exp(rng.uniform(np.log(.0025), np.log(.16), (n, 8)))
    variance[:, 6], variance[:, 7] = support_var[:, 0], 9.
    price = rng.uniform(.02, .3, (n, 8))
    price[rng.random(n) < .5] *= 50
    duplicate = np.zeros((n, 8))
    duplicate[:, 6] = 1
    irrelevant = np.zeros((n, 8))
    irrelevant[:, 7] = 1
    outcomes = response(uv)+rng.normal(size=(n, 8))*np.sqrt(variance)
    outcomes[:, 6], outcomes[:, 7] = support_y[:, 0], rng.normal(0, 3, n)
    perms = np.stack([rng.permutation(8) for _ in range(n)])
    index = np.arange(n)[:, None]
    clean = response(target_uv[:, None, :])[:, 0]
    result = {"mean": mean, "covariance": cov, "target": phi(target_uv), "target_uv": target_uv,
              "design": design[index, perms], "variance": variance[index, perms], "price": price[index, perms],
              "duplicate": duplicate[index, perms], "irrelevant": irrelevant[index, perms],
              "outcomes": outcomes[index, perms], "audit_clean": clean,
              "audit_y": clean+rng.normal(0, np.sqrt(AUDIT_VARIANCE), n),
              "support_uv": support_uv, "support_y": support_y, "support_variance": support_var,
              "theta_assessor": theta, "seed": np.array(seed)}
    return result


def features(bank, mean=None, covariance=None):
    cov = bank["covariance"] if covariance is None else covariance
    x, q = bank["design"], bank["target"]
    qv = np.einsum("bi,bij,bj->b", q, cov, q)
    av = np.einsum("bai,bij,baj->ba", x, cov, x)
    cross = np.einsum("bai,bij,bj->ba", x, cov, q)
    result = np.stack((np.broadcast_to(qv[:, None], av.shape), av, cross, bank["variance"],
                       bank["price"], bank["duplicate"], bank["irrelevant"], np.zeros_like(av)), axis=-1)
    stop = np.zeros((len(q), 1, 8))
    stop[:, 0, 0], stop[:, 0, 7] = qv, 1
    return np.concatenate((result, stop), axis=1)


def score(mean, variance, value):
    return -.5*(np.log(2*np.pi*variance)+(value-mean)**2/variance)


def consequences(bank, actions):
    index = np.arange(len(actions))
    a = np.minimum(actions, 7)
    design = bank["design"][index, a].copy()
    design[(actions == STOP) | (bank["duplicate"][index, a] == 1)] = 0
    m, c = update(bank["mean"], bank["covariance"], design,
                  bank["outcomes"][index, a], bank["variance"][index, a])
    q = bank["target"]
    before = np.einsum("bi,bi->b", q, bank["mean"])
    before_var = np.einsum("bi,bij,bj->b", q, bank["covariance"], q)+AUDIT_VARIANCE
    after = np.einsum("bi,bi->b", q, m)
    after_var = np.einsum("bi,bij,bj->b", q, c, q)+AUDIT_VARIANCE
    cost = .1*bank["price"][index, a]*(actions != STOP)
    reward = score(after, after_var, bank["audit_y"])-score(before, before_var, bank["audit_y"])-cost
    return {"actions": actions, "before": before, "before_variance": before_var,
            "after": after, "after_variance": after_var, "cost": cost, "reward": reward,
            "squared_error": (after-bank["audit_clean"])**2,
            "covered": np.abs(after-bank["audit_y"]) <= 1.95996398454*np.sqrt(after_var)}


def analytic(bank, kind="goal"):
    f = features(bank)
    qv, av, cross, nv, cost, dup, _, stop = np.moveaxis(f, -1, 0)
    if kind == "goal":
        reduction = cross**2/(av+nv+1e-30)*(1-dup)
        values = .5*np.log((qv+AUDIT_VARIANCE)/(qv-reduction+AUDIT_VARIANCE))-.1*cost
    else:
        values = .5*np.log1p(av/(nv+1e-30))*(1-dup)-.1*cost
    values[stop == 1] = 0
    return values.argmax(-1)


class PracticeSource:
    """Trusted local practice provider; public requests contain no hidden outcomes."""
    def __init__(self, seed):
        self.bank = make_bank(1, seed)
        self.identity = digest({"provider": "VC-001 conditional practice", "seed": seed,
                                "prior": read(OUT / "prior.json")["identity"]})

    def public(self):
        keys = ("target", "design", "variance", "price", "duplicate", "irrelevant",
                "support_uv", "support_y", "support_variance")
        return {key: self.bank[key][0].tolist() for key in keys} | {"source": self.identity,
                    "origin": "MODEL_CONDITIONAL_PRACTICE", "variable": "acceleration", "unit": "m/s^2"}

    def receipt(self, decision_id, action, purpose):
        if purpose not in {"acquisition", "independent_audit"}:
            raise ValueError("Practice receipt role required")
        if purpose == "acquisition" and not 0 <= action < STOP:
            raise ValueError("STOP cannot acquire an observation")
        value = float(self.bank["audit_y"][0] if purpose == "independent_audit" else self.bank["outcomes"][0, action])
        # Audit identity is provider-bound, not caller-name-bound; renaming a goal cannot duplicate credit.
        event = digest({"source": self.identity, "purpose": purpose,
                        "action": action if purpose == "acquisition" else None})
        record = {"source": self.identity, "purpose": purpose, "origin": "MODEL_CONDITIONAL_PRACTICE",
                  "event_id": event, "decision": decision_id, "action": action, "value": value}
        return record | {"identity": digest(record)}

    def verify(self, receipt):
        return receipt == self.receipt(receipt["decision"], receipt["action"], receipt["purpose"])
