"""Small empirical force/potential weights on the continuing R1 owner.

Coordinates, units and candidate forms are supplied. Observations fit executable
weights; an independent comparison qualifies where their predictions are useful.
"""

import copy
import re

import numpy as np
import torch
from torch import nn

from experiments.counterfactual_challenge import ChallengedR1
from experiments.gap_inquiry import ROOT, digest, sha
from experiments.verified_completion.credit import decode, encode
from sera.session_state import model_identity

OUT = ROOT / "research-continuation/46_structural_refinement"
KINDS = ("radial", "directional")


def contracts():
    return {p: sha(ROOT / p) for p in ("experiments/structural_field.py", "experiments/structural_check.py",
                                      "research-continuation/46_structural_refinement/PROTOCOL.md")}


def vector(value, size, bound=10.):
    x = np.asarray(value, dtype=np.float64)
    if x.shape != (size,) or not np.isfinite(x).all() or np.max(np.abs(x)) > bound:
        raise ValueError("A bounded finite typed vector is required")
    return x.tolist()


def observation(row, subject, purpose):
    row = copy.deepcopy(row)
    expected = {"id", "subject", "source", "kind", "purpose", "x", "y", "units"}
    if set(row) != expected or row["subject"] != subject or row["purpose"] != purpose:
        raise ValueError("Observation subject, purpose or schema changed")
    if row["kind"] not in {"SIMULATED_OBSERVATION", "MEASURED_OBSERVATION"}:
        raise ValueError("An imagined consequence is not an observation")
    if row["units"] != ["m", "m/s", "m/s^2", "m/s^2"]:
        raise ValueError("Position, velocity, input acceleration and measured acceleration units are required")
    if any(not isinstance(row[k], str) or not 1 <= len(row[k]) <= 512 for k in ("id", "source")):
        raise ValueError("Observation and source identities are required")
    row["x"], row["y"] = vector(row["x"], 6, 4.), vector(row["y"], 2, 100.)
    return row


def design(x, kind):
    """A force layer with tied spatial coefficients; input remains additive."""
    q, v = x[..., :2], x[..., 2:4]
    xx, yy = q.unbind(-1)
    zero = torch.zeros_like(xx)
    if kind == "radial":
        columns = [torch.stack((-xx, -yy), -1)]
    elif kind == "directional":
        columns = [torch.stack((-xx, zero), -1), torch.stack((zero, -yy), -1), torch.stack((-yy, -xx), -1)]
    else:
        raise ValueError("Unknown structural hypothesis")
    columns += [-q * q.square().sum(-1, keepdim=True), -v, torch.stack((-v[..., 1], v[..., 0]), -1)]
    return torch.stack(columns, -1)


class PhysicalWeights(nn.Module):
    def __init__(self, kind, integral_factors=(.5, .25)):
        super().__init__()
        if kind not in KINDS:
            raise ValueError("Unknown field")
        self.kind = kind
        n = 4 if kind == "radial" else 6
        self.weight = nn.Parameter(torch.zeros(n, dtype=torch.float64), requires_grad=False)
        self.register_buffer("gram", torch.eye(n, dtype=torch.float64) * 1e-6)
        self.register_buffer("rhs", torch.zeros(n, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("integral_factors", torch.tensor(integral_factors, dtype=torch.float64))

    @torch.no_grad()
    def observe(self, x, y):
        x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
        a = design(x, self.kind)
        gram, rhs = self.gram + a.T @ a, self.rhs + a.T @ (y - x[4:])
        weight = torch.linalg.solve(gram, rhs)
        if not torch.isfinite(weight).all():
            raise ValueError("Nonfinite field update")
        self.gram.copy_(gram)
        self.rhs.copy_(rhs)
        self.weight.copy_(weight)
        self.count.add_(1)

    def forward(self, x):
        return design(x, self.kind) @ self.weight + x[..., 4:]

    def coefficients(self):
        w = self.weight
        if self.kind == "radial":
            k, rest = torch.eye(2, dtype=w.dtype) * w[0], w[1:]
        else:
            k, rest = torch.stack((w[[0, 2]], w[[2, 1]])), w[3:]
        return k, *rest.unbind()

    def energy(self, state):
        q, v = state[..., :2], state[..., 2:]
        k, c, _, _ = self.coefficients()
        quadratic, quartic = self.integral_factors
        return quadratic * (q * (q @ k)).sum(-1) + quartic * c * q.square().sum(-1).square() + quadratic * v.square().sum(-1)

    def passive(self):
        k, c, gamma, _ = self.coefficients()
        return bool(torch.linalg.eigvalsh(k).min() >= 0 and c >= 0 and gamma >= 0)

    def identity(self):
        return digest({"kind": self.kind, "state": encode(self.state_dict())})


class FieldR1(ChallengedR1):
    @classmethod
    def attach(cls, owner):
        if type(owner) is not ChallengedR1:
            raise ValueError("Restore the qualified current counterfactual owner")
        from experiments.self_chosen.equations import learned_map
        quadratic = learned_map(owner, "integral", [0, 1])
        quartic = learned_map(owner, "integral", [0, 0, 0, 1])
        owner.field_integrals = (float(quadratic[2]), float(quartic[4]))
        if owner.field_integrals != (.5, .25) or any(v for i, v in enumerate(quadratic) if i != 2) or any(v for i, v in enumerate(quartic) if i != 4):
            raise ValueError("The retained exact integral route must qualify the potential coefficients")
        owner.__class__ = cls
        owner.physical_fields = nn.ModuleDict()
        owner.physical_contract = contracts()
        return owner

    def export_config(self):
        return {**super().export_config(), "physical_fields": self.physical_contract,
                "field_keys": sorted(self.physical_fields), "field_integrals": self.field_integrals}


class FieldSession:
    def __init__(self, base, saved=None):
        self.base, self.parent_owner = base, base.identity()
        self.owner = FieldR1.attach(base.owner)
        self.subjects, self.events, self.pending = {}, [], None
        if saved is not None:
            if saved["schema"] != "sera.structural-field.1" or saved["parent_owner"] != self.parent_owner or saved["contracts"] != contracts():
                raise ValueError("Changed field lineage or source requires explicit migration")
            self.subjects, self.events, self.pending = copy.deepcopy((saved["subjects"], saved["events"], saved["pending"]))
            for subject, record in self.subjects.items():
                self._install(subject)
                for row in record["observations"]:
                    observation(row, subject, "acquisition")
                if record["revision"] != len(record["observations"]) or len({r["id"] for r in record["observations"]}) != record["revision"]:
                    raise ValueError("Saved observation history changed")
                for kind in KINDS:
                    self.owner.physical_fields[self.key(subject, kind)].load_state_dict(decode(saved["weights"][self.key(subject, kind)]))
                    if int(self.model(subject, kind).count) != record["revision"]:
                        raise ValueError("Saved numerical evidence count changed")
                if record["qualification"] is not None:
                    from experiments.structural_check import qualification, same_qualification
                    q = record["qualification"]
                    weights = {k: self.model(subject, k).weight.detach().tolist() for k in KINDS}
                    if not same_qualification(q, qualification(subject, record, weights, q["selection"], q["audit"])):
                        raise ValueError("Saved independent qualification did not replay")
            if self.identity() != saved["owner"]:
                raise ValueError("Changed saved field weights")

    @staticmethod
    def key(subject, kind):
        return digest(subject)[:24] + "_" + kind

    def _install(self, subject):
        for kind in KINDS:
            self.owner.physical_fields[self.key(subject, kind)] = PhysicalWeights(kind, self.owner.field_integrals)

    def identity(self):
        return model_identity(self.owner)

    def model(self, subject, kind):
        return self.owner.physical_fields[self.key(subject, kind)]

    def start(self, subject, goal, source, original_state=None):
        if not isinstance(subject, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", subject) or subject in self.subjects:
            raise ValueError("Use a new bounded subject identity")
        if not isinstance(goal, str) or not 1 <= len(goal) <= 1000 or not isinstance(source, str) or not source:
            raise ValueError("The original goal and evidence source are required")
        original_state = vector(original_state, 6, 4.) if original_state is not None else None
        self._install(subject)
        self.subjects[subject] = {"goal": goal, "source": source, "observations": [], "revision": 0,
                                  "selected": "radial", "qualification": None, "history": [],
                                  "original_state": original_state}

    def observe(self, subject, row):
        record = self.subjects[subject]
        row = observation(row, subject, "acquisition")
        if row["source"] != record["source"] or row["id"] in {r["id"] for r in record["observations"]}:
            raise ValueError("Duplicate or different-source observation")
        prior_assessment = {r["id"] for h in record["history"] for key in ("selection", "audit") for r in h[key]}
        if row["id"] in prior_assessment:
            raise ValueError("Preserve sealed assessment separately from acquired training evidence")
        if len(record["observations"]) >= 512:
            raise ValueError("Declare a new bounded observation extension")
        backups = {kind: copy.deepcopy(self.model(subject, kind).state_dict()) for kind in KINDS}
        try:
            for kind in KINDS:
                self.model(subject, kind).observe(row["x"], row["y"])
        except Exception:
            for kind in KINDS:
                self.model(subject, kind).load_state_dict(backups[kind])
            raise
        record["observations"].append(row)
        record["revision"] += 1
        record["qualification"] = None
        self.events.append({"kind": "FIELD_OBSERVATION", "subject": subject, "observation": row["id"], "revision": record["revision"]})

    def qualify(self, subject, selection, audit):
        from experiments.structural_check import qualification
        record = self.subjects[subject]
        if any(h["revision"] == record["revision"] and h["selection"] == selection and h["audit"] == audit for h in record["history"]):
            raise ValueError("Repeated qualification earns no new progress")
        used = {r["id"] for h in record["history"] for key in ("selection", "audit") for r in h[key]}
        if used & {r["id"] for r in selection + audit}:
            raise ValueError("Fresh independent assessment is required after a model update")
        models = {kind: self.model(subject, kind).weight.detach().tolist() for kind in KINDS}
        receipt = qualification(subject, record, models, selection, audit)
        if any(r["id"] == receipt["id"] for r in record["history"]):
            raise ValueError("Repeated qualification earns no new progress")
        record["selected"], record["qualification"] = receipt["selected"], receipt
        record["history"].append(receipt)
        self.events.append({"kind": "FIELD_QUALIFICATION", "subject": subject, "receipt": receipt["id"]})
        return receipt

    def view(self, subject):
        return FieldView(self, subject)

    def state(self):
        return {"schema": "sera.structural-field.1", "parent_owner": self.parent_owner, "contracts": contracts(),
                "owner": self.identity(), "subjects": copy.deepcopy(self.subjects), "events": copy.deepcopy(self.events),
                "pending": copy.deepcopy(self.pending),
                "weights": {k: encode(v.state_dict()) for k, v in self.owner.physical_fields.items()}}


class FieldView:
    """A branch pins its model and evidence revision; all queries are read-only."""
    def __init__(self, session, subject):
        self.session, self.subject = session, subject
        record = session.subjects[subject]
        if record["qualification"] is not None:
            from experiments.structural_check import qualification, same_qualification
            q = record["qualification"]
            weights = {k: session.model(subject, k).weight.detach().tolist() for k in KINDS}
            try:
                current = qualification(subject, record, weights, q["selection"], q["audit"])
            except ValueError as error:
                raise ValueError("Stale qualification for the current physical weights") from error
            if not same_qualification(q, current) or q["selected"] != record["selected"]:
                raise ValueError("Stale qualification for the current physical weights")
        self.revision, self.kind = record["revision"], record["selected"]
        self.model = session.model(subject, self.kind)
        self.version = self.model.identity()
        self.qualification = digest(record["qualification"])

    def validate(self):
        record = self.session.subjects[self.subject]
        if (record["revision"] != self.revision or record["selected"] != self.kind
                or self.session.model(self.subject, self.kind) is not self.model or self.model.identity() != self.version
                or digest(record["qualification"]) != self.qualification):
            raise ValueError("Stale physical view; restore the updated evidence and model")
        return record

    def imagine(self, x):
        record = self.validate()
        x = vector(x, 6, 4.)
        q = record["qualification"]
        supported = bool(q and q["accepted"] and max(abs(v) for v in x[:4]) <= 1.5 and max(abs(v) for v in x[4:]) <= .5)
        value = self.model(torch.tensor(x, dtype=torch.float64)).detach().tolist()
        return {"kind": "MODEL_CONDITIONAL_PREDICTION", "status": "QUALIFIED_SCOPE" if supported else "INVESTIGATION_REQUIRED",
                "goal": record["goal"], "subject": self.subject, "acceleration": value, "model": self.version,
                "representation": self.kind, "evidence_revision": self.revision, "source": record["source"],
                "energy_interpretation": "specific energy under unit mass" if self.model.passive() else "unqualified energy parameters",
                "assumptions": ["supplied 2D state and origin", "fixed coefficients within this branch", "additive input acceleration"],
                "evidence_kind": sorted({r["kind"] for r in record["observations"]})}

    def trajectory(self, state, control, steps=20, dt=.05):
        self.validate()
        if type(steps) is not int or not 1 <= steps <= 200 or not 0 < dt <= .1:
            raise ValueError("Declare a finite trajectory horizon")
        z = torch.tensor(vector(state, 4, 4.), dtype=torch.float64)
        u = torch.tensor(vector(control, 2, .5), dtype=torch.float64)
        def derivative(value):
            return torch.cat((value[2:], self.model(torch.cat((value, u)))))
        path = [z.tolist()]
        for _ in range(steps):
            k1 = derivative(z)
            k2 = derivative(z + dt * k1 / 2)
            k3 = derivative(z + dt * k2 / 2)
            k4 = derivative(z + dt * k3)
            z = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if not torch.isfinite(z).all() or torch.max(z.abs()) > 10:
                return {"status": "NUMERICAL_STOP", "path": path, "model": self.version}
            path.append(z.tolist())
        scoped = all(max(map(abs, row)) <= 1.5 for row in path)
        qualified = self.imagine([*state, *control])["status"] == "QUALIFIED_SCOPE"
        status = "QUALIFIED_CONDITIONAL_TRAJECTORY" if scoped and qualified else "CONDITIONAL_EXTRAPOLATION" if qualified else "UNQUALIFIED_CONDITIONAL_TRAJECTORY"
        return {"status": status,
                "path": path, "model": self.version, "dt": dt, "control": list(control), "branch_model_fixed": True}

    def plan(self, state, target):
        target = np.asarray(vector(target, 2, 1.5))
        branches = [self.trajectory(state, u) for u in ([0., 0.], [.5, 0.], [-.5, 0.], [0., .5], [0., -.5])]
        valid = [b for b in branches if b["status"] == "QUALIFIED_CONDITIONAL_TRAJECTORY"]
        for b in branches:
            b["goal_error"] = float(np.linalg.norm(np.asarray(b["path"][-1][:2]) - target))
        best = min(valid, key=lambda b: b["goal_error"]) if valid else None
        return {"kind": "MODEL_BASED_PLAN", "target": target.tolist(), "branches": branches, "selected": best,
                "status": "PLAN_IN_MODEL" if best else "INVESTIGATION_REQUIRED", "executed_actions": 0}


@torch.no_grad()
def r1_features(owner, observations, queries):
    """Read the actual inherited R1 memory; never encode a query's target."""
    parameter = next(owner.sequence_encoder.parameters())
    def encoded(x, y=None):
        values = parameter.new_zeros((len(x), 20))
        values[:, :6] = torch.as_tensor(x, dtype=parameter.dtype) / 2
        if y is not None:
            values[:, 6:8] = torch.as_tensor(y, dtype=parameter.dtype) / 4
            values[:, 8] = 1.
        return owner.sequence_encoder(values)
    state = owner.initial(1)
    for row in observations:
        _, state = owner.shared_step(state, encoded([row["x"]], [row["y"]]), parameter.new_ones((1, 1)))
    state = {k: v.expand(len(queries), *v.shape[1:]).clone() for k, v in state.items()}
    hidden, _ = owner.shared_step(state, encoded(queries), parameter.new_zeros((len(queries), 1)))
    return np.c_[np.ones(len(queries)), np.asarray(queries), hidden.double().numpy()]
