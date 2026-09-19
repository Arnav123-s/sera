"""Owner-held temporal weights and retained intervention obligations.

The finite mechanism vocabulary and acquisition controller are supplied. Counts
learn numerical parameters; a separate assessor qualifies their use.
"""

import copy
import math
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.gap_inquiry import ROOT, digest, sha
from experiments.structural_field import FieldR1
from experiments.verified_completion.credit import decode, encode
from sera.session_state import model_identity

OUT = ROOT / "research-continuation/47_intervention_understanding"
KINDS = ("temporal", "pulse_loss")
DT = .25


def contracts():
    paths = ("experiments/intervention_model.py", "experiments/intervention_assess.py",
             "experiments/intervention_events.py", "research-continuation/47_intervention_understanding/PROTOCOL.md")
    return {p: sha(ROOT / p) for p in paths}


def program(value):
    if not isinstance(value, dict) or set(value) != {"ticks", "pulses"}:
        raise ValueError("Use ticks and explicit pulse boundaries")
    ticks, pulses = value["ticks"], value["pulses"]
    if type(ticks) is not int or not 2 <= ticks <= 24 or not isinstance(pulses, list):
        raise ValueError("Invalid bounded control duration")
    if any(type(p) is not int or not 0 < p < ticks for p in pulses) or pulses != sorted(set(pulses)):
        raise ValueError("Pulse boundaries must be unique, ordered and internal")
    if len(pulses) > 6:
        raise ValueError("At most six pulses are supported in this finite contract")
    return {"ticks": ticks, "pulses": list(pulses)}


def features(value):
    p = program(value)
    sign, area = 1, 0
    for tick in range(p["ticks"]):
        if tick in p["pulses"]:
            sign = -sign
        area += sign
    return [DT * p["ticks"], abs(DT * area), float(len(p["pulses"]))]


def rows_arrays(rows, width):
    groups = [g for row in rows for g in row["groups"]]
    x = np.array([features(g["applied"])[:width] for g in groups], dtype=float)
    n = np.array([g["shots"] for g in groups], dtype=float)
    k = np.array([g["plus"] for g in groups], dtype=float)
    return x, n, k


def checked_observation(row, subject, source, purpose):
    row = copy.deepcopy(row)
    expected = {"id", "subject", "source", "kind", "purpose", "decision", "requested", "groups", "monitor"}
    if set(row) != expected or row["subject"] != subject or row["source"] != source or row["purpose"] != purpose:
        raise ValueError("Evidence subject, source, purpose or schema changed")
    if row["kind"] not in {"SIMULATED_OBSERVATION", "MEASURED_OBSERVATION"}:
        raise ValueError("An imagined prediction is not an independent observation")
    if not isinstance(row["id"], str) or not row["id"] or not isinstance(row["decision"], str):
        raise ValueError("Evidence and decision identities are required")
    requested = program(row["requested"])
    if not 1 <= len(row["groups"]) <= 64:
        raise ValueError("An independent actuator monitor must report actual histories")
    seen, total, pulse_shots = set(), 0, 0
    for group in row["groups"]:
        if set(group) != {"applied", "shots", "plus"}:
            raise ValueError("Invalid outcome group")
        actual = program(group["applied"])
        if actual["ticks"] != requested["ticks"] or not set(actual["pulses"]).issubset(requested["pulses"]):
            raise ValueError("The supported monitor contract reports dropped requested pulses")
        key = digest(actual)
        if key in seen:
            raise ValueError("Duplicate actual history")
        seen.add(key)
        if type(group["shots"]) is not int or not 1 <= group["shots"] <= 65536:
            raise ValueError("Invalid shot count")
        if type(group["plus"]) is not int or not 0 <= group["plus"] <= group["shots"]:
            raise ValueError("Invalid independently measured outcome count")
        total += group["shots"]
        pulse_shots += group["shots"] * len(actual["pulses"])
    monitor = row["monitor"]
    expected_monitor = {"schema", "source", "shots", "requested_pulse_shots", "applied_pulse_shots", "receipt"}
    if set(monitor) != expected_monitor or monitor["schema"] != "sera.actuator-monitor.1" or not monitor["source"]:
        raise ValueError("A separate actuator receipt is required")
    if (monitor["shots"] != total or monitor["requested_pulse_shots"] != total * len(requested["pulses"])
            or monitor["applied_pulse_shots"] != pulse_shots):
        raise ValueError("Actuator receipt and observed histories disagree")
    receipt = digest({k: v for k, v in row.items() if k != "monitor"})
    if monitor["receipt"] != receipt:
        raise ValueError("Actuator and outcome association changed")
    return row


class DecayWeights(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError("Unknown installed temporal model")
        self.kind = kind
        size = 2 if kind == "temporal" else 3
        self.weight = nn.Parameter(torch.tensor([.2, .2, .03][:size], dtype=torch.float64), requires_grad=False)
        self.register_buffer("information", torch.zeros(size, size, dtype=torch.float64))
        self.register_buffer("shots", torch.zeros((), dtype=torch.int64))

    def forward(self, x):
        return .5 + .5 * torch.exp(-(x[..., :len(self.weight)] @ self.weight))

    def fit(self, rows):
        x, n, k = rows_arrays(rows, len(self.weight))
        tx, tn, tk = (torch.tensor(a, dtype=torch.float64) for a in (x, n, k))
        initial = self.weight.detach().clamp_min(.001)
        raw = nn.Parameter(torch.log(torch.expm1(initial)))
        optimizer = torch.optim.LBFGS([raw], max_iter=100, tolerance_grad=1e-10,
                                      tolerance_change=1e-13, line_search_fn="strong_wolfe")
        evaluations = 0

        def closure():
            nonlocal evaluations
            evaluations += 1
            optimizer.zero_grad()
            p = (.5 + .5 * torch.exp(-(tx @ F.softplus(raw)))).clamp(1e-12, 1-1e-12)
            loss = -(tk * p.log() + (tn-tk) * torch.log1p(-p)).sum() / tn.sum()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite temporal fit")
            loss.backward()
            return loss

        optimizer.step(closure)
        weight = F.softplus(raw).detach()
        if not torch.isfinite(weight).all() or torch.any(weight > 100):
            raise ValueError("Unbounded fit requires investigation")
        with torch.no_grad():
            self.weight.copy_(weight)
            p = self(tx).clamp(1e-9, 1-1e-9)
            derivative = .5 * torch.exp(-(tx @ weight))
            gram = tx.T @ ((tn * derivative.square() / (p * (1-p)))[:, None] * tx)
            self.information.copy_(gram)
            self.shots.fill_(int(n.sum()))
        return {"closure_evaluations": evaluations, "shots": int(n.sum()), "groups": len(n)}

    def identity(self):
        return digest({"kind": self.kind, "state": encode(self.state_dict())})


class InterventionR1(FieldR1):
    @classmethod
    def attach(cls, owner):
        if type(owner) is not FieldR1:
            raise ValueError("Restore the latest qualified FieldR1 owner")
        owner.__class__ = cls
        owner.intervention_models = nn.ModuleDict()
        owner.intervention_contract = contracts()
        return owner

    def export_config(self):
        return {**super().export_config(), "intervention_contract": self.intervention_contract,
                "intervention_keys": sorted(self.intervention_models)}


class InterventionSession:
    def __init__(self, base, saved=None):
        self.base, self.parent_owner = base, base.identity()
        self.owner = InterventionR1.attach(base.owner)
        self.subjects = {}
        if saved is not None:
            if saved["schema"] != "sera.intervention-session.1" or saved["parent_owner"] != self.parent_owner or saved["contracts"] != contracts():
                raise ValueError("Changed intervention lineage or executable contract")
            for subject, state in saved["subjects"].items():
                self.restore_subject(subject, state)
            if self.identity() != saved["owner"]:
                raise ValueError("Saved owner weights changed")

    @staticmethod
    def key(subject, kind):
        return subject + "__" + kind

    def model(self, subject, kind):
        return self.owner.intervention_models[self.key(subject, kind)]

    def identity(self):
        return model_identity(self.owner)

    def _install(self, subject):
        if not re.fullmatch(r"[a-z0-9_-]{1,100}", subject):
            raise ValueError("Invalid subject identifier")
        for kind in KINDS:
            self.owner.intervention_models[self.key(subject, kind)] = DecayWeights(kind)

    def start(self, subject, goal, source, original_program):
        if subject in self.subjects or not isinstance(goal, str) or not goal or not source:
            raise ValueError("A fresh immutable goal and source are required")
        original_program = program(original_program)
        self._install(subject)
        self.subjects[subject] = {"goal": goal, "goal_id": digest([subject, goal, original_program]),
                                  "original_program": program(original_program), "source": source,
                                  "observations": [], "revision": 0, "pending": None,
                                  "decisions": [], "fits": [], "qualification": None, "initial": None,
                                  "status": "OPEN", "alternatives": None, "history": [], "qualified_versions": None}

    def model_versions(self, subject):
        return {k: self.model(subject, k).identity() for k in KINDS}

    def commit_probe(self, subject, control, purpose="acquisition"):
        r = self.subjects[subject]
        if r["pending"] is not None:
            raise ValueError("Complete or explicitly preserve the pending intervention")
        if purpose != "acquisition":
            raise ValueError("Only acquisition changes fitted weights")
        decision = {"subject": subject, "source": r["source"], "goal_id": r["goal_id"],
                    "revision": r["revision"], "models": self.model_versions(subject),
                    "program": program(control), "purpose": purpose}
        decision["id"] = digest(decision)
        r["pending"] = copy.deepcopy(decision)
        return copy.deepcopy(decision)

    def observe(self, subject, row):
        r = self.subjects[subject]
        d = r["pending"]
        row = checked_observation(row, subject, r["source"], "acquisition")
        if d is None or row["decision"] != d["id"] or row["requested"] != d["program"]:
            raise ValueError("An observation must answer the committed intervention")
        if d["goal_id"] != r["goal_id"] or d["revision"] != r["revision"] or d["models"] != self.model_versions(subject):
            raise ValueError("Stale intervention or changed original goal")
        if any(old["id"] == row["id"] for old in r["observations"]):
            raise ValueError("Duplicate evidence")
        if row["id"] in {a["id"] for h in r["history"] for a in h["selection"]+h["audit"]}:
            raise ValueError("Assessment evidence stays sealed after revision")
        r["observations"].append(row)
        r["decisions"].append(d)
        r["pending"] = None
        r["revision"] += 1
        r["qualification"], r["status"] = None, "OPEN"
        r["qualified_versions"] = None

    def fit(self, subject):
        r = self.subjects[subject]
        if r["pending"] is not None or not r["observations"]:
            raise ValueError("Fit only complete observed evidence")
        old = {k: copy.deepcopy(self.model(subject, k).state_dict()) for k in KINDS}
        try:
            costs = {k: self.model(subject, k).fit(r["observations"]) for k in KINDS}
        except Exception:
            for k in KINDS:
                self.model(subject, k).load_state_dict(old[k])
            raise
        r["fits"].append({"revision": r["revision"], "cost": costs})
        r["qualification"], r["status"] = None, "OPEN"
        r["qualified_versions"] = None
        x, _, _ = rows_arrays(r["observations"], 3)
        rank = int(np.linalg.matrix_rank(x))
        if rank == 1:
            total = float(self.model(subject, "temporal").weight.sum())
            r["alternatives"] = {"type": "PASSIVE_EQUIVALENCE_CLASS", "constraint": "a+b=acquired decay rate; pulse loss unobserved",
                                 "sum": total, "a_interval": [0., total], "b_equals": "sum-a",
                                 "representatives": [[total, 0., 0.], [total/2, total/2, 0.], [0., total, 0.]],
                                 "pulse_loss": "unidentified", "design_rank": rank}
        else:
            r["alternatives"] = {"type": "FINITE_CANDIDATE_MODELS", "design_rank": rank,
                                 "models": {k: self.model(subject, k).weight.detach().tolist() for k in KINDS},
                                 "other_mechanisms": "preserved as an open adequacy obligation"}
        if r["initial"] is None:
            r["initial"] = self.answer(subject)
        return costs

    def qualify(self, subject, selection, audit):
        from experiments.intervention_assess import qualification
        r = self.subjects[subject]
        if r["pending"] is not None:
            raise ValueError("Resolve pending intervention before qualification")
        if r["qualification"] is not None:
            raise ValueError("Repeated qualification is not new evidence or credit")
        used = {a["id"] for h in r["history"] for a in h["selection"]+h["audit"]}
        if used & {a["id"] for a in selection+audit}:
            raise ValueError("Fresh independent assessment is required after revision")
        weights = {k: self.model(subject, k).weight.detach().tolist() for k in KINDS}
        q = qualification(subject, r, weights, selection, audit)
        r["qualification"] = q
        r["qualified_versions"] = self.model_versions(subject)
        r["history"].append(copy.deepcopy(q))
        r["status"] = "QUALIFIED_CONDITIONAL_ANSWER" if q["accepted"] else "OPEN"
        return copy.deepcopy(q)

    def answer(self, subject, control=None):
        r = self.subjects[subject]
        if r["goal_id"] != digest([subject, r["goal"], r["original_program"]]):
            raise ValueError("Original goal changed")
        requested = program(r["original_program"] if control is None else control)
        q = r["qualification"]
        if q is not None and (r["qualified_versions"] != self.model_versions(subject) or q["revision"] != r["revision"] or q["source"] != r["source"] or q["evidence_identity"] != digest(r["observations"]) or q["weight_identity"] != digest({k: self.model(subject, k).weight.detach().tolist() for k in KINDS})):
            raise ValueError("Qualification became stale")
        selected = q["selected"] if q else "temporal"
        model = self.model(subject, selected)
        x = torch.tensor(features(requested), dtype=torch.float64)
        p = float(model(x).detach())
        gram = model.information.detach().numpy()
        xx = x[:len(model.weight)].numpy()
        cov = np.linalg.pinv(gram, rcond=1e-10)
        error = (p-.5) * math.sqrt(max(0., float(xx @ cov @ xx)))
        identifiable = np.linalg.norm(xx - gram @ np.linalg.pinv(gram, rcond=1e-10) @ xx) < 1e-7
        alternatives = copy.deepcopy(r["alternatives"])
        if alternatives and alternatives["type"] == "PASSIVE_EQUIVALENCE_CLASS":
            alternatives["conditional_probabilities"] = [.5+.5*math.exp(-float(np.dot(features(requested), w))) for w in alternatives["representatives"]]
        supported = bool(q and q["accepted"] and requested["ticks"] <= 16 and len(requested["pulses"]) <= 4)
        return {"kind": "MODEL_CONDITIONAL_PREDICTION", "status": "QUALIFIED_SCOPE" if supported else "INVESTIGATION_REQUIRED",
                "original_goal": r["goal"], "goal_id": r["goal_id"], "program": requested, "plus_probability": p,
                "selected": selected, "model": model.identity(), "revision": r["revision"],
                "model_confidence": {"local_standard_error": error if identifiable else None,
                                     "identifiable_in_acquisition_design": bool(identifiable),
                                     "meaning": "local Fisher approximation conditional on this supplied model"},
                "model_adequacy": None if q is None else q["adequacy"], "alternatives": alternatives,
                "assumptions": ["the specified pulse history actually occurred", "binary plus-X readout", "fixed parameters within the assessed scope"],
                "evidence_kind": sorted({o["kind"] for o in r["observations"]}), "physical_uniqueness_proved": False}

    def subject_state(self, subject):
        return {"record": copy.deepcopy(self.subjects[subject]),
                "weights": {k: encode(self.model(subject, k).state_dict()) for k in KINDS}}

    def restore_subject(self, subject, state):
        if subject in self.subjects:
            raise ValueError("Do not overwrite an existing investigation")
        r = copy.deepcopy(state["record"])
        program(r["original_program"])
        if r["goal_id"] != digest([subject, r["goal"], r["original_program"]]):
            raise ValueError("Original goal changed")
        if r["revision"] != len(r["observations"]) or len({o["id"] for o in r["observations"]}) != r["revision"] or len(r["decisions"]) != r["revision"]:
            raise ValueError("Evidence history changed")
        for index, (row, decision) in enumerate(zip(r["observations"], r["decisions"], strict=True)):
            checked_observation(row, subject, r["source"], "acquisition")
            if decision["id"] != digest({k: v for k, v in decision.items() if k != "id"}) or row["decision"] != decision["id"] or decision["revision"] != index or decision["goal_id"] != r["goal_id"] or decision["source"] != r["source"] or decision["subject"] != subject or decision["program"] != row["requested"]:
                raise ValueError("Broken decision lineage")
        self._install(subject)
        self.subjects[subject] = r
        for kind in KINDS:
            self.model(subject, kind).load_state_dict(decode(state["weights"][kind]))
            fitted_revision = r["fits"][-1]["revision"] if r["fits"] else 0
            if fitted_revision:
                x, n, _ = rows_arrays(r["observations"][:fitted_revision], len(self.model(subject, kind).weight))
                w = self.model(subject, kind).weight.detach().numpy()
                p = np.clip(.5+.5*np.exp(-x@w), 1e-9, 1-1e-9)
                fisher = x.T @ ((n*(.5*np.exp(-x@w))**2/(p*(1-p)))[:, None]*x)
                if int(self.model(subject, kind).shots) != int(n.sum()) or not np.allclose(fisher, self.model(subject, kind).information, atol=1e-7, rtol=1e-9):
                    raise ValueError("Saved confidence or evidence count changed")
        if r["pending"] is not None:
            d = r["pending"]
            if d["id"] != digest({k: v for k, v in d.items() if k != "id"}) or d["revision"] != r["revision"] or d["goal_id"] != r["goal_id"] or d["models"] != self.model_versions(subject):
                raise ValueError("Pending decision changed")
        if r["qualification"] is not None:
            from experiments.intervention_assess import qualification, same_qualification
            q = r["qualification"]
            weights = {k: self.model(subject, k).weight.detach().tolist() for k in KINDS}
            actual = qualification(subject, r, weights, q["selection"], q["audit"])
            if not same_qualification(q, actual):
                raise ValueError("Changed saved qualification")
            if r["qualified_versions"] != self.model_versions(subject) or not r["history"] or r["history"][-1] != q:
                raise ValueError("Saved qualification history changed")

    def state(self):
        return {"schema": "sera.intervention-session.1", "parent_owner": self.parent_owner,
                "contracts": contracts(), "owner": self.identity(),
                "subjects": {s: self.subject_state(s) for s in self.subjects}}
