"""Registered online dynamics, paid factual history and revision-bound plans."""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from sera.contracts import EvidenceKind, Observation, Provenance
from sera.generative import GenerativeSharedR1
from sera.session_state import model_identity
from sera.shared import replace_shared_owner
from sera.storage import digest

PRIOR = np.array([.85, .06, 0.], dtype=np.float64)
PRECISION = np.diag([20., 200., 500.])
NOISE = .012
SPEED_LIMIT = .65


def fingerprint():
    directory = Path(__file__).parent
    return digest({name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                   for name in ("core.py", "control.py")})


def wrap(angle):
    return (np.asarray(angle)+np.pi) % (2*np.pi)-np.pi


class InteractiveR1(GenerativeSharedR1):
    """No independent network: the original owner gains registered sufficient statistics."""

    @classmethod
    def extend(cls, parent, window, detect_change=False):
        if type(parent) is not GenerativeSharedR1 or type(window) is not int or window < 3:
            raise ValueError("Extension needs the exact generative predecessor and a bounded window")
        result = copy.deepcopy(parent)
        result.__class__ = cls
        result.interaction_parent = model_identity(parent)
        result.interaction_source = fingerprint()
        result.interaction_window = window
        result.interaction_detect_change = bool(detect_change)
        values = {"gram": np.zeros((3, 3)), "rhs": np.zeros(3), "mean": PRIOR.copy(),
                  "covariance": np.linalg.inv(PRECISION), "position": np.zeros(2),
                  "angle": np.zeros(()), "velocity": np.zeros(())}
        for name, value in values.items():
            result.register_buffer("interaction_"+name, torch.as_tensor(value, dtype=torch.float64).clone())
        result.register_buffer("interaction_revision", torch.zeros((), dtype=torch.int64))
        result.register_buffer("interaction_segment_start", torch.zeros((), dtype=torch.int64))
        result.register_buffer("interaction_changes", torch.zeros((), dtype=torch.int64))
        return result

    def export_config(self):
        config = super().export_config()
        if not hasattr(self, "interaction_source"):
            return config
        return {**config, "type": "interactive_shared_r1", "interaction_parent": self.interaction_parent,
                "interaction_source": self.interaction_source, "interaction_window": self.interaction_window,
                "interaction_detect_change": self.interaction_detect_change,
                "interaction_numpy": np.__version__, "interaction_schema": 2}

    def geometry(self):
        coefficients = self.generator_mean[0, :4].detach().numpy().copy()
        radius = float(np.linalg.norm(coefficients[2:]))
        if not np.isfinite(coefficients).all() or radius < .05:
            raise ValueError("A supported nondegenerate acquired circle is required")
        return coefficients[:2], radius


def extend_solver(parent, window, detect_change=False):
    successor = copy.deepcopy(parent)
    owner = InteractiveR1.extend(parent.components["r1"], window, detect_change)
    replace_shared_owner(successor, owner)
    if successor.neural.owner is not successor.components["typed"].owner:
        raise ValueError("Shared interfaces lost their common owner")
    return successor


class ContinuingSession:
    def __init__(self, owner, *, adapt=True):
        if not isinstance(owner, InteractiveR1) or int(owner.interaction_revision) != 0:
            raise ValueError("Fresh session needs a fresh extension; use restore for continuing state")
        self.owner, self.adapt = owner, bool(adapt)
        self.events, self.rows = [], []
        self.work = {"paid_sensor_measurements": 0, "executed_actions": 0,
                     "posterior_fits": 0, "factual_replay_events": 0,
                     "conditional_transition_particles": 0, "controller_decisions": 0}

    def token(self):
        return (self.owner.interaction_parent, self.owner.interaction_source,
                int(self.owner.interaction_revision), len(self.events),
                digest([(n, t.data_ptr(), t._version) for n, t in self.owner.state_dict().items()]))

    def admit(self, observation, action=None, *, sensor_count=1, replay=False):
        if (not isinstance(observation, Observation) or observation.modality != "numeric"
                or len(observation.values) != 2 or observation.units != "m"
                or observation.position != len(self.events)
                or observation.provenance.kind != EvidenceKind.OBSERVATION
                or observation.provenance.source != "a08-paid-sensor"
                or type(sensor_count) is not int or sensor_count < 1):
            raise ValueError("Only ordered paid physical sensor observations are factual")
        if (self.events and (type(action) is not int or action not in (-1, 0, 1))) or (not self.events and action is not None):
            raise ValueError("Every continuing transition requires one executed action")
        center, _ = self.owner.geometry()
        xy = np.asarray(observation.values)
        angle = float(np.arctan2(xy[1]-center[1], xy[0]-center[0]))
        velocity = float(wrap(angle-float(self.owner.interaction_angle))) if self.events else 0.
        if len(self.events) >= 2:
            features = np.array([float(self.owner.interaction_velocity), float(action), 1.])
            innovation = velocity-float(features@self.owner.interaction_mean.numpy())
            scale = math.sqrt(NOISE**2+float(features@self.owner.interaction_covariance.numpy()@features))
            if (self.adapt and self.owner.interaction_detect_change
                    and len(self.rows)-int(self.owner.interaction_segment_start) >= 6
                    and abs(innovation) > max(.04, 4*scale)):
                self.owner.interaction_segment_start.fill_(len(self.rows))
                self.owner.interaction_changes.add_(1)
            self.rows.append((features.tolist(), velocity))
            if self.adapt:
                start = max(int(self.owner.interaction_segment_start), len(self.rows)-self.owner.interaction_window)
                values = self.rows[start:]
                x, y = np.array([v[0] for v in values]), np.array([v[1] for v in values])
                gram, rhs = x.T@x, x.T@y
                precision = PRECISION+gram/NOISE**2
                covariance = np.linalg.inv(precision)
                mean = np.linalg.solve(precision, PRECISION@PRIOR+rhs/NOISE**2)
                for name, value in (("gram", gram), ("rhs", rhs), ("mean", mean), ("covariance", covariance)):
                    getattr(self.owner, "interaction_"+name).copy_(torch.from_numpy(value))
                self.work["posterior_fits"] += 1
        self.owner.interaction_position.copy_(torch.from_numpy(xy))
        self.owner.interaction_angle.fill_(angle)
        self.owner.interaction_velocity.fill_(velocity)
        self.owner.interaction_revision.add_(1)
        self.events.append({"position": observation.position, "values": list(observation.values),
                            "record_id": observation.provenance.record_id, "action": action,
                            "sensor_count": sensor_count, "kind": observation.provenance.kind.value})
        if replay:
            self.work["factual_replay_events"] += 1
        else:
            self.work["paid_sensor_measurements"] += sensor_count
            self.work["executed_actions"] += int(action is not None)

    def validate_plan(self, plan):
        if tuple(plan["state_token"]) != self.token():
            raise ValueError("Stale conditional plan: factual state or learned dynamics changed")
        if plan["kind"] != EvidenceKind.PREDICTION.value:
            raise ValueError("A plan is conditional, not a factual sensor event")

    def snapshot(self):
        return {"schema": 1, "source": self.owner.interaction_source,
                "parent": self.owner.interaction_parent, "window": self.owner.interaction_window,
                "detect_change": self.owner.interaction_detect_change,
                "adapt": self.adapt, "events": copy.deepcopy(self.events), "work": dict(self.work),
                "buffers": {n: t.tolist() for n, t in self.owner.state_dict().items() if n.startswith("interaction_")},
                "owner_sha256": model_identity(self.owner)}

    @classmethod
    def restore(cls, parent, snapshot):
        if (snapshot.get("schema") != 1 or snapshot["source"] != fingerprint()
                or snapshot["parent"] != model_identity(parent)):
            raise ValueError("Stale parent or interaction interpreter")
        owner = InteractiveR1.extend(parent, snapshot["window"], snapshot["detect_change"])
        session = cls(owner, adapt=snapshot["adapt"])
        for row in snapshot["events"]:
            obs = Observation("numeric", tuple(row["values"]), row["position"],
                              Provenance("a08-paid-sensor", row["record_id"], EvidenceKind(row["kind"])), units="m")
            session.admit(obs, row["action"], sensor_count=row["sensor_count"], replay=True)
        actual = {n: t.tolist() for n, t in owner.state_dict().items() if n.startswith("interaction_")}
        if actual != snapshot["buffers"] or model_identity(owner) != snapshot["owner_sha256"]:
            raise ValueError("Factual replay disagrees with stored owner state")
        counts = snapshot["work"]
        if (counts["paid_sensor_measurements"] != sum(r["sensor_count"] for r in session.events)
                or counts["executed_actions"] != max(0, len(session.events)-1)
                or any(type(v) is not int or v < 0 for v in counts.values())):
            raise ValueError("Inconsistent cumulative observation/action costs")
        replay_count = session.work["factual_replay_events"]
        session.work = {**counts, "factual_replay_events": counts["factual_replay_events"]+replay_count}
        return session


def sensor(values, index, identity):
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Nonfinite sensor reading")
    return Observation("numeric", tuple(map(float, values)), index,
                       Provenance("a08-paid-sensor", identity, EvidenceKind.OBSERVATION), units="m")
