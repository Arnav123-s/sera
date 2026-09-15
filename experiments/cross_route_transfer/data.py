"""Separate conditional teacher samples from independently simulated outcomes.

The circle grammar, similarity transforms and constant angular speed are supplied.
The acquired coefficients come from the real parent's registered posterior. No
claim is made that this experiment discovers that grammar or its applicability.
"""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import torch

from sera.contracts import EvidenceKind, Observation, Provenance
from sera.storage import digest
from sera.typed_learning import TypedExample


@dataclass(frozen=True)
class MotionCase:
    observations: tuple
    target: tuple
    provenance: str
    split: str
    teacher: str | None
    conditions: tuple

    @property
    def identity(self):
        # No labels, provenance or split in the semantic key.
        return digest([(o.values, o.position, o.units) for o in self.observations])

    def record(self):
        return {**asdict(self), "identity": self.identity}


CONDITIONS = ("circle candidate", "constant angular speed", "similarity transform",
              "no measurement noise", "three equally spaced positions", "one next step")


def teacher_identity(owner):
    return digest({"graph": owner.generator_graph().identity,
                   "coefficients": owner.generator_mean[0].tolist(),
                   "observations": int(owner.generator_observations),
                   "corrections": int(owner.generator_labels)})


def circle_cases(owner, *, seed, count, split, conditional=False, extent=False):
    """New circle worlds, one query per world; no phase/center/velocity input.

    A supplied similarity transform reuses the fitted circle as a shape primitive.
    The observer receives three coordinates only. Independently simulated labels
    use numpy's trigonometry; conditional labels execute the actual generator.
    Normalizing the fitted primitive removes its center/radius: this is a test of
    reusing an instance within a supplied circle family, not discovering that family.
    """
    rng = np.random.default_rng(seed)
    coefficients = owner.generator_mean[0].detach().cpu().numpy()
    cx, cy, a, b = coefficients[:4]
    radius = float(np.hypot(a, b))
    if radius < .05 or int(owner.generator_observations) < 3:
        raise ValueError("A nondegenerate acquired circle candidate is required")
    intrinsic_phase = float(np.arctan2(b, a))
    theta = rng.uniform(-np.pi, np.pi, count)
    delta = rng.uniform(.60 if extent else .12, .95 if extent else .55, count)
    delta *= rng.choice([-1, 1], count)
    centers = rng.uniform(-.8, .8, (count, 2))
    radii = rng.uniform(.35, 1.1, count)
    angles = theta[:, None] + delta[:, None]*np.arange(4)
    truth = centers[:, None, :] + radii[:, None, None]*np.stack([np.cos(angles), np.sin(angles)], -1)
    if conditional:
        phases = (angles-intrinsic_phase)/np.pi
        with torch.no_grad():
            generated = owner.forward_generator(phases.reshape(-1).tolist())["class_means"][0]
        generated = generated.numpy().reshape(count, 4, 2)
        points = centers[:, None, :] + radii[:, None, None]*(generated-[cx, cy])/radius
        kind = EvidenceKind.PREDICTION
    else:
        points, kind = truth, EvidenceKind.OBSERVATION
    result = []
    for i in range(count):
        provenance = Provenance("conditional-circle-teacher" if conditional else "circle-simulator",
                                f"{split}/{seed}/{i}", kind)
        observations = tuple(Observation("numeric", tuple(map(float, p)), t, provenance, units="m")
                             for t, p in enumerate(points[i, :3]))
        result.append(MotionCase(observations, tuple(map(float, points[i, 3])),
                                 "conditional_model_target" if conditional else "simulator_ground_truth",
                                 split, teacher_identity(owner) if conditional else None, CONDITIONS))
    return result


def admit_training(cases, *, allow_conditional, teacher):
    if not cases:
        raise ValueError("Empty teaching corpus")
    for row in cases:
        if not row.split.startswith("support-"):
            raise ValueError("Development/final evidence cannot enter the optimizer")
        if row.provenance == "conditional_model_target":
            if not allow_conditional or row.teacher != teacher or row.conditions != CONDITIONS:
                raise ValueError("Unbound conditional teaching target")
        elif row.provenance != "simulator_ground_truth" or row.teacher is not None:
            raise ValueError("Unsupported target authority")
    # Deliberately do not coerce model targets into TypedEvidence/verified labels.
    return cases


def typed_motion(row):
    if row.provenance != "simulator_ground_truth":
        raise ValueError("Model targets are not independently observed evidence")
    return TypedExample(row.observations, "motion", row.target,
                        Provenance("circle-simulator", row.identity, EvidenceKind.SYNTHETIC), row.split)


def partition_audit(partitions):
    seen, report = set(), {}
    for name, rows in partitions.items():
        ids = [r.identity for r in rows]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Repeated inputs within/across partitions")
        seen.update(ids)
        report[name] = {"count": len(rows), "semantic_sha256": digest(sorted(ids)),
                        "record_sha256": digest([r.record() for r in rows])}
    return report


def state_digest(owner):
    h = hashlib.sha256()
    for name, value in sorted(owner.state_dict().items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
