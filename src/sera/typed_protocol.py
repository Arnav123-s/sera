"""Versioned semantic partitions for the seven bounded typed tasks.

The v1 generator remains in typed_learning for historical reproduction. These
partitions test fixed rules with held-out compositions, not newly invented rules.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict

import numpy as np

from sera.contracts import EvidenceKind, Observation, Provenance
from sera.data import seed_for
from sera.storage import digest
from sera.typed_learning import TASKS, UNIT_FACTORS, TypedExample
from sera.typed_programs import numeric_input

PROTOCOL = "typed-semantic-v2"
PARTITIONS = ("support", "validation", "test-id", "test-extent", "test-composition")
BYTE_HOLDOUT = {(0, 0), (1, 0), (2, 1), (3, 3)}
TEMPLATES = {
    "modular_sum": ["non-palindrome, 3-5 operands", "non-palindrome, 8-10 operands", "palindrome, 5 or 7 operands"],
    "spatial_relation": ["matching m/m or cm/cm units", "matching units, larger coordinates", "mixed m/cm or cm/m units"],
    "byte_sum": ["0-31 operands, 12 residue-pair classes", "32-63 operands, same 12 classes", "0-31 operands, four withheld residue-pair classes"],
    "patch_quadrant": ["solid bright quadrant", "solid quadrant, more noise", "diagonal bright pair inside quadrant"],
    "tone": ["one sinusoid", "one sinusoid, lower amplitude", "dominant sinusoid plus weaker second tone"],
    "motion": ["positions at times 0,1,2", "positions at times 0,2,4", "positions at times 0,1,3"],
    "binding": ["six writes, final query key appears once", "twelve writes, query key appears once", "six writes, query key overwritten at least once"],
}


def semantic_id(row):
    """Ignore provenance, split labels, invisible payload and equivalent spelling/units."""
    if row.task in {"modular_sum", "byte_sum"}:
        inputs = numeric_input(row.observations)
    else:
        inputs = []
        for obs in row.observations:
            mask = obs.available or (True,) * len(obs.values)
            factor = obs.scale * UNIT_FACTORS.get(obs.units, 1)
            inputs.append({"modality": obs.modality, "position": obs.position, "available": list(mask),
                           "values": [round(float(v) * factor, 10) if present else None
                                      for v, present in zip(obs.values, mask)]})
    # Inputs alone define identity: conflicting labels must not evade leakage checks.
    return digest([row.task, inputs])


def _bucket(identity):
    value = int(identity[:12], 16) % 20
    return "support" if value < 12 else "validation" if value < 15 else "test-id"


def _example(task, partition, rng, record_id):
    raw = Provenance(PROTOCOL, record_id, EvidenceKind.OBSERVATION)
    extent, composition = partition == "test-extent", partition == "test-composition"
    events = []
    if task == "modular_sum":
        if composition:
            half = rng.integers(4, size=int(rng.choice([2, 3]))).tolist()
            numbers = half + [int(rng.integers(4))] + half[::-1]
        else:
            length = int(rng.integers(8, 11) if extent else rng.integers(3, 6))
            numbers = rng.integers(4, size=length).tolist()
            if numbers == numbers[::-1]:
                return None
        events = [Observation("numeric", (float(v),), i, raw, units="dimensionless") for i, v in enumerate(numbers)]
        target = sum(numbers) % 4
    elif task == "spatial_relation":
        first = rng.uniform(-8 if extent else -2, 8 if extent else 2, size=2)
        target = int(rng.integers(4))
        delta = np.asarray([(1, 0), (-1, 0), (0, 1), (0, -1)][target]) * rng.uniform(.5, 2)
        first_unit = str(rng.choice(["m", "cm"]))
        second_unit = ("m" if first_unit == "cm" else "cm") if composition else first_unit
        for i, (point, unit) in enumerate(zip((first, first + delta), (first_unit, second_unit))):
            events.append(Observation("numeric", tuple(float(v) / UNIT_FACTORS[unit] for v in point), i, raw, units=unit))
    elif task == "byte_sum":
        a, b = (int(v) for v in rng.integers(32 if extent else 0, 64 if extent else 32, size=2))
        if ((a % 4, b % 4) in BYTE_HOLDOUT) != composition:
            return None
        spelling = f"{a}+{b}" if rng.random() < .5 else f"{a} + {b}"
        events = [Observation("text_bytes", tuple(float(ord(c)) for c in spelling), 0, raw)]
        target = (a + b) % 4
    elif task == "patch_quadrant":
        target = int(rng.integers(4))
        x, y = target % 2, target // 2
        patch = rng.normal(0, .08 if extent else .02, size=(4, 4))
        shape = np.eye(2) if composition else np.ones((2, 2))
        if composition and rng.random() < .5:
            shape = np.fliplr(shape)
        patch[2*y:2*y+2, 2*x:2*x+2] += rng.uniform(.6, 1) * shape
        events = [Observation("image_patch", tuple(float(v) for v in patch.flat), 0, raw, units="pixel")]
    elif task == "tone":
        target = int(rng.integers(4))
        amplitude = rng.uniform(.25, .5) if extent else rng.uniform(.5, 1)
        time = 2 * np.pi * np.arange(16) / 16
        values = amplitude * np.sin((target + 1) * time + rng.uniform(0, 2*np.pi))
        if composition:
            values += amplitude * rng.uniform(.2, .35) * np.sin(((target + 1) % 4 + 1) * time + rng.uniform(0, 2*np.pi))
        values += rng.normal(0, .01, size=16)
        events = [Observation("audio_frame", tuple(float(v) for v in values), 0, raw, units="amplitude")]
    elif task == "motion":
        initial, velocity = rng.uniform(-1, 1, size=2), rng.uniform(-.2, .2, size=2)
        times = (0, 1, 3) if composition else (0, 2, 4) if extent else (0, 1, 2)
        events = [Observation("numeric", tuple(float(v) for v in initial + t * velocity), t, raw, units="m") for t in times]
        target = tuple(float(v) for v in initial + (times[-1] + 1) * velocity)
    else:
        length, query = 12 if extent else 6, int(rng.integers(4))
        other = [k for k in range(4) if k != query]
        keys = rng.choice(other, size=length).tolist()
        positions = rng.choice(length, size=2 if composition else 1, replace=False)
        for pos in positions:
            keys[pos] = query
        values = {}
        for pos, key in enumerate(keys):
            value = int(rng.integers(4))
            values[key] = value
            vector = [float(i == key) for i in range(4)] + [float(i == value) for i in range(4)]
            events.append(Observation("symbolic", tuple(vector), pos, raw))
        events.append(Observation("symbolic", tuple([float(i == query) for i in range(4)] + [0.0]*4), length,
                                  raw, available=(True,)*4 + (False,)*4))
        target = values[query]
    return TypedExample(tuple(events), task, target,
                        Provenance(PROTOCOL, record_id, EvidenceKind.SYNTHETIC), partition)


def typed_suite(*, seed, support_count=192, validation_count=48, test_count=128, exclude=()):
    """Generate disjoint semantic cases with fixed, seed-independent development buckets.

    Finite tasks are sampled without replacement. Old training records can be
    excluded when evaluating frozen checkpoints; absence of overlap is verified.
    """
    counts = {"support": support_count, "validation": validation_count,
              **{p: test_count for p in PARTITIONS[2:]}}
    if any(type(n) is not int or not 0 <= n <= 192 for n in counts.values()) or test_count > 128:
        raise ValueError("Semantic protocol supports at most 192 development / 128 test cases per task")
    excluded = {semantic_id(row) for row in exclude}
    used = set(excluded)
    result = {p: [] for p in PARTITIONS}
    for task, partition in itertools.product(TASKS, PARTITIONS):
        rng = np.random.default_rng(seed_for(f"{PROTOCOL}/{task}/{partition}", seed))
        accepted = 0
        for attempt in range(100000):
            if accepted == counts[partition]:
                break
            row = _example(task, partition, rng, f"{seed}/{task}/{partition}/{attempt}")
            if row is None:
                continue
            identity = semantic_id(row)
            if identity in used or (partition in PARTITIONS[:3] and _bucket(identity) != partition):
                continue
            result[partition].append(row)
            used.add(identity)
            accepted += 1
        if accepted != counts[partition]:
            raise ValueError(f"Semantic partition exhausted: {task}/{partition}; do not duplicate cases")
    manifest = audit_partitions(result)
    manifest.update(seed=seed, templates=TEMPLATES, excluded_semantic_cases=len(excluded),
                    semantics="Fixed task rules; ID, extent and explicit withheld composition templates. No claim of unseen rule induction.")
    return result, manifest


def audit_partitions(partitions):
    seen, result = {}, {}
    for partition, rows in partitions.items():
        ids = [semantic_id(row) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Repeated semantic case within {partition}")
        overlap = set(ids).intersection(seen)
        if overlap:
            raise ValueError(f"Semantic cases reused across partitions: {partition}")
        seen.update({identity: partition for identity in ids})
        result[partition] = {"examples": len(rows), "semantic_sha256": digest(sorted(ids)),
                             "record_sha256": digest([asdict(row) for row in rows]),
                             "by_task": {task: sum(row.task == task for row in rows) for task in TASKS}}
    return {"protocol": PROTOCOL, "partitions": result, "unique_semantic_cases": len(seen),
            "cross_partition_overlap": 0, "within_partition_duplicates": 0}
