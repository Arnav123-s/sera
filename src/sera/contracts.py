"""Observation, learning, and retained-state contracts are separate on purpose."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum


class EvidenceKind(StrEnum):
    OBSERVATION = "observation"
    VERIFIED = "verified_outcome"
    SYNTHETIC = "simulator_ground_truth"
    PREDICTION = "model_prediction"
    ANNOTATION = "annotation"
    COUNTEREXAMPLE = "counterexample"
    WEAK_SUPERVISION = "weak_supervision"


@dataclass(frozen=True)
class Provenance:
    source: str
    record_id: str
    kind: EvidenceKind

    def __post_init__(self):
        if not self.source or not self.record_id or not isinstance(self.kind, EvidenceKind):
            raise ValueError("Evidence needs a source, stable record id, and a valid kind")


@dataclass(frozen=True)
class Observation:
    modality: str
    values: tuple[float, ...]
    position: int
    provenance: Provenance
    units: str | None = None
    scale: float = 1.0
    available: tuple[bool, ...] | None = None

    def __post_init__(self):
        if self.modality not in {"symbolic", "numeric", "text_bytes", "image_patch", "audio_frame"}:
            raise ValueError("Unsupported observation modality")
        if self.position < 0 or not self.values or not all(map(math.isfinite, self.values)):
            raise ValueError("Observation must have finite values and nonnegative position")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Scale must be positive and finite")
        if self.modality == "numeric" and not self.units:
            raise ValueError("Numerical observations require units, including 'dimensionless'")
        if self.available is not None and (len(self.available) != len(self.values)
                                           or any(type(value) is not bool for value in self.available)):
            raise ValueError("Observation availability must align with its values")


@dataclass(frozen=True)
class Experience:
    """A prediction is never implicitly promoted to an authoritative training target."""

    observations: tuple[Observation, ...]
    target: int
    evidence: Provenance

    def __post_init__(self):
        if not self.observations or self.target < 0:
            raise ValueError("Experience needs observations and a nonnegative target")
        if self.evidence.kind not in {EvidenceKind.VERIFIED, EvidenceKind.SYNTHETIC}:
            raise ValueError("Learning requires independently verified or simulator targets")


@dataclass(frozen=True)
class StateOwner:
    session_id: str
    model_version: str
    encoder_version: str = "symbolic-v1"
    schema_version: int = 1

    def validate(self):
        if not self.session_id or not self.model_version or self.schema_version != 1:
            raise ValueError("Invalid state ownership or unsupported state schema")
        return asdict(self)
