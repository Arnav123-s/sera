"""Additive physical measurement semantics around the unchanged R2 instrument.

Outcome availability and whether a measurement occurred are separate facts.
An unknown event returns alternatives instead of silently applying an identity.
"""

from dataclasses import asdict, dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class MeasurementEvent:
    subject: str
    action: str
    elapsed: float
    occurrence: str
    outcome: int | None
    source: str
    evidence_kind: str
    schema: str = "sera.measurement-event.1"

    def __post_init__(self):
        if self.schema != "sera.measurement-event.1":
            raise ValueError("Unknown measurement schema")
        if any(not isinstance(v, str) or not 1 <= len(v) <= 512 for v in (self.subject, self.action, self.source)):
            raise ValueError("Subject, action and provenance are required")
        if not isinstance(self.elapsed, (float, int)) or not isfinite(self.elapsed) or self.elapsed < 0:
            raise ValueError("A nonnegative finite elapsed time is required")
        if self.occurrence not in {"performed", "omitted", "unknown"}:
            raise ValueError("Explicit measurement occurrence is required")
        if self.outcome is not None and (type(self.outcome) is not int or not 0 <= self.outcome < 4):
            raise ValueError("Invalid instrument outcome")
        if self.occurrence != "performed" and self.outcome is not None:
            raise ValueError("An outcome requires a performed measurement")
        if self.evidence_kind not in {"MEASURED_OBSERVATION", "SIMULATED_OBSERVATION", "HYPOTHETICAL"}:
            raise ValueError("Explicit evidence kind is required")

    def record(self):
        return {**asdict(self), "outcome_available": self.outcome is not None}


def measurement_branches(instrument, rho, event, operators=None):
    """Apply measurement only; propagation/control remains an explicit operation."""
    if not isinstance(event, MeasurementEvent):
        raise TypeError("Use a versioned measurement event")
    operators = instrument.operators() if operators is None else operators
    result = {}
    if event.occurrence in {"omitted", "unknown"}:
        result["omitted"] = rho.clone()
    if event.occurrence in {"performed", "unknown"}:
        outcome = -1 if event.outcome is None else event.outcome
        colors = torch.full((len(rho),), outcome, dtype=torch.long, device=rho.device)
        updated, _ = instrument.observe(rho, colors, operators)
        result["performed_unread" if outcome == -1 else "performed_observed"] = updated
    return result
