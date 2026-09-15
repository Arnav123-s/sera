"""Immutable common events; codecs and evidence eligibility are supplied contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum

from sera.contracts import EvidenceKind, Observation, Provenance
from sera.storage import digest


class EventRole(StrEnum):
    OBSERVATION = "observation"
    TARGET = "target"
    QUERY = "query"
    PREDICTION = "prediction"
    HYPOTHETICAL = "hypothetical"


@dataclass(frozen=True, slots=True)
class EventSchema:
    name: str
    version: int
    modality: str
    width: int
    units: str | None = None
    entity_slots: int = 0
    encoding: str = "typed-values-v1"

    def __post_init__(self):
        if any(type(value) is not str or not value for value in (self.name, self.encoding)):
            raise ValueError("Schema name and encoding must be explicit")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Schema version must be a positive integer")
        if self.modality not in {"symbolic", "numeric", "text_bytes", "image_patch", "audio_frame"}:
            raise ValueError("Unsupported event modality")
        if type(self.width) is not int or not 1 <= self.width <= 4096:
            raise ValueError("Event width exceeds the declared contract")
        if type(self.entity_slots) is not int or not 0 <= self.entity_slots <= 64:
            raise ValueError("Invalid entity reference interface")
        if self.units is not None and (type(self.units) is not str or not self.units):
            raise ValueError("Units must be a nonempty string or absent")
        if self.modality == "numeric" and not self.units:
            raise ValueError("Numeric events require units, including dimensionless")

    def record(self):
        return asdict(self)

    @property
    def identity(self):
        return digest(self.record())

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict or set(record) != {
            "name", "version", "modality", "width", "units", "entity_slots", "encoding"
        }:
            raise ValueError("Invalid event schema record")
        return cls(**record)


@dataclass(frozen=True, slots=True)
class Event:
    schema: EventSchema
    sequence: int
    values: tuple[float, ...]
    provenance: Provenance
    available: tuple[bool, ...] | None = None
    entity_refs: tuple[str, ...] = ()
    role: EventRole = EventRole.OBSERVATION

    def __post_init__(self):
        if type(self.schema) is not EventSchema or type(self.provenance) is not Provenance:
            raise ValueError("Events require validated schema and provenance objects")
        if any(type(value) is not str or not value
               for value in (self.provenance.source, self.provenance.record_id)):
            raise ValueError("Event provenance needs stable string identifiers")
        if not isinstance(self.role, EventRole):
            raise ValueError("An explicit event role is required")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("Event sequence must be a nonnegative integer")
        values = tuple(self.values)
        mask = (True,) * len(values) if self.available is None else tuple(self.available)
        if len(values) != self.schema.width or len(mask) != len(values):
            raise ValueError("Event values and availability must match the schema width")
        if any(type(value) is not bool for value in mask):
            raise ValueError("Availability entries must be booleans")
        normalized = []
        for value, visible in zip(values, mask):
            if not visible:
                # Unavailable content has no feature or identity channel.
                normalized.append(0.0)
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Available event values must be finite real numbers")
            else:
                normalized.append(float(value))
        entities = tuple(self.entity_refs)
        if len(entities) != self.schema.entity_slots or any(
            type(entity) is not str or not entity for entity in entities
        ):
            raise ValueError("Entity references must match the declared interface")
        object.__setattr__(self, "values", tuple(normalized))
        object.__setattr__(self, "available", mask)
        object.__setattr__(self, "entity_refs", entities)

    @property
    def origin(self):
        return self.provenance.kind

    @property
    def evidence_key(self):
        return self.provenance.source, self.provenance.record_id

    @property
    def eligible_observation(self):
        return self.role == EventRole.OBSERVATION and self.origin in {
            EvidenceKind.OBSERVATION, EvidenceKind.VERIFIED, EvidenceKind.SYNTHETIC
        }

    @property
    def eligible_label(self):
        return self.role == EventRole.TARGET and self.origin in {
            EvidenceKind.VERIFIED, EvidenceKind.SYNTHETIC
        }

    def record(self):
        return {
            "schema": self.schema.record(), "sequence": self.sequence,
            "values": list(self.values), "available": list(self.available),
            "entity_refs": list(self.entity_refs), "role": self.role.value,
            "provenance": {"source": self.provenance.source,
                           "record_id": self.provenance.record_id, "kind": self.origin.value},
        }

    @property
    def identity(self):
        return digest(self.record())

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict or set(record) != {
            "schema", "sequence", "values", "available", "entity_refs", "role", "provenance"
        }:
            raise ValueError("Invalid event record")
        provenance = record["provenance"]
        if type(provenance) is not dict or set(provenance) != {"source", "record_id", "kind"}:
            raise ValueError("Invalid event provenance record")
        return cls(
            EventSchema.from_record(record["schema"]), record["sequence"], record["values"],
            Provenance(provenance["source"], provenance["record_id"],
                       EvidenceKind(provenance["kind"])),
            record["available"], record["entity_refs"], EventRole(record["role"]),
        )

    @classmethod
    def from_observation(cls, observation, schema, *, entity_refs=(), role=EventRole.OBSERVATION):
        if not isinstance(observation, Observation):
            raise ValueError("The adapter requires a SERA Observation")
        if (observation.modality != schema.modality or observation.units != schema.units
                or observation.scale != 1):
            raise ValueError("Observation units, modality or scale need an explicit conversion")
        return cls(schema, observation.position, observation.values, observation.provenance,
                   observation.available, entity_refs, role)
