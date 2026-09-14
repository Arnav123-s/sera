"""Admitted sensor/action/reward trajectories, distinct from evaluator-only truth."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from sera.contracts import EvidenceKind, Provenance
from sera.storage import digest, write_json


@dataclass(frozen=True)
class Trajectory:
    world_id: str
    observations: tuple[int, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    goal: int
    provenance: Provenance
    split: str

    def __post_init__(self):
        if not self.world_id or not self.split or not self.actions:
            raise ValueError("Trajectory needs a world, split and nonempty execution")
        if len(self.observations) != len(self.actions) + 1 or len(self.rewards) != len(self.actions):
            raise ValueError("Trajectory time alignment violated")
        if any(type(x) is not int or not -1 <= x < 4 for x in self.observations):
            raise ValueError("Sensor must be a declared color or the missing-observation marker")
        if any(type(a) is not int or not 0 <= a < 4 for a in self.actions) or not 0 <= self.goal < 4:
            raise ValueError("Invalid action or goal")
        if any(not math.isfinite(r) or not 0 <= r <= 1 for r in self.rewards):
            raise ValueError("Rewards must be finite in [0,1]")

    @property
    def identifier(self):
        return digest(asdict(self))


class EvidenceReplay:
    def __init__(self, records=()):
        self.records = []
        self.identifiers = set()
        for record in records:
            self.admit(record)

    def admit(self, record: Trajectory):
        if record.provenance.kind not in {EvidenceKind.VERIFIED, EvidenceKind.SYNTHETIC}:
            raise ValueError("Training requires verified execution or simulator provenance")
        if record.split.startswith(("evaluation", "query", "test", "promotion")):
            raise ValueError("Sealed evaluation evidence cannot enter the learning replay")
        if record.identifier in self.identifiers:
            return False
        self.records.append(record)
        self.identifiers.add(record.identifier)
        return True

    def save(self, path: Path):
        write_json(path, {"schema_version": 1, "records": [asdict(x) for x in self.records],
                          "dataset_id": digest(sorted(self.identifiers))})

    @classmethod
    def load(cls, path: Path):
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported replay schema")
        records = []
        for row in payload["records"]:
            p = row["provenance"]
            records.append(Trajectory(row["world_id"], tuple(row["observations"]),
                                      tuple(row["actions"]), tuple(row["rewards"]), row["goal"],
                                      Provenance(p["source"], p["record_id"], EvidenceKind(p["kind"])),
                                      row["split"]))
        replay = cls(records)
        if digest(sorted(replay.identifiers)) != payload["dataset_id"]:
            raise ValueError("Replay integrity failure")
        return replay
