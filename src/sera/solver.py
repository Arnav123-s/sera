"""Persistent executable solver versions and one cumulative admission ledger."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from sera.evaluation import AdmissionPolicy, assess
from sera.models import ModelConfig, StatefulModel
from sera.programs import TransitionProgram
from sera.storage import Journal, canonical, digest, write_json


@dataclass
class Work:
    """Count unlike operations separately; the sum is a declared gate proxy, not FLOPs."""

    counts: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    def add(self, kind, count=1):
        if type(count) is not int or count < 0:
            raise ValueError("Work counts must be nonnegative integers")
        self.counts[kind] = self.counts.get(kind, 0) + count

    def record(self):
        return {"operations": dict(self.counts),
                "operation_sum": sum(value for key, value in self.counts.items() if "_bytes" not in key),
                "wall_seconds": self.seconds,
                "units": "Heterogeneous counted operations; byte counters excluded from operation_sum. Wall time and storage bytes reported separately; not FLOPs"}


def tensor_digest(model):
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str((value.dtype, tuple(value.shape))).encode())
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


class Solver(nn.Module):
    """Ordinary predictions include verified skills; connected models share this snapshot."""

    def __init__(self, neural: StatefulModel, *, skills=None, components=None, version="unversioned"):
        super().__init__()
        self.neural = neural
        self.skills = copy.deepcopy(skills or {})
        self.components = nn.ModuleDict(components or {})
        self.version = version

    @property
    def config(self):
        return self.neural.config

    def forward(self, inputs):
        return self.predict_probabilities(inputs).clamp_min(1e-12).log()

    def predict_probabilities(self, inputs):
        probabilities = self.neural(inputs).softmax(-1)
        task_ids = inputs[:, -1, 8:13].argmax(-1)
        for key, record in self.skills.items():
            if not key.isdigit():
                continue
            rows = (task_ids == int(key)).nonzero().flatten()
            if not len(rows):
                continue
            selected = inputs[rows].detach().cpu()
            if record["kind"] == "transition":
                p = record["program"]
                program = TransitionProgram(p["environment_id"], p["initial_state"],
                                            tuple(tuple(row) for row in p["table"]), p["max_steps"])
                answers = [program.execute(x[:-1, 4:8].argmax(-1).tolist(),
                                           initial_state=int(x[0, 16:20].argmax()),
                                           environment_id="ordered-control-v1") for x in selected]
            elif record["kind"] == "sequence_rule":
                from sera.skills import execute_rule
                answers = [execute_rule(record["rule"], x) for x in selected]
            else:
                raise ValueError("Unsupported skill representation")
            probabilities = probabilities.clone()
            probabilities[rows] = F.one_hot(torch.tensor(answers, device=inputs.device), 4).to(
                probabilities.dtype
            )
        return probabilities

    def state_bytes(self, batch=1):
        return self.neural.state_bytes(batch)

    def identity(self):
        return digest({"weights": tensor_digest(self), "skills": self.skills,
                       "components": {name: module.export_config()
                                      for name, module in self.components.items()}})

    def validate(self):
        if any(not torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Solver contains nonfinite state or parameters")
        for component in self.components.values():
            if hasattr(component, "validity") and max(component.validity().values()) > 1e-4:
                raise ValueError("Solver instrument validity contract failed")
        checked_worlds = set()
        for record in self.skills.values():
            if record.get("kind") == "action_program":
                from sera.r2 import validate_library
                if record["world_id"] in checked_worlds:
                    continue
                library = {key: value for key, value in self.skills.items()
                           if value.get("kind") == "action_program"
                           and value["world_id"] == record["world_id"]}
                validate_library(library, record["world_id"])
                checked_worlds.add(record["world_id"])
        return True


class SolverStore:
    """Immutable snapshots, atomic pointer changes and serialized single-writer operations."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.root / "journal.sqlite")
        self.journal.verify()

    @contextmanager
    def writing(self):
        path = self.root / ".writer.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("Solver has an active or interrupted writer; inspect its lock") from error
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            path.unlink(missing_ok=True)

    @property
    def exists(self):
        return (self.root / "current.json").is_file()

    def current_record(self):
        pointer = json.loads((self.root / "current.json").read_text(encoding="utf-8"))
        record = self.record(pointer["version"])
        if pointer != record:
            raise ValueError("Current solver pointer differs from its immutable manifest")
        return record

    def record(self, version):
        if not isinstance(version, str) or not version.startswith("v") or not version[1:].isdigit():
            raise ValueError("Invalid solver version")
        return json.loads((self.root / "versions" / f"{version}.json").read_text(encoding="utf-8"))

    def _save(self, solver, parent):
        solver.validate()
        directory = self.root / "versions"
        directory.mkdir(exist_ok=True)
        indices = [int(p.stem[1:]) for p in directory.glob("v*.json") if p.stem[1:].isdigit()]
        version = f"v{max(indices, default=-1) + 1}"
        payload = {"schema_version": 2, "model_config": asdict(solver.config),
                   "neural_state": solver.neural.state_dict(), "skills": solver.skills,
                   "components": {name: {"config": module.export_config(),
                                          "state": module.state_dict()}
                                  for name, module in solver.components.items()}}
        path = directory / f"{version}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        record = {"schema_version": 2, "version": version, "parent": parent,
                  "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "solver_sha256": solver.identity(), "skills": sorted(solver.skills),
                  "components": sorted(solver.components)}
        write_json(directory / f"{version}.json", record)
        return record

    def initialize(self, solver):
        with self.writing():
            if self.exists:
                return self.current_record()
            if not isinstance(solver, Solver):
                solver = Solver(solver)
            record = self._save(solver, None)
            write_json(self.root / "current.json", record)
            self.journal.append("solver_initialized", record)
        return record

    def load(self, version=None):
        record = self.current_record() if version is None else self.record(version)
        if record.get("schema_version") != 2:
            raise ValueError("This directory needs explicit migration from the historical format")
        path = self.root / "versions" / f"{record['version']}.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
            raise ValueError("Solver checkpoint integrity failure")
        payload = torch.load(path, weights_only=True, map_location="cpu")
        neural = StatefulModel(ModelConfig(**payload["model_config"]))
        neural.load_state_dict(payload["neural_state"])
        components = {}
        if payload["components"]:
            from sera.connected import restore_component
            for name, component in payload["components"].items():
                module = restore_component(component["config"])
                module.load_state_dict(component["state"])
                components[name] = module
        solver = Solver(neural, skills=payload["skills"], components=components,
                        version=record["version"]).eval()
        if solver.identity() != record["solver_sha256"]:
            raise ValueError("Solver manifest identity failure")
        solver.validate()
        return solver

    def consider(self, candidate, evaluator, *, work=None, policy=None, description=None):
        """Freeze before drawing fresh evaluation randomness; every attempt consumes a round."""
        work = Work() if work is None else work
        policy = AdmissionPolicy() if policy is None else policy
        started = time.perf_counter()
        with self.writing():
            incumbent_record = self.current_record()
            incumbent = self.load()
            try:
                frozen = self._save(candidate, incumbent_record["version"])
            except (ValueError, RuntimeError) as error:
                work.seconds += time.perf_counter() - started
                self.journal.append("candidate_invalid", {"parent": incumbent_record["version"],
                                                          "reason": str(error), "work": work.record()})
                raise
            self.journal.append("candidate_frozen", {**frozen, "proposal": description or {}})
            seed = secrets.randbits(63)
            reservation = digest(["solver-admission-v2", frozen["version"], seed])
            round_index = self.journal.reserve_evaluation(reservation)
            try:
                baseline, before = evaluator(incumbent, seed)
                after_report, after = evaluator(self.load(frozen["version"]), seed)
                canonical(baseline)
                canonical(after_report)
                if baseline["dataset_id"] != after_report["dataset_id"]:
                    raise ValueError("Candidate and incumbent received different evidence")
                count = sum(len(values) for values in before.values())
                work.add("evaluation_examples", count * 2)
                decision = assess(after, before, round_index=round_index,
                                  invariants_ok=True, candidate_cost=work.record()["operation_sum"],
                                  policy=policy)
                result = {"status": "promoted" if decision["admitted"] else "rejected",
                          "version": frozen["version"], "parent": incumbent_record["version"],
                          "round_index": round_index, "seed": seed, "reservation": reservation,
                          "incumbent": baseline, "candidate": after_report, "decision": decision,
                          "dataset_id": baseline["dataset_id"],
                          "proposal": description or {}}
                if decision["admitted"]:
                    write_json(self.root / "current.json", frozen)
            except Exception as error:
                work.seconds += time.perf_counter() - started
                result = {"status": "failed", "version": frozen["version"],
                          "parent": incumbent_record["version"], "round_index": round_index,
                          "seed": seed, "reason": str(error), "work": work.record()}
                write_json(self.root / "rounds" / f"{round_index}.json", result)
                self.journal.append("evaluation_failed", result)
                raise
            work.seconds += time.perf_counter() - started
            result["work"] = work.record()
            write_json(self.root / "rounds" / f"{round_index}.json", result)
            self.journal.append("promotion_decision", result)
            return result

    def rollback(self):
        with self.writing():
            current = self.current_record()
            if current["parent"] is None:
                raise ValueError("Current version has no parent to roll back to")
            # Verify the executable payload before changing the pointer.
            self.load(current["parent"])
            parent = self.record(current["parent"])
            write_json(self.root / "current.json", parent)
            self.journal.append("rollback", {"from": current["version"], "to": parent["version"]})
            return parent


def load_solver(path: Path):
    path = Path(path)
    if path.is_dir():
        return SolverStore(path).load()
    from sera.training import load_model
    model, _ = load_model(path)
    return Solver(model).eval()
