"""Persistent conditional investigations, missing-data acquisition and exact replay."""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.language_inquiry.runtime import Runtime as ParentRuntime
from experiments.language_inquiry.study import ROOT, read, sha, write
from experiments.task_transfer.runtime import lock
from sera.session_state import model_identity
from sera.storage import digest
from workbench.storage import Store

from .compatibility import REPLAY_SOURCES
from .data import ENTITIES, corpus
from .model import ConstraintR1, apply, language_identity, source
from .settling import context, initialize, step, summarize
from .study import OUT, pin


def runtime_source():
    return sha(Path(__file__))


def admitted_runtime(value):
    return value == runtime_source() or value in REPLAY_SOURCES


def same_binding(expected, stored):
    left, right = copy.deepcopy(expected), copy.deepcopy(stored)
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    a, b = left.pop("scores", None), right.pop("scores", None)
    if left != right or not isinstance(a, list) or not isinstance(b, list) or len(a) != 3 or len(b) != 3:
        return False
    epsilon = float(np.finfo(np.float32).eps)
    for row, saved in zip(a, b, strict=True):
        if not isinstance(row, list) or not isinstance(saved, list) or len(row) != 4 or len(saved) != 4:
            return False
        for x, y in zip(row, saved, strict=True):
            if (type(x) is not float or type(y) is not float or not 0 <= x <= 1 or not 0 <= y <= 1
                    or not math.isclose(x, y, rel_tol=64*epsilon, abs_tol=64*epsilon**2)):
                return False
    return True


def same_starting_proposal(expected, stored):
    a, b = np.asarray(expected, dtype=np.float64), np.asarray(stored, dtype=np.float64)
    if (a.shape != (8, 2) or b.shape != (8, 2) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or (np.abs(a) > 1).any() or (np.abs(b) > 1).any()):
        return False
    # Random starts, their order and the jitter construction remain exact.
    if not np.array_equal(a[4:], b[4:]):
        return False
    offsets = np.asarray([[0., 0.], [.1, -.1], [-.1, .1], [.15, .15]], dtype=np.float32).astype(np.float64)
    for points in (a, b):
        if (not np.array_equal(points[0], points[0].astype(np.float32).astype(np.float64))
                or not np.array_equal(points[:4], np.clip(points[0] + offsets, -1, 1))):
            return False
    return bool((np.abs(a[0]-b[0]) <= np.finfo(np.float32).eps).all())


def supported_text(text):
    return text.lower().strip().rstrip(".?") in {
        r["text"] for part in ("train", "pairs") for r in corpus(part)
    }


class ConstraintRuntime:
    def __init__(self, checkpoint, record=None):
        path = (ROOT / checkpoint["checkpoint"]).resolve()
        if not path.is_relative_to(ROOT / "runs") or sha(path) != checkpoint["sha256"]:
            raise ValueError("Checkpoint identity or location changed")
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved["source"] != source() or saved["parent"] != sha(OUT / "parent.json"):
            raise ValueError("Changed model source or training predecessor")
        if saved["kind"] != "ordered" or saved["teaching_exact"] != 1. or "proposal" not in saved:
            raise ValueError("Unqualified teaching checkpoint")
        self.checkpoint = copy.deepcopy(checkpoint)
        self.parent_record = copy.deepcopy(pin() if record is None else record["parent"])
        self.base = ParentRuntime(self.parent_record["checkpoint"], self.parent_record)
        self.owner = self.base.owner
        learner = self.base.session.learner
        facts, library = learner.legacy.snapshot(), learner.library.record()
        ConstraintR1.attach(self.owner, saved["seed"], saved["kind"])
        apply(self.owner, saved["delta"])
        self.observations, self.jobs, self.history = [], {}, []
        self.replay_migrations = []
        self.work = {"paid_location_observations": 0, "gradient_batches": 0,
                     "conditional_transition_points": 0, "restore_gradient_batches": 0}
        if record is not None:
            if record["schema"] != "sera.constraint-inquiry.1" or not admitted_runtime(record["source"]):
                raise ValueError("Changed runtime; explicit migration is required")
            for row in record["observations"]:
                self._admit(row)
            if model_identity(self.owner) != record["owner"]:
                raise ValueError("Reconstructed owner differs")
            self.jobs = copy.deepcopy(record["jobs"])
            self.history = copy.deepcopy(record["history"])
            self.work = copy.deepcopy(record["work"])
            self.replay_migrations = copy.deepcopy(record.get("replay_migrations", []))
            if len(self.jobs) > 32 or len(self.history) > 64:
                raise ValueError("Unbounded investigation history")
            for branch in [*self.jobs.values(), *self.history]:
                self._validate_branch(branch)
                if "source" in branch and branch["source"] != runtime_source():
                    witness = {"source": branch["source"], "sha256": digest(branch)}
                    branch["source"] = runtime_source()
                    if "dependency" in branch and branch["status"] != "stale":
                        branch["dependency"] = self.dependency(branch["frame"]["target"])
                    branch["predecessor_replay"] = witness
            if record["source"] != runtime_source():
                self.replay_migrations.append({"from": record["source"], "to": runtime_source(),
                                               "record_sha256": digest(record)})
        self.reproof = learner.rebind(facts, library)
        if not (self.owner is learner.session.owner is learner.solver.neural.owner
                is learner.solver.components["typed"].owner is self.base.session.owner):
            raise AssertionError("The numerical and language routes lost their actual owner")

    def dependency(self, target, *, interpreter=None):
        from experiments.language_inquiry.graph import situation
        return digest({"language": language_identity(self.owner), "situation": situation(self.owner),
                       "target": target, "evidence": [r for r in self.observations if r["entity"] == target],
                       "checkpoint": self.checkpoint, "interpreter": runtime_source() if interpreter is None else interpreter})

    def _new(self, identifier, text):
        if not supported_text(text):
            return {"id": identifier, "text": text, "status": "MISSING_LANGUAGE",
                    "request": "Clarify using the declared finite vocabulary and productions",
                    "result": None, "steps": 0}
        frame = self.owner.interpret(text)
        target = frame["target"]
        branch = {"id": identifier, "text": text, "frame": frame,
                  "dependency": self.dependency(target), "steps": 0, "result": None,
                  "evidence_role": "conditional_imagination", "source": runtime_source(),
                  "premise_status": "user_request; not independently measured event"}
        if frame["mode"] == 2:
            branch.update(status="MISSING_EVENT_EVIDENCE", request="An observed event with actor, target and time")
        elif frame["actor"] != "orbit":
            branch.update(status="MISSING_DYNAMICS", request=f"Action-conditioned motion model for {frame['actor']}")
        elif target == "orbit":
            branch.update(status="UNRESOLVED_BINDING", request="Two distinct roles")
        elif int(self.owner.cloud_location_count[ENTITIES.index(target)]) < 3:
            branch.update(status="MISSING_LOCATION", request=f"Three source-qualified position readings for {target}, in metres")
        else:
            i = ENTITIES.index(target)
            xy = self.owner.cloud_location_sum[i] / self.owner.cloud_location_count[i]
            model = context(self.owner, xy.tolist(), frame["mode"] == 3)
            points = initialize(self.owner, model)
            branch.update(status="pending", model=model, initial=points.tolist(), points=points.tolist())
            branch["result"] = summarize(model, points)
        return branch

    def ask(self, identifier, text):
        if not isinstance(identifier, str) or not identifier or identifier in self.jobs or len(self.jobs) >= 32:
            raise ValueError("A new bounded investigation identifier is required")
        branch = self._new(identifier, text)
        self.jobs[identifier] = branch
        return copy.deepcopy(branch)

    def sync(self):
        for branch in self.jobs.values():
            if "dependency" in branch and branch["dependency"] != self.dependency(branch["frame"]["target"]):
                branch["status"] = "stale"

    def work_on(self, identifier, ticks=12):
        if type(ticks) is not int or not 0 <= ticks <= 12:
            raise ValueError("A finite refinement budget in 0..12 is required")
        self.sync()
        branch = self.jobs[identifier]
        if branch["status"] not in ("pending", "complete"):
            return copy.deepcopy(branch)
        points = torch.tensor(branch["points"], dtype=torch.float64)
        for _ in range(min(ticks, 12 - branch["steps"])):
            points = step(branch["model"], points)
            branch["steps"] += 1
            self.work["gradient_batches"] += 1
            self.work["conditional_transition_points"] += 8 * 7 * 2
        branch["points"] = points.tolist()
        branch["result"] = summarize(branch["model"], points)
        branch["status"] = "complete" if branch["steps"] == 12 else "pending"
        return copy.deepcopy(branch)

    def rebase(self, identifier):
        if len(self.history) >= 64:
            raise ValueError("Archive this finite investigation store before growing it")
        old = self.jobs[identifier]
        self.history.append(copy.deepcopy(old))
        self.jobs[identifier] = self._new(identifier, old["text"])
        return copy.deepcopy(self.jobs[identifier])

    def solve(self, identifier):
        """A fixed residual gate; this scheduler is engineered, not learned."""
        self.sync()
        branch = self.jobs[identifier]
        if branch["status"] not in ("pending", "complete"):
            return copy.deepcopy(branch)
        while branch["steps"] < 12 and branch["result"]["status"] != "CONDITIONAL_WITNESS":
            self.work_on(identifier, 1)
        branch["status"] = "complete"
        branch["scheduler"] = "fixed_nominal_residual_0.03m_max12; not a model-adequacy gate"
        return copy.deepcopy(branch)

    def _admit(self, row):
        required = {"kind", "source", "entity", "units", "record_id", "values"}
        if (set(row) != required or row["kind"] != "observation" or row["source"] != "ci-landmark-simulator-v1"
                or row["units"] != "m" or row["entity"] not in ENTITIES[1:]
                or any(r["record_id"] == row["record_id"] for r in self.observations)
                or not isinstance(row["record_id"], str) or not row["record_id"]
                or len(self.observations) >= 96):
            raise ValueError("Only new qualified position observations enter acquired knowledge")
        values = torch.tensor(row["values"], dtype=torch.float64)
        if values.shape != (2,) or not torch.isfinite(values).all():
            raise ValueError("A finite measured 2D location is required")
        index = ENTITIES.index(row["entity"])
        self.owner.cloud_location_sum[index].add_(values)
        self.owner.cloud_location_count[index].add_(1)
        self.observations.append(copy.deepcopy(row))

    def admit(self, row):
        learner = self.base.session.learner
        facts, library = learner.legacy.snapshot(), learner.library.record()
        self._admit(row)
        self.work["paid_location_observations"] += 1
        self.reproof = learner.rebind(facts, library)
        self.sync()

    def acquire(self, entity):
        if entity not in ENTITIES[1:] or int(self.owner.cloud_location_count[ENTITIES.index(entity)]):
            raise ValueError("Acquire one missing landmark; later corrections require new evidence IDs")
        # This is an authorized local simulated sensor, separate from the learner.
        rows = location_sensor(self.base.session.learner.world, entity)
        reopen = [i for i, b in self.jobs.items() if b.get("frame", {}).get("target") == entity]
        for row in rows:
            self.admit(row)
        for identifier in reopen:
            self.rebase(identifier)
        return {"entity": entity, "paid_observations": rows, "reopened": reopen,
                "scope": "Learned numeric location of a named simulator object; no visual ontology learned"}

    def advance(self, action):
        base = ParentRuntime(self.parent_record["checkpoint"], self.parent_record)
        response = base.observe(action)
        record = self.snapshot()
        record["parent"] = base.snapshot()
        # Build a fresh candidate; old branches are retained but explicitly stale.
        fresh = {**record, "jobs": {}, "history": []}
        # Reconstruct with no fabricated owner fingerprint after the physical update.
        candidate = self._from_new_parent(fresh)
        candidate.jobs, candidate.history = copy.deepcopy(self.jobs), copy.deepcopy(self.history)
        candidate.sync()
        self.__dict__.update(candidate.__dict__)
        return response

    def _from_new_parent(self, record):
        # Restore the changed parent and then attach the unchanged fitted interface.
        base = ParentRuntime(record["parent"]["checkpoint"], record["parent"])
        saved = torch.load(ROOT / self.checkpoint["checkpoint"], weights_only=True, map_location="cpu")
        learner = base.session.learner
        facts, library = learner.legacy.snapshot(), learner.library.record()
        ConstraintR1.attach(base.owner, saved["seed"], saved["kind"])
        apply(base.owner, saved["delta"])
        candidate = object.__new__(ConstraintRuntime)
        candidate.checkpoint = copy.deepcopy(self.checkpoint)
        candidate.parent_record, candidate.base, candidate.owner = copy.deepcopy(record["parent"]), base, base.owner
        candidate.observations, candidate.jobs, candidate.history = [], {}, []
        candidate.work = copy.deepcopy(record["work"])
        for row in record["observations"]:
            candidate._admit(row)
        candidate.reproof = learner.rebind(facts, library)
        return candidate

    def _validate_branch(self, branch):
        if "source" in branch and not admitted_runtime(branch["source"]):
            raise ValueError("Invalid branch interpreter")
        if "model" not in branch:
            return
        if not admitted_runtime(branch["source"]) or not 0 <= branch["steps"] <= 12:
            raise ValueError("Invalid branch interpreter or budget")
        points = torch.tensor(branch["initial"], dtype=torch.float64)
        if points.shape != (8, 2) or not torch.isfinite(points).all() or (points.abs() > 1).any():
            raise ValueError("Invalid initial cloud")
        for _ in range(branch["steps"]):
            points = step(branch["model"], points)
            self.work["restore_gradient_batches"] += 1
        if not torch.equal(points, torch.tensor(branch["points"], dtype=torch.float64)):
            raise ValueError("Saved refinement differs from replay")
        if summarize(branch["model"], points) != branch["result"]:
            raise ValueError("Saved conditional result differs from replay")
        if branch["status"] != "stale" and branch["dependency"] != self.dependency(branch["frame"]["target"], interpreter=branch["source"]):
            raise ValueError("A current branch has stale dependencies")
        if branch["status"] != "stale":
            expected = self._new(branch["id"], branch["text"])
            if (not same_binding(expected.get("frame"), branch.get("frame"))
                    or expected.get("model") != branch.get("model")
                    or not same_starting_proposal(expected.get("initial"), branch.get("initial"))):
                raise ValueError("Saved constraints differ from their learned binding or eligible evidence")

    def snapshot(self):
        self.sync()
        return {"schema": "sera.constraint-inquiry.1", "source": runtime_source(),
                "checkpoint": copy.deepcopy(self.checkpoint), "parent": copy.deepcopy(self.parent_record),
                "owner": model_identity(self.owner), "observations": copy.deepcopy(self.observations),
                "jobs": copy.deepcopy(self.jobs), "history": copy.deepcopy(self.history), "work": dict(self.work),
                "replay_migrations": copy.deepcopy(self.replay_migrations)}

    def status(self):
        self.sync()
        return {"owner": model_identity(self.owner), "same_owner": True, "work": self.work,
                "jobs": {i: {"status": b["status"], "text": b["text"], "steps": b["steps"],
                             "result": b["result"]} for i, b in self.jobs.items()},
                "locations": {e: int(self.owner.cloud_location_count[i]) for i, e in enumerate(ENTITIES)},
                "scope": "Finite language and conditional local simulator tasks; no applicability promotion"}


def location_sensor(world, entity):
    # Assessor/sensor state stays outside the trained interfaces. Only readings enter.
    from experiments.continuing_control.environment import CENTER, RADIUS
    # Locate a stationary beacon near a two-step coast trajectory in the local
    # simulator. This uses simulator truth only inside the measurement source.
    fixture = copy.deepcopy(world)
    fixture.step(0)
    fixture.step(0)
    angle = fixture.angle + .035 * (ENTITIES.index(entity) - 1)
    truth = CENTER + RADIUS * np.array([np.cos(angle), np.sin(angle)])
    rng = np.random.default_rng(25300 + ENTITIES.index(entity))
    return [{"kind": "observation", "source": "ci-landmark-simulator-v1", "entity": entity,
             "units": "m", "record_id": f"{entity}:25300:{i}",
             "values": (truth + rng.normal(0, .002, 2)).tolist()} for i in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=ROOT / "runs/sera-constraints")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.store.resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Use a local runs store")
    request = read(args.request)
    if request["operation"] == "initialize":
        directory.mkdir(parents=True, exist_ok=False)
    with lock(directory):
        store = Store(directory)
        previous = store.read()
        selection = read(ROOT / "runs/CI-fits-final-001/selected.json")
        runtime = ConstraintRuntime(selection if previous is None else previous["checkpoint"], previous)
        op = request["operation"]
        if op == "ask":
            response = runtime.ask(request["id"], request["text"])
        elif op == "work":
            response = runtime.work_on(request["id"], request.get("ticks", 12))
        elif op == "solve":
            response = runtime.solve(request["id"])
        elif op == "acquire":
            response = runtime.acquire(request["entity"])
        elif op == "rebase":
            response = runtime.rebase(request["id"])
        elif op == "observe":
            response = runtime.advance(request["action"])
        elif op == "result":
            runtime.sync()
            response = runtime.jobs[request["id"]]
        elif op in ("status", "initialize"):
            response = runtime.status()
        else:
            raise ValueError("Unknown operation")
        if op not in ("status", "result"):
            store.commit(runtime.snapshot(), previous)
        write(args.output, response)
        print(json.dumps(response, indent=2), flush=True)


if __name__ == "__main__":
    main()
