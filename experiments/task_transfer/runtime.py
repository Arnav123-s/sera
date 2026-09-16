"""Persistent numerical readouts owned by a preserved SERA descendant.

Predictions are empirical and conditional. The existing formal route remains
separate, with its original proof contract. No live workbench file is modified.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.grounded_language.model import LanguageR1
from sera.session_state import model_identity
from workbench.model import Learner

from .core import acquire, evidence, features, fingerprint, learned_basis
from .study import ROOT, sha, write

MIGRATABLE_SOURCES = {"de3b98e839afec3d8e139141abaf1e58dd43fd16567e1a6413d3d93fb94095f4"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source():
    return hashlib.sha256(Path(__file__).read_bytes()+fingerprint().encode()).hexdigest()


def name_check(name):
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", name) is None:
        raise ValueError("Use a task/group name beginning with a letter, then up to 47 letters, numbers or underscores")


def require_fresh_calibration(events, calibration):
    keys = set(map(tuple, calibration["x"]))
    for event in events:
        if keys & set(map(tuple, event.get("used_inputs", []))):
            raise ValueError("Each task version needs fresh calibration across the whole session, including other task names and observations")


class TaskReadout(nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        value = torch.tensor(coefficient, dtype=torch.float64)
        if value.shape != (20,) or not torch.isfinite(value).all():
            raise ValueError("Invalid task weights")
        self.weight = nn.Parameter(value, requires_grad=False)

    def forward(self, x):
        return torch.from_numpy(features(x))@self.weight


class TaskR1(LanguageR1):
    @classmethod
    def attach(cls, owner, tasks, implementation_source=None):
        if type(owner) is not LanguageR1:
            raise ValueError("Restore the exact language owner before attaching numerical tasks")
        owner.__class__ = cls
        owner.task_readouts = nn.ModuleDict()
        owner.task_metadata = {}
        owner.task_extension_source = implementation_source or source()
        for name, task in tasks.items():
            owner.install(name, task)

    def install(self, name, task):
        name_check(name)
        if task["acquisition"]["source"] != fingerprint():
            raise ValueError("Task interpreter changed; an explicit migration is required")
        self.task_readouts[name] = TaskReadout(task["acquisition"]["selected"]["coefficients"])
        self.task_metadata[name] = digest(task)

    def export_config(self):
        return {**super().export_config(), "task_extension_source": self.task_extension_source,
                "task_readouts": dict(self.task_metadata)}


class Session:
    def __init__(self, parent, revision=None, *, migrate=False):
        previous_source = revision["source"] if revision else source()
        migrating = migrate and previous_source in MIGRATABLE_SOURCES
        if previous_source != source() and not migrating:
            raise ValueError("Saved task source changed; explicitly migrate the preserved session")
        self.parent = copy.deepcopy(parent)
        self.learner = Learner(copy.deepcopy(parent))
        self.owner = self.learner.session.owner
        self.tasks = copy.deepcopy(revision["tasks"]) if revision else {}
        self.events = copy.deepcopy(revision["events"]) if revision else []
        self.number = revision["number"] if revision else 0
        factual, library = self.learner.legacy.snapshot(), self.learner.library.record()
        TaskR1.attach(self.owner, self.tasks, previous_source)
        self.reproof_work = self.learner.rebind(factual, library)
        if revision and revision["owner"] != model_identity(self.owner):
            raise ValueError("Saved task owner/source identity mismatch")
        if migrating:
            factual, library = self.learner.legacy.snapshot(), self.learner.library.record()
            self.owner.task_extension_source = source()
            self.reproof_work = self.learner.rebind(factual, library)
            self.events.append({"operation": "migrate", "from": previous_source, "to": source(), "labels": 0,
                                "reason": "Session-wide fresh calibration, including renamed tasks and prior observations; unchanged learned tensors"})
        assert self.learner.solver.neural.owner is self.learner.solver.components["typed"].owner is self.owner

    def rebind(self, factual, library):
        self.reproof_work = self.learner.rebind(factual, library)

    def teach(self, name, group, batches, tolerance=.05, noise=0.):
        name_check(name)
        name_check(group)
        if name not in self.tasks and len(self.tasks) >= 16:
            raise ValueError("This bounded descendant supports at most sixteen named task readouts")
        require_fresh_calibration(self.events, batches["calibration"])
        priors = [task["acquisition"]["selected"]["coefficients"] for key, task in self.tasks.items()
                  if key != name and task["group"] == group and task["status"] == "ACCEPTED_EMPIRICAL"]
        basis, singular = learned_basis(priors)
        acquired = acquire(batches["fit"], batches["selection"], batches["calibration"],
                           basis=basis, tolerance=tolerance, noise=noise)
        factual, library = self.learner.legacy.snapshot(), self.learner.library.record()
        task = {"group": group, "status": "ACCEPTED_EMPIRICAL" if acquired["gate"]["accepted"] else "WITHHELD",
                "acquisition": acquired, "basis": basis.tolist(), "source_tasks": [key for key, value in self.tasks.items()
                    if key != name and value["group"] == group and value["status"] == "ACCEPTED_EMPIRICAL"],
                "source_singular_values": singular, "observations": [],
                "generation": 1+self.tasks[name]["generation"] if name in self.tasks else 1}
        self.tasks[name] = task
        self.owner.install(name, task)
        self.rebind(factual, library)
        self.events.append({"operation": "teach", "task": name, "group": group,
                            "generation": task["generation"], "status": task["status"],
                            "labels": acquired["labels"], "evidence_sha256": acquired["evidence_sha256"],
                            "used_inputs": [row for batch in batches.values() for row in batch["x"]]})
        return {"task": name, "status": task["status"], "generation": task["generation"],
                "selected_method": acquired["selected"]["method"], "source_tasks": task["source_tasks"],
                "paid_labels": acquired["labels"], "gate": acquired["gate"],
                "owner": model_identity(self.owner), "reproof_work": self.reproof_work}

    def predict(self, name, x):
        if name not in self.tasks:
            raise ValueError("No acquired task with that name")
        task = self.tasks[name]
        if task["status"] != "ACCEPTED_EMPIRICAL":
            return {"task": name, "status": task["status"], "outputs": [], "reason": "No currently applicable calibrated task version"}
        if not 1 <= len(x) <= 256:
            raise ValueError("Supply 1..256 query rows")
        try:
            with torch.no_grad():
                values = self.owner.task_readouts[name](x).tolist()
        except ValueError as error:
            return {"task": name, "status": "OUTSIDE_DOMAIN", "outputs": [], "reason": str(error)}
        return {"task": name, "status": "EMPIRICAL_PREDICTION", "outputs": values,
                "generation": task["generation"], "owner": model_identity(self.owner),
                "gate": task["acquisition"]["gate"],
                "scope": "Learned numerical rule on three normalized inputs in [-1,1]. Assumes fresh iid calibration and unchanged deployment distribution. Rare exceptions and undetected drift remain possible. This is not a pointwise proof or a language-understanding claim."}

    def observe(self, name, x, y):
        if name not in self.tasks:
            raise ValueError("No acquired task with that name")
        batch = evidence(x, y, "observation")
        if not 1 <= len(x) <= 256:
            raise ValueError("Supply 1..256 new observations")
        task = self.tasks[name]
        identity = digest(batch)
        if any(row["digest"] == identity for row in task["observations"]):
            raise ValueError("This observation batch was already recorded")
        with torch.no_grad():
            predicted = self.owner.task_readouts[name](x).numpy()
        error = abs(predicted-np.asarray(y))
        failures = int(np.sum(error > task["acquisition"]["gate"]["tolerance"]))
        factual, library = self.learner.legacy.snapshot(), self.learner.library.record()
        if failures:
            task["status"] = "WITHDRAWN"
        task["observations"].append({"digest": identity, "batch": batch, "failures": failures,
                                     "max_error": float(error.max())})
        self.owner.install(name, task)
        self.rebind(factual, library)
        self.events.append({"operation": "observe", "task": name, "labels": len(x), "failures": failures,
                            "digest": identity, "status": task["status"], "used_inputs": batch["x"]})
        return {"task": name, "status": task["status"], "new_observations": len(x),
                "failures": failures, "max_error": float(error.max())}

    def snapshot(self):
        return {"schema": "sera.task-transfer-session.1", "source": source(), "number": self.number,
                "parent_owner": self.parent["owner_sha256"], "owner": model_identity(self.owner),
                "tasks": copy.deepcopy(self.tasks), "events": copy.deepcopy(self.events),
                "reproof_work": dict(self.reproof_work)}


def load(directory, *, migrate=False):
    pointer = json.loads((directory/"current.json").read_text())
    name = pointer["revision"]
    if Path(name).name != name or not re.fullmatch(r"\d{6}-[a-f0-9]{12}\.json", name):
        raise ValueError("Invalid local revision name")
    path = directory/"revisions"/name
    if sha(path) != pointer["sha256"]:
        raise ValueError("Revision integrity mismatch")
    parent_path = directory/"parent.json"
    if sha(parent_path) != pointer["parent_sha256"]:
        raise ValueError("Parent integrity mismatch")
    revision = json.loads(path.read_text())
    if migrate and revision["source"] in MIGRATABLE_SOURCES:
        # Old versions retained the factual observation in their task records;
        # restore those input identities before enforcing session-wide freshness.
        observations = {}
        for old_path in sorted((directory/"revisions").glob("*.json")):
            if not sha(old_path).startswith(old_path.stem.split("-")[-1]):
                raise ValueError("Historical revision integrity mismatch during migration")
            old = json.loads(old_path.read_text())
            for task in old["tasks"].values():
                for row in task["observations"]:
                    observations[row["digest"]] = row["batch"]["x"]
        for event in revision["events"]:
            if event["operation"] == "observe" and "used_inputs" not in event:
                if event["digest"] not in observations:
                    raise ValueError("Historical observation needed for migration is missing")
                event["used_inputs"] = observations[event["digest"]]
    return Session(json.loads(parent_path.read_text()), revision, migrate=migrate)


def save(directory, session):
    session.number += 1
    revision = session.snapshot()
    revision["saved_at"] = datetime.now(timezone.utc).isoformat()
    data = (json.dumps(revision, indent=2, allow_nan=False)+"\n").encode()
    identity = hashlib.sha256(data).hexdigest()
    name = f"{session.number:06d}-{identity[:12]}.json"
    with (directory/"revisions"/name).open("xb") as f:
        f.write(data)
    temporary = directory/"current.tmp"
    write(temporary, {"revision": name, "sha256": identity, "parent_sha256": sha(directory/"parent.json")})
    temporary.replace(directory/"current.json")
    return name


def initialize(directory):
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"revisions").mkdir()
    pointer = json.loads((ROOT/"runs/sera-workbench/current.json").read_text())
    parent_path = ROOT/"runs/sera-workbench/revisions"/pointer["revision"]
    if sha(parent_path) != pointer["sha256"]:
        raise ValueError("Existing workbench parent integrity mismatch")
    parent = json.loads(parent_path.read_text())
    write(directory/"parent.json", parent)
    session = Session(parent)
    save(directory, session)
    return session


@contextmanager
def lock(directory):
    path = directory/"task.lock"
    with path.open("x", encoding="utf-8") as f:
        f.write("Exclusive local task transaction\n")
    try:
        yield
    finally:
        path.unlink()


def read_csv(path, operation):
    text = Path(path).read_text(encoding="utf-8-sig")
    if len(text) > 200_000:
        raise ValueError("Use CSV smaller than 200 KB")
    reader = csv.DictReader(io.StringIO(text))
    required = ["x1", "x2", "x3"]+(["y", "role"] if operation == "teach" else ["y"] if operation == "observe" else [])
    if reader.fieldnames != required:
        raise ValueError("CSV header must be "+",".join(required))
    groups = {}
    for row in reader:
        if None in row or any(row[key] is None for key in required):
            raise ValueError("Malformed CSV row")
        role = row["role"] if operation == "teach" else operation
        x, y = groups.setdefault(role, ([], []))
        x.append([float(row[key]) for key in ("x1", "x2", "x3")])
        if "y" in row:
            y.append(float(row["y"]))
        if sum(len(value[0]) for value in groups.values()) > 1024:
            raise ValueError("Use at most 1024 example rows")
    if operation == "teach":
        if set(groups) != {"fit", "selection", "calibration"}:
            raise ValueError("Supply fit, selection and calibration roles")
        return {role: evidence(*value, role) for role, value in groups.items()}
    if operation not in groups:
        raise ValueError("CSV contains no rows")
    return groups[operation]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("init", "teach", "predict", "observe", "status", "migrate"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--group", default="default")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--tolerance", type=float, default=.05)
    parser.add_argument("--noise", type=float, default=0.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT/"runs"):
        raise ValueError("Keep task session state in the repository's runs directory")
    if args.operation == "init":
        session = initialize(directory)
        result = {"status": "INITIALIZED", "owner": model_identity(session.owner)}
    else:
        with lock(directory):
            session = load(directory, migrate=args.operation == "migrate")
            if args.operation == "migrate":
                save(directory, session)
                result = {"status": "MIGRATED", "source": source(), "owner": model_identity(session.owner)}
            elif args.operation == "status":
                result = {"owner": model_identity(session.owner), "revision": session.number,
                          "tasks": {name: {"status": task["status"], "generation": task["generation"],
                                           "group": task["group"]} for name, task in session.tasks.items()}}
            elif args.operation == "teach":
                result = session.teach(args.task, args.group, read_csv(args.csv, "teach"), args.tolerance, args.noise)
                save(directory, session)
            elif args.operation == "predict":
                x, _ = read_csv(args.csv, "predict")
                result = session.predict(args.task, x)
            else:
                x, y = read_csv(args.csv, "observe")
                result = session.observe(args.task, x, y)
                save(directory, session)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
