"""Bounded local task execution and persistent, shared-owner investigations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.continuing_control.core import sensor
from experiments.task_transfer.runtime import lock
from sera.session_state import model_identity, pack_tensors, unpack_tensors
from workbench.storage import Store
from .data import LESSON, TEACHING, encode
from .graph import Investigation
from .model import apply, delta, parser_identity
from .study import OUT, ROOT, read, restore_fit, sha


def source():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class Runtime:
    def __init__(self, checkpoint, record=None):
        path = (ROOT/checkpoint["path"]).resolve()
        if not path.is_relative_to(ROOT/"runs") or sha(path) != checkpoint["sha256"]:
            raise ValueError("Inquiry checkpoint identity differs")
        self.session, self.fit = restore_fit(path)
        self.owner = self.session.owner
        if self.fit["kind"] != "shared" or self.fit["training_exact"] != 1.:
            raise ValueError("Only the finite-qualified shared path enters the usable route")
        self.checkpoint = copy.deepcopy(checkpoint)
        self.observations, self.lessons = [], []
        self.restore_observations = 0
        if record is not None:
            if record["schema"] != "sera.language-inquiry.1" or record["source"] != source():
                raise ValueError("Changed runtime: preserve and explicitly migrate this store")
            apply(self.owner, unpack_tensors(record["language_delta"], delta(self.owner)))
            self.owner.inquiry_admitted = list(record["admitted"])
            self.lessons = copy.deepcopy(record["lessons"])
            for row in record["observations"]:
                self._observe(row["action"], expected=row)
                self.restore_observations += 1
            if model_identity(self.owner) != record["owner"]:
                raise ValueError("Restored owner differs")
            self.graph = Investigation.restore(self.owner, record["graph"])
        else:
            self.graph = Investigation(self.owner)
        self.reproof = self.session.learner.rebind(self.session.learner.legacy.snapshot(), self.session.learner.library.record())
        if not (self.owner is self.session.learner.session.owner is self.session.learner.solver.neural.owner
                is self.session.learner.solver.components["typed"].owner):
            raise ValueError("Language, investigation and execution lost their common owner")

    def _observe(self, action, *, expected=None):
        if type(action) is not int or action not in (-1, 0, 1) or len(self.observations) >= 96:
            raise ValueError("A permitted simulator control and bounded observation history are required")
        learner = self.session.learner
        learner.world.step(action)
        values = learner.world.measure().tolist()
        row = {"action": action, "position": values, "time": learner.world.time,
               "origin": "paid_local_simulator_sensor"}
        if expected is not None and row != expected:
            raise ValueError("Stored observation differs from its deterministic simulator replay")
        learner.session.admit(sensor(values, len(learner.session.events), f"LI:{learner.world.spec['seed']}:{learner.world.time}"),
                              action, replay=expected is not None)
        self.observations.append(row)
        return row

    def observe(self, action):
        # An exception cannot partially mutate the accepted owner or event history.
        candidate = copy.deepcopy(self)
        row = candidate._observe(action)
        learner = candidate.session.learner
        candidate.reproof = learner.rebind(learner.legacy.snapshot(), learner.library.record())
        candidate.graph.sync(candidate.owner)
        self.__dict__.update(candidate.__dict__)
        return {**row, "owner": model_identity(self.owner), "changed_situation": self.graph.current}

    def execute(self, identifier):
        self.graph.sync(self.owner)
        j = self.graph.jobs[identifier]
        if j["status"] != "complete" or j["anchor"] != "current":
            raise ValueError("Execute requires a completed current-situation interpretation")
        action = j["interpretation"]["frame"]["actions"][0]
        return {"executed_first_control": action, "observation": self.observe(action),
                "scope": "One authorized local simulator action; remaining plan needs a new current-state investigation"}

    def teach(self, phrase, action, *, checkpoint_folder=None):
        if phrase != LESSON or type(action) is not int or action not in (-1, 0, 1):
            raise ValueError("This finite lesson interface supports coast and an explicit control label")
        if self.lessons:
            raise ValueError("The single-correction protocol is complete; preserve it before designing a new learning cycle")
        candidate = copy.deepcopy(self.owner)
        for name, parameter in candidate.named_parameters():
            parameter.requires_grad_(name.startswith("inquiry_"))
        params = [p for p in candidate.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=.003)
        rows = TEACHING+[(phrase, action+1)]
        ids, y = torch.tensor(encode([t for t, _ in rows])), torch.tensor([v for _, v in rows])
        before = parser_identity(self.owner)
        history = []
        for step in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(candidate.span_logits(ids), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            history.append(float(loss.detach()))
            if checkpoint_folder is not None and (step+1) % 20 == 0:
                torch.save({"delta": delta(candidate), "optimizer": optimizer.state_dict(),
                            "step": step+1, "torch_rng": torch.get_rng_state(), "phrase": phrase,
                            "action": action, "parent_parser": before, "history": history},
                           Path(checkpoint_folder)/f"lesson-{step+1:03d}.pt")
        with torch.no_grad():
            predictions = candidate.span_logits(ids).argmax(-1)
        passed = bool(torch.equal(predictions, y))
        receipt = {"phrase": phrase, "action": action, "admitted": passed, "steps": 80,
                   "new_labels": 1, "replayed_labels": len(TEACHING), "presentations": 80*len(rows),
                   "old_correct": int((predictions[:-1] == y[:-1]).sum()), "old_count": len(TEACHING),
                   "new_correct": bool(predictions[-1] == y[-1]), "parent_parser": before,
                   "scope": "One supplied supervised correction with replay; no lifelong or learning-to-learn claim"}
        if passed:
            apply(self.owner, delta(candidate))
            self.owner.inquiry_admitted = [phrase]
            learner = self.session.learner
            self.reproof = learner.rebind(learner.legacy.snapshot(), learner.library.record())
            self.graph.sync(self.owner)
        self.lessons.append(receipt)
        return receipt

    def snapshot(self):
        return {"schema": "sera.language-inquiry.1", "source": source(), "checkpoint": self.checkpoint,
                "owner": model_identity(self.owner), "language_delta": pack_tensors(delta(self.owner)),
                "admitted": list(self.owner.inquiry_admitted), "observations": copy.deepcopy(self.observations),
                "lessons": copy.deepcopy(self.lessons), "graph": self.graph.snapshot(), "reproof": self.reproof}

    def status(self):
        self.graph.sync(self.owner)
        return {"owner": model_identity(self.owner), "parent_task_owner": self.fit["parent_owner"],
                "same_owner": True, "preserved_numerical_tasks": sorted(self.session.tasks),
                "paid_new_observations": len(self.observations), "restore_observations": self.restore_observations,
                "parser": parser_identity(self.owner), "lessons": copy.deepcopy(self.lessons),
                "situation": self.graph.current, "work": dict(self.graph.work),
                "investigations": {k: {"status": j["status"], "progress": j["cursor"], "text": j["text"],
                                      "result": j["result"]} for k, j in self.graph.jobs.items()},
                "scope": "Supplied grammar, learned span meanings, learned continuing dynamics; conditional forecasts are uncalibrated"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", type=Path, default=ROOT/"runs/sera-inquiry")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    directory = a.store.resolve()
    if not directory.is_relative_to(ROOT/"runs"):
        raise ValueError("Use a local owned runs directory")
    request = read(a.request)
    operation = request["operation"]
    if operation == "initialize":
        directory.mkdir(parents=True, exist_ok=False)
    if not directory.is_dir():
        raise ValueError("Initialize the investigation store first")
    with lock(directory):
        store = Store(directory)
        previous = store.read()
        if previous is None:
            if operation != "initialize":
                raise ValueError("No saved learner")
            selection = read(ROOT/"runs/LI-fits-final-001/selected.json")
            runtime = Runtime({"path": selection["checkpoint"], "sha256": selection["sha256"]})
            response = runtime.status()
        else:
            if operation == "initialize":
                raise ValueError("Already initialized")
            runtime = Runtime(previous["checkpoint"], previous)
            graph, owner = runtime.graph, runtime.owner
            if operation == "ask":
                response = graph.add(owner, request["id"], request["text"], entity=request.get("entity", "orbit"),
                                     anchor=request.get("anchor", "current"))
            elif operation == "work":
                graph.run(owner, request.get("ticks", 12)); response = runtime.status()
            elif operation == "result":
                graph.sync(owner); response = copy.deepcopy(graph.jobs[request["id"]])
            elif operation == "pause":
                graph.pause(request["id"]); response = runtime.status()
            elif operation == "resume":
                graph.resume(owner, request["id"]); response = runtime.status()
            elif operation == "rebase":
                response = graph.rebase(owner, request["id"])
            elif operation == "clarify":
                graph.clarify(request["id"], request["actions"]); response = runtime.status()
            elif operation == "observe":
                response = runtime.observe(request["action"])
            elif operation == "execute":
                response = runtime.execute(request["id"])
            elif operation == "teach":
                lesson_dir = a.output.parent/"lesson-checkpoints"
                lesson_dir.mkdir(exist_ok=False)
                (lesson_dir/"parent.json").write_text(json.dumps(runtime.snapshot()), encoding="utf-8")
                response = runtime.teach(request["phrase"], request["action"], checkpoint_folder=lesson_dir)
            elif operation == "status":
                response = runtime.status()
            else:
                raise ValueError("Unknown operation")
        # Read-only operations still charge numerical restoration externally, but
        # do not create a new persistent revision just for viewing state.
        if operation not in ("status", "result"):
            store.commit(runtime.snapshot(), previous)
        from .study import write
        write(a.output, response)
        print(json.dumps(response, indent=2), flush=True)


if __name__ == "__main__":
    main()
