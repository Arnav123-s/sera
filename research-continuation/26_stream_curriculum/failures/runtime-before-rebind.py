"""Persist learned request frames alongside the existing task-bearing owner."""

import argparse
import copy
import json
from pathlib import Path

import torch

from experiments.task_transfer.runtime import lock
from sera.session_state import model_identity
from workbench.storage import Store

from .data import ROOT, read, sha
from .model import apply, encode
from .study import attach, identities, load_parent, spans


class RequestSession:
    def __init__(self, checkpoint, record=None):
        path = (ROOT / checkpoint["path"]).resolve()
        if not path.is_relative_to(ROOT / "runs") or sha(path) != checkpoint["sha256"]:
            raise ValueError("Request checkpoint path or bytes changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["schema"] != "sera.stream-training.1" or saved["identities"] != identities():
            raise ValueError("Request training source or parent changed")
        self.base = load_parent()
        self.owner = attach(self.base, saved["config"])
        learner = self.base.base.session.learner
        facts, library = learner.legacy.snapshot(), learner.library.record()
        apply(self.owner, saved["delta"])
        self.reproof = learner.rebind(facts, library)
        self.checkpoint, self.requests = copy.deepcopy(checkpoint), {}
        self.training_step, self.seen = saved["step"], len(saved["seen_ids"])
        if record is not None:
            if (record["schema"] != "sera.request-session.1" or record["source"] != sha(Path(__file__))
                    or record["owner"] != model_identity(self.owner) or record["checkpoint"] != checkpoint
                    or len(record["requests"]) > 32):
                raise ValueError("Changed saved request session")
            for identifier, frame in record["requests"].items():
                if frame != self.interpret(frame["text"]):
                    raise ValueError("Saved frame differs from exact checkpoint replay")
                self.requests[identifier] = copy.deepcopy(frame)

    @torch.no_grad()
    def interpret(self, text):
        ids = encode([text])
        intents, slots = self.owner.request_logits(ids)
        vocabulary = self.owner.stream_config["vocabulary"]
        tokens = text.split()
        intent = int(intents.argmax(-1).item())
        tags = [vocabulary["tags"][i] for i in slots.argmax(-1)[0].tolist()]
        return {"text": text, "intent": vocabulary["intents"][intent],
                "intent_score": float(intents.softmax(-1)[0, intent]),
                "entities": [{"type": label, "value": " ".join(tokens[start:end]),
                              "token_start": start, "token_end": end}
                             for label, start, end in sorted(spans(tags), key=lambda x: (x[1], x[2], x[0]))],
                "tags": tags, "evidence_kind": "learned_interpretation_of_user_request",
                "qualification": "development_checkpoint", "training_step": self.training_step,
                "training_examples_seen": self.seen, "checkpoint": self.checkpoint,
                "actions_executed": 0}

    def ask(self, identifier, text):
        if not isinstance(identifier, str) or not identifier or identifier in self.requests or len(self.requests) >= 32:
            raise ValueError("A new bounded request identifier is required")
        self.requests[identifier] = self.interpret(text)
        return copy.deepcopy(self.requests[identifier])

    def snapshot(self):
        return {"schema": "sera.request-session.1", "source": sha(Path(__file__)),
                "checkpoint": self.checkpoint, "owner": model_identity(self.owner),
                "requests": copy.deepcopy(self.requests)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("initialize", "ask", "result"))
    parser.add_argument("--store", type=Path, default=ROOT / "runs/sera-requests")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--text")
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = args.store.resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Local request stores must remain inside runs")
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory):
        store, previous = Store(directory), Store(directory).read()
        if args.action == "initialize":
            if previous is not None or args.checkpoint is None:
                raise ValueError("Use a fresh store and an exact training checkpoint")
            path = args.checkpoint.resolve()
            runtime = RequestSession({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
            answer = {"owner": model_identity(runtime.owner), "training_step": runtime.training_step}
        else:
            if previous is None or not args.id:
                raise ValueError("Initialize the store and supply a request ID")
            runtime = RequestSession(previous["checkpoint"], previous)
            answer = runtime.ask(args.id, args.text) if args.action == "ask" else runtime.requests[args.id]
        if args.action != "result":
            store.commit(runtime.snapshot(), previous)
        print(json.dumps(answer, ensure_ascii=False))


if __name__ == "__main__":
    main()
