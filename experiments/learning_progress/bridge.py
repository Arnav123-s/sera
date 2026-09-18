"""Usable reading and requalified method portfolios on the continuing learner."""

import argparse
import copy
import json
from pathlib import Path

import torch

from experiments.concept_refinement.audit import languages
from experiments.quest_portfolio.board import Board
from experiments.quest_portfolio.runtime import call_assessor, protected_hash
from experiments.task_transfer.runtime import lock
from experiments.verified_completion.credit import encode
from sera.session_state import model_identity
from workbench.storage import Store

from .common import OUT, ROOT, contracts, read, sha
from .runtime import LearningSession


class LiveSession(LearningSession):
    def __init__(self, saved):
        super().__init__(saved=saved)
        self.bridge = copy.deepcopy(saved.get("bridge"))
        if self.bridge:
            if self.bridge["source"] != sha(Path(__file__)):
                raise ValueError("Changed live bridge contract")
            self.base.board = Board.restore(self.bridge["board"])
            self.base.anchor = self.bridge["anchor"]
            self.base.pending_assessment = self.bridge["pending_assessment"]
            receipts = [r for quest in self.base.board.quests.values() for r in quest["assessments"]]
            call_assessor(self.base.authority, {"action": "verify", "receipts": receipts})

    def qualify(self, identifier="velocity-routes"):
        if not self.bridge:
            audit = read(OUT / "audit.json")
            if (audit["contracts"] != contracts() or audit["owner"] != model_identity(self.owner)
                    or not audit["protected_equal"] or not audit["language_equal"]
                    or not audit["snapshot_equal"] or audit["novel_definition_gate"]):
                raise ValueError("The exact independently audited successor is required")
            values, probes = languages(self.owner)
            # This successor anchor is established once, after the independent
            # acquisition/retention audit. Subsequent mutations stale qualification.
            self.base.anchor = {"protected": protected_hash(self.owner), "languages": encode(values),
                                "probes": probes}
            self.bridge = {"source": sha(Path(__file__)), "audit": sha(OUT / "audit.json")}
        return self.base.assess(identifier)

    def snapshot(self):
        result = super().snapshot()
        if self.bridge:
            result["bridge"] = {**self.bridge, "board": self.base.board.snapshot(),
                                "anchor": self.base.anchor,
                                "pending_assessment": copy.deepcopy(self.base.pending_assessment)}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "qualify", "solve", "read"))
    parser.add_argument("--store", default="runs/sera-learning-live")
    parser.add_argument("--id", default="velocity-routes")
    parser.add_argument("--input")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = (ROOT / args.store).resolve()
    if not path.is_relative_to(ROOT / "runs"):
        raise ValueError("Owned local learner store required")
    with lock(path):
        store = Store(path)
        previous = store.read()
        if previous is None:
            raise ValueError("Restore the independently admitted learner first")
        session = LiveSession(previous)
        if args.action == "qualify":
            result = session.qualify(args.id)
            store.commit(session.snapshot(), previous)
        elif args.action == "solve":
            result = session.base.solve(args.id, read(args.input))
        elif args.action == "read":
            request = read(args.input)
            if not request.get("source") or not isinstance(request.get("question"), str):
                raise ValueError("An attributed source and question are required")
            with torch.no_grad():
                values, mask = session.owner.reading_features([request])
                chosen = int(session.owner.reading_logits(values, mask).argmax(-1)[0])
            result = {"status": "ATTRIBUTED_SENTENCE_SELECTION", "question": request["question"],
                      "sentence": request["sentences"][chosen]["text"], "source": request["source"],
                      "owner": model_identity(session.owner), "factual_updates": 0}
        else:
            result = {"owner": model_identity(session.owner), "practice_updates": len(session.events),
                      "default_controller": read(OUT / "final.json")["default"],
                      "quests": session.base.board.snapshot()["view"], "goal": session.goal}
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
