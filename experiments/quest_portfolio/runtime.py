"""Persistent verified alternatives through the continuing SERA owner."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.concept_refinement.audit import languages
from experiments.task_transfer.runtime import lock
from experiments.verified_completion.credit import decode, encode
from experiments.verified_completion.model import CompletionR1
from experiments.verified_completion.runtime import CompletionSession
from sera.session_state import model_identity
from workbench.storage import Store

from .board import Board
from .common import OUT, ROOT, contracts, digest, model_hash, read, sha
from .methods import SCOPES, SPECS, STOP, canonical, deploy, execute, features
from .model import QuestR1


def protected_hash(owner):
    import hashlib
    value = hashlib.sha256()
    for key, tensor in sorted(owner.state_dict().items()):
        if key.startswith(("quest_policy.", "completion_policy.", "completion_clouds.")):
            continue
        value.update(key.encode())
        value.update(tensor.detach().numpy().tobytes())
    return value.hexdigest()


def call_assessor(path, request):
    result = subprocess.run([sys.executable, "-m", "experiments.quest_portfolio.assessor", "--store", str(path)],
                            input=json.dumps(request, allow_nan=False), text=True, capture_output=True,
                            check=True, cwd=ROOT, timeout=30)
    return json.loads(result.stdout)


class QuestSession:
    def __init__(self, saved=None, *, parent=None, authority=None, seed=3301, selected=False):
        if saved and (saved["schema"] != "sera.verified-quests.1" or saved["contracts"] != contracts()):
            raise ValueError("Quest contract changed; preserve and explicitly migrate")
        self.base = CompletionSession(saved["base"] if saved else parent)
        learner = self.base.grounded.base.learner
        facts, library = learner.legacy.snapshot(), learner.library.record()
        self.owner = QuestR1.attach(self.base.owner, saved["seed"] if saved else seed)
        learner.rebind(facts, library)
        self.base._refresh()
        self.seed = saved["seed"] if saved else seed
        self.authority = Path(saved["authority"] if saved else authority or ROOT / "runs/sera-quest-assessor").resolve()
        self.board = Board.restore(saved["board"]) if saved else Board()
        self.optimizer = torch.optim.Adam(self.owner.quest_policy.parameters(), lr=.0003)
        self.rng = torch.Generator().manual_seed(33401)
        self.steps, self.baseline = 0, 0.
        self.pending = copy.deepcopy(saved.get("pending")) if saved else None
        self.pending_assessment = copy.deepcopy(saved.get("pending_assessment")) if saved else None
        self.persist = None
        self.selection = None
        if saved:
            self.base._mutation(lambda: self.owner.quest_policy.load_state_dict(decode(saved["weights"])))
            self.optimizer.load_state_dict(decode(saved["optimizer"]))
            self.rng.set_state(decode(saved["rng"]))
            self.steps, self.baseline = saved["steps"], saved["baseline"]
            self.anchor, self.selection = saved["anchor"], saved["selection"]
            receipts = [r["receipt"] for q in self.board.quests.values() for r in q["attempts"]]
            receipts += [r for q in self.board.quests.values() for r in q["assessments"]]
            if receipts:
                call_assessor(self.authority, {"action": "verify", "receipts": receipts})
            learned = [e["data"] for e in self.board.events if e["kind"] == "practice" and e["data"]["updated"]]
            baseline = 0.
            for event in learned:
                baseline = .98*baseline+.02*event["reward"]
            if self.steps != len(learned) or self.baseline != baseline or saved["owner"] != model_identity(self.owner):
                raise ValueError("Quest learning state lost its checked history")
        else:
            logits, probes = languages(self.owner)
            self.anchor = {"protected": protected_hash(self.owner), "languages": encode(logits), "probes": probes}
            if selected:
                self.selection = read(OUT / "selection.json")
                if self.selection["contracts"] != contracts() or sha(ROOT / self.selection["checkpoint"]) != self.selection["sha256"]:
                    raise ValueError("Selected quest policy changed")
                checkpoint = torch.load(ROOT / self.selection["checkpoint"], weights_only=True, map_location="cpu")
                self.base._mutation(lambda: self.owner.quest_policy.load_state_dict(checkpoint["weights"]))
        self.base._refresh()
        if self.owner is not self.base.grounded.base.study.owner:
            raise AssertionError("Quest control must use the actual continuing learner")

    def retention(self):
        logits, probes = languages(self.owner)
        expected = decode(self.anchor["languages"])
        error = max(float((a-b).abs().max()) for key in logits
                    for a, b in zip(logits[key], expected[key], strict=True))
        if probes != self.anchor["probes"]:
            raise ValueError("Retention probes changed")
        return {"owner": model_identity(self.owner), "protected_equal": protected_hash(self.owner) == self.anchor["protected"],
                "language_max_error": error, "probes": sum(map(len, probes.values())),
                "source": digest(self.anchor), "old_definition_gate": self.base.grounded.gate["autonomous"]}

    def sync_owner(self):
        self.base._refresh()
        self.board.append("owner", {"identity": model_identity(self.owner)})

    def _save_boundary(self):
        if self.persist:
            self.persist(self.snapshot())

    def practice(self, identifier, *, controller="learned"):
        if self.pending_assessment:
            raise ValueError("Finish the frozen assessment before further practice")
        key = self.board.aliases[identifier]
        quest = self.board.quests[key]
        step = 5-quest["remaining_steps"]
        if step >= 5:
            raise ValueError("Five-attempt practice allocation spent; preserve and assess")
        visits = np.zeros((1, 7))
        covered = np.zeros((1, 4))
        for row in quest["attempts"]:
            visits[0, row["method"]] += 1
            for scope in row["receipt"]["valid_scopes"]:
                covered[0, SCOPES.index(scope)] = 1
        values = features(np.full((1, 4), .25), covered, visits, step)[0]
        forced = step == 4
        with torch.no_grad():
            probabilities = self.owner.quest_policy(torch.from_numpy(values)).softmax(-1)
            if self.pending:
                action = self.pending["decision"]["action"]
            elif forced:
                action = int(visits[0, :6].argmin())
            elif controller == "balanced":
                action = int(visits[0, :4].argmin())
            elif controller == "learned":
                action = int(torch.multinomial(probabilities, 1, generator=self.rng))
            else:
                raise ValueError("Choose the balanced control or on-policy learned practice")
        if action == STOP:
            self.board.append("stop", {"quest": key, "owner": model_identity(self.owner), "step": step})
            self._save_boundary()
            return {"status": "SKIPPED_OPTIONAL_ATTEMPT", "remaining_steps": quest["remaining_steps"], "reward": 0., "updated": False}
        owner_before = model_identity(self.owner)
        decision = {"quest": key, "original_goal": quest["original_goal"], "owner": owner_before,
                    "policy": model_hash(self.owner.quest_policy), "features": values.tolist(),
                    "probabilities": probabilities.tolist(), "action": action, "step": step,
                    "forced_exploration": forced, "controller": controller}
        identity = digest(decision)
        if self.pending:
            decision = self.pending["decision"]
            identity, action = digest(decision), decision["action"]
            values = np.asarray(decision["features"])
            controller, forced = decision["controller"], decision["forced_exploration"]
            if decision["owner"] != owner_before or decision["quest"] != key or decision["step"] != step:
                raise ValueError("Preserve the unfinished practice decision; owner or goal changed")
        else:
            self.pending = {"decision": decision}
            self._save_boundary()
        candidate = canonical(SPECS[action])
        request = {"mode": "practice", "quest": key, "candidate": candidate, "owner": owner_before, "decision": identity}
        inputs = call_assessor(self.authority, {"action": "recover", "request": request})
        if inputs is None:
            inputs = call_assessor(self.authority, {"action": "begin", "request": request})
        if "receipt" in inputs:
            receipt = inputs["receipt"]
        else:
            predictions = [execute(SPECS[action], case, self.base) for case in inputs["cases"]]
            receipt = call_assessor(self.authority, {"action": "grade", "request": {
                "id": inputs["id"], "owner": owner_before, "candidate": candidate, "predictions": predictions}})
        reward = self.board.reward(key, action, receipt)
        updated = controller == "learned" and not forced
        if updated:
            def update():
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.owner.quest_policy(torch.from_numpy(values))
                loss = -(reward-self.baseline)*logits.log_softmax(-1)[action]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.owner.quest_policy.parameters(), 2., error_if_nonfinite=True)
                self.optimizer.step()
            self.base._mutation(update)
            self.steps += 1
            self.baseline = .98*self.baseline+.02*reward
        event = {"quest": key, "method": action, "decision": decision, "decision_id": identity,
                 "receipt": receipt, "reward": reward, "updated": updated,
                 "owner_after": model_identity(self.owner), "factual_updates": 0}
        self.board.append("practice", event)
        self.sync_owner()
        self.pending = None
        self._save_boundary()
        return copy.deepcopy(event)

    def assess(self, identifier):
        if self.pending:
            raise ValueError("Finish the committed practice decision before assessment")
        key = self.board.aliases[identifier]
        methods = self.board.quests[key]["methods"]
        owner = model_identity(self.owner)
        candidate = digest(sorted(set(methods)))
        request = {"mode": "boss", "quest": key, "candidate": candidate, "owner": owner, "retention": self.retention(),
                   "decision": digest({"quest": key, "attempt": len(self.board.quests[key]["assessments"]), "owner": owner})}
        if self.pending_assessment and self.pending_assessment != request:
            raise ValueError("Resume the frozen assessment before changing its owner or scope")
        self.pending_assessment = request
        self._save_boundary()
        inputs = call_assessor(self.authority, {"action": "recover", "request": request})
        if inputs is None:
            inputs = call_assessor(self.authority, {"action": "begin", "request": request})
        if "receipt" in inputs:
            receipt = inputs["receipt"]
        else:
            predictions = [deploy(methods, case, self.base)[0] for case in inputs["cases"]]
            receipt = call_assessor(self.authority, {"action": "grade", "request": {
                "id": inputs["id"], "owner": owner, "candidate": candidate, "predictions": predictions}})
        self.board.append("assessment", {"quest": key, "receipt": receipt})
        self.pending_assessment = None
        self._save_boundary()
        return copy.deepcopy(receipt)

    def solve(self, identifier, case):
        if not self.board.eligible(identifier, model_identity(self.owner), case.get("scope")):
            raise ValueError("Fresh qualification for this owner and scope is required")
        quest = self.board.quests[self.board.aliases[identifier]]
        answer, method = deploy(quest["methods"], case, self.base)
        return {"answer": answer, "method": method, "unit": "m/s", "status": "CONDITIONAL_CHECKED_ROUTE" if answer is not None else "NEEDS_GROUNDED_EVIDENCE",
                "assumptions": case.get("assumptions"), "original_goal": quest["original_goal"], "factual_updates": 0}

    def snapshot(self):
        identity = model_identity(self.owner)
        if any(q["state"] == "QUALIFIED" and q["assessments"][-1]["owner"] != identity for q in self.board.quests.values()):
            self.sync_owner()
        # Stage 32 remains byte-preserved and restores its own exact identity.
        policy = self.owner._modules.pop("quest_policy")
        self.owner.__class__ = CompletionR1
        try:
            base = self.base.snapshot()
        finally:
            self.owner.__class__ = QuestR1
            self.owner.quest_policy = policy
            self.base._refresh()
        return {"schema": "sera.verified-quests.1", "contracts": contracts(), "base": base,
                "seed": self.seed, "authority": str(self.authority), "board": self.board.snapshot(),
                "owner": model_identity(self.owner), "weights": encode(policy.state_dict()),
                "optimizer": encode(self.optimizer.state_dict()), "rng": encode(self.rng.get_state()),
                "steps": self.steps, "baseline": self.baseline, "anchor": self.anchor, "selection": self.selection,
                "pending": copy.deepcopy(self.pending), "pending_assessment": copy.deepcopy(self.pending_assessment)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("learn", "assess", "status", "solve"))
    parser.add_argument("--store", default="runs/sera-quests-live")
    parser.add_argument("--id", default="velocity-routes")
    parser.add_argument("--controller", choices=("balanced", "learned"), default="balanced")
    parser.add_argument("--input", help="Path to typed conditional mechanics JSON")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = (ROOT / args.store).resolve()
    if not path.is_relative_to(ROOT / "runs") or path in {ROOT / "runs/sera-completion-live", ROOT / "runs/sera-grounded-live"}:
        raise ValueError("Use a separate continuing quest store inside runs")
    path.mkdir(parents=True, exist_ok=True)
    with lock(path):
        store = Store(path)
        previous = store.read()
        parent = Store(ROOT / "runs/sera-completion-live").read() if previous is None else None
        if previous is None and parent is None:
            parent = read(ROOT / "runs/QP-study-001/parent.json")
        session = QuestSession(previous, parent=parent, authority=path / "assessor", selected=previous is None)
        def persist(value):
            nonlocal previous
            previous = store.commit(value, previous)
        session.persist = persist
        if args.id not in session.board.aliases and args.action != "status":
            session.board.open(args.id, "Find independently checked routes to final velocity from time, impulse or work.")
        if args.action == "learn":
            while session.board.quests[session.board.aliases[args.id]]["remaining_steps"]:
                result = session.practice(args.id, controller=args.controller)
                previous = store.commit(session.snapshot(), previous)
                if result.get("status") == "STOP":
                    break
            result = session.assess(args.id)
        elif args.action == "assess":
            result = session.assess(args.id)
        elif args.action == "solve":
            result = session.solve(args.id, read(args.input))
        else:
            result = {"owner": model_identity(session.owner), "quests": session.board.snapshot()["view"],
                      "online_selector_updates": session.steps, "lower_investigator_updates": session.base.credit.steps}
        if args.action != "status":
            store.commit(session.snapshot(), previous)
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
