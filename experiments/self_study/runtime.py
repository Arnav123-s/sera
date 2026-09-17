"""Persistent diagnose/acquire/practice/verify/adapt/retry on the live owner lineage."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import torch

from experiments.stream_curriculum.model import apply
from experiments.stream_curriculum.runtime import RequestSession
from experiments.task_transfer.runtime import lock
from sera.session_state import model_identity
from workbench.storage import Store

from .algebra import check, evaluate, independent, vector
from .lessons import calculus, knuth
from .model import StudyR1
from .sources import ROOT, Arxiv

OUT = ROOT / "research-continuation/27_self_study"
MIGRATABLE_SOURCES = {
    "72a3de9623c2fd547246f72225e72981a70504a83283b2f8413e481b2cb01a6f":
        "F03: preserve shuffled lesson order on resume; bound rational input and cache paths; correct lineage metadata",
    "8b1526987b51e26cbce4d6df0f73ba98400462735e6a3d643df3c8cd52c353d5":
        "F04: distinguish English/new-language IDs and the extension-only scope of the rehearsal checkpoint counter"
}


def admitted_source(value):
    return value == fingerprint() or value in MIGRATABLE_SOURCES


def request_lineage(parent_steps, parent_ids, checkpoint):
    ids = set(map(str, checkpoint["seen_ids"]))
    new = sum(identifier.startswith(("es-ES:", "fr-FR:", "de-DE:")) for identifier in ids)
    english = sum(identifier.isdecimal() for identifier in ids)
    return {"parent_updates": parent_steps, "continuation_updates": checkpoint["step"],
            "checkpoint_seen_ids_scope": "rehearsal_extension" if "extension" in checkpoint else "multilingual_curriculum",
            "unique_ids_in_checkpoint": len(ids), "new_language_ids_in_checkpoint": new,
            "english_rehearsal_ids_in_checkpoint": english,
            "other_ids_in_checkpoint": len(ids) - new - english,
            "parent_unique_training_ids": parent_ids}


def fingerprint():
    folder = Path(__file__).parent
    return hashlib.sha256(b"".join((folder / n).read_bytes() for n in
                         ("algebra.py", "lessons.py", "model.py", "runtime.py", "sources.py"))).hexdigest()


class StudySession:
    def __init__(self, parent, saved=None, request_update=None):
        self.parent = copy.deepcopy(parent)
        self.base = RequestSession(parent["checkpoint"], parent)
        self.owner = self.base.owner
        self.learner = self.base.base.base.session.learner
        self.goals, self.events = {}, []
        self.research_goals = copy.deepcopy((saved or {}).get("research_goals", {}))
        self.migrations = copy.deepcopy((saved or {}).get("migrations", []))
        facts, library = self.learner.legacy.snapshot(), self.learner.library.record()
        self.request_update = copy.deepcopy(request_update if request_update is not None else
                                            (saved or {}).get("request_update"))
        if self.request_update is not None:
            path = (ROOT / self.request_update["path"]).resolve()
            if not path.is_relative_to(ROOT / "runs") or hashlib.sha256(path.read_bytes()).hexdigest() != self.request_update["sha256"]:
                raise ValueError("Changed multilingual checkpoint")
            update = torch.load(path, map_location="cpu", weights_only=True)
            if update["schema"] != "sera.multilingual.1" or not admitted_source(update["contracts"]["runtime"]):
                raise ValueError("Changed multilingual source contract")
            apply(self.owner, update["delta"])
            self.request_training = request_lineage(self.base.training_step, self.base.seen, update)
        StudyR1.attach(self.owner)
        if saved is not None:
            if saved["schema"] != "sera.self-study.1" or not admitted_source(saved["source"]) or saved["parent"] != parent:
                raise ValueError("Changed study source or parent; preserve and explicitly migrate")
            if saved["source"] != fingerprint():
                self.migrations.append({"from": saved["source"], "to": fingerprint(),
                                        "reason": MIGRATABLE_SOURCES[saved["source"]],
                                        "previous_owner": saved["owner"]})
            for event in saved["events"]:
                lesson = event["lesson"]
                verified = check(lesson["domain"], lesson["input"], lesson["output"])
                if verified != event["check"]:
                    raise ValueError("Saved lesson certificate changed")
                if event["learned"]:
                    if not verified["accepted"] and event["policy"] != "unchecked_control":
                        raise ValueError("Unverified lesson in an admitted learner")
                    self.owner.study_maps[lesson["domain"]].update(lesson["input"], lesson["output"])
            self.events, self.goals = copy.deepcopy(saved["events"]), copy.deepcopy(saved["goals"])
        self.owner.eval()
        self.reproof = self.learner.rebind(facts, library)
        if not (self.owner is self.learner.session.owner is self.learner.solver.neural.owner
                is self.learner.solver.components["typed"].owner is self.base.base.owner):
            raise AssertionError("Lost the actual parameter owner")
        if saved is not None and request_update is None and (saved["owner"] != model_identity(self.owner)
                                  or saved["weights"] != self.weights()):
            raise ValueError("Retained weights disagree with verified acquisition history")

    def weights(self):
        return {k: v.tolist() for k, v in self.owner.state_dict().items() if k.startswith("study_")}

    def register_research_goal(self, identifier, contract):
        if identifier in self.research_goals and self.research_goals[identifier] != contract:
            raise ValueError("A parent research goal's contract is immutable")
        if not isinstance(contract.get("statement"), str) or not contract.get("source"):
            raise ValueError("Pin the intended statement and its definition source")
        self.research_goals[identifier] = copy.deepcopy(contract)

    def propose(self, domain, p):
        p = list(map(str, vector(p)))
        q = self.owner.study_maps[domain].propose(p)
        proof = check(domain, p, q)
        if proof["accepted"] != independent(domain, p, q):
            raise AssertionError("Independent certificate check disagrees")
        return {"status": "VERIFIED" if proof["accepted"] else "NEEDS_PRACTICE", "domain": domain,
                "input": p, "proposal": q, "certificate": proof,
                "retained_lessons": int(self.owner.study_maps[domain].observations),
                "interpretation": "supplied rational polynomial semantics; numerical operator learned from lessons"}

    def acquire(self, lesson, policy="verified"):
        if policy not in {"verified", "read_only", "unchecked_control"}:
            raise ValueError("Unknown acquisition control")
        if any(e["lesson"]["id"] == lesson["id"] for e in self.events):
            raise ValueError("A source lesson is counted only once per learner")
        proof = check(lesson["domain"], lesson["input"], lesson["output"])
        if proof["accepted"] != independent(lesson["domain"], lesson["input"], lesson["output"]):
            raise AssertionError("Source checker disagreement")
        before = model_identity(self.owner)
        learned = policy != "read_only" and (proof["accepted"] or policy == "unchecked_control")
        if learned:
            facts, library = self.learner.legacy.snapshot(), self.learner.library.record()
            self.owner.study_maps[lesson["domain"]].update(lesson["input"], lesson["output"])
            self.reproof = self.learner.rebind(facts, library)
        event = {"lesson": copy.deepcopy(lesson), "policy": policy, "check": proof,
                 "learned": learned, "owner_before": before, "owner_after": model_identity(self.owner),
                 "source_actions": 1, "exact_checker_calls": 2, "parameter_update": int(learned)}
        self.events.append(event)
        return copy.deepcopy(event)

    def autonomous_round(self, identifier, domain, p, client, *, reads=6, seed=2701,
                         policy="verified", lessons=None, persist=None):
        """Current successor to autonomous_round; fixed measured-progress control.

        The learner's failed certificate triggers acquisition. ArXiv equations
        are interpreted by a supplied reader; independently checked examples
        change retained owner weights. This does not train a new eta policy.
        """
        if not 1 <= reads <= 12 or (identifier not in self.goals and len(self.goals) >= 32):
            raise ValueError("A fresh goal ID and bounded source budget are required")
        if identifier in self.goals:
            goal = self.goals[identifier]
            if (goal["domain"] != domain or goal["coefficients"] != list(map(str, vector(p)))
                    or goal["policy"] != policy or goal["source_budget"] != reads or goal["seed"] != seed):
                raise ValueError("An existing goal's meaning and budget are immutable")
            if goal.get("status") == "SOLVED_WITH_CERTIFICATE":
                return copy.deepcopy(goal)
        else:
            goal = {"domain": domain, "coefficients": list(map(str, vector(p))),
                "before": self.propose(domain, p), "policy": policy, "seed": seed,
                "original_goal_preserved": True, "source_budget": reads, "source_ids": [],
                "status": "RUNNING"}
        self.goals[identifier] = goal
        if persist:
            persist(self.snapshot())
        if not goal["before"]["certificate"]["accepted"]:
            if lessons is None:
                if domain == "sum":
                    meta, raw = client.paper("math/9207222v1")
                    lessons = knuth(meta, raw)
                    goal["source_receipt"] = copy.deepcopy(meta)
                else:
                    meta = json.loads((OUT / "calculus-source.json").read_text())
                    lessons = calculus(meta)
                    goal["source_receipt"] = copy.deepcopy(meta)
            candidates = [copy.deepcopy(l) for l in lessons if l["domain"] == domain]
            random.Random(seed).shuffle(candidates)
            candidates = [l for l in candidates if not any(e["lesson"]["id"] == l["id"] for e in self.events)]
            for lesson in candidates[:reads - len(goal["source_ids"])]:
                self.acquire(lesson, policy)
                goal["source_ids"].append(lesson["id"])
                # Keep practice outcomes and return to precisely the same goal.
                goal.setdefault("practice", []).append(self.propose(domain, lesson["input"]))
                if persist:
                    persist(self.snapshot())
        goal["after"] = self.propose(domain, p)
        goal["status"] = "SOLVED_WITH_CERTIFICATE" if goal["after"]["certificate"]["accepted"] else "RETAINED_OPEN"
        goal["remaining_reads"] = reads - len(goal["source_ids"])
        if persist:
            persist(self.snapshot())
        return copy.deepcopy(goal)

    def motion(self, acceleration, time, position=0, velocity=0):
        """Polynomial acceleration under supplied one-dimensional kinematic semantics."""
        first = self.propose("integral", acceleration)
        if not first["certificate"]["accepted"]:
            return {"status": "NEEDS_PRACTICE", "gap": "integrate acceleration", "attempt": first}
        v = vector(first["proposal"])
        v[0] = vector([velocity])[0]
        second = self.propose("integral", list(map(str, v)))
        if not second["certificate"]["accepted"]:
            return {"status": "NEEDS_PRACTICE", "gap": "integrate velocity", "attempt": second}
        x = vector(second["proposal"])
        x[0] = vector([position])[0]
        return {"status": "VERIFIED", "position": str(evaluate(list(map(str, x)), time)),
                "velocity": str(evaluate(list(map(str, v)), time)), "time": str(time),
                "position_polynomial": list(map(str, x)), "velocity_polynomial": list(map(str, v)),
                "assumptions": ["one dimension", "supplied polynomial acceleration valid for this interval",
                                "consistent units", "initial conditions at t=0"],
                "certificates": [first["certificate"], second["certificate"]]}

    def snapshot(self):
        return {"schema": "sera.self-study.1", "source": fingerprint(), "parent": copy.deepcopy(self.parent),
                "owner": model_identity(self.owner), "weights": self.weights(),
                "events": copy.deepcopy(self.events), "goals": copy.deepcopy(self.goals),
                "request_update": copy.deepcopy(self.request_update),
                "research_goals": copy.deepcopy(self.research_goals),
                "migrations": copy.deepcopy(self.migrations)}

    def interpret(self, text):
        answer = self.base.interpret(text)
        answer["current_owner"] = model_identity(self.owner)
        if self.request_update is not None:
            answer["checkpoint"] = copy.deepcopy(self.request_update)
            answer.pop("training_step", None)
            answer.pop("training_examples_seen", None)
            answer["training_lineage"] = copy.deepcopy(self.request_training)
            answer["qualification"] = "multilingual request interpretation; qualification recorded in selected experiment"
        return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("study", "solve", "motion", "status", "interpret"))
    parser.add_argument("--store", type=Path, default=ROOT / "runs/sera-study-live")
    parser.add_argument("--id", default="new-goal")
    parser.add_argument("--domain", choices=("sum", "integral"), default="sum")
    parser.add_argument("--coefficients", default="0,2,0,3,0,1")
    parser.add_argument("--time", default="3")
    parser.add_argument("--position", default="0")
    parser.add_argument("--velocity", default="0")
    parser.add_argument("--text")
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = args.store.resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Local stores must remain under runs")
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory):
        store = Store(directory)
        previous = store.read()
        parent = json.loads((OUT / "parent-request.json").read_text())
        runtime = StudySession(parent, previous)
        p = args.coefficients.split(",")
        if args.action == "study":
            latest = [previous]
            def persist(value):
                latest[0] = store.commit(value, latest[0])
            answer = runtime.autonomous_round(args.id, args.domain, p, Arxiv(ROOT / "runs/SS-sources"), persist=persist)
        elif args.action == "solve":
            answer = runtime.propose(args.domain, p)
        elif args.action == "motion":
            answer = runtime.motion(p, args.time, args.position, args.velocity)
        elif args.action == "interpret":
            answer = runtime.interpret(args.text)
        else:
            answer = {"owner": model_identity(runtime.owner), "goals": runtime.goals,
                      "verified_lessons": sum(e["check"]["accepted"] for e in runtime.events),
                      "learned_lessons": sum(e["learned"] for e in runtime.events)}
        print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
