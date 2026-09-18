"""Attributed definitions, learned bindings and checked conditional motion on one owner."""

import argparse
import copy
import json
import re
from fractions import Fraction
from pathlib import Path

import torch

from experiments.human_reading.data import ROOT, digest, read, sha
from experiments.human_reading.runtime import ReadingSession
from experiments.task_transfer.runtime import lock
from sera.session_state import model_identity
from workbench.storage import Store

from .data import OUT
from .ground_data import GROUND, ROLES
from .ground_model import GroundR1, apply, restore_books
from .grounding_v2 import SELECTION, contracts, text

UNITS = {"position": "m", "velocity": "m/s", "acceleration": "m/s^2", "force": "N", "mass": "kg"}
PREMISES = ("physical_context", "one_dimension", "constant_quantity", "inertial_frame", "nonrelativistic", "si_units")


def number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Use a finite rational quantity")
    value = str(value)
    if len(value) > 40 or not re.fullmatch(r"-?\d+(?:/\d+|\.\d+)?", value):
        raise ValueError("Use a bounded integer, decimal or fraction")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Invalid rational quantity") from error
    if abs(result) > 1000000 or result.denominator > 1000000000:
        raise ValueError("Quantity exceeds this conditional model's numerical scope")
    return result


class GroundedSession(ReadingSession):
    def __init__(self, saved=None):
        previous = restore_books()
        self.__dict__.update(previous.__dict__)
        facts, library = self.base.learner.legacy.snapshot(), self.base.learner.library.record()
        selected = read(SELECTION)
        if selected["contracts"] != contracts() or sha(ROOT / selected["checkpoint"]) != selected["sha256"]:
            raise ValueError("Grounded reading lineage changed")
        self.owner = GroundR1.attach(self.owner)
        apply(self.owner, torch.load(ROOT / selected["checkpoint"], weights_only=True, map_location="cpu")["delta"])
        self.kind, self.gate = selected["kind"], copy.deepcopy(selected["gate"])
        self.owner.ground_config.update(selected=self.kind, checkpoint=selected["sha256"], named_input=True)
        for p in self.owner.parameters():
            p.requires_grad_(False)
        self.owner.eval()
        self.base.learner.rebind(facts, library)
        self.base.assert_owner()
        self.identity = {"owner": model_identity(self.owner), "selection": sha(SELECTION),
                         "runtime": sha(Path(__file__)), "admission": sha(OUT / "ADMISSION.md")}
        self.bank = read(GROUND / "teaching.json")["train"]
        self.tasks, self.concepts = {}, {}
        if saved:
            if saved.get("schema") != "sera.grounded-reading.1" or saved.get("identity") != self.identity:
                raise ValueError("Stored grounded owner changed")
            self.goals, self.events, self.tasks, self.concepts = (
                copy.deepcopy(saved[k]) for k in ("goals", "events", "tasks", "concepts"))
            for goal in self.goals.values():
                if goal["question_sha256"] != digest(goal["question"]):
                    raise ValueError("Stored investigation changed")
            for identifier, task in self.tasks.items():
                for attempt in task["attempts"]:
                    if attempt["request_sha256"] != digest(json.dumps(attempt["request"], sort_keys=True)):
                        raise ValueError("Stored original task changed")
                    if attempt["request"]["id"] != identifier or attempt["request"]["goal"] != task["original_goal"]:
                        raise ValueError("Stored task identity changed")
            for entry_id, concept in self.concepts.items():
                row = next((r for r in self.bank if r["id"] == entry_id), None)
                if row is None or any(concept[k] != row[v] for k, v in
                                      (("source_sha256", "source_sha256"), ("text_sha256", "text_sha256"),
                                       ("term", "term"), ("type", "label"), ("source", "source"))) or concept["learned_checkpoint"] != selected["sha256"]:
                    raise ValueError("Retained concept lost its human source")

    def snapshot(self):
        return {"schema": "sera.grounded-reading.1", "identity": copy.deepcopy(self.identity),
                **{k: copy.deepcopy(getattr(self, k)) for k in ("goals", "events", "tasks", "concepts")}}

    @torch.no_grad()
    def bind(self, term, definition, source):
        if not isinstance(term, str) or not 1 <= len(term) <= 128 or not isinstance(definition, str) or not 1 <= len(definition) <= 6000:
            raise ValueError("Provide a bounded named definition")
        if not isinstance(source, str) or not 1 <= len(source) <= 1024:
            raise ValueError("Attribute the original definition")
        row = {"term": term, "text": definition}
        x = self.owner.ground_features([text(row)])
        scores = self.owner.ground_logits(x, self.kind)[0].softmax(-1)
        index = int(scores.argmax())
        candidate = ROLES[index]
        matched = next((r for r in self.bank if (r["term"], r["text"], r["source"]) == (term, definition, source)), None)
        known = matched is not None and matched["label"] == candidate and candidate in UNITS
        novel = matched is None and self.gate["autonomous"] and float(scores[index]) >= self.gate["threshold"] and candidate in UNITS
        return {"candidate": candidate, "admitted": known or novel,
                "admission": "TAUGHT_SOURCE_REUSE" if known else "DEVELOPMENT_GATE" if novel else "TENTATIVE",
                "source_entry": matched["id"] if matched else None,
                "human_label_agreement": bool(matched and matched["label"] == candidate),
                "alternatives": [{"role": ROLES[i], "score": float(scores[i])} for i in scores.argsort(descending=True, stable=True)],
                "score_meaning": "learned classifier scores; not calibrated truth probabilities",
                "owner": self.identity["owner"], "definition_sha256": digest(definition), "source": source}

    def checked_motion(self, acceleration, time, position, velocity):
        result = self.base.exact_motion([str(acceleration)], str(time), str(position), str(velocity))
        expected = {"position": position+velocity*time+acceleration*time*time/2, "velocity": velocity+acceleration*time}
        if result["status"] != "CERTIFIED_ALGEBRA" or any(Fraction(result["result"][k]) != v for k, v in expected.items()):
            raise ValueError("Independent rational motion check failed")
        return {**result, "independent_check": {k: str(v) for k, v in expected.items()},
                "observed_event": False, "lexical_interpretation_certified": False}

    def imagine(self, request, *, acquire=False, client=None, persist=None):
        request = copy.deepcopy(request)
        identifier, goal = request.get("id"), request.get("goal")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier) or not isinstance(goal, str) or not 1 <= len(goal) <= 512:
            raise ValueError("Retain a bounded task identifier and original goal")
        if len(self.tasks) >= 64 and identifier not in self.tasks:
            raise ValueError("Export this session before adding more tasks")
        task = self.tasks.setdefault(identifier, {"original_goal": goal, "attempts": []})
        if task["original_goal"] != goal:
            raise ValueError("Do not overwrite the original task")
        if len(task["attempts"]) >= 64:
            raise ValueError("Export this task before adding more attempts")
        attempt = {"request": request, "request_sha256": digest(json.dumps(request, sort_keys=True)), "status": "OPEN"}
        task["attempts"].append(attempt)
        if persist:
            persist(self.snapshot())
        try:
            binding = self.bind(request["term"], request["definition"], request["source"])
            attempt["binding"] = binding
            if not binding["admitted"]:
                raise ValueError("This source meaning needs checked teaching evidence before physical execution")
            role = binding["candidate"]
            if request.get("unit") != UNITS[role]:
                raise ValueError("Quantity unit conflicts with the learned physical type")
            context = request.get("assumptions", {})
            if any(context.get(k) is not True for k in PREMISES):
                raise ValueError("The named physical model premises are incomplete")
            value, time = number(request["value"]), number(request["time"])
            if time < 0:
                raise ValueError("Use a nonnegative elapsed time")
            position = number(request.get("initial_position", "0"))
            velocity = number(request.get("initial_velocity", "0"))
            acceleration = number(request.get("acceleration", "0"))
            if role == "position":
                if "initial_position" in request and position != value:
                    raise ValueError("Contradictory initial positions")
                position = value
            elif role == "velocity":
                if "initial_velocity" in request and velocity != value:
                    raise ValueError("Contradictory initial velocities")
                velocity = value
            elif role == "acceleration":
                if "acceleration" in request and acceleration != value:
                    raise ValueError("Contradictory accelerations")
                acceleration = value
            else:
                if context.get("net_force") is not True:
                    raise ValueError("The force value must denote the constant net force")
                mass = value if role == "mass" else number(request["mass"])
                force = value if role == "force" else number(request["force"])
                if mass <= 0:
                    raise ValueError("Newtonian motion here requires positive mass")
                acceleration = force/mass
                if "acceleration" in request and number(request["acceleration"]) != acceleration:
                    raise ValueError("Supplied acceleration contradicts supplied net force and mass")
            branches = {"base": self.checked_motion(acceleration, time, position, velocity),
                        "opposite_drive": self.checked_motion(-acceleration, time, position, velocity),
                        "double_mass" if role in {"force", "mass"} else "half_acceleration":
                            self.checked_motion(acceleration/2, time, position, velocity)}
            attempt.update(status="CONDITIONAL_RESULT", role=role, branches=branches,
                           original_goal_preserved=True, source_interpretation="supervised taught-entry reuse" if binding["admission"] == "TAUGHT_SOURCE_REUSE" else "tentative learned interpretation",
                           supplied_physics="one-dimensional Newtonian kinematics; F/m when force or mass is supplied",
                           imagined_evidence_added_to_observations=False, learned_weight_updates=0)
            if binding["source_entry"]:
                row = next(r for r in self.bank if r["id"] == binding["source_entry"])
                self.concepts[row["id"]] = {"term": row["term"], "type": role, "source": row["source"],
                                            "source_sha256": row["source_sha256"], "text_sha256": row["text_sha256"],
                                            "learned_checkpoint": read(SELECTION)["sha256"],
                                            "evidence_status": "HUMAN_TEACHER_ASSOCIATION_AND_CONDITIONAL_ALGEBRA",
                                            "generalization_claim": False, "physical_occurrence_verified": False}
        except (KeyError, ValueError) as error:
            attempt.update(status="RETAINED_OPEN", reason=str(error), original_goal_preserved=True, learned_weight_updates=0)
            if acquire and isinstance(request.get("public_gap"), str):
                attempt["investigation"] = self.investigate(identifier, goal, request["public_gap"], client=client, persist=persist)
        if persist:
            persist(self.snapshot())
        return copy.deepcopy(attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("library", "imagine", "read", "research"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--text")
    parser.add_argument("--source")
    parser.add_argument("--gap")
    parser.add_argument("--id", default="book-research-1")
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--store", default="runs/sera-grounded-live")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = (ROOT / args.store).resolve()
    if not path.is_relative_to(ROOT / "runs"):
        raise ValueError("Session storage must remain inside runs")
    path.mkdir(parents=True, exist_ok=True)
    with lock(path):
        store = Store(path)
        previous = store.read()
        session = GroundedSession(previous)
        def persist(value):
            nonlocal previous
            previous = store.commit(value, previous)
        if args.action == "library":
            result = [{"id": r["id"], "term": r["term"], "definition": r["text"], "source": r["source"],
                       "supplied_type": r["label"]} for r in session.bank if r["label"] != "other"]
        elif args.action == "imagine":
            result = session.imagine(read(args.request), acquire=args.acquire, persist=persist)
        elif args.action == "read":
            result = session.read_passage(args.question, args.text, args.source)
            persist(session.snapshot())
        else:
            result = session.investigate(args.id, args.question, args.gap, persist=persist)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
