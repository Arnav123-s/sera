"""Owned conditional investigations, scoped dependencies and verified persistence."""
import copy
import hashlib
import itertools
from pathlib import Path

import numpy as np

from experiments.continuing_control.core import SPEED_LIMIT
from sera.storage import digest

from .compatibility import GRAPH_SOURCES
from .data import surface
from .model import parser_identity

WEIGHTS = np.array([.25]+[.125]*6)
MAX_JOBS, MAX_VERSIONS, MAX_HISTORY = 256, 128, 512


def source():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def language_identity(owner):
    """The span decoder depends on the recurrent path as well as its new head."""
    values = {n: v for n, v in owner.state_dict().items()
              if n.startswith(("memory.", "fusion.", "adapter."))}
    signature = tuple((n, t.data_ptr(), t._version) for n, t in values.items())
    previous = getattr(owner, "_inquiry_core_identity", None)
    if previous is None or previous[0] != signature:
        h = hashlib.sha256(digest(owner.settings).encode())
        for n, v in sorted(values.items()):
            h.update(n.encode())
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        previous = (signature, h.hexdigest())
        owner._inquiry_core_identity = previous
    return digest([parser_identity(owner), previous[1]])


def situation(owner):
    center, radius = owner.geometry()
    return {"center": center.tolist(), "radius": radius,
            "mean": owner.interaction_mean.tolist(), "covariance": owner.interaction_covariance.tolist(),
            "angle": float(owner.interaction_angle), "velocity": float(owner.interaction_velocity),
            "revision": int(owner.interaction_revision), "source": owner.interaction_source,
            "units": {"position": "m", "velocity": "rad/tick"}}


def start(model):
    return np.tile([model["angle"], model["velocity"]], (7, 1))


def advance(model, state, action):
    if type(action) is not int or action not in (-1, 0, 1):
        raise ValueError("Unrecognized action")
    mean, cov = np.array(model["mean"]), np.array(model["covariance"])
    chol = np.linalg.cholesky(cov+np.eye(3)*1e-15)
    parameters = np.vstack([mean, *(mean+2*chol[:, j] for j in range(3)),
                           *(mean-2*chol[:, j] for j in range(3))])
    old = np.asarray(state)
    if old.shape != (7, 2) or not np.isfinite(old).all():
        raise ValueError("Nonfinite conditional branch")
    velocity = np.clip(parameters[:, 0]*old[:, 1]+parameters[:, 1]*action+parameters[:, 2],
                       -SPEED_LIMIT, SPEED_LIMIT)
    return np.stack([old[:, 0]+velocity, velocity], 1)


def answer(model, states, frame):
    states = np.asarray(states)
    if frame["field"] == "position":
        values = np.array(model["center"])+model["radius"]*np.stack([np.cos(states[:, 0]), np.sin(states[:, 0])], 1)
    elif frame["field"] == "velocity":
        values = states[:, 1:2]
    else:
        raise ValueError("Unknown requested field")
    mean = WEIGHTS@values
    return {"kind": "conditional_prediction", "status": "CONDITIONAL_UNCALIBRATED",
            "field": frame["field"], "value": mean.tolist(),
            "parameter_variance": (WEIGHTS@(values-mean)**2).tolist(),
            "units": model["units"][frame["field"]], "actions": list(frame["actions"]),
            "uncertainty_scope": "Parameter quadrature only; process noise, parser mistakes and omitted mechanisms are not calibrated",
            "applicability": "No GG-GUARD-002 threshold or physical validity guarantee inherited"}


def interpret(owner, text, entity="orbit"):
    proposal = surface(text, entity, admitted=owner.inquiry_admitted)
    spans = [proposal["field_span"]]+[c for i, c in enumerate(proposal["clauses"]) if str(i) not in proposal["ambiguities"]]
    probabilities = owner.spans(spans)
    field_p = probabilities[0][3:5]
    action_p, index = [], 1
    for i, _ in enumerate(proposal["clauses"]):
        options = proposal["ambiguities"].get(str(i))
        if options:
            action_p.append([(a, 1/len(options)) for a in options])
        else:
            action_p.append([(a-1, float(p)) for a, p in enumerate(probabilities[index][:3])])
            index += 1
    alternatives = []
    for f, choices in itertools.product(range(2), itertools.product(*action_p)):
        frame = {"entity": entity, "field": ("position", "velocity")[f], "actions": [a for a, _ in choices]}
        alternatives.append({"frame": frame, "proposal_score": float(field_p[f]*np.prod([p for _, p in choices]))})
    alternatives.sort(key=lambda x: -x["proposal_score"])
    return {"status": "clarification" if proposal["ambiguities"] else "queued", "support": proposal,
            "frame": alternatives[0]["frame"], "alternatives": alternatives[:4],
            "parser": language_identity(owner), "confidence_scope": "Uncalibrated interpretation proposal scores",
            "assumptions": ["supplied surface grammar and entity binding", "control signs act on angular acceleration",
                            "velocity denotes angular displacement per simulator tick"]}


class Investigation:
    def __init__(self, owner, *, reuse=True):
        if type(reuse) is not bool:
            raise ValueError("Reuse must be explicit")
        model = situation(owner)
        self.models, self.current = {digest(model): model}, digest(model)
        self.parser = language_identity(owner)
        self.jobs, self.cache, self.history = {}, {}, []
        self.clock, self.reuse = 0, reuse
        self.work = {"ticks": 0, "particle_transitions": 0, "cache_hits": 0,
                     "candidate_inspections": 0, "restore_transitions": 0, "interpretations": 0}

    def sync(self, owner):
        model, parser = situation(owner), language_identity(owner)
        key = digest(model)
        if key not in self.models:
            if len(self.models) >= MAX_VERSIONS:
                raise ValueError("Model archive full; preserve this store before starting a successor")
            self.models[key] = model
        self.current, self.parser = key, parser
        for j in self.jobs.values():
            if j["anchor"] == "current" and j["status"] not in ("rejected", "stale"):
                if j["model"] != key or j["interpretation"].get("parser") != parser:
                    j["status"], j["reason"] = "stale", "Relevant situation or interpretation changed; explicit rebase required"

    def add(self, owner, identifier, text, *, entity="orbit", anchor="current"):
        if (not isinstance(identifier, str) or not identifier or len(identifier) > 80
                or identifier in self.jobs or len(self.jobs) >= MAX_JOBS or anchor not in ("current", "historical")):
            raise ValueError("Duplicate inquiry, invalid identity or bounded store full")
        self.sync(owner)
        try:
            parsed = interpret(owner, text, entity)
        except ValueError as error:
            parsed = {"status": "rejected", "reason": str(error), "parser": self.parser, "alternatives": []}
        self.work["interpretations"] += 1
        self.jobs[identifier] = {"id": identifier, "text": text, "entity": entity, "anchor": anchor,
            "interpretation": parsed, "model": self.current, "status": parsed["status"],
            "cursor": 0, "states": start(self.models[self.current]).tolist(), "result": None,
            "last_served": -1, "reason": parsed.get("reason"), "clarification": None}
        return copy.deepcopy(self.jobs[identifier])

    def clarify(self, identifier, actions):
        j = self.jobs[identifier]
        if j["status"] != "clarification" or len(actions) != len(j["interpretation"]["frame"]["actions"]):
            raise ValueError("Only an unresolved interpretation may be clarified")
        for i, a in enumerate(actions):
            allowed = j["interpretation"]["support"]["ambiguities"].get(str(i), [j["interpretation"]["frame"]["actions"][i]])
            if type(a) is not int or a not in allowed:
                raise ValueError("Clarification contradicts the unambiguous request")
        j["interpretation"]["frame"]["actions"] = list(actions)
        j["clarification"] = {"origin": "user_supplied", "actions": list(actions)}
        j["status"] = "queued"

    def run(self, owner, ticks):
        if type(ticks) is not int or not 0 <= ticks <= 768:
            raise ValueError("A finite 0..768 tick budget is required")
        self.sync(owner)
        for _ in range(ticks):
            pending = [j for j in self.jobs.values() if j["status"] in ("queued", "running")]
            self.work["candidate_inspections"] += len(pending)
            if not pending:
                break
            j = min(pending, key=lambda j: (j["last_served"], j["id"]))
            frame = j["interpretation"]["frame"]
            actions = frame["actions"][:j["cursor"]+1]
            key = digest([j["model"], actions])
            if self.reuse and key in self.cache:
                states = np.array(self.cache[key]["states"])
                self.work["cache_hits"] += 1
            else:
                states = advance(self.models[j["model"]], j["states"], actions[-1])
                self.work["particle_transitions"] += 7
                if self.reuse:
                    self.cache[key] = {"model": j["model"], "actions": actions, "states": states.tolist()}
            j["states"], j["cursor"], j["last_served"] = states.tolist(), len(actions), self.clock
            j["status"] = "complete" if len(actions) == len(frame["actions"]) else "running"
            if j["status"] == "complete":
                j["result"] = answer(self.models[j["model"]], states, frame)
            self.clock += 1
            self.work["ticks"] += 1

    def pause(self, identifier):
        j = self.jobs[identifier]
        if j["status"] not in ("queued", "running"):
            raise ValueError("Only active work can pause")
        j["status"] = "paused"

    def resume(self, owner, identifier):
        self.sync(owner)
        j = self.jobs[identifier]
        if j["status"] != "paused":
            raise ValueError("Only a current paused inquiry can resume")
        j["status"] = "running" if j["cursor"] else "queued"

    def rebase(self, owner, identifier):
        self.sync(owner)
        old = self.jobs[identifier]
        if old["status"] != "stale" or len(self.history) >= MAX_HISTORY:
            raise ValueError("Only stale work can rebase, within the history budget")
        self.history.append(copy.deepcopy(old))
        del self.jobs[identifier]
        return self.add(owner, identifier, old["text"], entity=old["entity"], anchor=old["anchor"])

    def snapshot(self):
        value = {"source": source(), "models": self.models, "current": self.current, "parser": self.parser,
                 "jobs": self.jobs, "cache": self.cache, "history": self.history, "clock": self.clock,
                 "reuse": self.reuse, "work": self.work}
        return copy.deepcopy({"payload": value, "sha256": digest(value)})

    @classmethod
    def restore(cls, owner, record, *, migrate=False):
        if set(record) != {"payload", "sha256"} or digest(record["payload"]) != record["sha256"]:
            raise ValueError("Inquiry snapshot checksum mismatch")
        p = copy.deepcopy(record["payload"])
        obj = cls(owner, reuse=p["reuse"])
        permitted_source = p["source"] == source() or (migrate and p["source"] in GRAPH_SOURCES)
        if not permitted_source or p["current"] != obj.current or p["parser"] != obj.parser:
            raise ValueError("Saved inquiry interpreter or current owner dependencies differ")
        if len(p["jobs"]) > MAX_JOBS or len(p["models"]) > MAX_VERSIONS or len(p["history"]) > MAX_HISTORY:
            raise ValueError("Archive exceeds its bounded contract")
        if len(p["cache"]) > (MAX_JOBS+MAX_HISTORY)*3 or any(k != j["id"] for k, j in p["jobs"].items()):
            raise ValueError("Invalid cache budget or job keys")
        if set(p["work"]) != set(obj.work) or any(type(v) is not int or v < 0 for v in p["work"].values()) or p["work"]["ticks"] != p["clock"]:
            raise ValueError("Invalid work accounting")
        for key, model in p["models"].items():
            if key != digest(model):
                raise ValueError("Model key differs")
        replayed = 0
        for j in [*p["jobs"].values(), *p["history"]]:
            model = p["models"][j["model"]]
            states = start(model)
            if j["status"] not in ("queued", "running", "paused", "clarification", "complete", "rejected", "stale"):
                raise ValueError("Unknown persisted state")
            frame = j["interpretation"].get("frame")
            if frame is not None and j["interpretation"]["parser"] == obj.parser:
                expected = interpret(owner, j["text"], j["entity"])
                if j["clarification"] is not None:
                    actions = j["clarification"]["actions"]
                    if len(actions) != len(expected["frame"]["actions"]):
                        raise ValueError("Invalid saved clarification")
                    for i, action in enumerate(actions):
                        allowed = expected["support"]["ambiguities"].get(str(i), [expected["frame"]["actions"][i]])
                        if type(action) is not int or action not in allowed:
                            raise ValueError("Saved clarification contradicts interpretation")
                    expected["frame"]["actions"] = list(actions)
                if expected != j["interpretation"]:
                    raise ValueError("Saved interpretation disagrees with its current parser and text")
            actions = [] if frame is None else frame["actions"]
            if type(j["cursor"]) is not int or not 0 <= j["cursor"] <= len(actions):
                raise ValueError("Invalid persisted progress")
            for a in actions[:j["cursor"]]:
                states = advance(model, states, a)
                replayed += 7
            if not np.array_equal(states, np.asarray(j["states"])):
                raise ValueError("Saved intermediate states disagree with execution")
            if j["status"] == "complete" and (j["cursor"] != len(actions) or j["result"] is None):
                raise ValueError("False completion")
            if j["result"] is not None and (j["cursor"] != len(actions) or j["result"] != answer(model, states, frame)):
                raise ValueError("Saved result disagrees with derived computation")
        for key, cache in p["cache"].items():
            if key != digest([cache["model"], cache["actions"]]):
                raise ValueError("Wrong cache key")
            states = start(p["models"][cache["model"]])
            for a in cache["actions"]:
                states = advance(p["models"][cache["model"]], states, a)
                replayed += 7
            if not np.array_equal(states, np.asarray(cache["states"])):
                raise ValueError("Wrong cached state")
        for key in ("models", "current", "parser", "jobs", "cache", "history", "clock", "reuse", "work"):
            setattr(obj, key, p[key])
        obj.work["restore_transitions"] += replayed
        return obj
