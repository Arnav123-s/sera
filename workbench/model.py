"""Real learner operations, checked finite answers and continuing paid interaction."""

import copy
import hashlib
import math
import re
from datetime import datetime, timezone

import numpy as np

from experiments.continuing_control.control import decide
from experiments.continuing_control.core import ContinuingSession, sensor
from experiments.continuing_control.environment import World
from experiments.continuing_control.integration import OUTPUT, restore
from experiments.continuing_control.study import parent, read
from experiments.guarded_consolidation.core import Library, Work, observe_solutions
from sera.event_ir import Event
from sera.generative import SharedGenerativeSession
from sera.session_state import model_identity
from sera.shared import replace_shared_owner
from sera.world_graph import ExecutionBudget, LiveSituation, SharedOwnerRef

from .owner import MIGRATABLE_SOURCES, LiveR1, base_snapshot, source_identity
from .storage import Store
from .streams import analyze, parse_csv
from .task_learning import learn


def finite_number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number from {low} to {high}")
    return float(value)


def integer(value, name, low, high):
    finite_number(value, name, low, high)
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


class Learner:
    def __init__(self, record=None, *, migrate=False):
        if record is None:
            self.solver, self.session, self.library = restore()
            self.legacy = SharedGenerativeSession.restore(self.solver, read(OUTPUT/"factual-situation.json"))
            source = read(OUTPUT/"four-new-interactions.json")
            self.world = World(source["spec"])
            state = source["world_at_end"]
            self.world.angle, self.world.velocity, self.world.time = state["angle"], state["velocity"], state["time"]
            self.goal_degrees = 45.
            self.records, self.answers, self.transactions = [], [], []
            self.streams, self.stream_costs = {}, {"new_observations": 0, "replayed_observations": 0, "fits": 0}
            self.task_attempts = []
            factual, library = self.legacy.snapshot(), self.library.record()
            LiveR1.attach(self.session.owner)
            self.rebind(factual, library)
        else:
            if record.get("schema") != "sera.workbench.session.1":
                raise ValueError("Unsupported workbench session")
            self.solver = parent()
            self.session = ContinuingSession.restore(self.solver.components["r1"], record["interaction"])
            migrating = migrate and record["live_source"] in MIGRATABLE_SOURCES
            if record["live_source"] != source_identity() and not migrating:
                raise ValueError("Live task interpreter changed; preserve this session and migrate explicitly")
            LiveR1.attach(self.session.owner, record["contexts"])
            if migrating:
                self.session.owner.live_source = record["live_source"]
            if model_identity(self.session.owner) != record["owner_sha256"]:
                raise ValueError("Live owner identity differs from the saved state")
            replace_shared_owner(self.solver, self.session.owner)
            self.legacy = SharedGenerativeSession.restore(self.solver, record["legacy"])
            self.library = Library.restore(record["library"], SharedOwnerRef.from_solver(self.solver), Work(), proof_backend="indexed")
            self.world = World(record["world"]["spec"])
            state = record["world"]["state"]
            self.world.angle, self.world.velocity, self.world.time = state["angle"], state["velocity"], state["time"]
            if self.world.time != len(self.session.events)-1:
                raise ValueError("World and factual session clocks disagree")
            self.goal_degrees = record["goal_degrees"]
            self.records, self.answers, self.transactions = copy.deepcopy((record["records"], record["answers"], record["transactions"]))
            self.streams, self.stream_costs = copy.deepcopy((record["streams"], record["stream_costs"]))
            self.task_attempts = copy.deepcopy(record.get("task_attempts", []))
            if migrating:
                factual, library = self.legacy.snapshot(), self.library.record()
                self.session.owner.live_source = source_identity()
                self.rebind(factual, library)

    def rebind(self, factual_record, library_record):
        """Charge factual replay and fully reprove finite skills after owner changes."""
        reference = SharedOwnerRef.from_solver(self.solver)
        payload = factual_record["payload"]
        situation = LiveSituation(payload["situation_id"], self.session.owner.generator_graph(), reference,
                                  budget=ExecutionBudget.restore(factual_record["budget"]), max_events=payload["max_events"])
        for event in payload["observations"]:
            situation._admit(Event.from_record(event), label=False, replay=True)
        for event in payload["labels"]:
            situation._admit(Event.from_record(event), label=True, replay=True)
        self.legacy = SharedGenerativeSession.restore(self.solver, situation.snapshot())
        rebound = {**library_record, "owner_sha256": reference.identity}
        work = Work()
        self.library = Library.restore(rebound, reference, work, proof_backend="indexed")
        return dict(work.counts)

    def goal(self):
        center, radius = self.session.owner.geometry()
        angle = math.radians(self.goal_degrees)
        return center+radius*np.array([math.cos(angle), math.sin(angle)])

    def advance(self, count, degrees, *, manual=None):
        integer(count, "Steps", 1, 24)
        self.goal_degrees = finite_number(degrees, "Goal angle", -360, 360)
        if manual is not None:
            integer(manual, "Action", -1, 1)
        factual, library = self.legacy.snapshot(), self.library.record()
        for _ in range(count):
            plan = decide(self.session, np.repeat(self.goal()[None], 4, axis=0), "mpc")
            self.session.validate_plan(plan)
            action = plan["action"] if manual is None else manual
            before = self.world.time
            self.world.step(action)
            observed = self.world.measure()
            self.session.admit(sensor(observed, len(self.session.events), f"workbench:{self.world.spec['seed']}:{self.world.time}"), action)
            predicted = np.asarray(plan["particles"])
            means = (predicted*np.asarray(plan["weights"])[None, :, None]).sum(1)
            self.records.append({"time": self.world.time, "before_time": before, "action": action,
                                 "observation": observed.tolist(), "goal": self.goal().tolist(),
                                 "goal_error": float(np.linalg.norm(observed-self.goal())),
                                 "prediction": means.tolist(), "predicted_action": plan["action"],
                                 "prediction_applies_to_action": manual is None or manual == plan["action"],
                                 "coefficients": self.session.owner.interaction_mean.tolist(),
                                 "changes_detected": int(self.session.owner.interaction_changes)})
        proof_work = self.rebind(factual, library)
        return {"steps": count, "last": self.records[-1], "reproof_work": proof_work}

    def solve(self, a, b, c):
        values = {name: integer(value, name, -1_000_000_000, 1_000_000_000) % 11
                  for name, value in (("a", a), ("b", b), ("c", c))}
        definition = self.library.store.current.definition("affine")
        work = Work()
        answer = self.library.answer(definition.identity, values, work)
        independent = list(observe_solutions(definition.body["left"], values, work))
        if answer is not None and independent != [answer]:
            raise ValueError("Compiled answer failed independent substitution")
        status = "verified_unique" if answer is not None else ("all_values" if len(independent) == 11 else "no_solution")
        result = {"inputs": values, "original_inputs": {"a": a, "b": b, "c": c}, "answer": answer,
                  "solutions": independent, "status": status, "modulus": 11,
                  "equation": f"{a} × x − {b} ≡ {c} (mod 11)", "work": dict(work.counts),
                  "method": "preserved guarded program + independent forward substitution",
                  "proof_scope": "All residues in the supplied field F11; arithmetic rules are supplied."}
        self.answers.append(result)
        return result

    def learn_csv(self, name, content, horizon=12):
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,47}", name) is None:
            raise ValueError("Use a task name of 1–48 letters, digits, dots, dashes or underscores")
        if name not in self.streams and len(self.streams) >= 8:
            raise ValueError("This bounded workspace supports eight live data tasks")
        rows = parse_csv(content)
        previous = self.streams.get(name)
        if previous is not None and rows == previous["observations"] and horizon == previous["horizon"]:
            return {"task": name, "unchanged": True, "report": previous["report"]}
        report = analyze(rows, horizon)
        prior = [] if previous is None else previous["observations"]
        common = 0
        for first, second in zip(prior, rows):
            if first != second:
                break
            common += 1
        self.stream_costs["new_observations"] += len(rows)-common
        self.stream_costs["replayed_observations"] += common
        self.stream_costs["fits"] += report["fits"]
        context = hashlib.sha256(name.encode()).hexdigest()[:16]
        factual, library = self.legacy.snapshot(), self.library.record()
        self.session.owner.set_context(context, [*report["coefficients"], len(rows), rows[-1]["value"]])
        owned_forecast = self.session.owner.forecast_context(context, [row["value"] for row in rows], horizon)
        if not np.allclose([r["value"] for r in owned_forecast], [r["value"] for r in report["forecast"]], rtol=1e-12, atol=1e-12):
            raise ValueError("Actual owner forecast disagrees with the fitted procedure")
        report["forecast"] = owned_forecast
        report["execution"] = "registered coefficients executed by the actual shared R1 owner"
        self.rebind(factual, library)
        self.streams[name] = {"observations": rows, "report": report, "context": context,
                              "horizon": horizon, "updated_at": datetime.now(timezone.utc).isoformat(),
                              "update_kind": "correction" if common < len(prior) else "append",
                              "version": 1 if previous is None else previous["version"]+1}
        return {"task": name, "report": report, "update_kind": self.streams[name]["update_kind"]}

    def view(self):
        center, radius = self.session.owner.geometry()
        return {"model": "SERA", "session_observations": len(self.session.events),
                "actions": self.session.work["executed_actions"], "clock": self.world.time,
                "owner": self.session.snapshot()["owner_sha256"], "solver": self.solver.identity(),
                "center": center.tolist(), "radius": radius, "position": self.session.owner.interaction_position.tolist(),
                "goal_degrees": self.goal_degrees, "goal": self.goal().tolist(),
                "coefficients": self.session.owner.interaction_mean.tolist(),
                "changes_detected": int(self.session.owner.interaction_changes),
                "work": dict(self.session.work), "records": self.records[-120:], "answers": self.answers[-20:],
                "uncertainty": "Conditional forecasts; uncertainty is not calibrated.",
                "controller": "Qualified four-step adaptive controller",
                "task_attempts": self.task_attempts[-10:],
                "streams": {name: {"report": stream["report"], "observations": stream["observations"][-160:],
                                    "updated_at": stream["updated_at"], "version": stream["version"], "update_kind": stream["update_kind"]}
                            for name, stream in self.streams.items()}, "stream_costs": self.stream_costs,
                "saved": True, "transaction_count": len(self.transactions)}

    def snapshot(self):
        return {"schema": "sera.workbench.session.1", "interaction": base_snapshot(self.session),
                "live_source": source_identity(), "contexts": self.session.owner.contexts(),
                "owner_sha256": model_identity(self.session.owner), "streams": self.streams, "stream_costs": self.stream_costs,
                "task_attempts": self.task_attempts,
                "legacy": self.legacy.snapshot(), "library": self.library.record(),
                "world": {"spec": copy.deepcopy(self.world.spec), "state": self.world.assessment()},
                "goal_degrees": self.goal_degrees, "records": self.records, "answers": self.answers,
                "transactions": self.transactions, "view": self.view()}


def transact(directory, request):
    store = Store(directory)
    old = store.read()
    identity = request.get("request_id")
    if not isinstance(identity, str) or not 1 <= len(identity) <= 100:
        raise ValueError("A bounded request identity is required")
    if old is not None and identity in old["transactions"]:
        return {"state": old["view"], "result": {"reused_transaction": True}}
    operation = request.get("operation")
    learner = Learner(old, migrate=operation == "migrate")
    if operation == "initialize":
        result = {"restored": old is not None}
    elif operation == "migrate":
        if old is None or old["live_source"] not in MIGRATABLE_SOURCES:
            raise ValueError("No matching preserved interpreter migration")
        result = {"migration": "same tensors and acquired contexts; explicit additional task execution routes",
                  "old_source": old["live_source"], "new_source": source_identity(),
                  "old_owner": old["owner_sha256"], "new_owner": model_identity(learner.session.owner)}
    elif operation == "advance":
        result = learner.advance(request.get("steps", 1), request.get("degrees", learner.goal_degrees), manual=request.get("manual"))
    elif operation == "reverse":
        learner.world.spec["gain"] *= -1
        result = {"intervention": "Actuator response reversed; the learner receives no parameter label."}
    elif operation == "solve":
        result = learner.solve(request.get("a"), request.get("b"), request.get("c"))
    elif operation == "learn_csv":
        result = learner.learn_csv(request.get("name"), request.get("csv"), request.get("horizon", 12))
    elif operation == "learn_task":
        result = learn(request.get("task"), request.get("queries"), request.get("examples", ""), request.get("tolerance", .01))
        if result["status"] == "ACCEPTED":
            context = hashlib.sha256(("task:"+request["task"]).encode()).hexdigest()[:16]
            factual, library = learner.legacy.snapshot(), learner.library.record()
            learner.session.owner.set_context(context, [*result["coefficients"], result["examples"], result["queries"][-1]])
            result["outputs"] = learner.session.owner.transform_context(context, result["queries"])
            result["context"] = context
            learner.rebind(factual, library)
        learner.task_attempts.append(result)
    else:
        raise ValueError("Unknown workbench operation")
    learner.transactions.append(identity)
    committed = store.commit(learner.snapshot(), old)
    return {"state": committed["view"], "result": result, "revision": committed["revision"]}
