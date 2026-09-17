"""Persistent empirical refinement and exact algebra through one actual StudyR1 owner."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.self_study.model import StudyR1
from experiments.task_transfer.runtime import lock
from sera.session_state import model_identity
from workbench.storage import Store

from . import physical
from .common import EXPERIMENT, OUT, ROOT, contracts, read, restore_parent, sha
from .data import DT, identity


def fingerprint():
    return identity(
        {
            "runtime": sha(Path(__file__)),
            "physical": sha(Path(physical.__file__)),
            "parent": sha(OUT / "parent-study.json"),
        }
    )


class ConceptStudyR1(StudyR1):
    """No new recurrent component: learned empirical operators on the existing owner."""

    @classmethod
    def attach(cls, owner):
        if type(owner) is not StudyR1:
            raise ValueError("A verified Stage 27 StudyR1 is required")
        owner.__class__ = cls
        owner.empirical_weights = nn.ParameterDict()
        owner.empirical_registry = {}
        owner.empirical_source = fingerprint()
        return owner

    def export_config(self):
        return {
            **super().export_config(),
            "empirical_refinement": {
                "source": self.empirical_source,
                "route": "bounded_observation_fit",
                "schema": 1,
                "models": copy.deepcopy(self.empirical_registry),
            },
        }


def admission():
    selected = read(EXPERIMENT / "selection.json")
    if selected["contracts"] != contracts() or selected["selected"] != "physical":
        raise ValueError("This runtime requires the supported physical identification route")
    from .study import gate

    result = gate(read(EXPERIMENT / "final-physical.json"), read(EXPERIMENT / "final-instant.json"))
    if not result["admitted"]:
        raise ValueError("The frozen final admission gate has not passed")
    return {
        "selection": sha(EXPERIMENT / "selection.json"),
        "final": sha(EXPERIMENT / "final-physical.json"),
        "baseline": sha(EXPERIMENT / "final-instant.json"),
        "gate": result,
    }


def validate_observations(subject, observations, previous=()):
    if not isinstance(subject, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", subject):
        raise ValueError("Use a bounded alphanumeric subject ID")
    if not isinstance(observations, list) or not 1 <= len(observations) <= 512:
        raise ValueError("Provide between one and 512 observations")
    rows = copy.deepcopy(list(previous))
    if len(rows) + len(observations) > 2048:
        raise ValueError("Subject archive limit reached; export it before a declared migration")
    for item in observations:
        if set(item) != {
            "subject",
            "time",
            "position",
            "velocity",
            "command",
            "available",
            "units",
            "source",
            "evidence",
        }:
            raise ValueError("Observation schema changed")
        if item["subject"] != subject or item["units"] != ["m", "m/s", "1"]:
            raise ValueError("Subject or unit mismatch")
        if item["evidence"] not in ("OBSERVED", "SYNTHETIC"):
            raise ValueError("Imagination cannot be admitted as an observation")
        if not isinstance(item["source"], str) or not 1 <= len(item["source"]) <= 512:
            raise ValueError("Observation source identity is required")
        mask = item["available"]
        if not isinstance(mask, list) or len(mask) != 2 or any(type(v) is not bool for v in mask):
            raise ValueError("Explicit two-sensor availability is required")
        if not rows and not all(mask):
            raise ValueError("The first observation must anchor both sensor values")
        for name, available in zip(("position", "velocity"), mask):
            value = item[name]
            if available and (type(value) not in (int, float) or not np.isfinite(value)):
                raise ValueError("Available sensors require finite numerical values")
            if not available and value is not None:
                raise ValueError("Unavailable sensor values must be null")
        if any(
            type(item[k]) not in (int, float) or not np.isfinite(item[k])
            for k in ("time", "command")
        ):
            raise ValueError("Finite observation time and command required")
        if abs(item["command"]) > 8:
            raise ValueError("Command exceeds the declared empirical range")
        if rows and abs(item["time"] - rows[-1]["time"] - DT) > 1e-8:
            raise ValueError("Observations must be ordered on the declared .05-second clock")
        rows.append(copy.deepcopy(item))
    return rows


def arrays(observations):
    values, masks = [], []
    for row in observations:
        value = [row["position"], row["velocity"]]
        if values:
            value = [
                x if present else values[-1][i]
                for i, (x, present) in enumerate(zip(value, row["available"]))
            ]
        values.append(value)
        masks.append(row["available"])
    return (
        np.asarray(values, dtype=float),
        np.asarray(masks, dtype=bool),
        [r["command"] for r in observations[:-1]],
    )


class RefinementSession:
    def __init__(self, saved=None):
        self.admission = admission()
        self.study = restore_parent()
        self.owner, self.learner = self.study.owner, self.study.learner
        facts, library = self.learner.legacy.snapshot(), self.learner.library.record()
        ConceptStudyR1.attach(self.owner)
        self.subjects, self.goals, self.events = {}, {}, []
        if saved is not None:
            if (
                saved["schema"] != "sera.concept.session.1"
                or saved["source"] != fingerprint()
                or saved["admission"] != self.admission
                or saved["parent"] != sha(OUT / "parent-study.json")
            ):
                raise ValueError("Changed source, admission or parent; explicit migration required")
            self.subjects, self.goals, self.events = (
                copy.deepcopy(saved[k]) for k in ("subjects", "goals", "events")
            )
            expected_keys = set()
            for subject, record in self.subjects.items():
                validate_observations(subject, record["observations"][:512])
                for offset in range(512, len(record["observations"]), 512):
                    validate_observations(
                        subject,
                        record["observations"][offset : offset + 512],
                        record["observations"][:offset],
                    )
                for model in record["models"]:
                    key = model["weight_key"]
                    expected_keys.add(key)
                    coef = saved["weights"][key]
                    if identity(coef) != model["parameter_hash"] or coef != model["coef"]:
                        raise ValueError("Empirical weights disagree with retained evidence record")
                    self.owner.empirical_weights[key] = nn.Parameter(
                        torch.tensor(coef, dtype=torch.float64), requires_grad=False
                    )
                    self.owner.empirical_registry[key] = self.model_contract(model)
                    end = next(
                        (
                            i
                            for i, r in enumerate(record["observations"])
                            if r["time"] == model["evidence_end"]
                        ),
                        -1,
                    )
                    if (
                        end < 80
                        or identity(record["observations"][end - 80 : end + 1])
                        != model["evidence_hash"]
                    ):
                        raise ValueError("Saved model observation support changed")
            if expected_keys != set(saved["weights"]):
                raise ValueError("Unowned empirical coefficients")
            if saved["owner"] != model_identity(self.owner):
                raise ValueError("Changed owner weights or configuration")
        self.reproof = self.learner.rebind(facts, library)
        self.assert_owner()
        self.owner.eval()

    @staticmethod
    def model_contract(model):
        return {
            k: copy.deepcopy(model[k])
            for k in ("kind", "tau", "parameter_hash", "evidence_hash", "scale", "subject")
        }

    def assert_owner(self):
        if not (
            self.owner
            is self.study.owner
            is self.study.base.owner
            is self.learner.session.owner
            is self.learner.solver.neural.owner
            is self.learner.solver.components["typed"].owner
        ):
            raise AssertionError("Empirical route lost the continuing shared owner")

    def observe(self, subject, observations):
        previous = self.subjects.get(subject, {}).get("observations", [])
        rows = validate_observations(subject, observations, previous)
        record = self.subjects.setdefault(
            subject, {"observations": [], "models": [], "active": None}
        )
        record["observations"] = rows
        self.events.append(
            {
                "operation": "observation",
                "subject": subject,
                "count": len(observations),
                "source_ids": sorted({r["source"] for r in observations}),
                "evidence_hash": identity(rows),
            }
        )
        return {"status": "OBSERVATIONS_RETAINED", "observations": len(rows)}

    def refine(self, subject):
        record = self.subjects.get(subject)
        if record is None or len(record["observations"]) < 81:
            return {
                "status": "MISSING_KNOWLEDGE",
                "needed": "81 ordered observations with varied past commands",
            }
        observations = record["observations"][-81:]
        if not all(observations[0]["available"]):
            return {
                "status": "MISSING_KNOWLEDGE",
                "needed": "a complete sensor anchor at the fitting-window start",
            }
        evidence_hash = identity(observations)
        if (
            record["active"] is not None
            and record["models"][record["active"]]["evidence_hash"] == evidence_hash
        ):
            return {"status": "REUSED", "model": copy.deepcopy(record["models"][record["active"]])}
        values, masks, controls = arrays(observations)
        try:
            model = physical.fit(values, masks, controls)
        except ValueError as error:
            self.events.append(
                {"operation": "rejected_fit", "subject": subject, "reason": str(error)}
            )
            return {"status": "MISSING_KNOWLEDGE", "needed": str(error)}
        # Independent empirical rollout implementation must agree before retention.
        challenge = [0.3, 0.8, -0.2, 1.0] * 3
        np.testing.assert_allclose(
            physical.predict(model, values, challenge),
            physical.predict(model, values, challenge, independent=True),
            atol=2e-12,
            rtol=0,
        )
        if model["selection_validation_mse"] > 0.25:
            self.events.append(
                {
                    "operation": "rejected_fit",
                    "subject": subject,
                    "candidate": model,
                    "reason": "Held-out observed residual exceeds the declared adequacy gate",
                }
            )
            return {
                "status": "MISSING_KNOWLEDGE",
                "needed": "a better model or more informative observations",
                "candidate": model,
            }
        if model["design_rank"] < len(model["coef"]):
            self.events.append(
                {"operation": "unidentifiable", "subject": subject, "candidate": model}
            )
            return {
                "status": "UNIDENTIFIABLE_CAUSE",
                "needed": "varied command probes to separate candidate coefficients",
                "candidate": model,
            }
        facts, library = self.learner.legacy.snapshot(), self.learner.library.record()
        key = hashlib.sha256(subject.encode()).hexdigest()[:16] + f"_{len(record['models']):04d}"
        before = model_identity(self.owner)
        self.owner.empirical_weights[key] = nn.Parameter(
            torch.tensor(model["coef"], dtype=torch.float64), requires_grad=False
        )
        model.update(
            weight_key=key,
            parameter_hash=identity(model["coef"]),
            subject=subject,
            variables=["position", "velocity", "command", "relaxing_force_if_supported"],
            units=["m", "m/s", "1"],
            scale={"dt": DT, "maximum_horizon_steps": 12, "gravity": 1.0},
            evidence_hash=evidence_hash,
            parent_approximation=record["active"],
            evidence_end=observations[-1]["time"],
            unresolved_obligations=[
                "physical premises",
                "omitted mechanisms",
                "uncalibrated uncertainty",
            ],
            temporal_state={
                "force_estimate": physical.initial_force(model, values),
                "history_length": len(values),
            },
            provenance=sorted({r["evidence"] for r in observations}),
        )
        record["models"].append(model)
        record["active"] = len(record["models"]) - 1
        self.owner.empirical_registry[key] = self.model_contract(model)
        self.reproof = self.learner.rebind(facts, library)
        self.events.append(
            {
                "operation": "learn",
                "subject": subject,
                "owner_before": before,
                "owner_after": model_identity(self.owner),
                "weight_key": key,
                "candidate_fits": 10,
                "observation_count": len(observations),
            }
        )
        return {"status": "EMPIRICAL_MODEL_LEARNED", "model": copy.deepcopy(model)}

    def predict(self, subject, commands, *, branch="query"):
        if (
            not isinstance(commands, list)
            or not 1 <= len(commands) <= 12
            or any(
                type(c) not in (int, float) or not np.isfinite(c) or abs(c) > 8 for c in commands
            )
        ):
            raise ValueError("One to twelve finite commands in [-8,8] are required")
        record = self.subjects.get(subject)
        if record is None or record["active"] is None:
            return {
                "status": "MISSING_KNOWLEDGE",
                "needed": "observations and an admitted empirical fit",
            }
        model = copy.deepcopy(record["models"][record["active"]])
        history = record["observations"][-81:]
        if identity(history) != model["evidence_hash"]:
            return {
                "status": "NEEDS_REFINEMENT",
                "needed": "new observations invalidate the cached model state",
            }
        if not all(history[-1]["available"]):
            return {
                "status": "MISSING_KNOWLEDGE",
                "needed": "current position and velocity observations",
            }
        model["coef"] = self.owner.empirical_weights[model["weight_key"]].detach().tolist()
        if identity(model["coef"]) != model["parameter_hash"]:
            raise ValueError("Weights changed without revalidation")
        if self.owner.empirical_registry[model["weight_key"]] != self.model_contract(model):
            raise ValueError("Mechanism changed without revalidation")
        values, _, _ = arrays(history)
        answer = physical.predict(model, values, commands)
        return {
            "status": "EMPIRICAL_PREDICTION",
            "evidence_kind": "CONDITIONAL_IMAGINATION",
            "branch": branch,
            "subject": subject,
            "position_velocity": answer.tolist(),
            "commands": commands,
            "model": model["kind"],
            "parameter_hash": model["parameter_hash"],
            "parent_owner": model_identity(self.owner),
            "uncertainty": {
                "kind": "held-out observed acceleration residual, not calibrated coverage",
                "mse": model["selection_validation_mse"],
            },
            "assumptions": [
                "same airborne mechanism and subject",
                "one-dimensional consistent units",
                "gravity=1",
                "future command executed as specified",
                "no unmodeled contact or force change",
            ],
            "source_kinds": model["provenance"],
            "occurrence_established": False,
        }

    def autonomous_refine(self, goal, subject, commands, observations=None):
        contract = {"subject": subject, "commands": commands}
        if goal in self.goals and self.goals[goal]["task"] != contract:
            raise ValueError("The original empirical goal is immutable")
        item = self.goals.setdefault(goal, {"task": copy.deepcopy(contract), "attempts": []})
        before = self.predict(subject, commands)
        if observations:
            acquisition = self.observe(subject, observations)
        else:
            acquisition = {"status": "NO_NEW_OBSERVATIONS"}
        learning = self.refine(subject)
        after = self.predict(subject, commands)
        item["attempts"].append(
            {"before": before, "acquisition": acquisition, "learning": learning, "after": after}
        )
        item["status"] = after["status"]
        return copy.deepcopy(item)

    def exact_motion(self, acceleration, time, position=0, velocity=0):
        result = self.study.motion(acceleration, time, position, velocity)
        return {
            "status": "CERTIFIED_ALGEBRA" if result["status"] == "VERIFIED" else result["status"],
            "result": result,
            "empirical_premises_verified": False,
        }

    def snapshot(self):
        return {
            "schema": "sera.concept.session.1",
            "source": fingerprint(),
            "admission": self.admission,
            "parent": sha(OUT / "parent-study.json"),
            "owner": model_identity(self.owner),
            "subjects": copy.deepcopy(self.subjects),
            "goals": copy.deepcopy(self.goals),
            "events": copy.deepcopy(self.events),
            "weights": {k: v.detach().tolist() for k, v in self.owner.empirical_weights.items()},
        }


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("task", "observe", "refine", "predict", "motion", "interpret", "status")
    )
    parser.add_argument("--store", type=Path, default=ROOT / "runs/sera-refinement-live")
    parser.add_argument("--subject", default="body-1")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--goal", default="forecast-motion")
    parser.add_argument("--commands", default="0.8,0.8,0.8,0.8")
    parser.add_argument("--coefficients", default="2,3,1")
    parser.add_argument("--time", default="3")
    parser.add_argument("--position", default="5")
    parser.add_argument("--velocity", default="-1")
    parser.add_argument("--text", default="turn on the lights")
    args = parser.parse_args()
    directory = args.store.resolve()
    if (
        not directory.is_relative_to(ROOT / "runs")
        or directory == (ROOT / "runs/sera-study-live").resolve()
    ):
        raise ValueError("Use a separate refinement store within runs")
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory):
        store = Store(directory)
        previous = store.read()
        session = RefinementSession(previous)
        commands = [float(x) for x in args.commands.split(",")]
        if args.action == "task":
            result = session.autonomous_refine(
                args.goal, args.subject, commands, read(args.input) if args.input else None
            )
        elif args.action == "observe":
            result = session.observe(args.subject, read(args.input))
        elif args.action == "refine":
            result = session.refine(args.subject)
        elif args.action == "predict":
            result = session.predict(args.subject, commands)
        elif args.action == "motion":
            result = session.exact_motion(
                args.coefficients.split(","), args.time, args.position, args.velocity
            )
        elif args.action == "interpret":
            result = session.study.interpret(args.text)
        else:
            result = {
                "owner": model_identity(session.owner),
                "subjects": list(session.subjects),
                "goals": session.goals,
            }
        if args.action in ("task", "observe", "refine"):
            store.commit(session.snapshot(), previous)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
