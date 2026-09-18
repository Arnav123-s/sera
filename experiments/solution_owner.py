"""Actual continuing owner: checked solutions, unique credit, and goal closure."""

import copy
from fractions import Fraction as Q

import torch
from torch import nn

from experiments.gap_inquiry import ROOT, GapR1, digest, sha
from experiments.self_chosen import equations as eq
from experiments.solution_check import validate
from experiments.solution_search import evaluate
from experiments.verified_completion.credit import decode, encode
from sera.session_state import model_identity

DOMAINS = ("motion_0", "motion_1", "motion_2", "polynomials")
METHODS = ("sparse", "dense", "implicit_simple", "implicit_reverse", "implicit_permuted")


def contracts():
    return {name: sha(ROOT / name) for name in ("experiments/solution_owner.py", "experiments/solution_search.py",
                                               "experiments/solution_check.py", "research-continuation/44_solution_portfolios/PROTOCOL.md")}


def features(question, method, degree):
    units = eq.layout(question["domain"])
    return ([float(question["domain"] == d) for d in DOMAINS]
            + [v / 4 for v in units[question["target"]]]
            + [degree / 4, (len(units) - 1 - len(question["missing"] or [])) / 10]
            + [float(method == m) for m in METHODS])


class SolutionR1(GapR1):
    @classmethod
    def attach(cls, owner):
        if type(owner) is not GapR1:
            raise ValueError("Continue the actual latest gap owner")
        owner.__class__ = cls
        owner.solution_models = nn.ParameterDict()
        owner.solution_policy = nn.Linear(14, 1, dtype=torch.float64)
        nn.init.zeros_(owner.solution_policy.weight)
        nn.init.zeros_(owner.solution_policy.bias)
        owner.solution_contract = contracts()
        return owner

    def export_config(self):
        return {**super().export_config(), "solutions": self.solution_contract,
                "solution_keys": sorted(self.solution_models)}


class SolutionSession:
    def __init__(self, base, saved=None):
        self.base, self.parent_owner = base, base.identity()
        self.owner = SolutionR1.attach(base.owner)
        self.records, self.credits, self.questions = {}, {}, {}
        self.pending = None
        self.optimizer = torch.optim.Adam(self.owner.solution_policy.parameters(), lr=.02)
        if saved is not None:
            if saved["parent_owner"] != self.parent_owner or saved["contracts"] != contracts():
                raise ValueError("Changed solution lineage")
            self.records, self.credits, self.questions = copy.deepcopy((saved["records"], saved["credits"], saved["questions"]))
            self.pending = copy.deepcopy(saved["pending"])
            for key, record in self.records.items():
                self._weights(key, record["candidate"])
            self.owner.solution_policy.load_state_dict(decode(saved["policy"]))
            self.optimizer.load_state_dict(decode(saved["optimizer"]))
            if self.identity() != saved["owner"]:
                raise ValueError("Solution owner failed exact restoration")

    def identity(self):
        return model_identity(self.owner)

    def _weights(self, key, candidate):
        values = candidate["numerator"]["weights"] + candidate["denominator"]["weights"]
        tensor = torch.tensor([float(Q(c)) for c in values], dtype=torch.float64)
        if [Q(float(v)).limit_denominator(1000000) for v in tensor] != list(map(Q, values)):
            raise ValueError("Proposed coefficients exceed the retained exact decoder")
        self.owner.solution_models[key] = nn.Parameter(tensor, requires_grad=False)

    def retain(self, record, final):
        c, receipt = record["candidate"], record["receipt"]
        predictor = record.get("predictor", self.parent_owner)
        if predictor not in {self.parent_owner, self.identity()}:
            raise ValueError("Stale solution predictor")
        validate(receipt, record["question"], c, record["commitment"], predictor)
        if not final["accepted"] or final["candidate"] != c["id"]:
            raise ValueError("Independent final replay must qualify this answer")
        key = c["id"]
        if key in self.records or receipt["id"] in self.credits:
            raise ValueError("Repeated or stale solution credit")
        self._weights(key, c)
        self.records[key] = copy.deepcopy(record | {"final": final})
        self.credits[receipt["id"]] = {"points": record["points"], "candidate": key,
                                       "original_goal": record["question"]["id"], "proof": receipt["id"]}

    def solve(self, domain, target, observations):
        if domain not in DOMAINS or target not in eq.layout(domain):
            raise ValueError("Use a retained typed domain and target")
        if target in observations:
            raise ValueError("A missing target cannot be supplied as its own answer")
        row = {k: Q(str(v)) for k, v in observations.items()}
        if any(v <= 0 for n, v in row.items() if n == "t" or n == "m" and domain != "polynomials"):
            raise ValueError("Retained domain requires positive time and mass")
        if domain == "polynomials" and "t" in row and row["t"].denominator != 1:
            raise ValueError("Finite-sum routes require integer time")
        routes = []
        earlier = self.base.base.base.solve(domain, target, observations)
        for item in earlier["routes"]:
            routes.append({**item, "origin": "retained earlier route"})
        for key, record in self.records.items():
            c = record["candidate"]
            if c["domain"] != domain or c["target"] != target or not set(c["requires"]) <= row.keys():
                continue
            weights = [Q(float(v)).limit_denominator(1000000) for v in self.owner.solution_models[key]]
            if list(map(str, weights)) != c["numerator"]["weights"] + c["denominator"]["weights"]:
                raise ValueError("A solution's owned weights changed after qualification")
            value = evaluate(c, row, weights)
            if value is not None:
                routes.append({"id": key, "result": str(value), "requires": c["requires"], "guard": c["guard"],
                               "expression": c["expression"], "proof": record["receipt"], "origin": "new owned rational route"})
        distinct = sorted({r["result"] for r in routes})
        return {"status": "CHECKED_SOLUTION_PORTFOLIO" if len(distinct) == 1 else "INCONSISTENT_PREMISES" if routes else "INVESTIGATION_REQUIRED",
                "domain": domain, "target": target, "answers": distinct, "routes": routes,
                "physical_fact_added": False, "original_observations": observations}

    def state(self):
        return {"schema": "sera.solution-portfolios.1", "parent_owner": self.parent_owner, "contracts": contracts(),
                "records": copy.deepcopy(self.records), "credits": copy.deepcopy(self.credits), "questions": copy.deepcopy(self.questions),
                "pending": copy.deepcopy(self.pending), "policy": encode(self.owner.solution_policy.state_dict()),
                "optimizer": encode(self.optimizer.state_dict()), "owner": self.identity()}

    def autonomous_round(self, question, observations, persist):
        """A curiosity trigger runs through checked answers, not just a note."""
        from experiments.solution_check import certify, final_check
        from experiments.solution_search import investigate

        if self.pending and self.pending["question"] != question:
            raise ValueError("Resume the current question before switching goals")
        if question["id"] in self.questions and self.questions[question["id"]].get("runtime_completed"):
            return self.solve(question["domain"], question["target"], observations)
        predictor = self.identity()
        if self.pending is None:
            # Observations stay in original-goal state; no target answers are
            # read by the imagined proposal generator.
            event = investigate(self.owner, question)
            event["predictor"] = predictor
            event["original_observations"] = observations
            event["commitment"] = digest(event)
            self.pending = event
            persist(self.state())
        event = self.pending
        if event["predictor"] != predictor or event["original_observations"] != observations:
            raise ValueError("Resume the exact committed predictor and original task")
        receipts, additions = [], []
        existing = {(r["candidate"]["domain"], r["candidate"]["target"], tuple(r["candidate"]["requires"])) for r in self.records.values()}
        for proposal in event["proposals"]:
            c = proposal["candidate"]
            proof = certify(question, c, event["commitment"], predictor)
            checked = final_check(c, 449001 + int(digest([event["commitment"], c["id"]])[:8], 16)) if proof["accepted"] else None
            receipts.append({"proof": proof, "check": checked})
            if checked and checked["accepted"] and c["id"] not in self.records:
                key = (c["domain"], c["target"], tuple(c["requires"]))
                points = float(key not in existing)
                existing.add(key)
                additions.append({"candidate": c, "question": question, "receipt": proof, "commitment": event["commitment"],
                                  "predictor": predictor, "questions": [question["id"]], "derivations": proposal["derivations"],
                                  "points": points, "runtime_final": checked})
        # Validate all new credit against the same committed owner before any
        # owned weights change. Each subsequent append still binds that owner.
        for record in additions:
            validate(record["receipt"], question, record["candidate"], event["commitment"], predictor)
        for record in additions:
            c, proof = record["candidate"], record["receipt"]
            if proof["id"] in self.credits:
                raise ValueError("Repeated runtime proof credit")
            self._weights(c["id"], c)
            self.records[c["id"]] = copy.deepcopy(record | {"final": record["runtime_final"]})
            self.credits[proof["id"]] = {"points": record["points"], "candidate": c["id"],
                                           "original_goal": question["id"], "proof": proof["id"]}
        for attempt in event["attempts"]:
            ids = {c["id"] for c in attempt["proposals"]}
            reward = sum(r["points"] for r in additions if r["candidate"]["id"] in ids)
            x = torch.tensor(features(question, attempt["method"], attempt["degree"]), dtype=torch.float64)
            self.optimizer.zero_grad(set_to_none=True)
            (self.owner.solution_policy(x).squeeze() - reward).square().backward()
            self.optimizer.step()
        answer = self.solve(question["domain"], question["target"], observations)
        self.questions[question["id"]] = {"question": question, "runtime_completed": True, "commitment": event["commitment"],
                  "proposals": event, "grades": receipts, "answer": answer, "solutions": [r["candidate"]["id"] for r in additions],
                  "status": answer["status"], "reward": sum(r["points"] for r in additions),
                  "next_action": "Use the checked solutions" if answer["routes"] else "Obtain independently relevant missing evidence after exhausted exploration"}
        self.pending = None
        persist(self.state())
        return answer | {"investigation": question["id"], "reward": self.questions[question["id"]]["reward"],
                         "search_completed_before_feedback": True}


def train_policy(session, rows):
    """Delayed verified-progress regression; held-out questions remain untouched."""
    train = [r for r in rows if int(r["question"]["id"][:8], 16) % 5]
    test = [r for r in rows if not int(r["question"]["id"][:8], 16) % 5]
    def tensors(records):
        return (torch.tensor([features(r["question"], r["method"], r["degree"]) for r in records], dtype=torch.float64),
                torch.tensor([r["reward"] for r in records], dtype=torch.float64))
    x, y = tensors(train)
    xt, yt = tensors(test)
    for _ in range(80):
        session.optimizer.zero_grad(set_to_none=True)
        loss = (session.owner.solution_policy(x).squeeze(-1) - y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(session.owner.solution_policy.parameters(), 2.)
        session.optimizer.step()
    with torch.no_grad():
        learned = float((session.owner.solution_policy(xt).squeeze(-1) - yt).square().mean())
        constant = float((yt - y.mean()).square().mean())
    return {"train_rows": len(train), "held_out_rows": len(test), "learned_mse": learned, "constant_mse": constant,
            "selected": learned < constant, "target": "independently verified new-route progress", "epochs": 80,
            "sealed_final_used": False}
