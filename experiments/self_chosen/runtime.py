"""A persistent conjecture portfolio and learned investigator on the actual owner."""

import copy
import math

import numpy as np
import torch
from torch import nn

from experiments.continuing_growth.model import GrowingR1
from experiments.continuing_growth.runtime import GrowthSession
from experiments.verified_completion.credit import decode
from sera.session_state import model_identity

from . import equations, language, teaching
from .common import DOMAINS, RUN, contracts, digest, read


class InvestigatingR1(GrowingR1):
    @classmethod
    def attach(cls, owner, seed):
        if type(owner) is not GrowingR1:
            raise ValueError("Continue the actual admitted growth owner")
        for parameter in owner.parameters():
            parameter.requires_grad_(False)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            owner.discovery_action = nn.Linear(3, 3, dtype=torch.float64)
            nn.init.zeros_(owner.discovery_action.weight)
            nn.init.zeros_(owner.discovery_action.bias)
            owner.self_question_policy = nn.Sequential(nn.Linear(len(DOMAINS)+7, 24, dtype=torch.float64),
                                                      nn.Tanh(), nn.Linear(24, 1, dtype=torch.float64))
            nn.init.zeros_(owner.self_question_policy[-1].weight)
            nn.init.zeros_(owner.self_question_policy[-1].bias)
        owner.__class__ = cls
        owner.self_discoveries = nn.ParameterDict()
        owner.self_discovery_config = {"contracts": contracts(), "seed": seed}
        return owner

    def export_config(self):
        return {**super().export_config(), "self_chosen_discovery": copy.deepcopy(self.self_discovery_config),
                "self_discovery_keys": sorted(self.self_discoveries)}


def route_key(question, proposal):
    if question["domain"] in DOMAINS[:4]:
        return digest({"domain": question["domain"], "target": proposal["target"],
                       "required": proposal["requires"], "nonzero": proposal["nonzero"]})
    return digest([question["domain"], proposal["target"], proposal["pattern"]])


def reward_for(question, proposal, receipt, records):
    if not receipt["accepted"]:
        return {"total": -.02, "new_connection": 0., "new_route": 0., "new_coverage": 0., "cheaper_execution": 0., "retain": False}
    relation = digest([question["domain"], proposal["canonical"]])
    route = route_key(question, proposal)
    related = [r for r in records if r["relation"] == relation]
    routes = [r for r in records if r["route"] == route]
    dominated = any(r["question"]["domain"] == question["domain"] and r["proposal"]["target"] == proposal["target"]
                    and "coefficients" in proposal and set(r["proposal"]["requires"]) <= set(proposal["requires"])
                    and set(r["proposal"].get("nonzero", [])) <= set(proposal.get("nonzero", []))
                    and r["proposal"]["execution_cost"] <= proposal["execution_cost"] for r in records)
    # These retained forward routes already exist before this experiment.
    # Expanded forms using their original inputs do not earn discovery credit.
    if question["domain"] in DOMAINS[:4]:
        domain, target = question["domain"], proposal["target"]
        if domain == "polynomials":
            native = {"t", "c0", "c1", "c2"} if target in {"p", "i", "s"} else None
        else:
            acceleration = {f"a{k}" for k in range(int(domain[-1])+1)}
            native = ({"t", "v0"} | acceleration if target == "v" else
                      {"t", "v0", "x0"} | acceleration if target == "x" else
                      {"m"} | acceleration | ({"t"} if domain != "motion_0" else set()) if target == "f" else None)
        dominated = dominated or (native is not None and native <= set(proposal["requires"]))
    new_connection = 1. if not related and not dominated else 0.
    new_route = .4 if not routes and not dominated else 0.
    coverage = 0.
    if receipt["kind"] == "HUMAN_ANNOTATED_ASSOCIATION":
        old_covered = {identifier for r in records if r["question"]["domain"] == question["domain"]
                       and r["proposal"]["target"] == proposal["target"]
                       for identifier in r["receipt"].get("verified_ids", [])}
        fresh = set(receipt["verified_ids"])-old_covered
        coverage = min(.5, len(fresh)/100)
        # Equivalent empirical coverage is not a new connection merely because
        # a different token happens to match the same examples.
        if not fresh:
            new_connection = new_route = 0.
    cost = proposal["execution_cost"]
    saving = min(.3, max(0., min(r["proposal"]["execution_cost"] for r in routes)/cost-1)) if routes else 0.
    reward = new_connection+new_route+coverage+saving
    return {"total": reward, "new_connection": new_connection, "new_route": new_route,
            "new_coverage": coverage, "cheaper_execution": saving, "retain": reward > 0,
            "relation": relation, "route": route}


class Session:
    def __init__(self, parent, seed=4101, saved=None):
        self.parent = parent
        self.base = GrowthSession(parent["parent"], saved=decode(parent["state"]))
        self.owner = InvestigatingR1.attach(self.base.owner, seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.optimizer = torch.optim.Adam(self.owner.self_question_policy.parameters(), lr=.005)
        self.public = read(RUN / "language-public.json")
        self.inventory = [q for domain in DOMAINS[:4] for q in equations.questions(domain)]+language.questions(self.public)
        self.events, self.records, self.visits = [], [], {}
        self.recent = {d: 0. for d in DOMAINS}
        self.teaching = None
        self.goal = "Choose unknown relationships, imagine distinct explanations, check them independently, and preserve useful scoped discoveries."
        if saved is not None:
            self.restore(saved)
        self.base.base.base.base.base.refresh()

    def teach(self):
        if self.teaching is not None:
            raise ValueError("Do not restart completed teaching")
        self.teaching = teaching.teach(self.owner)
        if self.teaching["transfer_correct"] != 12:
            raise ValueError("Preserve unsuccessful process teaching before research")
        return self.teaching

    def features(self, questions):
        rows = []
        for q in questions:
            visits = self.visits.get(q["id"], 0)
            routes = sum(r["question"]["domain"] == q["domain"] and r["proposal"]["target"] == q["target"] for r in self.records)
            domain_visits = sum(e["question"]["domain"] == q["domain"] for e in self.events)
            rows.append([float(q["domain"] == d) for d in DOMAINS] +
                        [1/(1+visits), 1/(1+routes), self.recent[q["domain"]],
                         math.log1p(domain_visits)/6, float(q["missing"] is not None),
                         float(q["domain"] in DOMAINS[:4]), 1.])
        return torch.tensor(rows, dtype=torch.float64)

    def choose(self, arm, phase="main"):
        # Future question identities are separated before any investigation.
        questions = [q for q in self.inventory if (int(q["id"][:8], 16) % 5 == 0) == (phase == "future")]
        lowest = min(self.visits.get(q["id"], 0) for q in questions)
        questions = [q for q in questions if self.visits.get(q["id"], 0) == lowest]
        x = self.features(questions)
        with torch.no_grad():
            if arm == "balanced" or self.rng.random() < .2:
                visits = {d: sum(e["question"]["domain"] == d for e in self.events) for d in DOMAINS}
                index = min(range(len(questions)), key=lambda i: (visits[questions[i]["domain"]], questions[i]["id"]))
            elif arm == "learned":
                scores = self.owner.self_question_policy(x).squeeze(-1).numpy()
                # At equal learned utility, randomized ties preserve broad exploration.
                best = np.flatnonzero(scores >= scores.max()-1e-12)
                index = int(self.rng.choice(best))
            else:
                raise ValueError("Unregistered investigator")
        return questions[index], x[index]

    def step(self, arm, commit, *, phase="main", learn=True):
        if self.teaching is None:
            raise ValueError("Teach and transfer-test the process first")
        q, x = self.choose(arm, phase)
        sequence = len(self.events)
        seed = int(q["id"][:8], 16)+self.visits.get(q["id"], 0)*100003
        if q["domain"] in DOMAINS[:4]:
            imagined = equations.imagine(self.owner, q["domain"], seed)
            proposals = [equations.propose(q, imagined, method) for method in ("sparse", "dense")]
            observations = [{k: str(v) for k, v in row.items()} for row in imagined]
        else:
            rejected = {tuple(p["pattern"]) for event in self.events if event["question"]["id"] == q["id"]
                        for p, outcome in zip(event["proposals"], event["receipts"], strict=True)
                        if "pattern" in p and not outcome["receipt"]["accepted"]}
            proposals = [language.propose(q, self.public[q["domain"]], method, rejected) for method in ("single", "pair")]
            observations = {"source": "language-public.json", "domain": q["domain"], "kind": "owner predictions on retained human text"}
        pending = {"sequence": sequence, "question": q, "goal": self.goal, "parent_owner": self.parent["owner"],
                   "seed": seed, "proposals": proposals, "imagined": observations,
                   "action": teaching.action(self.owner, "unverified"), "phase": phase,
                   "investigator": digest({k: v.detach().tolist() for k, v in self.owner.self_question_policy.state_dict().items()})}
        pending["commitment"] = digest(pending)
        commit(pending)  # Durable before assessment labels or symbolic proof semantics are called.
        receipts, total = [], 0.
        for proposal in proposals:
            receipt = equations.certify(q["domain"], proposal) if q["domain"] in DOMAINS[:4] else language.assess(q, proposal)
            credit = reward_for(q, proposal, receipt, self.records)
            if credit["retain"]:
                identifier = digest([pending["commitment"], proposal, receipt])
                record = {"id": identifier, "question": q, "proposal": proposal, "receipt": receipt,
                          "relation": credit["relation"], "route": credit["route"], "commitment": pending["commitment"],
                          "origin": "learned fit to own retained executions; separately assessed", "sequence": sequence}
                weights = list(map(lambda c: float(equations.Q(c)), proposal["coefficients"])) if "coefficients" in proposal else [receipt["accuracy"], receipt["count"]]
                self.owner.self_discoveries[identifier] = nn.Parameter(torch.tensor(weights, dtype=torch.float64), requires_grad=False)
                self.records.append(record)
            next_status = "counterexample" if not receipt["accepted"] else "duplicate" if not credit["retain"] else "unverified"
            receipts.append({"receipt": receipt, "credit": credit, "next_action": teaching.action(self.owner, next_status)})
            total += credit["total"]
        # The reward is based on independent receipts, not the imagined fit.
        before = {k: v.detach().clone() for k, v in self.owner.self_question_policy.state_dict().items()}
        if arm == "learned" and learn:
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.owner.self_question_policy(x).squeeze()
            loss = (prediction-total)**2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.owner.self_question_policy.parameters(), 2.)
            self.optimizer.step()
        event = {**pending, "receipts": receipts, "reward": total, "features": x.tolist(), "arm": arm,
                 "learn": learn, "policy_updated": any(not torch.equal(before[k], v) for k, v in self.owner.self_question_policy.state_dict().items())}
        event["id"] = digest(event)
        self.events.append(event)
        self.visits[q["id"]] = self.visits.get(q["id"], 0)+1
        self.recent[q["domain"]] = .8*self.recent[q["domain"]]+.2*total
        return event

    def state(self):
        return {"contracts": contracts(), "seed": self.seed, "goal": self.goal, "parent_owner": self.parent["owner"],
                "policy": copy.deepcopy(self.owner.self_question_policy.state_dict()),
                "action": copy.deepcopy(self.owner.discovery_action.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()), "rng": copy.deepcopy(self.rng.bit_generator.state),
                "events": copy.deepcopy(self.events), "records": copy.deepcopy(self.records), "visits": self.visits.copy(),
                "recent": self.recent.copy(), "teaching": copy.deepcopy(self.teaching)}

    def restore(self, saved):
        if saved["contracts"] != contracts() or saved["parent_owner"] != self.parent["owner"]:
            raise ValueError("Changed discovery source or parent")
        self.seed, self.goal = saved["seed"], saved["goal"]
        self.owner.self_discovery_config["seed"] = self.seed
        self.owner.self_question_policy.load_state_dict(saved["policy"])
        self.owner.discovery_action.load_state_dict(saved["action"])
        self.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        self.rng.bit_generator.state = copy.deepcopy(saved["rng"])
        for key in ("events", "records", "visits", "recent", "teaching"):
            setattr(self, key, copy.deepcopy(saved[key]))
        self.owner.self_discoveries = nn.ParameterDict()
        for r in self.records:
            weights = [float(equations.Q(c)) for c in r["proposal"]["coefficients"]] if "coefficients" in r["proposal"] else [r["receipt"]["accuracy"], r["receipt"]["count"]]
            self.owner.self_discoveries[r["id"]] = nn.Parameter(torch.tensor(weights, dtype=torch.float64), requires_grad=False)

    def identity(self):
        return model_identity(self.owner)

    def solve(self, domain, target, observations):
        """Execute learned owned coefficients; preserve all distinct applicable routes."""
        row = {k: equations.Q(v) for k, v in observations.items()}
        if domain not in DOMAINS[:4] or target not in equations.layout(domain):
            raise ValueError("Use a retained typed domain and target")
        if "t" in row and row["t"] <= 0 or "m" in row and row["m"] <= 0:
            raise ValueError("Time and mass must satisfy the declared positive scope")
        answers = []
        for record in self.records:
            p = record["proposal"]
            if record["question"]["domain"] != domain or p["target"] != target or not set(p["requires"]) <= row.keys():
                continue
            if any(not row[k] for k in p["nonzero"]):
                continue
            weights = self.owner.self_discoveries[record["id"]]
            coefficients = [equations.Q(float(v)).limit_denominator(720) for v in weights]
            if list(map(str, coefficients)) != p["coefficients"]:
                raise ValueError("Learned route weights changed")
            result = sum(c*equations.value(t, row) for c, t in zip(coefficients, p["terms"], strict=True))
            answers.append({"id": record["id"], "result": str(result), "requires": p["requires"],
                            "nonzero": p["nonzero"], "expression_cost": p["execution_cost"]})
        if len({a["result"] for a in answers}) > 1:
            return {"status": "INCONSISTENT_PREMISES", "routes": answers, "fact_added": False}
        return {"status": "CONDITIONAL_LEARNED_ROUTES" if answers else "RETAINED_OPEN", "domain": domain,
                "target": target, "routes": sorted(answers, key=lambda a: (a["expression_cost"], a["id"])),
                "fact_added": False, "source": "independently certified polynomial-model identity"}
