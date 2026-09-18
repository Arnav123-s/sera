"""Replay-derived quests, capability receipts and global discovery high-water marks."""

import copy

from .common import digest
from .methods import SCOPES, SPECS, canonical

SOURCE = "supplied-mechanics-contract-QP-001"
ROUTES = [["time", "constant_acceleration"], ["impulse", "constant_mass"],
          ["work", "constant_mass", "final_velocity_sign"]]


def goal_contract(question, scopes):
    if not isinstance(question, str) or not 1 <= len(question) <= 512:
        raise ValueError("A bounded original question is required")
    if not scopes or set(scopes)-set(SCOPES):
        raise ValueError("Only declared conditional mechanics scopes can be assessed")
    # Question wording and display aliases do not buy new statistical attempts.
    return {"obligation": "conditional final velocity", "scopes": sorted(set(scopes)),
            "source": SOURCE, "prerequisite_routes": copy.deepcopy(ROUTES)}


class Board:
    def __init__(self, events=None):
        self.events, self.quests, self.aliases = [], {}, {}
        self.methods, self.coverage, self.credits = set(), set(), set()
        self.display_points = 0
        for event in events or []:
            self.apply(event)

    def apply(self, event):
        if event.get("previous") != (digest(self.events[-1]) if self.events else None):
            raise ValueError("Quest event chain changed")
        kind, row = event["kind"], event["data"]
        if kind == "open":
            contract = goal_contract(row["question"], row["scopes"])
            key = digest(contract)
            if row["key"] != key or (row["id"] in self.aliases and self.aliases[row["id"]] != key):
                raise ValueError("Original quest cannot be substituted")
            if key not in self.quests:
                if len(self.quests) >= 64:
                    raise ValueError("Export before extending the finite quest board")
                self.quests[key] = {"key": key, "original_goal": row["question"], "contract": contract,
                                    "state": "OPEN", "methods": [], "attempts": [], "assessments": [],
                                    "remaining_steps": 5, "next_action": "practice a missing route"}
            self.aliases[row["id"]] = key
        elif kind == "practice":
            quest = self.quests[row["quest"]]
            receipt, method = row["receipt"], row["method"]
            decision = row["decision"]
            if (digest(decision) != row["decision_id"] or receipt["decision"] != row["decision_id"]
                    or receipt["owner"] != decision["owner"] or decision["quest"] != row["quest"]
                    or decision["action"] != method or decision["original_goal"] != quest["original_goal"]):
                raise ValueError("Practice credit lost its original decision or owner")
            expected = self.reward(row["quest"], method, receipt)
            if row["reward"] != expected or quest["remaining_steps"] <= 0:
                raise ValueError("Invalid high-water credit or exhausted quest budget")
            self.credits.add(receipt["id"])
            quest["remaining_steps"] -= 1
            quest["attempts"].append(copy.deepcopy(row))
            if receipt["valid_scopes"]:
                identity = canonical(SPECS[method])
                self.methods.add(identity)
                self.coverage.update(receipt["valid_scopes"])
                if method not in quest["methods"]:
                    quest["methods"].append(method)
            quest["state"] = "PRACTISING"
            quest["next_action"] = "independent assessment" if quest["remaining_steps"] == 0 else "practice a missing route"
        elif kind == "assessment":
            quest, receipt = self.quests[row["quest"]], row["receipt"]
            if receipt["mode"] != "boss" or receipt["quest"] != row["quest"]:
                raise ValueError("A matching independent capability assessment is required")
            if any(r["id"] == receipt["id"] for q in self.quests.values() for r in q["assessments"]):
                raise ValueError("Assessment receipt already used")
            if receipt["candidate"] != digest(sorted(set(quest["methods"]))):
                raise ValueError("Qualification belongs to another portfolio")
            quest["assessments"].append(copy.deepcopy(receipt))
            quest["state"] = "QUALIFIED" if receipt["qualified"] else "NEEDS_WORK"
            quest["next_action"] = "solve within assessed scope" if receipt["qualified"] else "retain failures; acquire missing route"
        elif kind == "stop":
            quest = self.quests[row["quest"]]
            if not 1 < quest["remaining_steps"] <= 5:
                raise ValueError("STOP cannot consume the reserved exploration attempt")
            quest["remaining_steps"] -= 1
            quest.setdefault("stops", []).append(copy.deepcopy(row))
        elif kind == "owner":
            for quest in self.quests.values():
                if quest["state"] == "QUALIFIED" and quest["assessments"][-1]["owner"] != row["identity"]:
                    quest["state"] = "STALE"
                    quest["next_action"] = "fresh independent revalidation"
        elif kind == "points":
            if type(row["amount"]) is not int:
                raise ValueError("Display points must be integer cosmetics")
            self.display_points += row["amount"]
        else:
            raise ValueError("Unknown quest event")
        self.events.append(copy.deepcopy(event))

    def append(self, kind, data):
        self.apply({"kind": kind, "data": data, "previous": digest(self.events[-1]) if self.events else None})

    def open(self, identifier, question, scopes=SCOPES):
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 64:
            raise ValueError("Bounded quest display identifier required")
        contract = goal_contract(question, scopes)
        key = digest(contract)
        self.append("open", {"id": identifier, "question": question, "scopes": list(scopes), "key": key})
        return key

    def reward(self, key, method, receipt):
        if type(method) is not int or not 0 <= method < len(SPECS):
            raise ValueError("Closed candidate method required")
        if (receipt["mode"] != "practice" or receipt["quest"] != key or receipt["id"] in self.credits
                or receipt["candidate"] != canonical(SPECS[method])):
            raise ValueError("Duplicate or mismatched practice receipt")
        if set(receipt["valid_scopes"])-set(SCOPES):
            raise ValueError("Practice cannot expand the admitted scope")
        gain = len(set(receipt["valid_scopes"])-self.coverage)/len(SCOPES)
        bonus = .03 if receipt["valid_scopes"] and canonical(SPECS[method]) not in self.methods else 0.
        return gain+bonus-.01*SPECS[method]["cost"]

    def eligible(self, identifier, owner, scope):
        quest = self.quests[self.aliases[identifier]]
        return (quest["state"] == "QUALIFIED" and scope in quest["contract"]["scopes"]
                and quest["assessments"][-1]["owner"] == owner)

    def snapshot(self):
        return {"events": copy.deepcopy(self.events), "view": copy.deepcopy(self.quests),
                "display_points": self.display_points}

    @classmethod
    def restore(cls, saved):
        result = cls(saved["events"])
        if result.snapshot() != saved:
            raise ValueError("Derived quest state disagrees with its evidence history")
        return result
