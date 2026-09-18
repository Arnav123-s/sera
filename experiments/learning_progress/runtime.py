"""Resumable verified practice and procedure credit through the latest owner."""

import argparse
import copy
import json

import numpy as np
import torch
from torch.nn import functional as F

from experiments.quest_portfolio.runtime import QuestSession
from experiments.task_transfer.runtime import lock
from experiments.verified_completion.credit import decode, encode
from sera.session_state import model_identity
from workbench.storage import Store

from .common import PREFIXES, RATES, ROOT, RUN, SKILLS, contracts, digest, read, sha
from .data import load_data, subset
from .model import ProgressR1, apply_knowledge, candidates, knowledge


def logits(owner, skill, rows):
    if skill == "reading":
        return owner.reading_logits(rows["x"], rows["mask"])
    if skill == "methods":
        return owner.quest_policy(rows["x"])
    return owner.book_logits(rows["x"], "shared")


def loss(owner, skill, rows):
    value = logits(owner, skill, rows)
    if skill == "reading":
        return -value.log_softmax(-1).masked_fill(~rows["gold"], -torch.inf).logsumexp(-1).mean()
    return F.cross_entropy(value, rows["y"])


@torch.no_grad()
def measure(owner, data, predictions=False):
    scores, accuracy, records = [], [], {}
    for skill in SKILLS:
        rows = data[skill]
        value = logits(owner, skill, rows)
        prediction = value.argmax(-1)
        good = rows["gold"].gather(1, prediction[:, None]).squeeze(1) if skill == "reading" else prediction.eq(rows["y"])
        ce = float(loss(owner, skill, rows))
        scores.append(1/(1+ce))
        accuracy.append(float(good.double().mean()))
        records[skill] = {"loss": ce, "score": scores[-1], "accuracy": accuracy[-1], "count": len(prediction)}
        if predictions:
            records[skill].update(logits=value.tolist(), labels=rows["y"].tolist(),
                                 gold=rows.get("gold", torch.empty(0)).tolist())
    return {"scores": scores, "accuracy": accuracy, "macro": float(np.mean(scores)), "skills": records}


def credit(before, after, highwater):
    old, new = np.asarray(before["scores"]), np.asarray(after["scores"])
    peak = np.asarray(highwater)
    if old.shape != (6,) or new.shape != (6,) or peak.shape != (6,) or not np.isfinite([old, new, peak]).all():
        raise ValueError("Six finite independently measured scores required")
    signed = float(np.mean(new-old))
    bonus = float(.5*np.mean(np.maximum(new-peak, 0)))
    correct = float(.25*(np.mean(after["accuracy"])-np.mean(before["accuracy"])))
    return {"reward": signed+bonus+correct, "maintenance": signed, "discovery": bonus,
            "accuracy_credit": correct, "highwater": np.maximum(peak, new).tolist()}


class LearningSession:
    def __init__(self, parent=None, saved=None, seed=3401):
        if saved and saved["contracts"] != contracts():
            raise ValueError("Changed learning contract; explicit migration required")
        self.parent_record = copy.deepcopy(saved["parent"] if saved else parent)
        self.base = QuestSession(self.parent_record)
        self.owner = self.base.owner
        learner = self.base.base.grounded.base.learner
        self.facts, self.library = learner.legacy.snapshot(), learner.library.record()
        self.seed = saved["seed"] if saved else seed
        ProgressR1.attach(self.owner, self.seed)
        self.optimizer = torch.optim.Adam(self.owner.learning_policy.parameters(), lr=.003)
        self.rng = torch.Generator().manual_seed(self.seed)
        self.events = []
        self.highwater = None
        self.recent = [0.]*6
        self.phase, self.remaining, self.cursors = None, [8]*6, [0]*6
        self.goal = "Improve verified acquisition across human reading and checked mechanics while preserving valid approaches."
        if saved:
            apply_knowledge(self.owner, decode(saved["knowledge"]))
            self.owner.learning_policy.load_state_dict(decode(saved["eta"]))
            self.optimizer.load_state_dict(decode(saved["optimizer"]))
            self.rng.set_state(decode(saved["rng"]))
            for name in ("events", "highwater", "recent", "phase", "remaining", "cursors", "goal"):
                setattr(self, name, copy.deepcopy(saved[name]))
            if len({e["id"] for e in self.events}) != len(self.events):
                raise ValueError("Repeated progress-credit identity")
            peak = None
            for event in self.events:
                unsigned = {k: v for k, v in event.items() if k != "id"}
                peak = event["before"]["scores"] if peak is None else peak
                expected = credit(event["before"], event["after"], peak)
                if digest(unsigned) != event["id"] or expected != event["credit"]:
                    raise ValueError("Changed checked progress-credit history")
                peak = expected["highwater"]
            if peak != self.highwater:
                raise ValueError("Changed persistent high-water credit")
        self.refresh()
        if saved and saved["owner"] != model_identity(self.owner):
            raise ValueError("Restored learning owner disagrees with checkpoint")

    def refresh(self):
        self.base.base.grounded.base.learner.rebind(self.facts, self.library)
        self.base.base._refresh()
        self.base.board.append("owner", {"identity": model_identity(self.owner)})

    def start(self, phase, quota=8):
        if self.phase == phase:
            return
        if self.phase is not None and any(self.remaining):
            raise ValueError("Preserve the unfinished acquisition phase")
        self.phase, self.remaining, self.cursors = phase, [quota]*6, [0]*6

    def step(self, data, probe, controller="learned", train_eta=True):
        if not any(self.remaining):
            raise ValueError("Finite practice quota spent")
        before = measure(self.owner, probe)
        owner_before = model_identity(self.owner)
        if self.highwater is None:
            self.highwater = before["scores"][:]
        x = candidates(before["scores"], before["accuracy"], self.recent, self.remaining)
        available = [i for i in range(6) if self.remaining[i]]
        with torch.no_grad():
            if controller == "balanced":
                skill = min(available, key=lambda i: (self.cursors[i], i))
                action = skill*3+1
            elif controller == "random":
                skill = available[int(torch.randint(len(available), (), generator=self.rng))]
                action = skill*3+int(torch.randint(3, (), generator=self.rng))
            elif controller == "progress":
                skill = max(available, key=lambda i: (self.recent[i], -self.cursors[i], -i))
                action = skill*3+1
            elif controller == "learned":
                permitted = [3*i+j for i in available for j in range(3)]
                if float(torch.rand((), generator=self.rng)) < .2:
                    action = permitted[int(torch.randint(len(permitted), (), generator=self.rng))]
                else:
                    predicted = self.owner.learning_policy(x).squeeze(-1)
                    action = max(permitted, key=lambda i: (float(predicted[i]), -i))
                skill = action//3
            else:
                raise ValueError("Unknown curriculum controller")
        name, rate = SKILLS[skill], RATES[action % 3]
        rows = data[name]
        ids = [(self.cursors[skill]+i) % len(rows["y"]) for i in range(8)]
        self.owner.zero_grad(set_to_none=True)
        teaching_loss = loss(self.owner, name, subset(rows, ids))
        teaching_loss.backward()
        parameters = [p for n, p in self.owner.named_parameters() if n.startswith(PREFIXES) and p.grad is not None]
        torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
        with torch.no_grad():
            for parameter in parameters:
                parameter.add_(parameter.grad, alpha=-rate)
        after = measure(self.owner, probe)
        result = credit(before, after, self.highwater)
        self.highwater = result["highwater"]
        self.recent[skill] = .8*self.recent[skill]+.2*result["reward"]
        if train_eta and controller == "learned":
            self.optimizer.zero_grad(set_to_none=True)
            predicted = self.owner.learning_policy(x[action]).squeeze()
            # Fixed scale makes the small, externally scored improvement resolvable.
            fit = (predicted-100*result["reward"])**2
            fit.backward()
            torch.nn.utils.clip_grad_norm_(self.owner.learning_policy.parameters(), 2., error_if_nonfinite=True)
            self.optimizer.step()
        self.remaining[skill] -= 1
        self.cursors[skill] += 8
        event = {"phase": self.phase, "sequence": len(self.events), "goal": self.goal,
                 "owner_before": owner_before, "owner_after": model_identity(self.owner),
                 "evidence": sha(RUN / "data-manifest.json"),
                 "skill": name, "action": action, "rate": rate, "rows": ids,
                 "controller": controller, "eta_updated": train_eta and controller == "learned",
                 "before": before, "after": after, "credit": result, "features": x[action].tolist()}
        event["id"] = digest(event)
        if any(e["id"] == event["id"] for e in self.events):
            raise ValueError("Repeated decision credit")
        self.events.append(event)
        return event

    def snapshot(self):
        self.refresh()
        return {"schema": "sera.learning-progress.1", "contracts": contracts(), "parent": self.parent_record,
                "seed": self.seed, "owner": model_identity(self.owner), "knowledge": encode(knowledge(self.owner)),
                "eta": encode(self.owner.learning_policy.state_dict()), "optimizer": encode(self.optimizer.state_dict()),
                "rng": encode(self.rng.get_state()), **{k: copy.deepcopy(getattr(self, k)) for k in
                    ("events", "highwater", "recent", "phase", "remaining", "cursors", "goal")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "practice", "read"))
    parser.add_argument("--store", default="runs/sera-learning-live")
    parser.add_argument("--phase", choices=("teach1", "teach2"), default="teach2")
    parser.add_argument("--controller", choices=("balanced", "learned", "random", "progress"), default="balanced")
    parser.add_argument("--input", help="JSON containing question, sentences and attributed source")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = (ROOT / args.store).resolve()
    if not path.is_relative_to(ROOT / "runs"):
        raise ValueError("A local continuing learner store is required")
    path.mkdir(parents=True, exist_ok=True)
    with lock(path):
        store = Store(path)
        previous = store.read()
        session = LearningSession(saved=previous) if previous else LearningSession(parent=read(RUN / "parent.json"))
        if args.action == "practice":
            data = load_data()
            session.start(args.phase)
            while any(session.remaining):
                session.step(data[args.phase], data["probe"], args.controller)
                previous = store.commit(session.snapshot(), previous)
        if args.action == "read":
            request = read(args.input)
            if not request.get("source") or not isinstance(request.get("question"), str):
                raise ValueError("An attributed source and a question are required")
            examples = [{"question": request["question"], "sentences": request["sentences"]}]
            with torch.no_grad():
                x, mask = session.owner.reading_features(examples)
                selected = int(session.owner.reading_logits(x, mask).argmax(-1)[0])
            print(json.dumps({"status": "ATTRIBUTED_SENTENCE_SELECTION", "question": request["question"],
                              "sentence": request["sentences"][selected]["text"], "source": request["source"],
                              "owner": model_identity(session.owner), "factual_updates": 0}, indent=2))
            return
        print(json.dumps({"owner": model_identity(session.owner), "steps": len(session.events),
                          "remaining": session.remaining, "phase": session.phase, "goal": session.goal}, indent=2))


if __name__ == "__main__":
    main()
