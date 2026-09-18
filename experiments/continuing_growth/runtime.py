"""Persistent gap selection, imagined alternatives, practice, checking and credit."""

import copy

import numpy as np
import torch

from experiments.autonomous_discovery.acquire import AcquisitionSession
from sera.session_state import model_identity

from .common import BATCH, BLOCK, PRACTICE_STEPS, RATES, RUN, SKILLS, contracts, digest, sha
from .data import take
from .model import GrowingR1, apply, candidates, credit, errors, identity, knowledge, measure


class GrowthSession:
    def __init__(self, parent, seed=3801, saved=None):
        self.parent = parent
        self.base = AcquisitionSession(saved=parent)
        self.owner = GrowingR1.attach(self.base.owner, seed)
        self.optimizer = torch.optim.Adam(self.owner.growth_policy.parameters(), lr=.003)
        self.rng = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.events = []
        self.visits = [0]*len(SKILLS)
        self.recent = [0.]*len(SKILLS)
        self.highwater = None
        self.current = None
        self.goal = "Advance across retained capabilities; verify useful progress and preserve unfinished goals."
        if saved is not None:
            self.restore(saved)

    def state(self):
        return {"contracts": contracts(), "seed": self.seed, "knowledge": knowledge(self.owner),
                "eta": copy.deepcopy(self.owner.growth_policy.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()), "rng": self.rng.get_state().clone(),
                "events": copy.deepcopy(self.events), "visits": self.visits[:], "recent": self.recent[:],
                "highwater": copy.deepcopy(self.highwater), "current": copy.deepcopy(self.current), "goal": self.goal}

    def restore(self, saved):
        if saved["contracts"] != contracts():
            raise ValueError("Changed continuing protocol requires explicit migration")
        apply(self.owner, saved["knowledge"])
        self.owner.growth_policy.load_state_dict(saved["eta"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.rng.set_state(saved["rng"])
        for name in ("seed", "events", "visits", "recent", "highwater", "current", "goal"):
            setattr(self, name, copy.deepcopy(saved[name]))
        self.owner.growth_config["seed"] = self.seed
        validate_events(self.events, self.highwater)

    def step(self, data, probe, controller="self_directed", train_eta=True):
        if len(self.events) % BLOCK == 0:
            self.visits = [0]*len(SKILLS)
        before = self.current or measure(self.owner, probe)
        if self.highwater is None:
            self.highwater = before["scores"][:]
        x = candidates(before, self.recent, self.visits)
        remaining = BLOCK-len(self.events) % BLOCK
        missing = sum(max(0, 2-v) for v in self.visits)
        forced = remaining <= missing
        with torch.no_grad():
            if controller == "balanced":
                chosen = min(range(len(SKILLS)), key=lambda i: (self.visits[i], i))
                action = chosen*6+4  # Strong fixed-largest-rate, uniform-practice control.
            elif controller == "self_directed":
                if forced or float(torch.rand((), generator=self.rng)) < .2:
                    chosen = min(range(len(SKILLS)), key=lambda i: (self.visits[i], i))
                    action = chosen*6+int(torch.randint(6, (), generator=self.rng))
                else:
                    permitted = [i for i in range(len(x)) if self.visits[i//6] < 32]
                    scores = self.owner.growth_policy(x).squeeze(-1)
                    action = max(permitted, key=lambda i: (float(scores[i]), -self.visits[i//6], -i))
                    chosen = action//6
            else:
                raise ValueError("Unregistered continuing controller")
        skill = SKILLS[chosen]
        rate, method = RATES[(action % 6)//2], action % 2
        old = knowledge(self.owner)
        prior = identity(old)
        used = []
        for _ in range(PRACTICE_STEPS):
            rows = data[skill]
            if method:
                with torch.no_grad():
                    probabilities = errors(self.owner, skill, rows).clamp(0, 12)+.1
                ids = torch.multinomial(probabilities.double(), BATCH, replacement=True, generator=self.rng)
            else:
                ids = torch.randint(len(rows["y"]), (BATCH,), generator=self.rng)
            self.owner.zero_grad(set_to_none=True)
            loss = errors(self.owner, skill, take(rows, ids)).mean()
            loss.backward()
            params = [p for n, p in self.owner.named_parameters()
                      if p.requires_grad and p.grad is not None and not n.startswith("growth_policy.")]
            torch.nn.utils.clip_grad_norm_(params, 2., error_if_nonfinite=True)
            with torch.no_grad():
                for parameter in params:
                    parameter.add_(parameter.grad, alpha=-rate)
            used.append(ids.tolist())
        trial = measure(self.owner, probe)
        delta = np.asarray(trial["scores"])-np.asarray(before["scores"])
        accepted = bool(delta.min() >= -.015 and delta.mean() >= -.0001)
        if not accepted:
            apply(self.owner, old)
        after = trial if accepted else before
        result = credit(before, after, self.highwater, chosen)
        # Rejections are preserved and charged; confidence and repetition earn nothing.
        if not accepted:
            result["reward"] -= .001
        self.highwater = result["highwater"]
        self.recent[chosen] = .8*self.recent[chosen]+.2*result["reward"]
        if train_eta and controller == "self_directed":
            self.optimizer.zero_grad(set_to_none=True)
            predicted = self.owner.growth_policy(x[action]).squeeze()
            fit = (predicted-100*result["reward"])**2
            fit.backward()
            torch.nn.utils.clip_grad_norm_(self.owner.growth_policy.parameters(), 2., error_if_nonfinite=True)
            self.optimizer.step()
        event = {"sequence": len(self.events), "parent_owner": self.parent["owner"], "goal": self.goal,
                 "evidence": sha(RUN / "data-manifest.json"), "knowledge_before": prior,
                 "knowledge_after": identity(knowledge(self.owner)), "controller": controller,
                 "skill": skill, "action": action, "rate": rate, "method": ("uniform", "error_focused")[method],
                 "rows": used, "before": before, "trial": trial, "after": after, "accepted": accepted,
                 "credit": result, "eta_updated": train_eta and controller == "self_directed",
                 "features": x[action].tolist(), "forced_coverage": forced}
        event["id"] = digest(event)
        self.events.append(event)
        self.visits[chosen] += 1
        self.current = after
        return event

    def snapshot(self):
        from experiments.verified_completion.credit import encode
        return {"schema": "sera.continuing-growth.1", "parent": self.parent,
                "state": encode(self.state()), "owner": model_identity(self.owner)}


def validate_events(events, highwater):
    peak, previous = None, None
    seen = set()
    for i, event in enumerate(events):
        unsigned = {k: v for k, v in event.items() if k != "id"}
        if event["id"] in seen or digest(unsigned) != event["id"] or event["sequence"] != i:
            raise ValueError("Duplicate, reordered or changed independent credit")
        if previous is not None and event["knowledge_before"] != previous:
            raise ValueError("Stale knowledge received procedure credit")
        peak = event["before"]["scores"] if peak is None else peak
        expected = credit(event["before"], event["after"], peak, SKILLS.index(event["skill"]))
        if not event["accepted"]:
            expected["reward"] -= .001
        if expected != event["credit"]:
            raise ValueError("Changed independent progress reward")
        peak, previous = expected["highwater"], event["knowledge_after"]
        seen.add(event["id"])
    if events and peak != highwater:
        raise ValueError("Changed persistent high-water credit")
