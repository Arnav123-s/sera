"""Long continuing practice with retained-ability floors and learned optimizer choice."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.continuing_growth.common import PREFIXES, SKILLS
from experiments.continuing_growth.data import load as growth_data
from experiments.continuing_growth.data import take
from experiments.continuing_growth.model import (
    GrowingR1,
    apply,
    credit,
    errors,
    identity,
    knowledge,
    measure,
)
from experiments.continuing_growth.runtime import GrowthSession
from experiments.learning_progress.data import method_rows
from experiments.verified_completion.common import ROOT, digest, read, sha, write
from experiments.verified_completion.credit import decode, encode
from sera.session_state import model_identity
from workbench.storage import Store

OUT = ROOT / "research-continuation/39_sustained_refinement"
RUN = ROOT / "runs/SR-study-001"
STEPS = 2048
SEEDS = (3901, 3902)
ARMS = ("balanced_adam", "autonomous")


def contracts():
    return {"source": sha(Path(__file__)), "protocol": sha(OUT / "PROTOCOL.md")}


def grouping(skill, row):
    if skill == "reading":
        return "title:"+row["title"]
    if "group" in row:
        return row["group"]
    return "request:"+str(row.get("source_id", row["id"]))


def bucket(group):
    v = int(digest(["SR-001", group])[:8], 16) % 10
    return "train" if v < 4 else "final" if v < 6 else "future_train" if v < 8 else "future_final"


def prepare():
    if (RUN / "manifest.json").exists():
        return
    source = growth_data("train")
    metadata = read(ROOT / "runs/CG-study-001/metadata.json")
    pools = {n: {} for n in ("train", "final", "future_train", "future_final")}
    manifest = {"parent_manifest": sha(ROOT / "runs/CG-study-001/data-manifest.json"),
                "parent_exposure": "all human records were available in parent CG training; prospective holdback starts here",
                "closed_prior_final_files_opened": False, "groups": {}, "indices": {}}
    for skill in SKILLS:
        if skill == "methods":
            for i, name in enumerate(pools):
                pools[name][skill] = method_rows(39300+i, 512 if name == "train" else 128)
            continue
        rows = metadata[skill]["train"]
        groups = [grouping(skill, row) for row in rows]
        for name in pools:
            ids = [i for i, g in enumerate(groups) if bucket(g) == name][:512 if name == "train" else 128]
            if len(ids) < 8:
                raise ValueError("Insufficient new continuation holdback: "+skill+":"+name)
            pools[name][skill] = take(source[skill], ids)
            manifest["indices"][skill+":"+name] = ids
            manifest["groups"][skill+":"+name] = sorted({groups[i] for i in ids})
    for name, data in pools.items():
        path = RUN / (name+".pt")
        if path.exists():
            raise FileExistsError("Preserve partially prepared continuation")
        torch.save(data, path)
        manifest[name] = sha(path)
    write(RUN / "manifest.json", manifest)


def load(name):
    path = RUN / (name+".pt")
    if sha(path) != read(RUN / "manifest.json")[name]:
        raise ValueError("Changed prospectively assigned source cohort")
    return torch.load(path, weights_only=True)


def admissible(before, trial, anchor):
    change = np.array(trial["scores"])-before["scores"]
    enough_correct = all(trial["skills"][s]["correct"] >= anchor["skills"][s]["correct"]-1 for s in SKILLS)
    return bool(enough_correct and change.min() >= -.01 and change.mean() >= -.00005)


class ContinuingR1(GrowingR1):
    def export_config(self):
        return {**super().export_config(), "sustained_refinement": self.refinement_config}


class Session:
    def __init__(self, parent, seed=3901, saved=None):
        self.parent = parent
        self.base = GrowthSession(parent["parent"], saved=decode(parent["state"]))
        self.owner = self.base.owner
        for n, p in self.owner.named_parameters():
            p.requires_grad_(n.startswith(PREFIXES))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.owner.refinement_policy = nn.Sequential(nn.Linear(len(SKILLS)+6, 32, dtype=torch.float64),
                                                         nn.Tanh(), nn.Linear(32, 1, dtype=torch.float64))
            nn.init.zeros_(self.owner.refinement_policy[-1].weight)
            nn.init.zeros_(self.owner.refinement_policy[-1].bias)
        self.owner.__class__ = ContinuingR1
        self.owner.refinement_config = {"contracts": contracts(), "seed": seed}
        self.parameters = [p for n, p in self.owner.named_parameters() if n.startswith(PREFIXES)]
        self.adam = torch.optim.Adam(self.parameters, lr=.0005)
        self.eta = torch.optim.Adam(self.owner.refinement_policy.parameters(), lr=.003)
        self.rng = torch.Generator().manual_seed(seed)
        self.seed, self.events = seed, []
        self.visits, self.recent = [0]*len(SKILLS), [0.]*len(SKILLS)
        self.current = self.anchor = self.highwater = None
        if saved:
            self.restore(saved)

    def state(self):
        return {"contracts": contracts(), "seed": self.seed, "knowledge": knowledge(self.owner),
                "policy": copy.deepcopy(self.owner.refinement_policy.state_dict()),
                "adam": copy.deepcopy(self.adam.state_dict()), "eta": copy.deepcopy(self.eta.state_dict()),
                "rng": self.rng.get_state().clone(), "events": copy.deepcopy(self.events),
                "visits": self.visits[:], "recent": self.recent[:], "current": copy.deepcopy(self.current),
                "anchor": copy.deepcopy(self.anchor), "highwater": copy.deepcopy(self.highwater)}

    def restore(self, saved):
        if saved["contracts"] != contracts():
            raise ValueError("Changed continuing-refinement source")
        apply(self.owner, saved["knowledge"])
        self.owner.refinement_policy.load_state_dict(saved["policy"])
        self.adam.load_state_dict(saved["adam"])
        self.eta.load_state_dict(saved["eta"])
        self.rng.set_state(saved["rng"])
        for name in ("seed", "events", "visits", "recent", "current", "anchor", "highwater"):
            setattr(self, name, copy.deepcopy(saved[name]))
        self.owner.refinement_config["seed"] = self.seed
        previous, peak = None, None
        for i, e in enumerate(self.events):
            if e["id"] != digest({k: v for k, v in e.items() if k != "id"}) or e["sequence"] != i:
                raise ValueError("Changed or repeated sustained credit")
            if previous is not None and previous != e["knowledge_before"]:
                raise ValueError("Stale sustained decision")
            peak = e["before"]["scores"] if peak is None else peak
            expected = credit(e["before"], e["after"], peak, SKILLS.index(e["skill"]))
            if not e["accepted"]:
                expected["reward"] -= .001
            if expected != e["credit"]:
                raise ValueError("Forged sustained progress")
            peak, previous = expected["highwater"], e["knowledge_after"]
        if self.events and self.highwater != peak:
            raise ValueError("Changed sustained high-water mark")

    def step(self, train, probe, arm, learn=True):
        if len(self.events) % 128 == 0:
            self.visits = [0]*len(SKILLS)
        before = self.current or measure(self.owner, probe)
        if self.anchor is None:
            self.anchor = copy.deepcopy(before)
            self.highwater = before["scores"][:]
        x = torch.tensor([[float(j == i) for j in range(len(SKILLS))]+
            [before["scores"][i], before["accuracy"][i], self.recent[i], self.visits[i]/32, float(method), 1.]
            for i in range(len(SKILLS)) for method in range(2)], dtype=torch.float64)
        left = 128-len(self.events) % 128
        force = left <= sum(max(0, 2-v) for v in self.visits)
        with torch.no_grad():
            if arm == "balanced_adam":
                chosen = min(range(len(SKILLS)), key=lambda i: (self.visits[i], i))
                action = 2*chosen+1
            elif arm == "autonomous":
                if force or float(torch.rand((), generator=self.rng)) < .2:
                    chosen = min(range(len(SKILLS)), key=lambda i: (self.visits[i], i))
                    action = 2*chosen+int(torch.randint(2, (), generator=self.rng))
                else:
                    values = self.owner.refinement_policy(x).squeeze(-1)
                    action = max((a for a in range(len(x)) if self.visits[a//2] < 32),
                                 key=lambda a: (float(values[a]), -self.visits[a//2], -a))
                    chosen = action//2
            else:
                raise ValueError("Unregistered refinement controller")
        method = action % 2
        # Replay asks which retained strand has fallen furthest below its own
        # verified high-water mark. This has no access to hidden final labels.
        other = max((i for i in range(len(SKILLS)) if i != chosen),
                    key=lambda i: (self.highwater[i]-before["scores"][i], -self.visits[i], -i))
        old, old_adam = knowledge(self.owner), copy.deepcopy(self.adam.state_dict())
        before_id, rows_used = identity(old), []
        for n in range(8):
            skill = SKILLS[chosen if n % 2 == 0 else other]
            ids = torch.randint(len(train[skill]["y"]), (16,), generator=self.rng)
            self.owner.zero_grad(set_to_none=True)
            errors(self.owner, skill, take(train[skill], ids)).mean().backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 2., error_if_nonfinite=True)
            if method:
                self.adam.step()
            else:
                with torch.no_grad():
                    for p in self.parameters:
                        if p.grad is not None:
                            p.add_(p.grad, alpha=-.008)
            rows_used.append({"skill": skill, "ids": ids.tolist()})
        trial = measure(self.owner, probe)
        accepted = admissible(before, trial, self.anchor)
        if not accepted:
            apply(self.owner, old)
            self.adam.load_state_dict(old_adam)
        after = trial if accepted else before
        result = credit(before, after, self.highwater, chosen)
        if not accepted:
            result["reward"] -= .001
        if learn and arm == "autonomous":
            self.eta.zero_grad(set_to_none=True)
            loss = (self.owner.refinement_policy(x[action]).squeeze()-100*result["reward"])**2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.owner.refinement_policy.parameters(), 2.)
            self.eta.step()
        e = {"sequence": len(self.events), "goal": "Advance all retained strands while preserving earlier correct performance",
             "parent_owner": self.parent["owner"], "evidence": sha(RUN / "manifest.json"), "skill": SKILLS[chosen],
             "replay": SKILLS[other], "method": ("sgd", "adam")[method], "arm": arm, "action": action,
             "rows": rows_used, "before": before, "trial": trial, "after": after, "accepted": accepted,
             "knowledge_before": before_id, "knowledge_after": identity(knowledge(self.owner)), "credit": result,
             "eta_updated": learn and arm == "autonomous", "features": x[action].tolist()}
        e["id"] = digest(e)
        self.events.append(e)
        self.highwater, self.current = result["highwater"], after
        self.recent[chosen] = .8*self.recent[chosen]+.2*result["reward"]
        self.visits[chosen] += 1

    def snapshot(self):
        return {"schema": "sera.sustained-refinement.1", "parent": self.parent,
                "state": encode(self.state()), "owner": model_identity(self.owner)}


def save(session, path):
    if path.exists():
        raise FileExistsError("Preserve completed sustained checkpoint")
    tmp = path.with_suffix(".tmp")
    torch.save(session.state(), tmp)
    tmp.replace(path)


def state(path):
    return torch.load(path, weights_only=True)


def train(seconds):
    if (OUT / "selection.json").exists():
        raise FileExistsError("Sustained selection is complete")
    RUN.mkdir(exist_ok=True)
    if not (RUN / "parent.json").exists():
        parent = Store(ROOT / "runs/sera-growth-live").read()
        if parent is None or parent["owner"] != "11653a31b45b36be4d02a81c1a0d2aae5a30561ea9b4a305e0dc617a4751dd81":
            raise ValueError("Reconcile and integrate the audited Stage 38 owner")
        write(RUN / "parent.json", parent)
    freeze = {"contracts": contracts(), "parent": sha(RUN / "parent.json"), "steps": STEPS,
              "seeds": list(SEEDS), "arms": list(ARMS)}
    if (OUT / "freeze.json").exists() and read(OUT / "freeze.json") != freeze:
        raise ValueError("Preserve frozen refinement source")
    write(OUT / "freeze.json", freeze)
    prepare()
    started = time.monotonic()
    data, probe = load("train"), growth_data("probe")
    for seed in SEEDS:
        if all((RUN / f"{arm}-{seed}-{STEPS:04d}.pt").exists() for arm in ARMS):
            continue
        session = Session(read(RUN / "parent.json"), seed)
        initial = RUN / f"initial-{seed}.pt"
        if not initial.exists():
            save(session, initial)
        for arm in ARMS:
            paths = sorted(RUN.glob(f"{arm}-{seed}-*.pt"))
            session.restore(state(paths[-1] if paths else initial))
            while len(session.events) < STEPS:
                session.step(data, probe, arm)
                if len(session.events) % 128 == 0:
                    save(session, RUN / f"{arm}-{seed}-{len(session.events):04d}.pt")
                    print(json.dumps({"arm": arm, "seed": seed, "decisions": len(session.events),
                                      "macro": session.current["macro"], "accepted": sum(e["accepted"] for e in session.events)}), flush=True)
                    if time.monotonic()-started > seconds:
                        return
    results = []
    for seed in SEEDS:
        for arm in ARMS:
            path = RUN / f"{arm}-{seed}-{STEPS:04d}.pt"
            s = state(path)
            gain = s["current"]["macro"]-s["anchor"]["macro"]
            results.append({"name": f"{arm}-{seed}", "arm": arm, "seed": seed, "before": s["anchor"],
                            "after": s["current"], "gain": gain, "eligible": gain > 0,
                            "checkpoint": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                            "accepted": sum(e["accepted"] for e in s["events"]),
                            "methods": {m: sum(e["method"] == m for e in s["events"]) for m in ("sgd", "adam")}})
    eligible = [r for r in results if r["eligible"]]
    chosen = max(eligible, key=lambda r: (r["after"]["macro"], r["arm"] == "balanced_adam", -r["seed"])) if eligible else None
    write(OUT / "selection.json", {"contracts": contracts(), "candidates": results,
                                  "selected": chosen["name"] if chosen else None, "final_opened": False})


def witness(owner, data, name):
    result = measure(owner, data, True)
    arrays = {}
    for skill, row in result["skills"].items():
        for key in ("logits", "labels", "gold"):
            arrays[skill+":"+key] = row.pop(key)
    path = RUN / (name+".npz")
    if path.exists():
        raise FileExistsError("Preserve independent held-back witness")
    np.savez_compressed(path, **arrays)
    result["witness"] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
    return result


def evaluate():
    if (RUN / "final-opened.json").exists():
        raise FileExistsError("Refinement final is already opened")
    selection = read(OUT / "selection.json")
    if selection["contracts"] != contracts():
        raise ValueError("Changed selection source")
    write(RUN / "final-opened.json", {"selection": sha(OUT / "selection.json")})
    session = Session(read(RUN / "parent.json"))
    final = load("final")
    initial = witness(session.owner, final, "parent")
    results = []
    for row in selection["candidates"]:
        session.restore(state(ROOT / row["checkpoint"]))
        result = witness(session.owner, final, row["name"])
        delta = np.array(result["scores"])-initial["scores"]
        accurate = all(result["skills"][s]["accuracy"] >= initial["skills"][s]["accuracy"]-.04 for s in SKILLS)
        results.append({"name": row["name"], "result": result, "delta": delta.tolist(),
                        "admitted": bool(delta.mean() > 0 and delta.min() >= -.02 and accurate)})
        write(RUN / "final-partial.json", {"initial": initial, "results": results})
    selected = next((r for r in results if r["name"] == selection["selected"]), None)
    write(OUT / "final.json", {"contracts": contracts(), "initial": initial, "results": results,
                              "selected": selection["selected"], "admitted": bool(selected and selected["admitted"]),
                              "prior_parent_exposure": True, "prior_final_retuning": False})


def future(seconds):
    if (OUT / "procedure.json").exists():
        raise FileExistsError("Preserve completed procedure evaluation")
    started = time.monotonic()
    rows = read(RUN / "future-partial.json") if (RUN / "future-partial.json").exists() else []
    train, probe, final = load("future_train"), growth_data("probe"), load("future_final")
    session = Session(read(RUN / "parent.json"))
    for seed in SEEDS:
        old, new = state(RUN / f"initial-{seed}.pt"), state(RUN / f"autonomous-{seed}-{STEPS:04d}.pt")
        for age in ("old", "new"):
            for policy in ("old", "new", "balanced_adam"):
                name = f"future-{seed}-{age}-{policy}"
                if any(r["name"] == name for r in rows):
                    continue
                current = copy.deepcopy(old if age == "old" else new)
                current["policy"] = copy.deepcopy(old["policy"] if policy == "old" else new["policy"])
                # K excludes optimizer momentum. Every future procedure gets
                # the same clean inner optimizer at a given knowledge state.
                current["adam"] = copy.deepcopy(old["adam"])
                current.update(events=[], current=None, anchor=None, highwater=None,
                               visits=[0]*len(SKILLS), recent=[0.]*len(SKILLS))
                session.restore(current)
                session.rng.manual_seed(39700+seed)
                before = measure(session.owner, final)
                for _ in range(96):
                    session.step(train, probe, "balanced_adam" if policy == "balanced_adam" else "autonomous", learn=False)
                after = witness(session.owner, final, name)
                save(session, RUN / (name+".pt"))
                rows.append({"name": name, "seed": seed, "knowledge": age, "policy": policy,
                             "before": before, "after": after, "gain": after["macro"]-before["macro"], "eta_frozen": True})
                write(RUN / "future-partial.json", rows)
                print(json.dumps({"future": name, "gain": rows[-1]["gain"]}), flush=True)
                if time.monotonic()-started > seconds:
                    return
    admitted = all(next(r["gain"] for r in rows if r["seed"] == s and r["knowledge"] == k and r["policy"] == "new") >
                   next(r["gain"] for r in rows if r["seed"] == s and r["knowledge"] == k and r["policy"] == control)
                   for s in SEEDS for k in ("old", "new") for control in ("old", "balanced_adam"))
    write(OUT / "procedure.json", {"trials": rows, "admitted": admitted,
                                   "default": "autonomous" if admitted else "balanced_adam"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "evaluate", "future"))
    p.add_argument("--seconds", type=float, default=690)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.action == "train":
        train(args.seconds)
    elif args.action == "future":
        future(args.seconds)
    else:
        evaluate()


if __name__ == "__main__":
    main()
