"""Resolve oversized proposed updates against the unchanged independent gate."""

import argparse
import copy
import json
from pathlib import Path

import torch

from experiments import sustained_refinement as sr
from experiments.continuing_growth.data import load as growth_data
from experiments.continuing_growth.model import apply, knowledge, measure
from experiments.verified_completion.common import ROOT, digest, read, sha, write
from experiments.verified_completion.credit import encode
from sera.session_state import model_identity

OUT = ROOT / "research-continuation/40_step_resolution"
RUN = ROOT / "runs/RS-study-001"
STEPS = 1024
ORIGINAL_GATE = sr.admissible


def contracts():
    return {"source": sha(Path(__file__)), "protocol": sha(OUT / "PROTOCOL.md"), "parent": sr.contracts()}


def resolve(owner, old, before, trial, anchor, probe):
    """Line-search predictions; acceptance always calls the original verifier."""
    proposed = knowledge(owner)
    attempts = []
    for level in range(9):
        scale = 2.**-level
        if level:
            apply(owner, {k: old[k]+scale*(proposed[k]-old[k]) for k in old})
            measured = measure(owner, probe)
        else:
            measured = copy.deepcopy(trial)
        passes = ORIGINAL_GATE(before, measured, anchor)
        improves = measured["macro"] > before["macro"]+1e-10
        attempts.append({"scale": scale, "measurement": measured, "original_gate": passes,
                         "strict_macro_progress": improves, "accepted": passes and improves})
        if passes and improves:
            trial.clear()
            trial.update(measured)
            return True, attempts
    # The inherited transaction restores weights and optimizer on False.
    return False, attempts


class ResolvedR1(sr.ContinuingR1):
    def export_config(self):
        return {**super().export_config(), "step_resolution": contracts()}


class ResolutionSession:
    def __init__(self, parent, initial=None, saved=None):
        self.parent = parent
        self.base = sr.Session(parent)
        if saved:
            if saved["resolution_contracts"] != contracts():
                raise ValueError("Changed retained step-resolution contract")
            self.base.restore(saved["state"])
        elif initial:
            self.base.restore(initial)
        self.owner = self.base.owner
        self.owner.__class__ = ResolvedR1

    def step(self, train, probe):
        old = knowledge(self.owner)
        attempts = []
        def callback(before, trial, anchor):
            accepted, records = resolve(self.owner, old, before, trial, anchor, probe)
            attempts.extend(records)
            return accepted
        # This single-worker adapter supplies a numerical transaction hook. It
        # never replaces or relaxes ORIGINAL_GATE, which judges every candidate.
        if sr.admissible is not ORIGINAL_GATE:
            raise ValueError("Another numerical transaction is active")
        sr.admissible = callback
        try:
            self.base.step(train, probe, "autonomous")
        finally:
            sr.admissible = ORIGINAL_GATE
        event = self.base.events[-1]
        event.pop("id")
        event["step_resolution"] = attempts
        event["resolution_contracts"] = contracts()
        event["id"] = digest(event)

    def state(self):
        return {"resolution_contracts": contracts(), "state": self.base.state()}

    def snapshot(self):
        return {"schema": "sera.step-resolution.1", "parent": self.parent,
                "state": encode(self.state()), "owner": model_identity(self.owner)}


def save(session, path):
    if path.exists():
        raise FileExistsError("Preserve retained step-search checkpoint")
    torch.save(session.state(), path)


def train():
    if (OUT / "selection.json").exists():
        raise FileExistsError("Completed resolution selection is sealed")
    if (sr.RUN / "final-opened.json").exists():
        raise ValueError("Register this prospective repair before its shared final cohort opens")
    RUN.mkdir(exist_ok=True)
    selection = read(sr.OUT / "selection.json")
    selected = selection["selected"]
    if selected is None:
        raise ValueError("No development-eligible continuing parent")
    path = ROOT / next(r["checkpoint"] for r in selection["candidates"] if r["name"] == selected)
    frozen = {"contracts": contracts(), "selected_parent": selected, "checkpoint": sha(path),
              "shared_final_not_opened": True, "decisions_per_seed": STEPS, "seeds": [4001, 4002]}
    if (OUT / "freeze.json").exists() and read(OUT / "freeze.json") != frozen:
        raise ValueError("Changed prospective step-resolution experiment")
    write(OUT / "freeze.json", frozen)
    parent = read(sr.RUN / "parent.json")
    source = sr.state(path)
    data, probe = sr.load("train"), growth_data("probe")
    results = []
    for seed in (4001, 4002):
        initial = copy.deepcopy(source)
        initial.update(seed=seed, events=[], current=None, anchor=None, highwater=None,
                       visits=[0]*len(sr.SKILLS), recent=[0.]*len(sr.SKILLS))
        paths = sorted(RUN.glob(f"{seed}-*.pt"))
        session = ResolutionSession(parent, initial=initial, saved=sr.state(paths[-1]) if paths else None)
        if not paths:
            session.base.rng.manual_seed(seed)
            save(session, RUN / f"{seed}-0000.pt")
        while len(session.base.events) < STEPS:
            session.step(data, probe)
            if len(session.base.events) % 128 == 0:
                path = RUN / f"{seed}-{len(session.base.events):04d}.pt"
                save(session, path)
                print(json.dumps({"seed": seed, "decisions": len(session.base.events),
                                  "macro": session.base.current["macro"],
                                  "accepted": sum(e["accepted"] for e in session.base.events)}), flush=True)
        path = RUN / f"{seed}-{STEPS:04d}.pt"
        result = {"seed": seed, "name": str(seed), "before": session.base.anchor, "after": session.base.current,
                  "checkpoint": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                  "accepted": sum(e["accepted"] for e in session.base.events),
                  "search_checks": sum(len(e["step_resolution"]) for e in session.base.events),
                  "rescued": sum(e["accepted"] and len(e["step_resolution"]) > 1 for e in session.base.events)}
        results.append(result)
    chosen = max(results, key=lambda r: (r["after"]["macro"], -r["seed"]))
    write(OUT / "selection.json", {"contracts": contracts(), "selected": chosen["name"],
                                  "candidates": results, "final_opened": False})


def evaluate():
    if (OUT / "final.json").exists() or (RUN / "final-opened.json").exists():
        raise FileExistsError("Step-resolution final is already open")
    selection = read(OUT / "selection.json")
    if selection["contracts"] != contracts():
        raise ValueError("Changed prospective step-resolution selection")
    write(RUN / "final-opened.json", {"selection": sha(OUT / "selection.json")})
    final, parent = sr.load("final"), read(sr.RUN / "parent.json")
    results = []
    # Use the same prospectively reserved cohort as its parent comparison; no
    # result from either evaluation is used for further training or selection.
    for row in selection["candidates"]:
        session = ResolutionSession(parent, saved=sr.state(ROOT / row["checkpoint"]))
        result = measure(session.owner, final, True)
        arrays = {}
        for skill, record in result["skills"].items():
            for key in ("logits", "labels", "gold"):
                arrays[skill+":"+key] = record.pop(key)
        import numpy as np
        path = RUN / (row["name"]+".npz")
        np.savez_compressed(path, **arrays)
        result["witness"] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
        original = read(sr.OUT / "final.json")["initial"]
        delta = np.array(result["scores"])-original["scores"]
        accurate = all(result["skills"][s]["accuracy"] >= original["skills"][s]["accuracy"]-.04 for s in sr.SKILLS)
        results.append({"name": row["name"], "result": result, "delta": delta.tolist(),
                        "admitted": bool(delta.mean() > 0 and delta.min() >= -.02 and accurate)})
    chosen = next(r for r in results if r["name"] == selection["selected"])
    write(OUT / "final.json", {"contracts": contracts(), "selected": selection["selected"], "results": results,
                              "admitted": chosen["admitted"], "comparison": "same reserved cohort as SR-001; no final-driven tuning"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "evaluate"))
    args = p.parse_args()
    torch.set_num_threads(1)
    (train if args.action == "train" else evaluate)()


if __name__ == "__main__":
    main()
