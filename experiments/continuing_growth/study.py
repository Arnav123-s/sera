"""Finite sustained training, immutable checkpoints and a single sealed evaluation."""

import argparse
import copy
import json
import time

import numpy as np
import torch

from workbench.storage import Store

from .common import (
    ARMS,
    BLOCK,
    DECISIONS,
    OUT,
    ROOT,
    RUN,
    SEEDS,
    SKILLS,
    contracts,
    read,
    sha,
    write,
)
from .data import load, prepare
from .model import measure
from .runtime import GrowthSession


def load_state(path):
    return torch.load(path, weights_only=True)


def save(session, path):
    if path.exists():
        raise FileExistsError("Preserve completed checkpoints")
    temporary = path.with_suffix(".tmp")
    torch.save(session.state(), temporary)
    temporary.replace(path)


def freeze():
    RUN.mkdir(exist_ok=True)
    if not (RUN / "parent.json").exists():
        parent = Store(ROOT / "runs/sera-acquisition-live").read()
        if parent["owner"] != "e127e23509a9134cdc38dfd77be641be66ce2b82063d4e6aefd9a06dd0462636":
            raise ValueError("Reconcile newer acquired owner first")
        write(RUN / "parent.json", parent)
    current = {"contracts": contracts(), "parent": sha(RUN / "parent.json"),
               "seeds": list(SEEDS), "arms": list(ARMS), "decisions_per_arm": DECISIONS}
    path = OUT / "freeze.json"
    if path.exists() and read(path) != current:
        raise ValueError("Frozen continuation changed; preserve it and record a versioned repair")
    if not path.exists():
        write(path, current)


def train(seconds=720):
    if (OUT / "selection.json").exists():
        raise FileExistsError("Completed selection remains sealed")
    freeze()
    started = time.monotonic()
    parent = read(RUN / "parent.json")
    summaries = []
    for seed in SEEDS:
        if all((RUN / f"{arm}-{seed}-{DECISIONS:04d}.pt").exists() for arm in ARMS):
            continue
        session = GrowthSession(parent, seed)
        prepare(session.owner)
        data, probe = load("train"), load("probe")
        initial_path = RUN / f"initial-{seed}.pt"
        if not initial_path.exists():
            save(session, initial_path)
        initial = load_state(initial_path)
        for arm in ARMS:
            paths = sorted(RUN.glob(f"{arm}-{seed}-*.pt"))
            session.restore(load_state(paths[-1]) if paths else initial)
            while len(session.events) < DECISIONS:
                session.step(data, probe, arm)
                if len(session.events) % BLOCK == 0:
                    save(session, RUN / f"{arm}-{seed}-{len(session.events):04d}.pt")
                    record = {"arm": arm, "seed": seed, "decisions": len(session.events),
                              "macro": session.current["macro"],
                              "accepted": sum(e["accepted"] for e in session.events),
                              "elapsed_seconds": time.monotonic()-started}
                    print(json.dumps(record), flush=True)
                    if time.monotonic()-started > seconds:
                        write(OUT / "resume.json", {**record, "status": "CHECKPOINTED", "next_action": "train"})
                        return
        del session
    # Selection sees only the prospectively assigned development cohort.
    for seed in SEEDS:
        for arm in ARMS:
            path = RUN / f"{arm}-{seed}-{DECISIONS:04d}.pt"
            state = load_state(path)
            before, after = state["events"][0]["before"], state["current"]
            delta = np.asarray(after["scores"])-np.asarray(before["scores"])
            summaries.append({"name": f"{arm}-{seed}", "arm": arm, "seed": seed, "before": before,
                              "after": after, "delta": delta.tolist(), "eligible": bool(delta.mean() > 0 and delta.min() >= -.02),
                              "checkpoint": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                              "accepted": sum(e["accepted"] for e in state["events"]),
                              "skill_visits": {s: sum(e["skill"] == s for e in state["events"]) for s in SKILLS}})
    eligible = [r for r in summaries if r["eligible"]]
    chosen = max(eligible, key=lambda r: (r["after"]["macro"], r["arm"] == "balanced", -r["seed"])) if eligible else None
    write(OUT / "selection.json", {"contracts": contracts(), "data": sha(RUN / "data-manifest.json"),
                                  "candidates": summaries, "selected": None if chosen is None else chosen["name"],
                                  "final_opened": False})
    print(json.dumps({"training_complete": True, "selected": None if chosen is None else chosen["name"]}), flush=True)


def witness(owner, data, name):
    result = measure(owner, data, True)
    arrays = {}
    for skill, row in result["skills"].items():
        for key in ("logits", "labels", "gold"):
            arrays[skill+":"+key] = row.pop(key)
        row["predictions"] = arrays[skill+":logits"].argmax(-1).tolist()
    path = RUN / "witnesses" / (name+".npz")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("Preserve evaluation witness")
    np.savez_compressed(path, **arrays)
    result["witness"] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
    return result


def evaluate():
    if (OUT / "final.json").exists() or (RUN / "final-opened.json").exists():
        raise FileExistsError("Final evaluation has already opened; audit interrupted state")
    selection = read(OUT / "selection.json")
    if selection["contracts"] != contracts():
        raise ValueError("Changed selected procedure")
    write(RUN / "final-opened.json", {"selection": sha(OUT / "selection.json"), "contracts": contracts()})
    session = GrowthSession(read(RUN / "parent.json"))
    data = load("final")
    initial = load_state(RUN / f"initial-{SEEDS[0]}.pt")
    session.restore(initial)
    baseline = witness(session.owner, data, "parent")
    results = []
    for row in selection["candidates"]:
        if sha(ROOT / row["checkpoint"]) != row["sha256"]:
            raise ValueError("Selected checkpoint changed")
        session.restore(load_state(ROOT / row["checkpoint"]))
        result = witness(session.owner, data, row["name"])
        delta = np.asarray(result["scores"])-np.asarray(baseline["scores"])
        results.append({"name": row["name"], "result": result, "delta": delta.tolist(),
                        "admitted": bool(delta.mean() > 0 and delta.min() >= -.02)})
        write(RUN / "final-partial.json", {"baseline": baseline, "results": results})
    chosen = next((r for r in results if r["name"] == selection["selected"]), None)
    # A lexical source-only comparator is evaluated without training or selection.
    reading = data["reading"]
    lexical = reading["x"][..., 1].masked_fill(~reading["mask"], -1e9).argmax(-1)
    correct = int(reading["gold"].gather(1, lexical[:, None]).sum())
    write(OUT / "final.json", {"contracts": contracts(), "selection": sha(OUT / "selection.json"),
        "baseline": baseline, "results": results, "selected": selection["selected"],
        "knowledge_admitted": bool(chosen and chosen["admitted"]),
        "lexical_reading_control": {"correct": correct, "count": len(lexical)},
        "inherited_source_exposure": True, "completed_old_finals_used": False})
    print(json.dumps({"final_complete": True, "selected": selection["selected"],
                      "knowledge_admitted": bool(chosen and chosen["admitted"])}), flush=True)


def future(seconds=720):
    if (OUT / "procedure.json").exists():
        raise FileExistsError("Completed future cohort remains sealed")
    started = time.monotonic()
    results = read(RUN / "future-partial.json") if (RUN / "future-partial.json").exists() else []
    data, probe, final = load("future_train"), load("probe"), load("future_final")
    session = GrowthSession(read(RUN / "parent.json"))
    for seed in SEEDS:
        old = load_state(RUN / f"initial-{seed}.pt")
        new = load_state(RUN / f"self_directed-{seed}-{DECISIONS:04d}.pt")
        for age in ("old", "new"):
            for eta in ("old", "new", "balanced"):
                name = f"future-{seed}-{age}-{eta}"
                if any(r["name"] == name for r in results):
                    continue
                state = copy.deepcopy(old if age == "old" else new)
                state["eta"] = copy.deepcopy(old["eta"] if eta == "old" else new["eta"])
                state.update(events=[], visits=[0]*len(SKILLS), recent=[0.]*len(SKILLS), highwater=None, current=None)
                session.restore(state)
                session.rng.manual_seed(38500+seed)
                before = measure(session.owner, final)
                procedure_before = copy.deepcopy(session.owner.growth_policy.state_dict())
                for _ in range(96):
                    session.step(data, probe, "balanced" if eta == "balanced" else "self_directed", train_eta=False)
                if any(not torch.equal(v, session.owner.growth_policy.state_dict()[k]) for k, v in procedure_before.items()):
                    raise AssertionError("Final future trial mutated procedure weights")
                after = witness(session.owner, final, name)
                save(session, RUN / (name+".pt"))
                results.append({"name": name, "seed": seed, "knowledge": age, "procedure": eta,
                                "before": before, "after": after, "gain": after["macro"]-before["macro"],
                                "decisions": 96, "eta_frozen": True})
                write(RUN / "future-partial.json", results)
                print(json.dumps({"future": name, "gain": results[-1]["gain"]}), flush=True)
                if time.monotonic()-started > seconds:
                    return
    admitted = all(next(r["gain"] for r in results if r["seed"] == s and r["knowledge"] == k and r["procedure"] == "new") >
                   next(r["gain"] for r in results if r["seed"] == s and r["knowledge"] == k and r["procedure"] == control)
                   for s in SEEDS for k in ("old", "new") for control in ("old", "balanced"))
    write(OUT / "procedure.json", {"contracts": contracts(), "trials": results, "admitted": admitted,
                                   "default": "self_directed" if admitted else "balanced"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "evaluate", "future"))
    p.add_argument("--seconds", type=float, default=720)
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
