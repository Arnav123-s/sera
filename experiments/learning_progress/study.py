"""Finite paired successor experiment; no final-driven parameter selection."""

import argparse
import copy
import json

import numpy as np
import torch

from experiments.verified_completion.common import model_hash
from workbench.storage import Store

from .common import OUT, ROOT, RUN, contracts, read, sha, write
from .data import load_data, prepare
from .model import apply_knowledge, knowledge
from .runtime import LearningSession, measure

ARMS = ("balanced", "random", "progress", "learned")


def pack(session):
    return {"contracts": contracts(), "seed": session.seed, "knowledge": knowledge(session.owner),
            "eta": copy.deepcopy(session.owner.learning_policy.state_dict()),
            "optimizer": copy.deepcopy(session.optimizer.state_dict()), "rng": session.rng.get_state().clone(),
            **{k: copy.deepcopy(getattr(session, k)) for k in
               ("events", "highwater", "recent", "phase", "remaining", "cursors")}}


def restore(session, saved):
    if saved["contracts"] != contracts() or session.seed != saved["seed"]:
        raise ValueError("Changed procedure checkpoint identity")
    apply_knowledge(session.owner, saved["knowledge"])
    session.owner.learning_policy.load_state_dict(saved["eta"])
    session.optimizer.load_state_dict(saved["optimizer"])
    session.rng.set_state(saved["rng"])
    for key in ("events", "highwater", "recent", "phase", "remaining", "cursors"):
        setattr(session, key, copy.deepcopy(saved[key]))


def save(session, name):
    path = RUN / (name+".pt")
    if path.exists():
        raise FileExistsError("Completed checkpoint is immutable")
    torch.save(pack(session), path)
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path)}


def train():
    if (OUT / "selection.json").exists():
        raise FileExistsError("Training selection is already frozen")
    if not (RUN / "data.pt").exists():
        prepare()
    if not (RUN / "parent.json").exists():
        parent = Store(ROOT / "runs/sera-quests-live").read()
        if parent["owner"] != "42d55d9a0cf0ae7be65164fae3296b538a5ee2d629886a239d43b181ca8584a3":
            raise ValueError("A newer live owner requires reconciliation")
        write(RUN / "parent.json", parent)
    data = load_data()
    results = []
    for seed in (3401, 3402):
        session = LearningSession(parent=read(RUN / "parent.json"), seed=seed)
        initial = pack(session)
        initial_path = RUN / f"initial-{seed}.pt"
        if not initial_path.exists():
            torch.save(initial, initial_path)
        start = measure(session.owner, data["probe"])
        for arm in ARMS:
            restore(session, initial)
            name = f"{arm}-{seed}"
            existing = sorted(RUN.glob(name+"-step-*.pt"))
            if existing:
                restore(session, torch.load(existing[-1], weights_only=True))
            for phase in (1, 2):
                if len(session.events) >= phase*48:
                    continue
                session.start("teach"+str(phase))
                while any(session.remaining):
                    session.step(data[session.phase], data["probe"], arm)
                    if len(session.events) % 8 == 0:
                        save(session, name+f"-step-{len(session.events):03d}")
                print(json.dumps({"arm": name, "transition": phase, "probe": measure(session.owner, data["probe"])["macro"]}), flush=True)
            result = {"name": name, "arm": arm, "seed": seed, "initial": start,
                      "after": measure(session.owner, data["probe"]),
                      "checkpoint": str((RUN / (name+"-step-096.pt")).relative_to(ROOT)).replace("\\", "/"),
                      "sha256": sha(RUN / (name+"-step-096.pt"))}
            results.append(result)
    selected = max(results, key=lambda r: (r["after"]["macro"], r["arm"] == "balanced", -r["seed"]))
    write(OUT / "selection.json", {"contracts": contracts(), "parent": sha(RUN / "parent.json"),
                                  "data": sha(RUN / "data-manifest.json"), "candidates": results,
                                  "selected": selected["name"], "final_opened": False})


def witness(result, name):
    """Keep full logits once in compressed arrays, not repeated dense JSON."""
    folder = RUN / "predictions"
    folder.mkdir(exist_ok=True)
    arrays = {}
    for skill, row in result["skills"].items():
        arrays[skill] = np.asarray(row.pop("logits"))
        row["predictions"] = arrays[skill].argmax(-1).tolist()
        if skill not in {"reading", "methods"}:
            covered = np.asarray(row["labels"]) != 0
            row["known_vocabulary_accuracy"] = float((arrays[skill].argmax(-1)[covered] == np.asarray(row["labels"])[covered]).mean())
            row["unknown_token_rate"] = float((~covered).mean())
    path = folder / (name+".npz")
    if not path.exists():
        np.savez_compressed(path, **arrays)
    else:
        with np.load(path, allow_pickle=False) as old:
            if any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError("Changed saved prediction witness")
    result["witness"] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path)}
    return result


def evaluate():
    if (OUT / "final.json").exists() or (RUN / "final-opened.json").exists():
        raise FileExistsError("Final evaluation has already opened; preserve partial work")
    selection = read(OUT / "selection.json")
    if selection["contracts"] != contracts():
        raise ValueError("Changed frozen selection")
    data = load_data()
    write(RUN / "final-opened.json", {"selection": sha(OUT / "selection.json"), "contracts": contracts()})
    records, factorial = [], []
    for seed in (3401, 3402):
        session = LearningSession(parent=read(RUN / "parent.json"), seed=seed)
        initial = torch.load(RUN / f"initial-{seed}.pt", weights_only=True)
        for arm in ARMS:
            state = torch.load(RUN / f"{arm}-{seed}-step-096.pt", weights_only=True)
            restore(session, state)
            result = {"name": f"{arm}-{seed}", "cohorts": {}}
            for phase in (1, 2):
                result["cohorts"][str(phase)] = witness(measure(session.owner, data[f"final{phase}"], True), f"{arm}-{seed}-{phase}")
            restore(session, initial)
            result["initial"] = {str(phase): witness(measure(session.owner, data[f"final{phase}"], True), f"initial-{seed}-{phase}") for phase in (1, 2)}
            records.append(result)
        # Each transition uses its actual consecutive learned-owner successors.
        for phase in (1, 2):
            older = initial if phase == 1 else torch.load(RUN / f"learned-{seed}-step-048.pt", weights_only=True)
            newer = torch.load(RUN / f"learned-{seed}-step-{48*phase:03d}.pt", weights_only=True)
            for knowledge_age in ("old", "new"):
                state = older if knowledge_age == "old" else newer
                for procedure in ("old_eta", "new_eta", "balanced", "random", "progress"):
                    restore(session, state)
                    if procedure in ("old_eta", "new_eta"):
                        eta = older if procedure == "old_eta" else newer
                        session.owner.learning_policy.load_state_dict(eta["eta"])
                    session.rng.manual_seed(34500+seed+phase)
                    session.highwater, session.recent = None, [0.]*6
                    session.phase, session.remaining = None, [0]*6
                    session.start(f"future{phase}", quota=4)
                    start = measure(session.owner, data[f"final{phase}"])
                    # Only the development probe controls decisions; final labels remain read-only.
                    while any(session.remaining):
                        session.step(data[f"future{phase}"], data["probe"],
                                     "learned" if "eta" in procedure else procedure, train_eta=False)
                    label = f"transfer-{seed}-{phase}-{knowledge_age}-{procedure}"
                    outcome = witness(measure(session.owner, data[f"final{phase}"], True), label)
                    torch.save({"knowledge": knowledge(session.owner), "events": session.events[-24:],
                                "eta": copy.deepcopy(session.owner.learning_policy.state_dict()),
                                "rng": session.rng.get_state(), "contracts": contracts()}, RUN / (label+".pt"))
                    item = {"seed": seed, "transition": phase, "knowledge": knowledge_age, "procedure": procedure,
                            "before": start, "after": outcome, "gain": outcome["macro"]-start["macro"],
                            "knowledge_identity": model_hash(session.owner), "updates": 24,
                            "examples": 192, "eta_trained_on_final": False}
                    factorial.append(item)
                    write(RUN / "final-partial.json", {"records": records, "factorial": factorial})
            print(json.dumps({"evaluated_seed": seed, "transition": phase}), flush=True)
    chosen = next(r for r in records if r["name"] == selection["selected"])
    delta = [chosen["cohorts"][str(p)]["scores"][s]-chosen["initial"][str(p)]["scores"][s]
             for p in (1, 2) for s in range(6)]
    knowledge_admitted = sum(delta) > 0 and min(delta) >= -.02
    eta_admitted = all(
        sum(r["gain"] for r in factorial if r["transition"] == p and r["procedure"] == "new_eta") >
        sum(r["gain"] for r in factorial if r["transition"] == p and r["procedure"] == control)
        for p in (1, 2) for control in ("balanced", "old_eta"))
    write(OUT / "final.json", {"contracts": contracts(), "selection": sha(OUT / "selection.json"),
                              "results": records, "factorial": factorial, "selected": selection["selected"],
                              "knowledge_admitted": knowledge_admitted, "learned_scheduler_admitted": eta_admitted,
                              "default": "learned" if eta_admitted else "balanced", "score_changes": delta,
                              "scope": "continued acquisition within retained task families; inherited corpus exposure recorded"})
    print(json.dumps({"selected": selection["selected"], "knowledge_admitted": knowledge_admitted,
                      "learned_scheduler_admitted": eta_admitted}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    (train if args.action == "train" else evaluate)()


if __name__ == "__main__":
    main()
