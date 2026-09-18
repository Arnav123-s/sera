"""Frozen source acquisition, definition learning, selection and independent final."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.human_reading.data import ROOT, read, sha, write

from .data import OUT
from .ground_data import GROUND, ROLES, acquire, prepare
from .ground_model import KINDS, GroundR1, apply, delta, restore_books


def contracts():
    here = Path(__file__).parent
    return {**{name: sha(here / (name+".py")) for name in ("ground_data", "ground_model", "grounding")},
            "protocol": sha(OUT / "GROUNDING_PROTOCOL.md"), "books": sha(OUT / "selection.json"),
            "teaching": sha(GROUND / "teaching.json")}


def targets(rows):
    return torch.tensor([ROLES.index(r["label"]) for r in rows])


def metric(logits, rows):
    y = targets(rows)
    p = logits.softmax(-1)
    predicted = p.argmax(-1)
    by_role = {k: float(predicted[y == i].eq(i).float().mean()) for i, k in enumerate(ROLES) if y.eq(i).any()}
    return {"accuracy": float(predicted.eq(y).float().mean()), "macro_accuracy": sum(by_role.values())/len(by_role),
            "cross_entropy": float(F.cross_entropy(logits, y)), "by_role": by_role,
            "predictions": [{"id": r["id"], "gold": r["label"], "predicted": ROLES[int(predicted[i])],
                             "probabilities": p[i].tolist(), "correct": bool(predicted[i] == y[i])} for i, r in enumerate(rows)]}


def gate(rows):
    for threshold in (0., .5, .65, .8, .9, .95):
        admitted = [r for r in rows if r["predicted"] != "other" and max(r["probabilities"]) >= threshold]
        if len(admitted) >= 3 and all(r["correct"] for r in admitted):
            return {"threshold": threshold, "development_admitted": len(admitted), "autonomous": True}
    return {"threshold": 1.1, "development_admitted": 0, "autonomous": False}


def train():
    if (OUT / "ground-selection.json").exists():
        raise FileExistsError("Keep the completed binding study")
    torch.manual_seed(3031)
    owner = GroundR1.attach(restore_books().owner)
    old = {k: v.clone() for k, v in owner.state_dict().items() if not k.startswith("ground_")}
    rows = read(GROUND / "teaching.json")
    x = {part: owner.ground_features([r["text"] for r in rows[part]]) for part in ("train", "dev")}
    owner.ground_mean.copy_(x["train"]["neural"].mean(0))
    owner.ground_scale.copy_(x["train"]["neural"].std(0).clamp_min(.01))
    y = targets(rows["train"])
    count = y.bincount(minlength=len(ROLES))
    if count.eq(0).any():
        raise ValueError("Each supplied physical role needs teaching evidence")
    weight = 1/count.float()
    optimizer = torch.optim.AdamW(owner.ground_heads.parameters(), lr=.01, weight_decay=.01)
    first = 1
    checkpoints = sorted(GROUND.glob("step-*.pt"))
    history = read(OUT / "ground-development.json") if (OUT / "ground-development.json").exists() else []
    if checkpoints:
        saved = torch.load(checkpoints[-1], weights_only=True, map_location="cpu")
        if saved["contracts"] != contracts():
            raise ValueError("Definition resume contract changed")
        apply(owner, saved["delta"])
        optimizer.load_state_dict(saved["optimizer"])
        first = saved["step"]+1
    for step in range(first, 321):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(F.cross_entropy(owner.ground_logits(x["train"], k), y, weight=weight) for k in KINDS)
        loss.backward()
        optimizer.step()
        if step in (40, 80, 160, 320):
            path = GROUND / f"step-{step:04d}.pt"
            torch.save({"contracts": contracts(), "step": step, "delta": delta(owner), "optimizer": optimizer.state_dict()}, path)
            with torch.no_grad():
                for kind in KINDS:
                    result = metric(owner.ground_logits(x["dev"], kind), rows["dev"])
                    history.append({"step": step, "kind": kind, "development": result, "checkpoint": path.relative_to(ROOT).as_posix(),
                                    "sha256": sha(path)})
            write(OUT / "ground-development.json", history)
            print(json.dumps({"step": step, "dev_macro": {r["kind"]: r["development"]["macro_accuracy"] for r in history[-3:]}}), flush=True)
    selected = min(history, key=lambda r: (-r["development"]["macro_accuracy"], r["development"]["cross_entropy"],
                                          owner.ground_heads[r["kind"]].weight.numel(), r["step"]))
    assert all(torch.equal(v, owner.state_dict()[k]) for k, v in old.items())
    write(OUT / "ground-selection.json", {"contracts": contracts(), **selected,
                                         "gate": gate(selected["development"]["predictions"]),
                                         "predecessor_tensors_exact": len(old), "final_opened": False})


def evaluate():
    destination = OUT / "ground-final.json"
    if destination.exists():
        raise FileExistsError("Keep the completed definition final closed")
    selected = read(OUT / "ground-selection.json")
    if selected["contracts"] != contracts() or sha(ROOT / selected["checkpoint"]) != selected["sha256"]:
        raise ValueError("Frozen binding selection changed")
    if not (GROUND / "final.json").exists():
        prepare(final=True)
    data = read(GROUND / "final.json")
    owner = GroundR1.attach(restore_books().owner)
    apply(owner, torch.load(ROOT / selected["checkpoint"], weights_only=True, map_location="cpu")["delta"])
    with torch.no_grad():
        x = owner.ground_features([r["text"] for r in data["rows"]])
        results = {k: metric(owner.ground_logits(x, k), data["rows"]) for k in KINDS}
        majority = targets(read(GROUND / "teaching.json")["train"]).bincount().argmax().item()
        scores = torch.zeros(len(data["rows"]), len(ROLES))
        scores[:, majority] = 1
        results["majority"] = metric(scores, data["rows"])
    predictions = results[selected["kind"]]["predictions"]
    admitted = [r for r in predictions if r["predicted"] != "other" and max(r["probabilities"]) >= selected["gate"]["threshold"]]
    write(destination, {"selection": sha(OUT / "ground-selection.json"), "data": sha(GROUND / "final.json"), "counts": data["counts"],
                        "selected": selected["kind"], "results": results, "admitted": len(admitted),
                        "admitted_errors": sum(not r["correct"] for r in admitted), "threshold_retuned": False})
    print(json.dumps({"selected": selected["kind"], "gate": selected["gate"], "counts": data["counts"],
                      "macro": {k: v["macro_accuracy"] for k, v in results.items()}, "admitted": len(admitted),
                      "admitted_errors": sum(not r["correct"] for r in admitted)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("acquire", "prepare", "train", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"acquire": acquire, "prepare": prepare, "train": train, "evaluate": evaluate}[args.action]()
