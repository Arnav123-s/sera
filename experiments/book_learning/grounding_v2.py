"""Named human-entry repair; the headword-blind study remains immutable."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.human_reading.data import ROOT, digest, normalized, read, sha, write

from . import grounding as first
from .data import OUT
from .ground_data import COLLEGE, GROUND, ROLES, glossary
from .ground_model import KINDS, GroundR1, apply, delta, restore_books

RUN = ROOT / "runs/GD-study-002"
SELECTION = OUT / "named-selection.json"
FINAL = OUT / "named-final.json"


def contracts():
    return {**first.contracts(), "repair": sha(OUT / "GD002_PROTOCOL.md"), "named_study": sha(Path(__file__))}


def text(row):
    return row["term"] + "\n" + row["text"]


def train():
    if SELECTION.exists():
        raise FileExistsError("Preserve the completed named-entry study")
    RUN.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3031)
    owner = GroundR1.attach(restore_books().owner)
    old = {k: v.clone() for k, v in owner.state_dict().items() if not k.startswith("ground_")}
    rows = read(GROUND / "teaching.json")
    x = {p: owner.ground_features([text(r) for r in rows[p]]) for p in ("train", "dev")}
    owner.ground_mean.copy_(x["train"]["neural"].mean(0))
    owner.ground_scale.copy_(x["train"]["neural"].std(0).clamp_min(.01))
    y = first.targets(rows["train"])
    weight = 1/y.bincount(minlength=len(ROLES)).float()
    optimizer = torch.optim.AdamW(owner.ground_heads.parameters(), lr=.01, weight_decay=.01)
    start = 1
    history = read(OUT / "named-development.json") if (OUT / "named-development.json").exists() else []
    checkpoints = sorted(RUN.glob("step-*.pt"))
    if checkpoints:
        saved = torch.load(checkpoints[-1], weights_only=True, map_location="cpu")
        if saved["contracts"] != contracts():
            raise ValueError("Named-entry resume contract changed")
        apply(owner, saved["delta"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]+1
    for step in range(start, 321):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(F.cross_entropy(owner.ground_logits(x["train"], k), y, weight=weight) for k in KINDS)
        loss.backward()
        optimizer.step()
        if step in (40, 80, 160, 320):
            path = RUN / f"step-{step:04d}.pt"
            torch.save({"contracts": contracts(), "step": step, "delta": delta(owner), "optimizer": optimizer.state_dict()}, path)
            with torch.no_grad():
                for kind in KINDS:
                    result = first.metric(owner.ground_logits(x["dev"], kind), rows["dev"])
                    history.append({"step": step, "kind": kind, "development": result,
                                    "checkpoint": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
            write(OUT / "named-development.json", history)
            print(json.dumps({"step": step, "dev_macro": {r["kind"]: r["development"]["macro_accuracy"] for r in history[-3:]}}), flush=True)
    selected = min(history, key=lambda r: (-r["development"]["macro_accuracy"], r["development"]["cross_entropy"],
                                          owner.ground_heads[r["kind"]].weight.numel(), r["step"]))
    assert all(torch.equal(v, owner.state_dict()[k]) for k, v in old.items())
    write(SELECTION, {"contracts": contracts(), **selected, "gate": first.gate(selected["development"]["predictions"]),
                      "predecessor_tensors_exact": len(old), "final_opened": False})


def prepare_final():
    from collections import Counter

    path = RUN / "final.json"
    if path.exists():
        return read(path)
    old = read(GROUND / "teaching.json")
    forbidden = {normalized(r["text"]) for p in ("train", "dev") for r in old[p]}
    seen, rows, count = set(), [], Counter()
    for row in sorted(glossary(read(COLLEGE / "receipt.json")), key=lambda r: digest(r["id"])):
        key = normalized(row["text"])
        if key in forbidden or key in seen or (row["label"] == "other" and count["other"] >= 40):
            continue
        rows.append(row)
        seen.add(key)
        count[row["label"]] += 1
    result = {"rows": rows, "counts": dict(count), "selection": sha(SELECTION), "source": sha(COLLEGE / "receipt.json")}
    write(path, result)
    return result


def evaluate():
    if FINAL.exists():
        raise FileExistsError("The named-entry final is closed")
    selected = read(SELECTION)
    if selected["contracts"] != contracts() or sha(ROOT / selected["checkpoint"]) != selected["sha256"]:
        raise ValueError("Frozen named-entry selection changed")
    data = prepare_final()
    owner = GroundR1.attach(restore_books().owner)
    apply(owner, torch.load(ROOT / selected["checkpoint"], weights_only=True, map_location="cpu")["delta"])
    with torch.no_grad():
        x = owner.ground_features([text(r) for r in data["rows"]])
        results = {k: first.metric(owner.ground_logits(x, k), data["rows"]) for k in KINDS}
        majority = first.targets(read(GROUND / "teaching.json")["train"]).bincount().argmax().item()
        constant = torch.zeros(len(data["rows"]), len(ROLES))
        constant[:, majority] = 1
        results["majority"] = first.metric(constant, data["rows"])
        headword = torch.zeros_like(constant)
        for i, row in enumerate(data["rows"]):
            hits = [k for k in ROLES[:-1] if k in row["term"].split()]
            label = hits[0] if len(hits) == 1 else "other"
            headword[i, ROLES.index(label)] = 1
        results["headword_only_rule"] = first.metric(headword, data["rows"])
    predictions = results[selected["kind"]]["predictions"]
    admitted = [r for r in predictions if r["predicted"] != "other" and max(r["probabilities"]) >= selected["gate"]["threshold"]]
    report = {"selection": sha(SELECTION), "data": sha(RUN / "final.json"), "counts": data["counts"],
              "selected": selected["kind"], "results": results, "admitted": len(admitted),
              "admitted_errors": sum(not r["correct"] for r in admitted), "threshold_retuned": False}
    write(FINAL, report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"} | {"macro": {k: v["macro_accuracy"] for k, v in results.items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    (train if args.action == "train" else evaluate)()
