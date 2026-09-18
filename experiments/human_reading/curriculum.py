"""Sequential human-text weight learning, frozen comparisons and exact resumption."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .corpus import CORPUS, DOMAINS, prepare
from .data import OUT, ROOT, digest, normalized, read, sha, write
from .model import ReadingR1, encode, lexical, parent, source

CHECKPOINTS = ROOT / "runs/HC-study-002"
STEPS = {"dictionary": 400, "sentences": 300, "conversation": 300, "textbook": 300}


def contracts():
    return {"model": source(), "curriculum": sha(Path(__file__)),
            "corpus": sha(Path(__file__).with_name("corpus.py")),
            "manifest": sha(CORPUS / "manifest.json"), "protocol": sha(OUT / "CURRICULUM.md"),
            "parent": sha(OUT / "parent-empirical.json")}


def load_rows(domain, split):
    manifest = read(CORPUS / ("final-manifest.json" if split == "final" else "manifest.json"))
    path = CORPUS / f"{domain}-{split}.json"
    if sha(path) != manifest["hashes"][domain+":"+split]:
        raise ValueError("Teaching/evaluation partition changed")
    return read(path)


def candidates(rows):
    target_groups = {}
    for i, row in enumerate(rows):
        target_groups.setdefault(normalized(row["target"]), i)
    representatives = list(target_groups.values())
    if len(representatives) < 8:
        raise ValueError("At least eight distinct human targets are required")
    result = []
    for i, row in enumerate(rows):
        rng = random.Random(int(digest("HC-candidates:"+row["id"]), 16))
        pool = [j for j in representatives if normalized(rows[j]["target"]) != normalized(row["target"])]
        indices = [i] + rng.sample(pool, 7)
        rng.shuffle(indices)
        result.append({"indices": indices, "gold": indices.index(i)})
    return result


@torch.no_grad()
def measure(owner, rows, choices, *, baseline=False):
    if baseline:
        scores = torch.stack([lexical(row["question"], [{"text": rows[j]["target"]} for j in c["indices"]])[:, 1]
                              for row, c in zip(rows, choices, strict=True)])
    else:
        q, t = [], []
        for start in range(0, len(rows), 128):
            batch = rows[start:start+128]
            q.append(owner.semantic(encode([r["question"] for r in batch])))
            t.append(owner.semantic(encode([r["target"] for r in batch])))
        q, t = torch.cat(q), torch.cat(t)
        indices = torch.tensor([c["indices"] for c in choices])
        scores = (q[:, None] * t[indices]).sum(-1)
    order = scores.argsort(dim=-1, descending=True, stable=True).tolist()
    records = [{"id": r["id"], "group": r["group"], "gold": c["gold"],
                "prediction": ranks[0], "correct": ranks[0] == c["gold"],
                "rank": ranks.index(c["gold"])+1,
                "candidates": [rows[j]["id"] for j in c["indices"]]}
               for r, c, ranks in zip(rows, choices, order, strict=True)]
    return {"accuracy": sum(r["correct"] for r in records)/len(records),
            "mrr": sum(1/r["rank"] for r in records)/len(records),
            "pairs": len(records), "predictions": records}


def snapshot(owner, domain, step, optimizer=None, order=None, cursor=0, generator=None):
    return {"schema": "sera.human-curriculum.1", "contracts": contracts(),
            "stage": domain, "step": step, "embedding": owner.reading_embedding.weight.detach().clone(),
            "optimizer": optimizer.state_dict() if optimizer else None, "order": order, "cursor": cursor,
            "rng": generator.get_state() if generator else None}


def restore(owner, path):
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if saved["contracts"] != contracts():
        raise ValueError("Curriculum source contract changed")
    value = saved["embedding"]
    if value.shape != owner.reading_embedding.weight.shape or not torch.isfinite(value).all():
        raise ValueError("Curriculum embedding schema changed")
    with torch.no_grad():
        owner.reading_embedding.weight.copy_(value)
    return saved


def fit(owner, domain):
    folder = CHECKPOINTS / domain
    folder.mkdir(parents=True, exist_ok=True)
    rows = load_rows(domain, "train")
    q, t = encode([r["question"] for r in rows]), encode([r["target"] for r in rows])
    targets = [normalized(r["target"]) for r in rows]
    target_ids = {s: i for i, s in enumerate(sorted(set(targets)))}
    labels = torch.tensor([target_ids[s] for s in targets])
    optimizer = torch.optim.AdamW([owner.reading_embedding.weight], lr=.003, weight_decay=.0001)
    generator = torch.Generator().manual_seed(2911+DOMAINS.index(domain))
    order = torch.randperm(len(rows), generator=generator)
    cursor, first = 0, 1
    existing = sorted(folder.glob("step-*.pt"))
    if existing:
        saved = restore(owner, existing[-1])
        optimizer.load_state_dict(saved["optimizer"])
        order, cursor, first = saved["order"], saved["cursor"], saved["step"]+1
        generator.set_state(saved["rng"])
    losses = []
    for step in range(first, STEPS[domain]+1):
        if cursor == len(order):
            order, cursor = torch.randperm(len(rows), generator=generator), 0
        indices = order[cursor:cursor+64]
        cursor += len(indices)
        optimizer.zero_grad(set_to_none=True)
        scores = owner.semantic(q[indices]) @ owner.semantic(t[indices]).T / .15
        duplicate = labels[indices, None].eq(labels[indices][None])
        duplicate.fill_diagonal_(False)
        scores = scores.masked_fill(duplicate, -1e9)
        loss = F.cross_entropy(scores, torch.arange(len(indices)))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            torch.save(snapshot(owner, domain, step, optimizer, order, cursor, generator), folder / f"step-{step:04d}.pt")
            print(json.dumps({"stage": domain, "step": step, "loss": losses[-1]}), flush=True)
    return folder / f"step-{STEPS[domain]:04d}.pt"


def train():
    if (OUT / "curriculum-selection.json").exists():
        raise FileExistsError("Completed curriculum must not restart")
    started = time.perf_counter()
    session = parent()
    old = {k: v.clone() for k, v in session.owner.state_dict().items()}
    owner = ReadingR1.attach(session.owner)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    initial = CHECKPOINTS / "initial.pt"
    if not initial.exists():
        torch.save(snapshot(owner, "initial", 0), initial)
    else:
        restore(owner, initial)
    development = {domain: load_rows(domain, "dev") for domain in DOMAINS}
    choices = {domain: candidates(rows) for domain, rows in development.items()}
    records = []
    for stage in ("initial", *DOMAINS):
        path = initial if stage == "initial" else fit(owner, stage)
        dev_path = OUT / f"curriculum-dev-{stage}.json"
        if dev_path.exists():
            scores = read(dev_path)
        else:
            scores = {d: measure(owner, rows, choices[d]) for d, rows in development.items()}
            write(dev_path, scores)
        record = {"stage": stage, "checkpoint": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                  "development": {d: r["accuracy"] for d, r in scores.items()}}
        record["mean_accuracy"] = sum(record["development"].values())/len(DOMAINS)
        records.append(record)
        print(json.dumps(record), flush=True)
    write(OUT / "curriculum-dev-bm25.json", {d: measure(owner, r, choices[d], baseline=True) for d, r in development.items()})
    if any(not torch.equal(owner.state_dict()[k], v) for k, v in old.items()):
        raise AssertionError("A predecessor tensor changed")
    winner = max(records, key=lambda r: r["mean_accuracy"])
    write(OUT / "curriculum-selection.json", {**winner, "contracts": contracts(), "stages": records,
                                             "preserved_tensors": len(old), "final_opened": False,
                                             "wall_seconds": time.perf_counter()-started})


def evaluate():
    destination = OUT / "curriculum-final.json"
    if destination.exists():
        raise FileExistsError("Final evaluation is already complete")
    selection = read(OUT / "curriculum-selection.json")
    if selection["contracts"] != contracts():
        raise ValueError("Selection changed")
    if not (CORPUS / "final-manifest.json").exists():
        prepare(final=True)
    owner = ReadingR1.attach(parent().owner)
    rows = {d: load_rows(d, "final") for d in DOMAINS}
    choices = {d: candidates(r) for d, r in rows.items()}
    results = {"bm25": {d: measure(owner, r, choices[d], baseline=True) for d, r in rows.items()}}
    for stage in selection["stages"]:
        path = ROOT / stage["checkpoint"]
        if sha(path) != stage["sha256"]:
            raise ValueError("Curriculum checkpoint bytes changed")
        restore(owner, path)
        results[stage["stage"]] = {d: measure(owner, r, choices[d]) for d, r in rows.items()}
    write(destination, {"selection": sha(OUT / "curriculum-selection.json"), "results": results})
    print(json.dumps({s: {d: r["accuracy"] for d, r in v.items()} for s, v in results.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    (train if args.action == "train" else evaluate)()
