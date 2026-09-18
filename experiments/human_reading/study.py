"""Frozen human-text comparisons, exact resumptions and once-only evaluation."""

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from sera.session_state import model_identity

from .data import OUT, ROOT, RUN, prepare, read, sha, write
from .model import ReadingR1, apply, delta, load_curriculum, parent, source


def contracts():
    return {"parent": sha(OUT / "parent-empirical.json"), "protocol": sha(OUT / "PROTOCOL.md"),
            "model": source(), "data": sha(Path(__file__).with_name("data.py")),
            "study": sha(Path(__file__)), "teaching": sha(RUN / "data-manifest.json"),
            "amendment": sha(OUT / "CURRICULUM.md"),
            "curriculum": sha(OUT / "curriculum-selection.json")}


def metrics(rows, logits):
    order = logits.argsort(dim=-1, descending=True, stable=True).tolist()
    predictions, articles = [], defaultdict(list)
    for row, indices in zip(rows, order, strict=True):
        correct = indices[0] in row["gold"]
        rank = min(indices.index(g)+1 for g in row["gold"])
        articles[row["title"]].append(int(correct))
        predictions.append({"id": row["id"], "article": row["title"], "prediction": indices[0],
                            "gold": row["gold"], "correct": correct, "rank": rank,
                            "context_sha256": row["context_sha256"]})
    return {"accuracy": sum(p["correct"] for p in predictions)/len(rows),
            "article_macro_accuracy": sum(sum(v)/len(v) for v in articles.values())/len(articles),
            "mrr": sum(1/p["rank"] for p in predictions)/len(rows),
            "questions": len(rows), "articles": len(articles), "predictions": predictions}


def checkpoint(owner, optimizer, step, order, cursor, generator):
    return {"schema": "sera.human-reading-checkpoint.1", "contracts": contracts(),
            "config": dict(owner.reading_config), "delta": delta(owner), "optimizer": optimizer.state_dict(),
            "step": step, "order": order, "cursor": cursor, "rng": generator.get_state()}


def fit(owner, features, labels, folder, kind, seed, updates=420, resume=None):
    folder.mkdir(parents=True, exist_ok=True)
    owner.reset_reading(kind, seed)
    optimizer = torch.optim.AdamW(owner.reading_head.parameters(), lr=.002, weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(labels), generator=generator)
    cursor, first = 0, 1
    if resume is not None:
        saved = torch.load(resume, weights_only=True, map_location="cpu")
        if saved["contracts"] != contracts() or saved["config"] != owner.reading_config:
            raise ValueError("Changed training source or configuration")
        apply(owner, saved["delta"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["rng"])
        order, cursor, first = saved["order"], saved["cursor"], saved["step"]+1
    x, mask = features
    losses = []
    for step in range(first, updates+1):
        if cursor == len(order):
            order, cursor = torch.randperm(len(labels), generator=generator), 0
        indices = order[cursor:cursor+32]
        cursor += len(indices)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(owner.reading_logits(x[indices], mask[indices]), labels[indices])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 140 == 0 or step == updates:
            path = folder / f"step-{step:04d}.pt"
            if path.exists():
                raise ValueError("A checkpoint must not overwrite completed work")
            torch.save(checkpoint(owner, optimizer, step, order, cursor, generator), path)
    return losses


def cache_features(owner, split, rows):
    path = RUN / f"features-{split}.pt"
    identity = {**contracts(), "rows": sha(RUN / (split+".json"))}
    if path.exists():
        data = torch.load(path, weights_only=True, map_location="cpu")
        if data["identity"] != identity:
            raise ValueError("Changed cached feature source")
        return data["x"], data["mask"]
    x, mask = owner.reading_features(rows, lambda n, total: print(json.dumps({"split": split, "pairs": n, "total": total}), flush=True))
    torch.save({"identity": identity, "x": x, "mask": mask}, path)
    return x, mask


def train():
    if (OUT / "selection.json").exists():
        raise ValueError("Completed selection must not be rerun")
    start = time.perf_counter()
    session = parent()
    old = {n: v.clone() for n, v in session.owner.state_dict().items()}
    before = model_identity(session.owner)
    owner = load_curriculum(ReadingR1.attach(session.owner))
    train_rows, dev_rows = read(RUN / "train.json"), read(RUN / "dev.json")
    train_features = cache_features(owner, "train", train_rows)
    dev_features = cache_features(owner, "dev", dev_rows)
    valid = train_features[0][train_features[1]]
    owner.reading_mean.copy_(valid.mean(0))
    owner.reading_scale.copy_(valid.std(0).clamp_min(.05))
    labels = torch.tensor([r["gold"][0] for r in train_rows])
    base = (4 * dev_features[0][..., 1]).masked_fill(~dev_features[1], -1e9)
    baseline = metrics(dev_rows, base)
    write(OUT / "dev-bm25.json", baseline)
    candidates = []
    for kind in ("lexical", "shared"):
        for seed in (2901, 2902):
            name = f"{kind}-{seed}"
            folder = RUN / name
            completed = folder / "step-0420.pt"
            existing = sorted(folder.glob("step-*.pt")) if folder.exists() else []
            if not completed.exists():
                losses = fit(owner, train_features, labels, folder, kind, seed, resume=existing[-1] if existing else None)
            else:
                losses = []
            saved = torch.load(completed, weights_only=True, map_location="cpu")
            if saved["contracts"] != contracts():
                raise ValueError("Completed fit source changed")
            owner.reading_config = copy.deepcopy(saved["config"])
            apply(owner, saved["delta"])
            with torch.no_grad():
                result = metrics(dev_rows, owner.reading_logits(*dev_features))
            result.update(name=name, checkpoint=str(completed.relative_to(ROOT)).replace("\\", "/"),
                          sha256=sha(completed), updates=420, last_new_loss=losses[-1] if losses else None)
            write(OUT / f"dev-{name}.json", result)
            candidates.append({k:result[k] for k in ("name", "checkpoint", "sha256", "accuracy", "article_macro_accuracy")})
            print(json.dumps({"completed":name,"dev_accuracy":result["accuracy"]}), flush=True)
    if any(not torch.equal(v, owner.state_dict()[n]) for n, v in old.items()):
        raise AssertionError("Predecessor weights changed")
    selected = max(candidates, key=lambda r: (r["accuracy"], r["name"]))
    # A tie favors the supplied baseline. The best fitted reader remains archived.
    mode = selected["name"] if selected["accuracy"] > baseline["accuracy"] else "bm25"
    write(OUT / "selection.json", {"contracts": contracts(), "selected": mode, "best_trained": selected,
                                    "candidates": candidates, "baseline_accuracy": baseline["accuracy"],
                                    "parent_owner": before, "preserved_tensors": len(old),
                                    "wall_seconds": time.perf_counter()-start, "final_labels_opened": False})


def evaluate():
    if (OUT / "final.json").exists():
        raise ValueError("Completed final evaluation must not be repeated or tuned")
    selection = read(OUT / "selection.json")
    if selection["contracts"] != contracts():
        raise ValueError("Selection source changed")
    if not (RUN / "final.json").exists():
        prepare(final=True)
    rows = read(RUN / "final.json")
    session = parent()
    owner = load_curriculum(ReadingR1.attach(session.owner))
    features = cache_features(owner, "final", rows)
    results = {"bm25": metrics(rows, (4 * features[0][..., 1]).masked_fill(~features[1], -1e9))}
    for item in selection["candidates"]:
        path = ROOT / item["checkpoint"]
        if sha(path) != item["sha256"]:
            raise ValueError("Selected checkpoint bytes changed")
        saved = torch.load(path, weights_only=True, map_location="cpu")
        owner.reading_config = saved["config"]
        apply(owner, saved["delta"])
        with torch.no_grad():
            results[item["name"]] = metrics(rows, owner.reading_logits(*features))
    chosen = results[selection["selected"]]
    write(OUT / "final.json", {"contracts": contracts(), "selection_sha256": sha(OUT / "selection.json"),
                              "results": results, "selected": selection["selected"],
                              "accuracy_difference_from_baseline": chosen["accuracy"]-results["bm25"]["accuracy"],
                              "numerical_admission": chosen["accuracy"] >= results["bm25"]["accuracy"]-.02,
                              "retention_and_persistence": "requires independent audit"})
    print(json.dumps({k:v["accuracy"] for k,v in results.items()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    (train if args.action == "train" else evaluate)()


if __name__ == "__main__":
    main()
