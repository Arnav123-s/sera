"""Frozen ordered-prose curriculum, lexical controls and retained stages."""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.human_reading.data import ROOT, read, sha, write
from experiments.human_reading.runtime import ReadingSession

from .data import DOMAINS, OUT, RUN, prepare
from .model import BookR1, apply, delta


def contracts():
    return {"protocol": sha(OUT / "PROTOCOL.md"), "preparation_repair": sha(OUT / "failures/HB-F01/DIAGNOSIS.md"),
            "data": sha(Path(__file__).with_name("data.py")), "model": sha(Path(__file__).with_name("model.py")),
            "study": sha(Path(__file__)), "manifest": sha(RUN / "manifest.json"),
            "parent": sha(ROOT / "research-continuation/29_human_reading/release-manifest.json")}


def rows(domain, split):
    manifest = read(RUN / ("final-manifest.json" if split == "final" else "manifest.json"))
    path = RUN / f"{domain}-{split}.json"
    if sha(path) != manifest["hashes"][domain+":"+split]:
        raise ValueError("Human book partition changed")
    return read(path)


def vocabulary():
    path = RUN / "vocabulary.json"
    if path.exists():
        return read(path)
    count, targets = Counter(), Counter()
    for domain in DOMAINS:
        for row in rows(domain, "train"):
            count.update(row["context"]+[row["target"]])
            targets[row["target"]] += 1
    words = ["<unknown>"]+[w for w, _ in sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))[:2047]]
    index = {word: i for i, word in enumerate(words)}
    frequency = [1.] * len(words)
    for word, n in targets.items():
        frequency[index.get(word, 0)] += n
    result = {"words": words, "counts": frequency, "contracts": contracts()}
    write(path, result)
    return result


def features(owner, domain, split):
    path = RUN / f"features-{domain}-{split}.pt"
    data = rows(domain, split)
    identity = {"contracts": contracts(), "rows": sha(RUN / f"{domain}-{split}.json")}
    if path.exists():
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved["identity"] != identity:
            raise ValueError("Cached language features changed")
        return data, saved["features"]
    result = torch.cat([owner.book_features([r["context"] for r in data[start:start+64]])
                        for start in range(0, len(data), 64)])
    torch.save({"identity": identity, "features": result}, path)
    return data, result


def labels(data, vocabulary):
    index = {word: i for i, word in enumerate(vocabulary)}
    return torch.tensor([index.get(row["target"], 0) for row in data])


def metric(logits, target):
    prediction = logits.argmax(-1)
    covered = target.ne(0)
    loss = F.cross_entropy(logits, target, reduction="none")
    return {"cross_entropy": float(loss.mean()), "covered_accuracy": float(prediction[covered].eq(target[covered]).float().mean()),
            "unknown_rate": float((~covered).float().mean()), "tokens": len(target),
            "losses": loss.tolist(), "predictions": prediction.tolist(), "targets": target.tolist()}


def baseline(data, words):
    index = {w: i for i, w in enumerate(words)}
    counts = torch.ones(len(words))
    bigrams = defaultdict(Counter)
    for domain in DOMAINS:
        for row in rows(domain, "train"):
            target = index.get(row["target"], 0)
            counts[target] += 1
            bigrams[row["context"][-1]][target] += 1
    unigram = (counts/counts.sum()).log().expand(len(data), -1)
    probability = []
    for row in data:
        frequency = counts/counts.sum()
        following = bigrams[row["context"][-1]]
        for key, value in following.items():
            frequency[key] += value
        probability.append((frequency/frequency.sum()).log())
    return {"unigram": unigram, "bigram": torch.stack(probability)}


def train():
    if (OUT / "selection.json").exists():
        raise FileExistsError("Completed book curriculum must not restart")
    started = time.perf_counter()
    torch.manual_seed(3011)
    session = ReadingSession()
    old = {k: v.clone() for k, v in session.owner.state_dict().items()}
    vocab = vocabulary()
    if vocab["contracts"] != contracts():
        raise ValueError("Vocabulary was built from different source bytes")
    counts = torch.tensor(vocab["counts"])
    owner = BookR1.attach(session.owner, vocab["words"], (counts/counts.sum()).log())
    train_data = {d: features(owner, d, "train") for d in DOMAINS}
    dev_data = {d: features(owner, d, "dev") for d in DOMAINS}
    all_features = torch.cat([f for _, f in train_data.values()])
    owner.book_mean.copy_(all_features.mean(0))
    owner.book_scale.copy_(all_features.std(0).clamp_min(.01))
    optimizer = torch.optim.AdamW(owner.book_heads.parameters(), lr=.003, weight_decay=.0001)
    generator = torch.Generator().manual_seed(3011)
    history = []
    for domain in DOMAINS:
        data, x = train_data[domain]
        target = labels(data, vocab["words"])
        folder = RUN / domain
        folder.mkdir(exist_ok=True)
        order, cursor, first = torch.randperm(len(data), generator=generator), 0, 1
        checkpoints = sorted(folder.glob("step-*.pt"))
        if checkpoints:
            saved = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
            if saved["contracts"] != contracts():
                raise ValueError("Saved language curriculum changed")
            apply(owner, saved["delta"])
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["rng"])
            order, cursor, first = saved["order"], saved["cursor"], saved["step"]+1
        for step in range(first, 121):
            if cursor == len(order):
                order, cursor = torch.randperm(len(data), generator=generator), 0
            ids = order[cursor:cursor+64]
            cursor += len(ids)
            optimizer.zero_grad(set_to_none=True)
            loss = sum(F.cross_entropy(owner.book_logits(x[ids], kind), target[ids]) for kind in ("shared", "pooled"))
            loss.backward()
            optimizer.step()
            if step % 40 == 0:
                torch.save({"schema": "sera.human-books.1", "contracts": contracts(), "stage": domain, "step": step,
                            "delta": delta(owner), "optimizer": optimizer.state_dict(), "order": order, "cursor": cursor,
                            "rng": generator.get_state(), "vocabulary": vocab["words"]}, folder / f"step-{step:04d}.pt")
        with torch.no_grad():
            result = {d: {k: metric(owner.book_logits(f, k), labels(r, vocab["words"])) for k in ("shared", "pooled")}
                      for d, (r, f) in dev_data.items()}
        write(OUT / f"dev-after-{domain}.json", result)
        means = {k: sum(r[k]["cross_entropy"] for r in result.values())/len(DOMAINS) for k in ("shared", "pooled")}
        history.append({"stage": domain, "mean_development_cross_entropy": means})
        print(json.dumps(history[-1]), flush=True)
    selected = min(history[-1]["mean_development_cross_entropy"], key=history[-1]["mean_development_cross_entropy"].get)
    path = RUN / DOMAINS[-1] / "step-0120.pt"
    assert all(torch.equal(v, owner.state_dict()[k]) for k, v in old.items())
    write(OUT / "selection.json", {"contracts": contracts(), "selected": selected, "checkpoint": path.relative_to(ROOT).as_posix(),
                                   "sha256": sha(path), "history": history, "predecessor_tensors_exact": len(old),
                                   "wall_seconds": time.perf_counter()-started, "final_opened": False})


def evaluate():
    if (OUT / "final.json").exists():
        raise FileExistsError("Completed book final must remain closed")
    selection = read(OUT / "selection.json")
    if selection["contracts"] != contracts():
        raise ValueError("Selected source contract changed")
    if not (RUN / "final-manifest.json").exists():
        prepare(final=True)
    vocab = vocabulary()
    counts = torch.tensor(vocab["counts"])
    owner = BookR1.attach(ReadingSession().owner, vocab["words"], (counts/counts.sum()).log())
    path = ROOT / selection["checkpoint"]
    if sha(path) != selection["sha256"]:
        raise ValueError("Selected weights changed")
    apply(owner, torch.load(path, weights_only=True, map_location="cpu")["delta"])
    results = {}
    with torch.no_grad():
        for domain in DOMAINS:
            data, x = features(owner, domain, "final")
            target = labels(data, vocab["words"])
            scores = {**baseline(data, vocab["words"]), **{k: owner.book_logits(x, k) for k in ("shared", "pooled")}}
            results[domain] = {k: metric(v, target) for k, v in scores.items()}
            print(json.dumps({"domain": domain, "cross_entropy": {k: r["cross_entropy"] for k, r in results[domain].items()}}), flush=True)
    write(OUT / "final.json", {"selection": sha(OUT / "selection.json"), "results": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    (train if args.action == "train" else evaluate)()
