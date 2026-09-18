"""Whole-group partitions of attributed human practice; old finals stay closed."""

import numpy as np
import torch

from experiments.quest_portfolio.methods import SPECS, contexts, features, screen

from .common import ROOT, RUN, SKILLS, digest, read, sha, write


def group_indices(rows, key, count, bucket, limit):
    groups = sorted({r[key] for r in rows}, key=lambda g: digest(["LP-001", g]))
    allowed = {g for i, g in enumerate(groups) if i % count == bucket}
    indices = sorted((i for i, r in enumerate(rows) if r[key] in allowed),
                     key=lambda i: digest(["LP-row", rows[i]["id"]]))[:limit]
    if len(indices) < 8:
        raise ValueError("Insufficient independently grouped human practice")
    return indices


def subset(x, ids):
    return {k: v[ids].clone() for k, v in x.items()}


def method_rows(seed, n):
    rng = np.random.default_rng(seed)
    weights = contexts(n, seed, shifted=seed % 2 == 0)
    covered = (rng.uniform(size=(n, 4)) < .4).astype(float)
    visits = rng.integers(0, 3, size=(n, 7))
    values = features(weights, covered, visits, 2)
    profiles, wrong = screen(34000)
    if not all(wrong[i] for i in (4, 5)):
        raise AssertionError("Independent defective-method screen changed")
    gains = ((weights*(1-covered)) @ profiles.T) - .01*np.array([s["cost"] for s in SPECS])
    gains = np.column_stack((gains, np.zeros(n)))
    return {"x": torch.tensor(values), "y": torch.tensor(gains.argmax(1))}


def prepare():
    if (RUN / "data.pt").exists():
        raise FileExistsError("Prepared curriculum is immutable")
    RUN.mkdir(parents=True, exist_ok=True)
    book = ROOT / "runs/HB-study-002"
    reader = ROOT / "runs/HR-study-001"
    vocabulary = read(book / "vocabulary.json")["words"]
    index = {w: i for i, w in enumerate(vocabulary)}
    pools = {k: {} for k in ("teach1", "teach2", "future1", "future2", "retention", "probe", "final1", "final2")}
    provenance, groups = {}, {}
    for skill in SKILLS[:-1]:
        for split in ("train", "dev"):
            if skill == "reading":
                path = reader / (split+".json")
                cache = reader / ("features-"+split+".pt")
            else:
                path = book / (skill+"-"+split+".json")
                cache = book / ("features-"+skill+"-"+split+".pt")
            rows = read(path)
            saved = torch.load(cache, weights_only=True, map_location="cpu")
            if saved["identity"]["rows"] != sha(path):
                raise ValueError("Precomputed human features lost source identity")
            if skill == "reading":
                values = {"x": saved["x"], "mask": saved["mask"],
                          "y": torch.tensor([r["gold"][0] for r in rows])}
                # Preserve every supplied correct sentence for independent scoring.
                gold = torch.zeros_like(saved["mask"])
                for i, row in enumerate(rows):
                    gold[i, row["gold"]] = True
                values["gold"] = gold
                key = "title"
            else:
                values = {"x": saved["features"], "y": torch.tensor([index.get(r["target"], 0) for r in rows])}
                key = "group"
            names = list(pools)[:5] if split == "train" else list(pools)[5:]
            for bucket, name in enumerate(names):
                ids = group_indices(rows, key, len(names), bucket, 64 if name != "probe" else 16)
                pools[name][skill] = subset(values, ids)
                groups[name+":"+skill] = {"groups": sorted({rows[i][key] for i in ids}),
                    "ids": [rows[i]["id"] for i in ids], "source": str(path.relative_to(ROOT)),
                    "count": len(ids), "historical_parent_exposure": split}
            provenance[skill+":"+split] = {"rows": sha(path), "features": sha(cache), "path": str(path.relative_to(ROOT))}
    for i, name in enumerate(pools):
        pools[name]["methods"] = method_rows(34100+i, 16 if name == "probe" else 64)
    manifest = {"sources": provenance, "partitions": groups, "skills": list(SKILLS),
                "human_text": True, "mechanics": "supplied conditional checked methods",
                "old_final_access": False, "parent_exposure": "training/development corpora previously used; partition separation is within LP-001"}
    torch.save(pools, RUN / "data.pt")
    manifest["tensor_sha256"] = sha(RUN / "data.pt")
    write(RUN / "data-manifest.json", manifest)
    return manifest


def load_data():
    manifest = read(RUN / "data-manifest.json")
    if sha(RUN / "data.pt") != manifest["tensor_sha256"]:
        raise ValueError("Changed curriculum tensors")
    return torch.load(RUN / "data.pt", weights_only=True, map_location="cpu")
