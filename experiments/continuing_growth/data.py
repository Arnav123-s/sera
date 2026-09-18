"""Only retained human training records and explicitly conditional imagination."""

import json

import torch

from experiments.learning_progress.data import method_rows
from experiments.stream_curriculum.model import batch

from .common import BOOKS, LANGUAGES, ROOT, RUN, digest, read, sha, write


def partition(groups):
    ordered = sorted(set(groups), key=lambda g: digest(["CG-001-groups", g]))
    # Translation variants use the same underlying source ID across languages.
    # Assignment is by the ID hash, not its rank within a locale-specific subset.
    return {g: ({0: "probe", 1: "final", 2: "final", 3: "future_train", 4: "future_final"}.get(
                int(digest(["CG-001-split", g])[:8], 16) % 10, "train"))
            for g in ordered}


def take(values, ids):
    return {k: v[ids].clone() for k, v in values.items()}


@torch.no_grad()
def request_features(owner, rows):
    saved = []
    def capture(module, inputs):
        saved.append(inputs[0].detach().clone())
    hook = owner.stream_intent.register_forward_pre_hook(capture)
    try:
        for i in range(0, len(rows), 32):
            ids, _, _ = batch(rows[i:i+32], owner.stream_config["vocabulary"])
            owner.request_logits(ids)
    finally:
        hook.remove()
    return torch.cat(saved)


def prepare(owner):
    if (RUN / "data-manifest.json").exists():
        return read(RUN / "data-manifest.json")
    pools = {k: {} for k in ("train", "probe", "final", "future_train", "future_final")}
    manifest = {"sources": {}, "partitions": {}, "old_final_access": False,
                "source_scope": "retained human training corpora; previous parent exposure is acknowledged",
                "conditional_scope": "methods are supplied mechanics programs, not new factual observations"}
    book = ROOT / "runs/HB-study-002"
    vocabulary = read(book / "vocabulary.json")["words"]
    words = {w: i for i, w in enumerate(vocabulary)}
    metadata = {}
    for skill in (*BOOKS, "reading", *LANGUAGES):
        if skill in BOOKS:
            path = book / (skill+"-train.json")
            cache = book / ("features-"+skill+"-train.pt")
            rows = read(path)
            saved = torch.load(cache, weights_only=True)
            if saved["identity"]["rows"] != sha(path):
                raise ValueError("Changed attributed human features")
            values = {"x": saved["features"], "y": torch.tensor([words.get(r["target"], 0) for r in rows])}
            groups = [r["group"] for r in rows]
        elif skill == "reading":
            path = ROOT / "runs/HR-study-001/train.json"
            cache = path.with_name("features-train.pt")
            rows = read(path)
            saved = torch.load(cache, weights_only=True)
            if saved["identity"]["rows"] != sha(path):
                raise ValueError("Changed attributed sentence features")
            gold = torch.zeros_like(saved["mask"])
            for i, row in enumerate(rows):
                gold[i, row["gold"]] = True
            values = {"x": saved["x"], "mask": saved["mask"], "gold": gold,
                      "y": torch.tensor([r["gold"][0] for r in rows])}
            groups = [r["title"] for r in rows]
        else:
            path = ROOT / ("runs/SC-data-002/train.jsonl" if skill == "en-US" else
                           "runs/SS-language-data/"+skill+"-train.jsonl")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            groups = ["request:"+str(r.get("source_id", r["id"])) for r in rows]
            values, cache = None, None
        assignment = partition(groups)
        metadata[skill] = {}
        for name in pools:
            limit = {"train": 1024, "probe": 48, "final": 192, "future_train": 128, "future_final": 96}[name]
            ids = sorted((i for i, g in enumerate(groups) if assignment[g] == name),
                         key=lambda i: digest(["CG-001-row", rows[i]["id"]]))[:limit]
            if len(ids) < 12:
                raise ValueError("Insufficient independent groups for "+skill+":"+name)
            if values is None:
                selected = [rows[i] for i in ids]
                pools[name][skill] = {"x": request_features(owner, selected), "y": torch.tensor([
                    owner.stream_config["vocabulary"]["intents"].index(r["intent"]) for r in selected])}
            else:
                pools[name][skill] = take(values, ids)
            manifest["partitions"][name+":"+skill] = {"groups": sorted({groups[i] for i in ids}),
                "ids": [rows[i]["id"] for i in ids], "count": len(ids), "source": path.relative_to(ROOT).as_posix()}
            metadata[skill][name] = [rows[i] for i in ids]
        manifest["sources"][skill] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                                     "features_sha256": sha(cache) if cache else None}
    for i, name in enumerate(pools):
        pools[name]["methods"] = method_rows(38100+i, {"train": 1024, "probe": 48, "final": 192,
                                                     "future_train": 128, "future_final": 96}[name])
    # These are independent frozen cohorts within this continuation, not claims
    # that its inherited owner has never seen any of the underlying sources.
    for name, value in pools.items():
        path = RUN / (name+".pt")
        if path.exists():
            raise FileExistsError("Preserve partial preparation")
        torch.save(value, path)
        manifest[name+"_sha256"] = sha(path)
    write(RUN / "metadata.json", metadata)
    manifest["metadata_sha256"] = sha(RUN / "metadata.json")
    write(RUN / "data-manifest.json", manifest)
    return manifest


def load(name):
    manifest = read(RUN / "data-manifest.json")
    path = RUN / (name+".pt")
    if sha(path) != manifest[name+"_sha256"]:
        raise ValueError("Changed frozen continuation cohort")
    return torch.load(path, weights_only=True)
