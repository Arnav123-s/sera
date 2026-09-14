"""Exact parent-referenced checkpoints without repeating unchanged tensors."""

import hashlib
import os
from pathlib import Path

import torch

from sera.connected import restore_component


def checkpoint_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_shared_checkpoint(path, model, *, parent=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Shared experiment checkpoints are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    payload = {"schema_version": 1, "config": model.export_config()}
    if parent is not None:
        original = load_shared_checkpoint(parent).state_dict()
        payload["parent"] = {"path": Path(os.path.relpath(Path(parent).resolve(), path.parent.resolve())).as_posix(),
                             "sha256": checkpoint_hash(parent)}
        payload["removed"] = [name for name in original if name not in state]
        state = {name: tensor for name, tensor in state.items()
                 if name not in original or tensor.shape != original[name].shape or not torch.equal(tensor, original[name])}
    payload["state"] = {name: tensor.detach().cpu().clone() for name, tensor in state.items()}
    torch.save(payload, path)
    return {"path": path.name, "sha256": checkpoint_hash(path), "bytes": path.stat().st_size,
            "stored_tensors": len(state), "parent": payload.get("parent")}


def load_shared_checkpoint(path, *, depth=0):
    path = Path(path)
    if depth > 8:
        raise ValueError("Checkpoint parent chain exceeds the declared bound")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("schema_version") != 1:
        raise ValueError("Unknown shared experiment checkpoint")
    state = {}
    if "parent" in payload:
        parent = path.parent / payload["parent"]["path"]
        if checkpoint_hash(parent) != payload["parent"]["sha256"]:
            raise ValueError("Shared checkpoint parent changed")
        state = load_shared_checkpoint(parent, depth=depth+1).state_dict()
        for name in payload["removed"]:
            state.pop(name)
    state.update(payload["state"])
    model = restore_component(payload["config"])
    model.load_state_dict(state, strict=True)
    return model.eval()
