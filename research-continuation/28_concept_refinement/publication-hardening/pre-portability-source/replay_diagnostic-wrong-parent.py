"""Measure platform replay differences without changing or admitting saved evidence."""

import json
import platform

import numpy as np
import torch

from experiments.constraint_inquiry.runtime import ConstraintRuntime
from experiments.language_inquiry.graph import advance, answer, interpret, language_identity, start
from experiments.language_inquiry.model import apply, delta
from experiments.language_inquiry.runtime import Runtime
from experiments.language_inquiry.study import ROOT, read
from sera.session_state import model_identity, unpack_tensors


def differences(expected, observed, path=""):
    if type(expected) is not type(observed):
        return [
            {
                "path": path,
                "kind": "type",
                "expected": str(type(expected)),
                "observed": str(type(observed)),
            }
        ]
    if isinstance(expected, dict):
        if expected.keys() != observed.keys():
            return [{"path": path, "kind": "keys"}]
        return [
            row
            for key in expected
            for row in differences(expected[key], observed[key], path + "/" + key)
        ]
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return [{"path": path, "kind": "length"}]
        return [
            row
            for i, (a, b) in enumerate(zip(expected, observed, strict=True))
            for row in differences(a, b, path + "/" + str(i))
        ]
    if expected == observed:
        return []
    row = {
        "path": path,
        "kind": type(expected).__name__,
        "expected": expected,
        "observed": observed,
    }
    if type(expected) is float:
        row["absolute_difference"] = abs(expected - observed)
    return [row]


def main():
    torch.set_num_threads(1)
    saved = read(ROOT / "research-continuation/25_constraint_inquiry/parent.json")
    runtime = Runtime(saved["checkpoint"])
    apply(runtime.owner, unpack_tensors(saved["language_delta"], delta(runtime.owner)))
    runtime.owner.inquiry_admitted = list(saved["admitted"])
    for row in saved["observations"]:
        runtime._observe(row["action"], expected=row)
    assert model_identity(runtime.owner) == saved["owner"]
    graph = saved["graph"]["payload"]
    assert graph["parser"] == language_identity(runtime.owner)
    rows = []
    for job in [*graph["jobs"].values(), *graph["history"]]:
        frame = job["interpretation"].get("frame")
        if frame is not None and job["interpretation"]["parser"] == graph["parser"]:
            replay = interpret(runtime.owner, job["text"], job["entity"])
            if job["clarification"] is not None:
                replay["frame"]["actions"] = list(job["clarification"]["actions"])
            rows.extend(differences(job["interpretation"], replay, job["id"] + "/interpretation"))
        model = graph["models"][job["model"]]
        states = start(model)
        actions = [] if frame is None else frame["actions"]
        for action in actions[: job["cursor"]]:
            states = advance(model, states, action)
        rows.extend(differences(job["states"], states.tolist(), job["id"] + "/states"))
        if job["result"] is not None:
            rows.extend(
                differences(job["result"], answer(model, states, frame), job["id"] + "/result")
            )
    for key, cache in graph["cache"].items():
        model = graph["models"][cache["model"]]
        states = start(model)
        for action in cache["actions"]:
            states = advance(model, states, action)
        rows.extend(differences(cache["states"], states.tolist(), "cache/" + key))
    print(
        json.dumps(
            {
                "schema": "sera.platform-replay-diagnostic.1",
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "owner_byte_identity": saved["owner"],
                "difference_count": len(rows),
                "differences": rows,
                "stored_evidence_changed": False,
            },
            allow_nan=False,
        )
    )
    constraint_saved = read(ROOT / "research-continuation/26_stream_curriculum/parent.json")
    constraint = ConstraintRuntime(constraint_saved["checkpoint"])
    for row in constraint_saved["observations"]:
        constraint._admit(row)
    assert model_identity(constraint.owner) == constraint_saved["owner"]
    rows = []
    for branch in [*constraint_saved["jobs"].values(), *constraint_saved["history"]]:
        if "model" not in branch or branch["status"] == "stale":
            continue
        replay = constraint._new(branch["id"], branch["text"])
        for key in ("frame", "model", "initial"):
            rows.extend(differences(branch[key], replay[key], branch["id"] + "/" + key))
    print(json.dumps({"schema": "sera.constraint-replay-diagnostic.1",
                      "platform": platform.platform(), "owner_byte_identity": constraint_saved["owner"],
                      "difference_count": len(rows), "differences": rows,
                      "stored_evidence_changed": False}, allow_nan=False))


if __name__ == "__main__":
    main()
