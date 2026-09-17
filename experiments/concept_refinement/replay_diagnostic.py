"""Measure platform replay differences without changing or admitting saved evidence."""

import json
import platform
from unittest.mock import patch

import numpy as np
import torch

from experiments.constraint_inquiry.runtime import ConstraintRuntime
from experiments.constraint_inquiry.settling import step, summarize
from experiments.language_inquiry.graph import advance, answer, interpret, language_identity, start
from experiments.language_inquiry.model import apply, delta
from experiments.language_inquiry.runtime import Runtime
from experiments.language_inquiry.study import ROOT, read
from experiments.stream_curriculum.runtime import RequestSession
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
    print(json.dumps({"torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
                      "numpy_simd": np.__config__.CONFIG["SIMD Extensions"]}), flush=True)
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
    # Restore the actual corrected parent and its observations, then independently
    # inspect original branches outside the constructor's exact replay check.
    owner_record = {**constraint_saved, "jobs": {}, "history": []}
    constraint = ConstraintRuntime(constraint_saved["checkpoint"], owner_record)
    assert model_identity(constraint.owner) == constraint_saved["owner"]
    rows = []
    for branch in [*constraint_saved["jobs"].values(), *constraint_saved["history"]]:
        if "model" not in branch:
            continue
        if branch["status"] != "stale":
            replay = constraint._new(branch["id"], branch["text"])
            for key in ("frame", "model", "initial"):
                rows.extend(differences(branch[key], replay[key], branch["id"] + "/" + key))
        points = torch.tensor(branch["initial"], dtype=torch.float64)
        for _ in range(branch["steps"]):
            points = step(branch["model"], points)
        rows.extend(differences(branch["points"], points.tolist(), branch["id"] + "/points"))
        rows.extend(differences(branch["result"], summarize(branch["model"], points), branch["id"] + "/result"))
    print(json.dumps({"schema": "sera.constraint-replay-diagnostic.1",
                      "platform": platform.platform(), "owner_byte_identity": constraint_saved["owner"],
                      "difference_count": len(rows), "differences": rows,
                      "stored_evidence_changed": False}, allow_nan=False))
    request_saved = read(ROOT / "research-continuation/27_self_study/parent-request.json")
    # This diagnostic compares the unchanged owner directly. Branches are checked
    # above; they are not admitted or rewritten to make the request check run.
    with patch("experiments.stream_curriculum.runtime.load_parent", return_value=constraint):
        request = RequestSession(request_saved["checkpoint"])
    assert model_identity(request.owner) == request_saved["owner"]
    rows = []
    for key, frame in request_saved["requests"].items():
        rows.extend(differences(frame, request.interpret(frame["text"]), key))
    print(json.dumps({"schema": "sera.request-replay-diagnostic.1",
                      "platform": platform.platform(), "owner_byte_identity": request_saved["owner"],
                      "difference_count": len(rows), "differences": rows,
                      "stored_evidence_changed": False}, allow_nan=False))


if __name__ == "__main__":
    main()
