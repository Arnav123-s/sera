"""Retain exact branch execution while qualifying float32 neural proposals."""

import copy
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments.constraint_inquiry.runtime import (
    ConstraintRuntime,
    runtime_source,
    same_binding,
    same_conditional_result,
    same_starting_proposal,
)
from sera.session_state import model_identity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def record():
    return json.loads((ROOT / "research-continuation/26_stream_curriculum/parent.json").read_text())


def test_measured_platform_proposals_preserve_model_and_inputs(record):
    original = record["jobs"]["reach"]
    replay = copy.deepcopy(original)
    path = ROOT / "research-continuation/28_concept_refinement/publication-hardening"
    diagnostic = json.loads((path / "linux-constraint-diagnostic.json").read_text())
    for row in diagnostic["differences"]:
        if not row["path"].startswith("reach/"):
            continue
        keys = row["path"].split("/")[1:]
        target = replay
        for key in keys[:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        target[int(keys[-1])] = row["observed"]
    before = copy.deepcopy(replay)
    assert original["frame"] != replay["frame"]
    assert original["initial"] != replay["initial"]
    assert original["model"] == replay["model"]
    assert same_binding(replay["frame"], original["frame"])
    assert same_starting_proposal(replay["initial"], original["initial"])
    assert before == replay


@pytest.mark.parametrize("field,value", [("actor", "alien"), ("target", "orbit"), ("mode", 3)])
def test_changed_executable_binding_is_rejected(record, field, value):
    frame = record["jobs"]["reach"]["frame"]
    changed = {**frame, field: value}
    assert not same_binding(frame, changed)


@pytest.mark.parametrize("change", ["random", "jitter", "large", "nan", "shape"])
def test_changed_starting_construction_is_rejected(record, change):
    original = record["jobs"]["reach"]["initial"]
    points = np.asarray(original).copy()
    if change == "random":
        points[4, 0] += 1e-12
    elif change == "jitter":
        points[1, 0] += 1e-12
    elif change == "large":
        points[:4, 0] += 4 * np.finfo(np.float32).eps
    elif change == "nan":
        points[0, 0] = float("nan")
    else:
        points = points[:7]
    assert not same_starting_proposal(original, points.tolist())


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 2.0, True])
def test_invalid_binding_scores_are_rejected(record, score):
    frame = record["jobs"]["reach"]["frame"]
    changed = copy.deepcopy(frame)
    changed["scores"][0][0] = score
    assert not same_binding(frame, changed)


def test_unreviewed_reader_cannot_admit_predecessor(tmp_path):
    folder = ROOT / "experiments/constraint_inquiry"
    for name in ("compatibility.py", "runtime.py"):
        shutil.copyfile(folder / name, tmp_path / name)
    assert runpy.run_path(str(tmp_path / "compatibility.py"))["REPLAY_SOURCES"]
    with (tmp_path / "runtime.py").open("a") as stream:
        stream.write("\n# Unreviewed source change\n")
    assert not runpy.run_path(str(tmp_path / "compatibility.py"))["REPLAY_SOURCES"]


def test_migration_keeps_original_record_owner_and_unfinished_progress(record):
    original = copy.deepcopy(record)
    runtime = ConstraintRuntime(record["checkpoint"], record)
    assert record == original
    assert model_identity(runtime.owner) == record["owner"]
    for name, branch in record["jobs"].items():
        for key in ("steps", "status", "initial", "points", "result", "frame", "model"):
            assert runtime.jobs[name].get(key) == branch.get(key)
    saved = runtime.snapshot()
    assert saved["source"] == runtime_source()
    assert len(saved["replay_migrations"]) == 1
    restored = ConstraintRuntime(record["checkpoint"], saved)
    assert restored.jobs == runtime.jobs
    assert restored.replay_migrations == runtime.replay_migrations


def test_measured_archived_energy_last_bit_preserves_full_result(record):
    original = next(row["result"] for row in record["history"] if row["id"] == "wording")
    assert original["energy"] == 0.0008018090864610274
    replay = {**original, "energy": 0.0008018090864610273}
    assert same_conditional_result(original, replay)


@pytest.mark.parametrize("field,value", [("status", "different"), ("occurrence", "ESTABLISHED"),
                                        ("nominal_distance_m", -1.0), ("controls", [0., 0.]),
                                        ("energy", float("nan")), ("energy", float("inf")),
                                        ("energy", -.0001), ("energy", .0009)])
def test_changed_conditional_evidence_or_excessive_energy_is_rejected(record, field, value):
    original = record["jobs"]["reach"]["result"]
    assert not same_conditional_result(original, {**original, field: value})
