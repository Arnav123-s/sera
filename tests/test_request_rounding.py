"""Exact executable request semantics with bounded float32 confidence replay."""

import copy
import json
import runpy
import shutil
from pathlib import Path

import pytest

from experiments.stream_curriculum.runtime import RequestSession, same_request_frame
from sera.session_state import model_identity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def record():
    return json.loads((ROOT / "research-continuation/27_self_study/parent-request.json").read_text())


def test_measured_one_float32_unit_preserves_semantics(record):
    original = record["requests"]["example-2"]
    replay = {**original, "intent_score": 0.9999946355819702}
    assert original["intent_score"] == 0.9999947547912598
    assert same_request_frame(replay, original)


@pytest.mark.parametrize("field,value", [("intent", "alien"), ("actions_executed", 1),
                                        ("evidence_kind", "observed_event"), ("text", "changed")])
def test_executable_or_evidence_change_is_rejected(record, field, value):
    original = record["requests"]["example-2"]
    assert not same_request_frame(original, {**original, field: value})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.01, 1.01, True, 0.99])
def test_invalid_or_excessive_confidence_change_is_rejected(record, score):
    original = record["requests"]["example-2"]
    assert not same_request_frame(original, {**original, "intent_score": score})


def test_request_owner_record_and_current_roundtrip_are_preserved(record):
    original = copy.deepcopy(record)
    runtime = RequestSession(record["checkpoint"], record)
    assert record == original
    assert runtime.requests == record["requests"]
    assert model_identity(runtime.owner) == record["owner"]
    assert len(runtime.replay_migrations) == 1
    saved = runtime.snapshot()
    restored = RequestSession(saved["checkpoint"], saved)
    assert restored.snapshot() == saved


def test_unreviewed_request_reader_does_not_admit_old_source(tmp_path):
    folder = ROOT / "experiments/stream_curriculum"
    for name in ("runtime.py", "compatibility.py"):
        shutil.copyfile(folder / name, tmp_path / name)
    assert runpy.run_path(str(tmp_path / "compatibility.py"))["REPLAY_SOURCES"]
    with (tmp_path / "runtime.py").open("a") as stream:
        stream.write("\n# Unreviewed source change\n")
    assert not runpy.run_path(str(tmp_path / "compatibility.py"))["REPLAY_SOURCES"]
