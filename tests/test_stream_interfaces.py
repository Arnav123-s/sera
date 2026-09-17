"""Actual-owner request inference, annotation scoring and source mixing contracts."""

import copy
import json

import pytest
import torch

from experiments.stream_curriculum.audit import independent_spans
from experiments.stream_curriculum.data import PREPARED, ROOT, parse, read
from experiments.stream_curriculum.evaluate import summarize
from experiments.stream_curriculum.model import batch
from experiments.stream_curriculum.runtime import RequestSession
from experiments.stream_curriculum.study import spans
from workbench.storage import Store


@pytest.fixture(scope="module")
def session():
    saved = Store(ROOT / "runs/sera-requests").read()
    if saved is None:
        pytest.skip("Prepared local request learner required")
    return RequestSession(saved["checkpoint"], saved)


def test_actual_owner_and_fact_views_are_current(session):
    learner = session.base.base.session.learner
    assert session.owner is learner.session.owner is learner.solver.neural.owner
    assert session.owner is learner.solver.components["typed"].owner
    assert learner.legacy.snapshot()["payload"]
    assert learner.library.record()


def test_only_text_is_encoded_for_prediction():
    raw = {"id": "source", "partition": "train", "scenario": "alarm", "intent": "alarm_set",
           "utt": "alarm at nine am", "annot_utt": "alarm at [time : nine am]"}
    one = parse(raw)
    two = {**copy.deepcopy(one), "intent": "weather_query", "tags": ["O"] * 4}
    vocabulary = read(PREPARED / "vocabulary.json")
    a, _, _ = batch([one], vocabulary)
    b, _, _ = batch([two], vocabulary)
    assert torch.equal(a, b)


@pytest.mark.parametrize("tags", [["O", "B-time", "I-time", "O"], ["I-time", "I-time"],
                                  ["B-time", "B-time"], ["B-date", "I-time"], ["O"], []])
def test_independent_span_counter_including_broken_bio(tags):
    assert spans(tags) == independent_spans(tags)


def test_request_is_recorded_as_interpretation(session):
    count = len(session.base.base.session.learner.session.events)
    value = session.ask("new-test-request", "set an alarm for nine am")
    assert value["intent"] == "alarm_set"
    assert value["actions_executed"] == 0
    assert value["evidence_kind"] == "learned_interpretation_of_user_request"
    assert len(session.base.base.session.learner.session.events) == count
    with pytest.raises(ValueError):
        session.ask("new-test-request", "different text")


def test_old_conditional_motion_interface_uses_the_same_owner(session):
    value = session.base.ask("new-test-motion", "can orbit reach beacon")
    assert value["frame"]["actor"] == "orbit" and value["frame"]["target"] == "beacon"
    assert value["evidence_role"] == "conditional_imagination"


def test_metrics_keep_duplicate_texts_visible():
    records = [{"id": "1", "scenario": "alarm", "text": "known", "predicted_intent": "alarm_set",
                "target_intent": "alarm_set", "predicted_tags": ["O"], "target_tags": ["O"]},
               {"id": "2", "scenario": "alarm", "text": "novel", "predicted_intent": "weather_query",
                "target_intent": "alarm_set", "predicted_tags": ["O"], "target_tags": ["B-time"]}]
    result = summarize(records, {"known"})
    assert result["examples"] == 2 and result["intent_accuracy"] == .5
    assert result["novel_text"]["examples"] == 1 and result["novel_text"]["intent_accuracy"] == 0.


def test_prepared_training_has_no_dev_or_final_ids():
    source = ROOT / "research/intake/massive-1.1-en-US-20260916"
    train = {json.loads(line)["id"] for line in (PREPARED / "train.jsonl").read_text(encoding="utf-8").splitlines()}
    dev = {json.loads(line)["id"] for line in (source / "dev.jsonl").read_text(encoding="utf-8").splitlines()}
    assert len(train) == 11514 and train.isdisjoint(dev)
    # The final partition is intentionally never opened by these development tests.
