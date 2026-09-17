"""Selection and one-time final admission are fixed before label access."""

import copy
import json

import pytest

from experiments.stream_curriculum import evaluate


def scores():
    return [{"checkpoint": {"path": name, "sha256": name, "fit": name},
             "frame_accuracy": frame, "intent_accuracy": intent, "slot_span_f1": slots}
            for name, frame, intent, slots in (("a", .3, .9, .4), ("b", .4, .71, .51))]


def test_frame_selection_and_secondary_qualification_are_distinct():
    rows = scores()
    assert evaluate.select(rows)["path"] == "b"
    assert evaluate.select(rows)["qualified_annotation_aid"]
    rows[1]["slot_span_f1"] = .49
    assert evaluate.select(rows)["path"] == "b"
    assert not evaluate.select(rows)["qualified_annotation_aid"]


def test_final_receipt_rejects_changed_selection_and_repeat_access(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    (tmp_path / "evaluation-protocol.md").write_text("fixture")
    rows = scores()
    models = [r["checkpoint"] for r in rows]
    development = {"partition": "dev", "scores": rows, "models": models, "selected": evaluate.select(rows)}
    source = tmp_path / "dev.json"
    source.write_text(json.dumps(development))
    changed = copy.deepcopy(development)
    changed["selected"] = evaluate.select(rows[:1])
    with pytest.raises(ValueError):
        evaluate.final_receipt(changed, models, [], tmp_path, source)
    assert not (tmp_path / "official-test-access.json").exists()
    assert evaluate.final_receipt(development, models, [], tmp_path, source) == development["selected"]
    with pytest.raises(FileExistsError):
        evaluate.final_receipt(development, models, [], tmp_path, source)
