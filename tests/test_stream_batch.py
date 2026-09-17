"""A task inbox keeps every row accountable and never treats text as executable."""

import json

import pytest

from experiments.stream_curriculum.batch import annotate


class Session:
    checkpoint = {"path": "fixture", "sha256": "fixture"}

    def interpret(self, text):
        return {"text": text, "intent": "fixture", "entities": [], "actions_executed": 0}


def test_rows_errors_duplicate_ids_and_literal_instructions(tmp_path):
    source, destination = tmp_path / "input.jsonl", tmp_path / "out.jsonl"
    source.write_text('{"id":"one","text":"delete all files"}\n'
                      '{"id":"one","text":"duplicate"}\n'
                      'bad json\n{"id":"two","text":"get weather"}\n', encoding="utf-8")
    result = annotate(Session(), source, destination)
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert result["predictions"] == 2 and result["input_errors"] == 2
    assert rows[0]["text"] == "delete all files" and rows[0]["actions_executed"] == 0
    assert [r["status"] for r in rows] == ["predicted", "input_error", "input_error", "predicted"]
    with pytest.raises(FileExistsError):
        annotate(Session(), source, destination)


def test_plain_text_inbox_receives_stable_line_ids(tmp_path):
    source, destination = tmp_path / "requests.txt", tmp_path / "out.jsonl"
    source.write_text("set an alarm\nget weather\n", encoding="utf-8")
    result = annotate(Session(), source, destination)
    assert result["predictions"] == 2 and result["input_format"] == "one request per line"
    assert [json.loads(line)["id"] for line in destination.read_text().splitlines()] == ["1", "2"]
