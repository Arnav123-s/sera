"""Source alignment and optimizer-stream isolation must hold before fitting."""

import json

import pytest

from experiments.stream_curriculum.data import RowStream, parse, token_id


def example(text, annotated=None, partition="train", identifier="a"):
    return {"id": identifier, "partition": partition, "scenario": "calendar", "intent": "calendar_set",
            "utt": text, "annot_utt": text if annotated is None else annotated}


def test_labels_reconstruct_multiword_slots_and_repeated_values():
    row = parse(example("meet sam and sam on friday", "meet [person : sam] and [person : sam] on [date : friday]"))
    assert row["tokens"] == ["meet", "sam", "and", "sam", "on", "friday"]
    assert row["tags"] == ["O", "B-person", "O", "B-person", "O", "B-date"]
    assert parse(example("in new york", "in [place : new york]"))["tags"] == ["O", "B-place", "I-place"]


@pytest.mark.parametrize("text, annotated", [("meet sam", "meet [person : bob]"),
                                            ("bookmark", "book[object : mark]"),
                                            ("hello", "[broken hello]"), ("", ""),
                                            (" ".join(["x"] * 41), None)])
def test_misaligned_or_overlength_rows_are_not_silently_relabelled(text, annotated):
    with pytest.raises(ValueError):
        parse(example(text, annotated))


def test_saved_buffer_and_cursor_repeat_exactly_across_end_of_file(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(parse(example("hello", identifier=str(i)))) + "\n" for i in range(11)))
    stream = RowStream(path, 2631, limit=9, buffer_size=4)
    first = stream.take(7)
    state = stream.snapshot()
    expected = stream.take(25)
    restored = RowStream(path, 2631, limit=9, buffer_size=4, state=state)
    assert restored.take(25) == expected
    assert restored.snapshot() == stream.snapshot()
    assert all(int(r["id"]) < 9 for r in first + expected)
    assert stream.epoch > 0


def test_changed_source_and_final_rows_are_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(parse(example("hello"))) + "\n")
    stream = RowStream(path, 4)
    saved = stream.snapshot()
    path.write_text(json.dumps(parse(example("hello", partition="test"))) + "\n")
    with pytest.raises(ValueError, match="Changed"):
        RowStream(path, 4, state=saved)
    with pytest.raises(ValueError, match="training"):
        RowStream(path, 4).take(1)


def test_hash_encoding_is_stable_case_insensitive_and_reserves_padding():
    assert token_id("Calendar") == token_id("calendar")
    assert all(1 <= token_id(w) < 8192 for w in ("hello", "calendar", "time", "naive"))
