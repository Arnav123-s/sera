"""Actual-owner interruption equivalence and untrusted input regressions."""
import hashlib
import json

import pytest

from experiments.self_study.algebra import evaluate, vector
from experiments.self_study.lessons import calculus
from experiments.self_study.runtime import OUT, StudySession, admitted_source, request_lineage


def test_resume_preserves_acquisition_order_and_owner():
    parent = json.loads((OUT / "parent-request.json").read_text())
    lessons = calculus(json.loads((OUT / "calculus-source.json").read_text()))
    complete = StudySession(parent)
    complete.autonomous_round("resume-probe", "integral", [1, 2, 3, 4, 5, 6], None,
                              lessons=lessons)
    interrupted = StudySession(parent)
    saved = []

    def stop_after_two(value):
        saved.append(value)
        if len(value["events"]) == 2:
            raise InterruptedError("simulated process interruption after committed lesson")

    with pytest.raises(InterruptedError):
        interrupted.autonomous_round("resume-probe", "integral", [1, 2, 3, 4, 5, 6],
                                      None, lessons=lessons, persist=stop_after_two)
    resumed = StudySession(parent, saved[-1])
    resumed.autonomous_round("resume-probe", "integral", [1, 2, 3, 4, 5, 6], None,
                             lessons=lessons)
    expected, actual = complete.snapshot(), resumed.snapshot()
    assert actual["goals"] == expected["goals"]
    assert actual["events"] == expected["events"]
    assert actual["weights"] == expected["weights"]
    assert actual["owner"] == expected["owner"]
    # The original task remains immutable, including after successful completion.
    with pytest.raises(ValueError, match="immutable"):
        resumed.autonomous_round("resume-probe", "integral", [2], None, lessons=lessons)


@pytest.mark.parametrize("value", ["1e20", "1.25", "9" * 1000])
def test_reject_noncontract_coefficient_syntax_before_conversion(value):
    with pytest.raises(ValueError):
        vector([value])
    with pytest.raises(ValueError):
        evaluate([1], value)


def test_cache_cannot_redirect_body_outside_content_address(tmp_path, monkeypatch):
    import experiments.self_study.sources as module
    monkeypatch.setattr(module, "ROOT", tmp_path)
    client = module.Arxiv(tmp_path / "runs/cache")
    url = "https://arxiv.org/html/math/9207222v1"
    raw = b"a separate local fixture"
    (tmp_path / "runs/separate.txt").write_bytes(raw)
    record = client.directory / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    record.write_text(json.dumps({"url": url, "body": "../separate.txt",
                                  "sha256": hashlib.sha256(raw).hexdigest()}))
    with pytest.raises(ValueError, match="Cached"):
        client.fetch(url)


def test_request_counter_separates_rehearsal_from_new_languages():
    record = request_lineage(5400, 11514, {"step": 2000, "extension": {},
                            "seen_ids": ["1", "2", "es-ES:1", "fr-FR:1", "de-DE:2"]})
    assert record["checkpoint_seen_ids_scope"] == "rehearsal_extension"
    assert record["unique_ids_in_checkpoint"] == 5
    assert record["new_language_ids_in_checkpoint"] == 3
    assert record["english_rehearsal_ids_in_checkpoint"] == 2
    assert record["other_ids_in_checkpoint"] == 0
    assert record["continuation_updates"] == 2000


def test_source_migration_is_explicit_not_a_general_bypass():
    assert admitted_source("72a3de9623c2fd547246f72225e72981a70504a83283b2f8413e481b2cb01a6f")
    assert admitted_source("8b1526987b51e26cbce4d6df0f73ba98400462735e6a3d643df3c8cd52c353d5")
    assert not admitted_source("unreviewed-source-change")
