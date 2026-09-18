import copy

import pytest

from experiments.published_receipts import PublishedVerifier, historical_replay, replay


def test_read_only_published_receipts_reject_changed_and_new_credit():
    verifier = PublishedVerifier()
    receipt = next(iter(verifier.receipts.values()))
    assert verifier("X:/unavailable/issuer", {"action": "verify", "receipts": [receipt]})["verified"] == 1
    changed = copy.deepcopy(receipt)
    changed["correct"] += 1
    with pytest.raises(ValueError, match="changed published"):
        verifier("X:/unavailable/issuer", {"action": "verify", "receipts": [changed]})
    with pytest.raises(ValueError, match="read-only"):
        verifier("X:/unavailable/issuer", {"action": "begin"})


def test_replay_rejects_prediction_corruption():
    import json
    import zipfile

    from experiments.quest_portfolio.common import ROOT
    with zipfile.ZipFile(ROOT / "research-continuation/33_capability_portfolio/research.zip") as archive:
        record = json.loads(archive.read(next(n for n in archive.namelist() if "/completed/" in n)))
    record["predictions"][0] = 987654321.
    with pytest.raises(ValueError, match="predictions changed"):
        replay(record)


def test_portable_restore_uses_no_private_issuer_process(monkeypatch):
    from experiments.quest_portfolio import runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("Portable replay attempted an external authority process")
    monkeypatch.setattr(runtime, "call_assessor", forbidden)
    with historical_replay() as verifier:
        receipt = next(iter(verifier.receipts.values()))
        assert runtime.call_assessor("D:/nonexistent", {"action": "verify", "receipts": [receipt]})["verified"] == 1
    assert runtime.call_assessor is forbidden
