from types import SimpleNamespace

import pytest

from scripts import continuing_use
from scripts.continuing_use import perform


def test_reading_requires_attribution_and_bounded_sentence_count():
    session = SimpleNamespace(owner=None)
    with pytest.raises(ValueError, match="attribution"):
        perform(session, None, {"kind": "read", "sentences": []})
    with pytest.raises(ValueError, match="sixteen"):
        perform(session, None, {"kind": "read", "source": "local-book", "sentences": [{}]*17})


def test_unknown_task_does_not_dispatch_arbitrary_actions():
    with pytest.raises(ValueError, match="Supported tasks"):
        perform(SimpleNamespace(owner=None), None, {"kind": "run_shell", "command": "untrusted"})


def test_latest_store_selection_preserves_unrelated_older_refinement(tmp_path, monkeypatch):
    for name in ("sera-refinement-live", "sera-growth-live"):
        path = tmp_path / "runs" / name
        path.mkdir(parents=True)
        (path / "current.json").write_text("{}")
    selected = []
    class CaptureStore:
        def __init__(self, path):
            selected.append(path)
        def read(self):
            return None
    monkeypatch.setattr(continuing_use, "ROOT", tmp_path)
    monkeypatch.setattr(continuing_use, "Store", CaptureStore)
    with pytest.raises(ValueError, match="No independently admitted"):
        continuing_use.restore()
    assert selected == [tmp_path / "runs/sera-growth-live"]
