"""A temporary Windows sharing collision must not abort numerical progress."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("module_name", ["run_counterfactual_persistent", "run_structural_persistent"])
def test_progress_replace_retries_a_transient_windows_handle(tmp_path, monkeypatch, module_name):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module(module_name)
    target = tmp_path / "state.json"
    target.write_text('{"status":"older"}')
    replace = Path.replace
    calls = []
    def delayed_replace(path, destination):
        calls.append(str(path))
        if len(calls) <= 3:
            raise PermissionError("Simulated Windows sharing collision")
        return replace(path, destination)
    monkeypatch.setattr(Path, "replace", delayed_replace)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    runner.write(target, {"status": "RUNNING", "charged_seconds": 41.})
    assert json.loads(target.read_text()) == {"status": "RUNNING", "charged_seconds": 41.}
    assert len(calls) == 4 and not target.with_suffix(".tmp").exists()
