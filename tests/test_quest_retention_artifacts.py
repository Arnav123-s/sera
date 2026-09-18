import hashlib
import json
import shutil

import pytest

from scripts.quest_retention_artifacts import MANIFEST, ROOT, restore


def test_retention_restores_exact_inputs_in_empty_checkout_and_preserves_conflicts(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for archive in manifest["archives"]:
        target = tmp_path / archive["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / archive["path"], target)
    assert restore(tmp_path) == {"restored": 5, "already_identical": 0, "models_changed": 0, "experiments_restarted": 0}
    for archive in manifest["archives"]:
        for entry in archive["files"]:
            assert hashlib.sha256((tmp_path / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert restore(tmp_path)["already_identical"] == 5
    changed = tmp_path / "runs/SC-data-002/dev.jsonl"
    changed.write_text("local work must survive", encoding="utf-8")
    with pytest.raises(ValueError, match="Preserve divergent local work"):
        restore(tmp_path)
    assert changed.read_text() == "local work must survive"
