"""Published fixture restoration must preserve local work and artifact identity."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/restore_test_artifacts.py"
    spec = importlib.util.spec_from_file_location("artifact_restore", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(module, "MANIFEST", manifest)
    archive = tmp_path / "research-continuation/fixtures.zip"
    archive.parent.mkdir()
    raw = b"preserved checkpoint witness"
    with zipfile.ZipFile(archive, "x") as output:
        output.writestr("witness", raw)
    row = {
        "path": "runs/example/witness.bin",
        "archive": archive.relative_to(tmp_path).as_posix(),
        "member": "witness",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    value = {
        "files": [row],
        "archives": {row["archive"]: hashlib.sha256(archive.read_bytes()).hexdigest()},
    }
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return module, value, raw


def test_restore_is_exact_and_idempotent(bundle, capsys):
    module, value, raw = bundle
    module.main()
    path = module.ROOT / value["files"][0]["path"]
    assert path.read_bytes() == raw
    original_time = path.stat().st_mtime_ns
    module.main()
    assert path.stat().st_mtime_ns == original_time
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "restored": 0,
        "already_identical": 1,
        "experiments_restarted": 0,
    }


def test_divergent_local_file_is_preserved(bundle):
    module, value, _ = bundle
    path = module.ROOT / value["files"][0]["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"newer uncommitted work")
    with pytest.raises(ValueError, match="Preserve divergent local work"):
        module.main()
    assert path.read_bytes() == b"newer uncommitted work"


@pytest.mark.parametrize("path", ["../outside", "/outside", "C:/outside", "runs/../src/x"])
def test_escape_is_rejected(bundle, path):
    module, value, _ = bundle
    value["files"][0]["path"] = path
    module.MANIFEST.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe artifact destination"):
        module.main()


@pytest.mark.parametrize("level", ["archive", "member"])
def test_changed_identity_is_rejected_before_write(bundle, level):
    module, value, _ = bundle
    if level == "archive":
        value["archives"][value["files"][0]["archive"]] = "0" * 64
    else:
        value["files"][0]["sha256"] = "0" * 64
    module.MANIFEST.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        module.main()
    assert not (module.ROOT / value["files"][0]["path"]).exists()
