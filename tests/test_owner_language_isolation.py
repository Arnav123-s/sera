"""Tests for laboratory import isolation, lineage integrity and supervision.

These run without restoring the owner, so they stay cheap enough for the ordinary
test pass. The expensive fresh-process restore is exercised by
``scripts/owner_language_baseline.py`` under the supervisor.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

from experiments.owner_language import isolation  # noqa: E402
from experiments.owner_language import labstore  # noqa: E402

SCRATCH = LAB_ROOT / "runs/owner-learning-001/test-scratch"


@pytest.fixture
def scratch():
    """A laboratory-owned scratch directory; see the supervision tests."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=SCRATCH))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_lab_root_is_this_checkout():
    assert isolation.LAB_ROOT == LAB_ROOT
    assert (isolation.LAB_SRC / "sera" / "__init__.py").is_file()


def test_required_packages_resolve_inside_the_laboratory():
    isolation.activate()
    resolved = isolation.check()
    for name, entry in resolved.items():
        assert Path(entry["file"]).is_relative_to(LAB_ROOT), (name, entry)


def test_fresh_process_reports_laboratory_modules_not_production():
    """The editable install must not win in a clean worker (review finding 5)."""
    result = subprocess.run([sys.executable, "-X", "utf8", "-c",
                             "import sys;sys.path.insert(0, r'%s');"
                             "from experiments.owner_language.isolation import activate, identity;"
                             "activate();import json;print(json.dumps(identity()))" % LAB_ROOT],
                            capture_output=True, text=True, cwd=str(LAB_ROOT))
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["lab_root"] == LAB_ROOT.as_posix()
    for name, entry in record["modules"].items():
        assert entry["file"].startswith(LAB_ROOT.as_posix()), (name, entry)
    assert record["executable_sha256"]


def test_activation_rejects_a_foreign_package_imported_first():
    production_src = Path("D:/ai/projects/sera/src")
    if not (production_src / "sera" / "__init__.py").is_file():
        pytest.skip("No production reference on this machine")
    script = ("import sys;sys.path.insert(0, r'%s');import sera;"
              "sys.path.insert(0, r'%s');"
              "from experiments.owner_language.isolation import activate;activate()"
              % (production_src, LAB_ROOT))
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", script],
                            capture_output=True, text=True, cwd=str(LAB_ROOT))
    assert result.returncode != 0
    assert "Non-laboratory modules were imported before isolation" in result.stderr


def test_baseline_store_has_a_readable_layout_and_the_recorded_hash():
    pointer = labstore.verify_pointer(labstore.BASELINE)
    assert pointer["revision"] == labstore.BASELINE_REVISION
    assert pointer["sha256"] == labstore.BASELINE_SHA256


def test_baseline_revision_carries_the_recorded_owner_identity():
    revision = labstore.BASELINE / "revisions" / labstore.BASELINE_REVISION
    state = json.loads(revision.read_text())["state"]
    assert state["owner"] == labstore.OWNER_IDENTITY
    assert state["schema"] == "sera.intervention-session.1"
    assert len(state["subjects"]) == 144


def test_lineage_manifest_matches_the_copied_bytes():
    if not labstore.MANIFEST.exists():
        pytest.skip("Lineage has not been materialised in this checkout")
    manifest = json.loads(labstore.MANIFEST.read_text())
    store = manifest["copied_stores"][0]
    actual = labstore.store_entries(Path(store["laboratory"]))
    assert actual == store["entries"]


def test_copy_store_refuses_a_divergent_existing_copy(scratch):
    source = scratch / "reference/runs/example"
    (source / "revisions").mkdir(parents=True)
    payload = b'{"revision": 0}\n'
    (source / "revisions/000000-aaaa.json").write_bytes(payload)
    import hashlib
    (source / "current.json").write_text(json.dumps(
        {"revision": "000000-aaaa.json", "sha256": hashlib.sha256(payload).hexdigest()}))
    runs = scratch / "lab/runs"
    labstore.copy_store("runs/example", reference=scratch / "reference", runs=runs)
    (runs / "example/revisions/000000-aaaa.json").write_bytes(b'{"revision": 1}\n')
    with pytest.raises(ValueError, match="differs from the reference"):
        labstore.copy_store("runs/example", reference=scratch / "reference", runs=runs)
