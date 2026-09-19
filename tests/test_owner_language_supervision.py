"""Tests for the supervised launch: real job-object cap, lease ownership, costs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "scripts"))

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")

import run_owner_supervised as supervisor  # noqa: E402

SCRATCH = LAB_ROOT / "runs/owner-learning-001/test-scratch"


@pytest.fixture
def scratch():
    """A laboratory-owned scratch directory.

    The shared system temporary root on this machine belongs to another account
    and pytest's own ``tmp_path`` cannot scan it, so the tests keep their
    scratch space inside the ignored laboratory runs directory.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=SCRATCH))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_release_preserves_a_foreign_lease(scratch, monkeypatch):
    lease = scratch / "active.lock"
    monkeypatch.setattr(supervisor, "SHARED_LEASE", lease)
    lease.write_text(json.dumps({"pid": 999999, "token": "someone-else"}))
    assert supervisor.release({"pid": os.getpid(), "token": "mine"}) == "FOREIGN_LEASE_PRESERVED"
    assert lease.exists()


def test_acquire_refuses_a_held_lease(scratch, monkeypatch):
    lease = scratch / "active.lock"
    monkeypatch.setattr(supervisor, "SHARED_LEASE", lease)
    lease.write_text(json.dumps({"pid": 999999, "token": "someone-else"}))
    with pytest.raises(SystemExit):
        supervisor.acquire({"pid": os.getpid(), "token": "mine"})
    assert json.loads(lease.read_text())["token"] == "someone-else"


def test_acquire_then_release_round_trip(scratch, monkeypatch):
    lease = scratch / "active.lock"
    monkeypatch.setattr(supervisor, "SHARED_LEASE", lease)
    record = {"pid": os.getpid(), "token": "mine", "output": str(scratch)}
    supervisor.acquire(record)
    assert lease.exists()
    assert supervisor.release(record) == "RELEASED"
    assert not lease.exists()
    assert supervisor.release(record) == "ALREADY_ABSENT"


def test_job_object_enforces_the_committed_memory_cap():
    """A lock is not a cap. Allocate past a small limit and require the kill."""
    from windows_job_v3 import WindowsJob

    limit = 300 * 1024 ** 2
    child = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-c",
         "b=[]\nwhile True:\n    b.append(bytearray(32*1024*1024))"],
        creationflags=supervisor.CREATE_SUSPENDED,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    job = WindowsJob(child, memory_bytes=limit)
    try:
        sample = job.sample()
        assert sample["configured_job_memory_limit_bytes"] == limit
        assert sample["configured_limit_flags"] & supervisor.JOB_MEMORY_LIMIT
        returncode = child.wait(timeout=120)
        assert returncode != 0
        # The peak the job reports includes the allocation that tripped the
        # limit, so it can exceed the limit by one request. What matters is that
        # the tree was stopped near the cap rather than growing without bound.
        peak = job.sample()["peak_job_committed_bytes"]
        assert peak < limit * 1.5, peak
    finally:
        if child.poll() is None:
            job.terminate()
            child.wait()
        job.close()


def test_supervisor_charges_a_failing_attempt():
    output = LAB_ROOT / "runs/owner-learning-001/attempts/test-failing-attempt"
    if output.exists():
        shutil.rmtree(output)
    ledger_before = json.loads(supervisor.LEDGER.read_text()) if supervisor.LEDGER.exists() else {"attempts": []}
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(LAB_ROOT / "scripts/run_owner_supervised.py"),
         "--label", "test-failing-attempt", "--output", str(output),
         "--module", "experiments.owner_language.deliberately_absent"],
        capture_output=True, text=True, cwd=str(LAB_ROOT))
    assert result.returncode == 1, result.stdout + result.stderr
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "FAILED"
    assert state["charged_seconds"] > 0
    assert state["lease"]["released"] == "RELEASED"
    ledger = json.loads(supervisor.LEDGER.read_text())
    assert len(ledger["attempts"]) == len(ledger_before["attempts"]) + 1
    assert ledger["attempts"][-1]["label"] == "test-failing-attempt"
    assert ledger["attempts"][-1]["status"] == "FAILED"
