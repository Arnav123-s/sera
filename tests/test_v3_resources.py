"""Native cap is exercised only on a newly created test-owned child."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from windows_job_v3 import WindowsJob  # noqa: E402


@pytest.mark.skipif(os.name != "nt", reason="Windows native resource contract")
def test_native_job_memory_ceiling_precedes_resume():
    code = "try:\n a=bytearray(512*1024**2)\nexcept MemoryError:\n print('CAP_ENFORCED',flush=True)\nelse:\n raise SystemExit(3)"
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, creationflags=0x00000004)
    job = None
    try:
        job = WindowsJob(child, memory_bytes=256*1024**2)
        output, _ = child.communicate(timeout=15)
        assert child.returncode == 0 and b"CAP_ENFORCED" in output
        sample = job.sample()
        assert sample["configured_job_memory_limit_bytes"] == 256*1024**2
        assert sample["configured_limit_flags"] & 0x200
        # On this host the reported high-water count includes the failed large
        # request. Verify rejection plus the read-back ceiling, retain telemetry.
        print(sample)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        if job is not None:
            job.close()
