"""Single owned local job with memory protection, checkpoints and wake request.

The user's 18 September instruction removes the local time allowance. Every
actual second is still recorded; historical remaining balances are preserved.
"""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from windows_job_v3 import WindowsJob

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "runs/v3-batch-001"
ALLOWED = {"scripts.solution_study", "scripts.solution_use", "scripts.solution_artifacts", "scripts.solution_replay",
           "scripts.solution_feedback", "scripts.solution_coverage_review", "pytest", "ruff"}


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--module", choices=sorted(ALLOWED), required=True)
    p.add_argument("args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.exists():
        raise ValueError("Use a fresh owned runs directory")
    with (BATCH / "active.lock").open("x", encoding="utf-8") as lock:
        lock.write(json.dumps({"pid": os.getpid(), "output": str(output), "time_limit": None}))
    job = process = None
    state = None
    started = time.perf_counter()
    wake = 0
    try:
        ledger = json.loads((BATCH / "budget.json").read_text())
        if not ledger.get("unlimited_local_time", {}).get("authorized"):
            raise ValueError("Explicit time-limit removal has not been recorded")
        output.mkdir(parents=True, exist_ok=False)
        extra = args.args[1:] if args.args[:1] == ["--"] else args.args
        command = [sys.executable, "-X", "utf8", "-m", args.module, *extra]
        state = {"status": "RUNNING", "argv": command, "time_limit_seconds": None, "charged_seconds": 0.,
                 "memory_limit_bytes": ledger["memory_limit_bytes"], "worker_threads": 1,
                 "peak_job_committed_bytes": 0, "peak_process_committed_bytes": 0, "observed_process_ids": [],
                 "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        key = output.relative_to(ROOT).as_posix()
        ledger["jobs"][key] = {"status": "RUNNING", "charged_seconds": 0., "time_limit_seconds": None}
        write(BATCH / "budget.json", ledger)
        wake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        state["windows_awake_requested"] = bool(wake)
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "NUMEXPR_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
        with (output / "process.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=0x00000004)
            job = WindowsJob(process, memory_bytes=ledger["memory_limit_bytes"])
            state["child_pid"] = process.pid
            while process.poll() is None:
                sample = job.sample()
                for name in ("peak_job_committed_bytes", "peak_process_committed_bytes"):
                    state[name] = max(state[name], sample[name])
                state["observed_process_ids"] = sorted(set(state["observed_process_ids"]) | set(sample["active_process_ids"]))
                state["charged_seconds"] = time.perf_counter() - started
                write(output / "state.json", state)
                try:
                    process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    pass
            state["status"] = "PASS" if process.returncode == 0 else "FAILED"
            state["returncode"] = process.returncode
    finally:
        if process is not None and process.poll() is None:
            if job is not None:
                job.terminate()
            else:
                process.kill()
            process.wait()
        if job is not None:
            sample = job.sample()
            for name in ("peak_job_committed_bytes", "peak_process_committed_bytes"):
                state[name] = max(state[name], sample[name])
            job.close()
        if wake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if state is not None:
            state["charged_seconds"] = time.perf_counter() - started
            if state["status"] == "RUNNING":
                state["status"] = "INTERRUPTED"
            state["wake_request_released"] = True
            write(output / "state.json", state)
            ledger = json.loads((BATCH / "budget.json").read_text())
            ledger["jobs"][key] = {k: state[k] for k in ("status", "charged_seconds", "time_limit_seconds")}
            ledger["unlimited_local_time"]["charged_seconds"] += state["charged_seconds"]
            write(BATCH / "budget.json", ledger)
        (BATCH / "active.lock").unlink()
    print(json.dumps(state, indent=2))
    raise SystemExit(0 if state["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
