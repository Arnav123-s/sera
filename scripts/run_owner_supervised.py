"""Supervised parent launch for laboratory numerical work on lab/owner-learning-001.

A lock is not a memory cap (review finding 8). This launcher does in one place
all four things the brief requires:

* acquire the *shared* exclusive numerical lease at the production path, which
  is the single authorised production-path write for this workstream;
* create the child suspended, assign it to a Windows Job Object carrying a
  committed process-tree limit, and only then resume it, so that no descendant
  can start outside the cap;
* verify the configured limit and the observed process count by querying the
  job object rather than trusting a docstring;
* charge every second and peak byte of the attempt to a laboratory ledger, for
  failures exactly as for successes, and release the lease only while this
  process still owns it.

The launched child must not acquire the lease again.
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

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "scripts"))

from windows_job_v3 import WindowsJob  # noqa: E402

SHARED_LEASE = Path("D:/ai/projects/sera/runs/v3-batch-001/active.lock")
LEDGER = LAB_ROOT / "runs/owner-learning-001/costs.json"
MEMORY_BYTES = 2 * 1024 ** 3
CREATE_SUSPENDED = 0x00000004
KILL_ON_JOB_CLOSE = 0x2000
JOB_MEMORY_LIMIT = 0x200


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.1)


def read_ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"schema": "sera.owner-learning.cost-ledger.1", "memory_limit_bytes": MEMORY_BYTES,
            "numerical_workers": 1, "paid_services": False, "charged_seconds": 0., "attempts": []}


def acquire(owner_record):
    """Exclusive create; never steal or remove another team's lease."""
    SHARED_LEASE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SHARED_LEASE.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(owner_record))
    except FileExistsError:
        holder = SHARED_LEASE.read_text(encoding="utf-8", errors="replace")
        raise SystemExit("The shared numerical lease is held by another job: " + holder)
    return json.loads(SHARED_LEASE.read_text())


def release(owner_record):
    """Ownership-checked release; a foreign lease is left untouched."""
    if not SHARED_LEASE.exists():
        return "ALREADY_ABSENT"
    try:
        current = json.loads(SHARED_LEASE.read_text())
    except ValueError:
        return "UNREADABLE_LEASE_PRESERVED"
    if current.get("pid") != owner_record["pid"] or current.get("token") != owner_record["token"]:
        return "FOREIGN_LEASE_PRESERVED"
    SHARED_LEASE.unlink()
    return "RELEASED"


def source_hashes(paths):
    record = {}
    for path in paths:
        path = Path(path)
        if path.is_file():
            record[path.relative_to(LAB_ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Short attempt name recorded in the ledger")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh owned directory under runs/owner-learning-001")
    parser.add_argument("--module", required=True, help="Module executed with -m inside this checkout")
    parser.add_argument("--memory-bytes", type=int, default=MEMORY_BYTES)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()

    output = options.output.resolve()
    if not output.is_relative_to(LAB_ROOT / "runs/owner-learning-001"):
        raise SystemExit("Use a fresh owned directory under runs/owner-learning-001")
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing attempt directory: " + str(output))
    output.mkdir(parents=True)

    token = os.urandom(8).hex()
    owner_record = {"pid": os.getpid(), "token": token, "output": str(output),
                    "workstream": "lab/owner-learning-001", "time_limit": None,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    extra = options.args[1:] if options.args[:1] == ["--"] else options.args
    command = [sys.executable, "-X", "utf8", "-u", "-m", options.module, *extra]
    state = {"schema": "sera.owner-learning.attempt.1", "label": options.label, "status": "RUNNING",
             "argv": command, "module": options.module, "output": str(output),
             "started_utc": owner_record["started_utc"],
             "memory_limit_bytes": options.memory_bytes, "worker_threads": 1, "time_limit_seconds": None,
             "charged_seconds": 0., "peak_job_committed_bytes": 0, "peak_process_committed_bytes": 0,
             "observed_process_ids": [], "max_concurrent_processes": 0,
             "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "lease": {"path": str(SHARED_LEASE), "acquired": False, "released": None},
             "source_sha256": source_hashes([*(LAB_ROOT / "experiments/owner_language").glob("*.py"),
                                             *(LAB_ROOT / "scripts").glob("owner_language_*.py"),
                                             Path(__file__)])}

    job = process = None
    started = time.perf_counter()
    wake = 0
    try:
        acquire(owner_record)
        state["lease"]["acquired"] = True
        write_json(output / "state.json", state)
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                       "PYTHONIOENCODING": "utf-8", "SERA_LAB_ATTEMPT": str(output),
                       "SERA_LAB_SUPERVISED": token}
        wake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        state["windows_awake_requested"] = bool(wake)
        with (output / "process.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=LAB_ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, creationflags=CREATE_SUSPENDED)
            job = WindowsJob(process, memory_bytes=options.memory_bytes)
            state["child_pid"] = process.pid
            first = job.sample()
            flags = first["configured_limit_flags"]
            state["verified_job_configuration"] = {
                "configured_job_memory_limit_bytes": first["configured_job_memory_limit_bytes"],
                "configured_limit_flags": flags,
                "limit_matches_policy": first["configured_job_memory_limit_bytes"] == options.memory_bytes,
                "kill_on_job_close": bool(flags & KILL_ON_JOB_CLOSE),
                "job_memory_limit_active": bool(flags & JOB_MEMORY_LIMIT)}
            if not state["verified_job_configuration"]["limit_matches_policy"]:
                raise RuntimeError("The job object did not accept the declared committed-memory limit")
            if not state["verified_job_configuration"]["job_memory_limit_active"]:
                raise RuntimeError("The job object did not activate its committed-memory limit")
            while process.poll() is None:
                sample = job.sample()
                for name in ("peak_job_committed_bytes", "peak_process_committed_bytes"):
                    state[name] = max(state[name], sample[name])
                state["observed_process_ids"] = sorted(set(state["observed_process_ids"])
                                                       | set(sample["active_process_ids"]))
                state["max_concurrent_processes"] = max(state["max_concurrent_processes"],
                                                        len(sample["active_process_ids"]))
                state["charged_seconds"] = time.perf_counter() - started
                write_json(output / "state.json", state)
                try:
                    process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    pass
            state["status"] = "PASS" if process.returncode == 0 else "FAILED"
            state["returncode"] = process.returncode
    except BaseException as error:
        state["status"] = "FAILED"
        state["error"] = type(error).__name__ + ": " + str(error)
    finally:
        if process is not None and process.poll() is None:
            if job is not None:
                job.terminate()
            else:
                process.kill()
            process.wait()
            state["status"] = "TERMINATED"
        if job is not None:
            sample = job.sample()
            for name in ("peak_job_committed_bytes", "peak_process_committed_bytes"):
                state[name] = max(state[name], sample[name])
            job.close()
        if wake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        state["charged_seconds"] = time.perf_counter() - started
        if state["status"] == "RUNNING":
            state["status"] = "INTERRUPTED"
        if state["lease"]["acquired"]:
            state["lease"]["released"] = release(owner_record)
        state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(output / "state.json", state)
        ledger = read_ledger()
        entry = {key: state[key] for key in ("label", "status", "module", "charged_seconds",
                                             "peak_job_committed_bytes", "max_concurrent_processes",
                                             "output", "started_utc", "finished_utc")}
        entry["returncode"] = state.get("returncode")
        entry["error"] = state.get("error")
        entry["lease_released"] = state["lease"]["released"]
        ledger["attempts"].append(entry)
        ledger["charged_seconds"] = round(sum(a["charged_seconds"] for a in ledger["attempts"]), 6)
        write_json(LEDGER, ledger)
    print(json.dumps(state, indent=2))
    raise SystemExit(0 if state["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
