"""One owned CPU job, pre-reserved aggregate allowance and native memory ceiling."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from windows_job_v3 import WindowsJob

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT/"runs/v3-batch-001"
ALLOWED = {"pytest", "experiments.world_calibration.study", "experiments.world_calibration.audit",
           "experiments.guarded_consolidation.study"}


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--module", choices=sorted(ALLOWED), required=True)
    p.add_argument("args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/"runs") or not 0 < args.seconds <= 900:
        raise ValueError("A fresh local output and a 1..900 second cap are required")
    # An existing lock is never cleared by this runner. A crash needs ownership review.
    with (BATCH/"active.lock").open("x", encoding="utf-8") as lock:
        lock.write(json.dumps({"pid": os.getpid(), "output": str(output)}))
    try:
        ledger = json.loads((BATCH/"budget.json").read_text())
        if ledger["remaining_seconds"] < args.seconds:
            raise ValueError("Aggregate allowance exhausted; save state and request a new allowance")
        output.mkdir(parents=True, exist_ok=False)
        extra = args.args[1:] if args.args[:1] == ["--"] else args.args
        command = [sys.executable, "-X", "utf8", "-m", args.module, *extra]
        state = {"status": "RESERVED", "argv": command, "reserved_seconds": args.seconds,
                 "charged_seconds": args.seconds, "memory_limit_bytes": ledger["memory_limit_bytes"],
                 "peak_job_committed_bytes": 0, "peak_process_committed_bytes": 0,
                 "observed_process_ids": [], "worker_threads": 1,
                 "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "job_helper_sha256": hashlib.sha256(Path(__file__).with_name("windows_job_v3.py").read_bytes()).hexdigest(),
                 "boundary": "Wall allowance and job COMMITTED memory, not RSS. All descendant processes belong to this owned job. "
                             "Full reservation persists if the supervisor is externally interrupted."}
        key = output.relative_to(ROOT).as_posix()
        if key in ledger["jobs"]:
            raise ValueError("Job identity already used")
        ledger["jobs"][key] = {"status": "RESERVED", "charged_seconds": args.seconds}
        ledger["remaining_seconds"] -= args.seconds
        write(BATCH/"budget.json", ledger)
        write(output/"state.json", state)
        process = job = None
        started = time.perf_counter()
        with (output/"process.log").open("w", encoding="utf-8") as log:
            try:
                environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, creationflags=0x00000004)
                job = WindowsJob(process, memory_bytes=ledger["memory_limit_bytes"])
                state.update(status="RUNNING", child_pid=process.pid)
                write(output/"state.json", state)
                while process.poll() is None:
                    sample = job.sample()
                    for name in ("peak_job_committed_bytes", "peak_process_committed_bytes"):
                        state[name] = max(state[name], sample[name])
                    state["observed_process_ids"] = sorted(set(state["observed_process_ids"]) | set(sample["active_process_ids"]))
                    if time.perf_counter()-started >= args.seconds:
                        state["status"] = "TIMEOUT"
                        job.terminate()
                        process.wait()
                        break
                    try:
                        process.wait(timeout=.05)
                    except subprocess.TimeoutExpired:
                        pass
                if state["status"] == "RUNNING":
                    state["status"] = "PASS" if process.returncode == 0 else "FAILED"
            except BaseException as error:
                state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "SUPERVISOR_ERROR",
                             exception_type=type(error).__name__, exception=str(error))
                raise
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
                state["returncode"] = None if process is None else process.returncode
                state["charged_seconds"] = time.perf_counter()-started
                log.flush()
                state["log_sha256"] = hashlib.sha256((output/"process.log").read_bytes()).hexdigest()
                write(output/"state.json", state)
                ledger["remaining_seconds"] += args.seconds-state["charged_seconds"]
                ledger["jobs"][key] = {"status": state["status"], "charged_seconds": state["charged_seconds"]}
                write(BATCH/"budget.json", ledger)
        print(json.dumps(state, indent=2))
        if state["status"] != "PASS":
            raise SystemExit(1)
    finally:
        # Only this process's exclusively acquired lock is removed.
        (BATCH/"active.lock").unlink()


if __name__ == "__main__":
    main()
