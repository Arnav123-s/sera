"""Supervise one allowlisted local research process with timeout and native telemetry.

The full allowance is reserved before launch and remains charged if this parent
is externally killed. A running supervisor enforces its wall deadline. This is
process supervision, not a security boundary against hostile research code.
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

from windows_job import WindowsJob

ALLOWED = {
    "experiments.generative_memory.refit",
    "experiments.generative_memory.acquisition",
    "pytest",
    "sera",
}


class WindowsTelemetry:
    def __init__(self, process):
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]

        self.Counters = Counters
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, process.pid)

    def sample(self):
        if not self.handle:
            return None
        value = self.Counters()
        value.cb = ctypes.sizeof(value)
        if not self.psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(value), value.cb):
            return None
        return int(value.PeakWorkingSetSize)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", required=True, choices=sorted(ALLOWED))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.seconds <= 900:
        raise ValueError("A positive allowance of at most 900 seconds is required")
    args.output.mkdir(parents=True, exist_ok=False)
    command_args = args.args[1:] if args.args[:1] == ["--"] else args.args
    command = [sys.executable, "-m", args.module, *command_args]
    state = {"status": "RESERVED", "argv": command, "reserved_seconds": args.seconds,
             "charged_seconds": args.seconds, "root_process_peak_working_set_bytes": None,
             "peak_job_committed_bytes": None, "peak_process_committed_bytes": None,
             "observed_process_ids": [],
             "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "job_helper_sha256": hashlib.sha256(Path(__file__).with_name("windows_job.py").read_bytes()).hexdigest(),
             "boundary": "Process-tree wall deadline and Windows Job Object peak COMMITTED memory (not RSS); direct launcher working set is separate. Child starts suspended and joins a kill-on-close job before execution. Full reserved time stays charged on external parent interruption. Research, energy and supervisor startup excluded."}
    write(args.output / "state.json", state)
    started = time.perf_counter()
    process = telemetry = job = None
    with (args.output / "process.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                       env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                                       creationflags=0x00000004 if os.name == "nt" else 0,
                                       stdout=log, stderr=subprocess.STDOUT)
            if os.name == "nt":
                job = WindowsJob(process)
            telemetry = WindowsTelemetry(process)
            state.update(status="RUNNING", child_pid=process.pid)
            write(args.output / "state.json", state)
            while True:
                sample = telemetry.sample()
                if sample is not None:
                    state["root_process_peak_working_set_bytes"] = max(state["root_process_peak_working_set_bytes"] or 0, sample)
                if job is not None:
                    values = job.sample()
                    for key in ("peak_job_committed_bytes", "peak_process_committed_bytes"):
                        state[key] = max(state[key] or 0, values[key])
                    state["observed_process_ids"] = sorted(set(state["observed_process_ids"]) | set(values["active_process_ids"]))
                elapsed = time.perf_counter() - started
                if elapsed >= args.seconds:
                    if job is not None:
                        job.terminate()
                    else:
                        process.kill()
                    process.wait()
                    state["status"] = "TIMEOUT"
                    break
                try:
                    result = process.wait(timeout=min(.2, args.seconds-elapsed))
                    sample = telemetry.sample()
                    if sample is not None:
                        state["root_process_peak_working_set_bytes"] = max(state["root_process_peak_working_set_bytes"] or 0, sample)
                    state["status"] = "PASS" if result == 0 else "FAILED"
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException as exc:
            state.update(status="INTERRUPTED", error=repr(exc))
            raise
        finally:
            if process is not None:
                if process.poll() is None:
                    if job is not None:
                        job.terminate()
                    else:
                        process.kill()
                    process.wait()
                state["returncode"] = process.returncode
            if telemetry is not None:
                try:
                    telemetry.close()
                except Exception as exc:
                    state.setdefault("telemetry_errors", []).append(repr(exc))
            if job is not None:
                try:
                    values = job.sample()
                    for key in ("peak_job_committed_bytes", "peak_process_committed_bytes"):
                        state[key] = max(state[key] or 0, values[key])
                except Exception as exc:
                    state.setdefault("telemetry_errors", []).append(repr(exc))
                finally:
                    job.close()
            state["charged_seconds"] = time.perf_counter() - started
            log.flush()
            state["log_sha256"] = hashlib.sha256((args.output / "process.log").read_bytes()).hexdigest()
            write(args.output / "state.json", state)
    print(json.dumps(state, indent=2))
    if state["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
