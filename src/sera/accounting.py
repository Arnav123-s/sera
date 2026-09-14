"""Whole-run and phase costs, including failed work, without a fictitious FLOP total."""

from __future__ import annotations

import os
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


def process_peak_bytes():
    """OS process high-water RSS; cumulative, not the isolated phase allocation."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "Cannot measure process peak memory")
        return counters.peak
    import resource
    amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(amount if platform.system() == "Darwin" else amount * 1024)


@dataclass
class Costs:
    started: float = field(default_factory=time.perf_counter)
    cpu_started: float = field(default_factory=time.process_time)
    phases: list = field(default_factory=list)

    @contextmanager
    def phase(self, name, work=None):
        start, cpu = time.perf_counter(), time.process_time()
        before = dict(work.counts) if work is not None else {}
        result = {"name": name, "status": "failed"}
        try:
            yield result
            result["status"] = "completed"
        except Exception as error:
            result["failure"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            result.update(wall_seconds=time.perf_counter() - start,
                          process_cpu_seconds=time.process_time() - cpu,
                          process_peak_rss_bytes=process_peak_bytes())
            if work is not None:
                result["operations"] = {key: count - before.get(key, 0) for key, count in work.counts.items()
                                        if count != before.get(key, 0)}
            self.phases.append(result)

    def record(self):
        return {"wall_seconds": time.perf_counter() - self.started,
                "process_cpu_seconds": time.process_time() - self.cpu_started,
                "process_peak_rss_bytes": process_peak_bytes(), "phases": self.phases,
                "hardware": {"system": platform.platform(), "processor": platform.processor(),
                             "logical_cpus": os.cpu_count(), "torch": torch.__version__,
                             "torch_threads": torch.get_num_threads(), "device": "cpu"},
                "scope": "Whole invocation including acquisition, updates, validation, search, evaluation and failures. RSS is the cumulative process high-water mark. Phase costs may overlap if nested. Counts are not FLOPs; design and implementation labor, prior development, energy and external service costs are not measured here."}
