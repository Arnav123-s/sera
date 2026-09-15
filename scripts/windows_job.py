"""Windows process-tree lifetime and commit accounting for local research jobs.

Uses documented Job Objects, Toolhelp thread enumeration and ResumeThread.
The child must be created suspended so it cannot spawn before job assignment.
"""

import ctypes
import os


class WindowsJob:
    def __init__(self, process):
        if os.name != "nt":
            raise RuntimeError("Windows Job Objects require Windows")
        from ctypes import wintypes as w

        size = ctypes.c_size_t

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", w.DWORD),
                        ("MinimumWorkingSetSize", size), ("MaximumWorkingSetSize", size),
                        ("ActiveProcessLimit", w.DWORD), ("Affinity", size),
                        ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                        ("ProcessMemoryLimit", size), ("JobMemoryLimit", size),
                        ("PeakProcessMemoryUsed", size), ("PeakJobMemoryUsed", size)]

        class Thread(ctypes.Structure):
            _fields_ = [("dwSize", w.DWORD), ("cntUsage", w.DWORD),
                        ("th32ThreadID", w.DWORD), ("th32OwnerProcessID", w.DWORD),
                        ("tpBasePri", w.LONG), ("tpDeltaPri", w.LONG), ("dwFlags", w.DWORD)]

        class ProcessIds(ctypes.Structure):
            _fields_ = [("NumberOfAssignedProcesses", w.DWORD),
                        ("NumberOfProcessIdsInList", w.DWORD), ("ProcessIdList", size * 128)]

        self.Extended, self.ProcessIds = Extended, ProcessIds
        self.kernel = k = ctypes.WinDLL("kernel32", use_last_error=True)
        prototypes = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
            "Thread32First": ([w.HANDLE, ctypes.POINTER(Thread)], w.BOOL),
            "Thread32Next": ([w.HANDLE, ctypes.POINTER(Thread)], w.BOOL),
            "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "ResumeThread": ([w.HANDLE], w.DWORD),
        }
        for name, (arguments, result) in prototypes.items():
            getattr(k, name).argtypes = arguments
            getattr(k, name).restype = result
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = Extended()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            process_handle = k.OpenProcess(0x0100 | 0x0001, False, process.pid)
            if not process_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not k.AssignProcessToJobObject(self.handle, process_handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                k.CloseHandle(process_handle)
            # Popen closes its primary thread handle. Reopen the suspended child's thread.
            snapshot = k.CreateToolhelp32Snapshot(0x00000004, 0)
            if snapshot == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            found = False
            try:
                thread = Thread()
                thread.dwSize = ctypes.sizeof(thread)
                more = k.Thread32First(snapshot, ctypes.byref(thread))
                while more:
                    if thread.th32OwnerProcessID == process.pid:
                        handle = k.OpenThread(0x0002, False, thread.th32ThreadID)
                        if not handle:
                            raise ctypes.WinError(ctypes.get_last_error())
                        try:
                            if k.ResumeThread(handle) == 0xFFFFFFFF:
                                raise ctypes.WinError(ctypes.get_last_error())
                            found = True
                        finally:
                            k.CloseHandle(handle)
                    more = k.Thread32Next(snapshot, ctypes.byref(thread))
            finally:
                k.CloseHandle(snapshot)
            if not found:
                raise RuntimeError("Suspended child thread could not be resumed")
        except BaseException:
            self.close()
            raise

    def sample(self):
        value = self.Extended()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, ctypes.byref(value), ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        ids = self.ProcessIds()
        if not self.kernel.QueryInformationJobObject(self.handle, 3, ctypes.byref(ids), ctypes.sizeof(ids), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"peak_job_committed_bytes": int(value.PeakJobMemoryUsed),
                "peak_process_committed_bytes": int(value.PeakProcessMemoryUsed),
                "active_process_ids": list(ids.ProcessIdList)[:ids.NumberOfProcessIdsInList]}

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
