# Direct-launcher telemetry limitation

The first supervised refit passed all 5,120 exact model hashes. Its reported 4,345,856-byte peak working set belongs to the Windows virtual-environment launcher, not the actual Python worker. It is not a valid full-training-memory measurement. The same launcher boundary also means killing only that direct process cannot certify descendant termination.

The original supervisor source and its recorded result are preserved. Subsequent supervision starts the child suspended, assigns a Windows Job Object with kill-on-close, then resumes its thread. Descendants belong to that job. The new metric is explicitly **peak committed memory for the process tree**, separate from direct-launcher working set and from resident memory.

The implementation follows Microsoft's documented [Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects) and [ResumeThread](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-resumethread) interfaces. Native supervision is a process-lifetime boundary, not a scientific or hostile-code evaluator sandbox.

There is a narrow setup window between creating the suspended process and assigning it to the job. An abrupt external kill of the supervisor in that window can leave a suspended child; process creation and assignment are not one atomic operation. After successful assignment, kill-on-close covers the job's descendants. The three-second forced-timeout test ended after 3.013 seconds with none of the recorded process IDs surviving.
