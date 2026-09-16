# SERA live workbench

Run from the repository in the existing local environment:

```powershell
.venv/Scripts/python.exe -X utf8 -m workbench --port 8765
```

Open **http://127.0.0.1:8765**. The current session already has a local host running there. Only the loopback interface is bound. No external model service is used.

## Tasks you can perform

- **Learn on request:** enter a numerical transformation and values. `Convert Celsius to Fahrenheit` retrieves source-labeled local examples, learns the rule, checks separate withheld examples, then executes. Supply an `x,y` CSV with 18–256 distinct examples to teach your own linear or quadratic calibration. Results outside the example domain or beyond the requested error tolerance are withheld and preserved.
- **Live data tasks:** upload or paste a `timestamp,value` CSV with 8–1,024 regularly spaced observations. Receive a forecast, causal prediction-error comparison with persistence, surprise flags, and downloadable results. ISO timestamps or numeric observation indices are supported.
- **Automatic updates:** save a CSV in `runs/sera-workbench/inbox`. Changes trigger a new bounded job and produce JSON, Markdown and CSV reports in `runs/sera-workbench/outputs`. Appending data and correcting earlier data both preserve predecessor revisions. No work runs for an unchanged file.
- **Adaptive control:** choose a goal, execute up to 24 actions, reverse the actuator and observe correction. This resumes the actual saved learner in its simulated circle environment.
- **Verified reasoning:** solve `a*x-b=c (mod 11)` through the preserved guarded program plus independent substitution. Degenerate equations return all residues or no solution.

The live session is in `runs/sera-workbench`. `current.json` points to a checksummed immutable revision; older revisions remain under `revisions`. Exporting a session does not include the immutable neural base weights. The trained local A06 checkpoint is required to restore; a source-only clone cannot recreate those weights automatically. No completed experiment is restarted by launching the application.

All numerical requests share the project resource ledger and use one CPU thread and a 2 GiB worker process-tree cap. The HTTP host and idle file watcher are lightweight application services; they are not unbounded training loops. The explicit additional 60-minute grant is recorded in [resource-grants](resource-grants/grant.json). An exhausted allowance stops new numerical jobs with an error while retaining saved results. Stop the owned host by closing its launching terminal; this session's hidden host PID is recorded under `runs/workbench-host-20260915/pid.txt`.

This is a useful bounded learner, not a general agent that understands arbitrary tutorials. The [larger research target](RESEARCH_LEARNER_TARGET.md), [results](report.md), [architecture audit](architecture-audit.md) and [failures](failures.md) distinguish implemented behavior from open capabilities.
