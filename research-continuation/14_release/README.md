# Compact continuation evidence

This directory publishes the measured results, raw-score summaries, verification records and exact frozen source needed to identify the completed cohorts. Full model checkpoints, every intermediate posterior, raw query arrays, original archives and isolated source checkouts remain locally preserved and indexed. This distinction avoids duplicating hundreds of megabytes into Git; it does not make those storage costs disappear.

The [report](../RESULTS_2026-09-15.md), [architecture audit](../01_audit/final-architecture-comparison.md), [registry](../EXPERIMENT_REGISTRY.jsonl) and [failure ledger](../FAILURE_LEDGER.jsonl) state what was executed and what remains unproved. The release manifest records hashes and original local locations for compact copies. The separate preservation audit checks the pre-continuation inventory.

## Frozen sources

`frozen-protocol-sources.zip` contains three independently rooted source snapshots. `frozen-source-members.json` records every member hash. The original GG-P0 source differs from current source because its later audit corrected metadata and initialization wording; original scores must use the frozen implementation. GG-ACT pins its decoder, common interfaces, test and supervisor closure. SHARED-GG pins all SERA Python sources, integration code, tests and supervisor. Source/run versions are never mixed implicitly.

For GG-P0, unpack into a fresh directory and run its frozen `experiments.generative_memory.study` module with the included protocol and a new output directory. Use the command interface printed by `--help`. Its original runner has a soft admission deadline; that limitation remains in the record. The later refit uses the corrected external supervisor.

For GG-ACT, set `PYTHONPATH` to the unpacked `GG-ACT-001/src`, change to the unpacked `GG-ACT-001` directory and use the published runtime (Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1 and CPU Torch 2.10.0). The package includes the frozen supervisor:

```powershell
python scripts/run_bounded.py --seconds 180 --output runs/act-supervisor --module experiments.generative_memory.acquisition -- --protocol research-continuation/04_protocols/GG-ACT-001.json --output runs/act --phase final
python -m experiments.generative_memory.acquisition --output runs/act --replay
```

Use a fresh output for each primary invocation. The completed local primary is `runs/GG-ACT-001-final`. Its published compact manifest lists 365 raw files and their hashes, and the summary preserves all 300 final rows, learning curves and paired differences. Exact numerical replay requires either those local raw files or a regenerated cohort; a hash alone does not supply missing bytes. The independent `acquisition_audit` command additionally checks expected cohort completeness and batch algebra.

SHARED-GG requires the preserved trained parent at `runs/sera-0.5-current`, matching the pinned protocol identity. It is not silently replaced with a random model if absent. The integration procedure is an explicit pytest target so the frozen supervisor can apply its existing allowlist:

```powershell
python scripts/run_bounded.py --seconds 120 --output runs/shared-gg-supervisor --module pytest -- -q -s experiments/generative_memory/integration.py
```

The default integration output is `runs/SHARED-GG-001` and must be fresh. `SERA_INTEGRATION_OUTPUT` can select a fresh alternative. The full original parent remains unchanged. The saved generator interpreter pins the complete local SERA source and Python/Torch runtime; use the matching source snapshot to restore it after later code revisions.

## Resource interpretation

Windows supervision starts the child suspended, assigns it to a kill-on-close Job Object and then resumes it. Measured memory is **peak committed job bytes**, not RSS. The original launcher-only measurement and forced-timeout failure records remain preserved. The supervisor is a trusted-research process deadline, not a hostile-code sandbox; there is a narrow pre-assignment interruption window, and non-Windows supervision does not provide the same process-tree guarantee. Full time reservations remain charged if the supervisor is externally interrupted.

The scientific reports distinguish numeric workspace, serialized model, shared decoder/runtime, learning work and raw archive bytes. No total-compression or quantum-advantage claim follows from a small parameter buffer.
