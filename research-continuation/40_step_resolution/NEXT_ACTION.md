# Exact continuation state

CG-001, SR-001 and RS-001 training, final evaluation and independent numerical audits are complete. Do not restart their training or retune their completed final cohorts. The live ledger is `runs/v3-batch-001/budget.json`; historical balances in reports are snapshots, not grants.

The usable owner is `11653a31b45b36be4d02a81c1a0d2aae5a30561ea9b4a305e0dc617a4751dd81` in `runs/sera-growth-live`. It contains the full acquired-parent chain, continued knowledge, procedure weights, optimizer, RNG, 1,536 selected-run decisions and persistent rewards. Later experimental successors remain in `runs/SR-study-001` and `runs/RS-study-001`, including their terminal checkpoints, exact intermediate states, all candidates and failed updates. The old empirical store `runs/sera-refinement-live` remains unrelated and unchanged.

To use the current owner without repeating research, supply new tasks in a JSON list and run:

```powershell
.venv/Scripts/python.exe scripts/run_resolution_bounded.py --seconds 65 --output runs/my-new-task-job-001 --module scripts.continuing_use -- --input path/to/new-tasks.json --output runs/my-new-task-results-001.json
```

The user's latest steering prioritizes self-chosen discovery after three elementary process examples. Continue with [SD-001](../41_self_chosen_discovery/PROTOCOL.md): questions and candidates from retained executions, independently checked credit, distinct useful routes, prospective real-evidence checks and one actual shared owner. The standing local-resource approval and the live ledger govern this work. The source-reading question in `research-continuation/34_learning_progress/reading-correction-open.json` remains preserved with its latest prediction in `example-results.json`; it has not been silently marked solved or discarded.

The sustained studies identify two distinct uncertainties to address: development retention from a small repeated probe did not guarantee retention on its reserved groups, and per-step loss tolerance permitted accumulated drift. RS-001 removed the latter within its transaction contract. Its final still did not qualify the resulting weights. Future work should test retention evidence and representation changes against fresh, source-separated cohorts; do not loosen completed thresholds to promote these checkpoints.
