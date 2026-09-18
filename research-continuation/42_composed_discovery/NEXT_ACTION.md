# Exact continuing state

The latest usable store is `runs/sera-observed-discovery-live`, owner `91f6c21ffcf80e8aab05ca8eea8e34499a27803cd823aa5a087839ac44957230`. The SD and CF comparisons, their final evaluations and the corrected observation check are completed. They must not be restarted or retuned.

Run the saved owner on a fresh task batch:

```powershell
.venv/Scripts/python.exe scripts/run_frontier_bounded.py --seconds 120 --output runs/CF-user-batch-001 --module scripts.continuing_use -- --input research-continuation/42_composed_discovery/example-tasks.json --output runs/CF-user-results-001.json
```

The next research cycle should acquire genuinely new human-language request groups with source provenance, assess the two unconfirmed English associations on development evidence, and compare acquisition efficiency with a balanced procedure on newly frozen questions. The post-integration example `mets une alarme pour demain matin` was interpreted as `play_music`; retain it as a diagnosed acquisition goal, not as a new held-out test after teaching. All prior request IDs across translations must remain excluded from new final groups. Keep the older source-reading goal open in its preserved history. Use the current owner as parent, a new experiment/output directory and a fresh protocol; do not rerun `prepare`, `train` or `final` for SD-001/CF-001.

Resources have standing user approval for bounded local allocations: one CPU thread, 2 GiB committed process-tree cap, at most 900 seconds per owned job and no paid services. Check `runs/v3-batch-001/budget.json` and record any additional allocation and its previous hash. An old numerical balance is not a new allocation.
