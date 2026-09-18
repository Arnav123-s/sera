# Compose learned discoveries and investigate missing information

This continuation expands the learner's questions to combinations of missing inputs and adds composition of two previously learned, owned rules. Every generated candidate is independently checked. [Protocol](PROTOCOL.md) · [Results](report.md) · [Discoveries](discoveries.json) · [Independent audit](audit.json).

The previous model and all its results remain in [Stage 41](../41_self_chosen_discovery/README.md). A separate [credit reconciliation](credit-reconciliation.json) identifies an older built-in calculation, preserving its useful weights while withholding fresh novelty points for reproducing it.

From the prepared workspace, run this example through the latest admitted owner:

```powershell
.venv/Scripts/python.exe scripts/run_resolution_bounded.py --seconds 90 --output runs/my-discovery-batch-001 --module scripts.continuing_use -- --input research-continuation/42_composed_discovery/example-tasks.json --output runs/my-discovery-results-001.json
```

The JSON supplies observations and asks for a missing quantity. The `solve_discovery` route uses retained learned coefficients. It returns all applicable alternatives with assumptions, and reports contradictory supplied premises when their answers disagree. `observed_motion` uses the separately fitted and checked physical parameters, retaining its source identity and observed range. Predictions remain labeled according to their evidence.

Every output path must be fresh. The bounded wrapper records one CPU thread, the 2 GiB process-tree cap, elapsed cost and exact source identities. The user has authorized continued local numerical work within those limits; the live aggregate ledger remains authoritative and records further allocations.
