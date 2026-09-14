# Reproducing the connected SERA study

I separate rebuilding a report, rerunning a learning experiment and reproducing an already exposed admission decision. These operations answer different questions.

The published three-seed study used **0.2.0**, preserved at commit `00d61b8ce4de4c27afabb743cb54b5ddae7ea682` with source SHA-256 `59c0835fba356932d945dc67c04b781aa5549702ee40389b09b9e03c73016594`. To reproduce its saved admission decisions, use that commit in a separate checkout and install that checkout. Version 0.2.1 changes evidence validation and fixed-search construction cost; it is not a rerun of the published study. `audit_connected_run.py` intentionally rejects mismatched implementation hashes. Use a fresh output directory for a new 0.2.1 learning study.

## Rebuild the published tables

Install `.[reports]`, then run:

```text
python scripts/build_connected_report.py
python scripts/verify_release.py
```

With no arguments, the report builder reads committed `reports/connected-study-data.json` and `reports/connected-policy-episodes.json`. It computes tables and plots from these records. It does not need local checkpoints, original PDFs, the ZIP, a sign-in, a model API or a network call. Plot bytes can vary with rendering-library versions; JSON evidence hashes are the integrity anchor.

## Rerun from scratch

Use a fresh directory and the declared configuration:

```text
python -m sera study --output runs/reproduction --seeds 0 1 2 --steps 900 --meta-train 8 --meta-validation 4 --meta-test 4 --inner-steps 16 --instrument-steps 200
python scripts/audit_connected_run.py --input runs/reproduction
python scripts/build_connected_report.py --input runs/reproduction
```

The last command deliberately replaces the selected report evidence with that run's results. Keep a separate checkout if retaining the published report unchanged matters. The study trains all parameters from scratch and runs on CPU with one PyTorch thread. Training seeds, versions and source hashes are recorded. Admission draws fresh randomness after each frozen candidate and records those seeds, so a new study is not expected to reproduce every promotion bit-for-bit. Source edits or changed configurations require a new study directory. Completed seeds can be skipped only under an identical manifest; interrupted partial seeds require inspection and a new directory.

Each seed directory contains:

| Path | Contents |
|---|---|
| `study.json` | Metrics, teaching configuration, model/policy training histories, controls, generations and wall time |
| `initial-evidence.json` | The initial 512 admitted world trajectories |
| `planned-evidence.json` | Actual consequences of 32 on-policy practice trajectories |
| `r2-evidence.json` | The 256 instrument-training trajectories |
| `policy-episodes/` | Every measured method outcome, support identifiers and query identifiers for 8/4/4 development/validation/test episodes |
| `solver/versions/` | Immutable neural/component/skill payloads and checksum manifests |
| `solver/rounds/` | Fresh seeds, paired results, retention decisions and work for every evaluation round |
| `solver/journal.sqlite` | Cumulative reservations and chained lifecycle records |
| `solver/current.json` | The currently admitted executable version |
| `rollback-check/` | A separate copy used to verify parent restoration |
| `r2/` | Controlled-instrument weights, learning history and held-out search trials |

The audit script reloads saved candidates, regenerates their recorded admission data, compares the paired decisions, checks evidence separation and verifies current solver identity. This is reproduction of exposed software behavior, not a new generalization estimate. Its additional work is recorded separately from the original learning study.

## Use a trained solver

```text
python -m sera solve runs/reproduction/0/solver --family rotation --world-seed 50000 --start 0 --goal 3
python -m sera evaluate runs/reproduction/0/solver --output runs/reloaded-evaluation
python -m sera status runs/reproduction/0/solver
```

The symbolic CLI evaluation defaults to the original four tasks. The final study explicitly evaluates all five tasks, including the acquired earliest-binding rule. World goals use visible color indices in 0–3. Seed/family identifies the simulator; its table is environment state, never a neural input. Goal success in these finite worlds must not be described as general language reasoning.

To continue learning, prepare a separate executable copy with actual support from all completed generations:

```text
python scripts/prepare_solver.py --input runs/reproduction/0 --output runs/current-solver
sera learn runs/current-solver --family reset --world-seed 123456 --steps 32
```

The formal study replays its fixed initial-world buffer; later support is archived separately. Preparation merges actual initial/planned/generation records into the continued-learning replay without changing weights or admitting meta/query/test data. The saved policy chooses the next intervention. `sera rollback runs/current-solver` restores its accepted parent. Actual evidence remains after rejection or rollback. Historical checkpoint-only directories lack the connected R1/controller required by these commands.

## Verification and remaining research

Run `python -m ruff check src tests scripts`, `python -m pytest` and the release verifier. Tests cover mathematical invariants and executable behavior, including reload and rollback; final learning tables test performance. Full-width reference-state correctness is tested separately from compact-model training. The [checklist](implementation-checklist.md) and [roadmap](roadmap.md) distinguish completed engineering from open research acceptance criteria.

For the original-document audit, run `python scripts/audit_architecture_contracts.py --output runs/my-architecture-audit.json`. Its numerical reduction, admission counterexample, search-budget checks, structural inspection and valid-input fixture need no saved study. Add `--study runs/connected-v2-final` to reproduce library ablations when those checkpoints exist. The [audit report](../reports/original-source-alignment-audit.md) explains how its post-publication evidence differs from a new held-out performance estimate.
