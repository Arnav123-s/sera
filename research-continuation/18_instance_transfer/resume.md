# Use the qualified experimental descendant

Run from the SERA repository with its recorded Python 3.12.14 / PyTorch 2.10.0 CPU environment. The trained artifacts are preserved locally under ignored `runs/`; the Git release contains their SHA-256 inventory and all protocols, compact outcomes and source snapshots. A fresh clone alone does not contain the trained checkpoints.

The complete solver is `runs/A06-integrated-001/solver`. Its solver identity is `d774aa5175488634043f75711ffee4f943313aa92a396351dcdcb57690ba8285`; its shared-owner identity is `37a025461acdfb077580e9d97aba857e80b8dbde968193c6b91b787b28bf7e25`. These are different identity contracts, not interchangeable hashes. Factual state and the re-proved finite library are preserved under `17_transfer/integration/`.

All three commands below have already passed. They restore the real saved solver, factual state and finite library. Motion answers remain conditional and uncalibrated. Finite answers are conditional on the supplied finite field and proof assumptions.

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.resume_transfer status
.venv/Scripts/python.exe -X utf8 -m scripts.resume_transfer finite --a 4 --b 5 --c 6
.venv/Scripts/python.exe -X utf8 -m scripts.resume_transfer motion --coordinates 1.05 -0.1 0.918140114 0.132530549 0.72087802 0.3129333
```

The finite example returns 0 for `4*x-5=6 modulo 11`. Inputs are user-supplied observations, not newly admitted independent evidence. These commands leave the owner and stores unchanged.

For another supervised use, choose a fresh output directory and retain the existing shared ledger:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/run_integration_bounded.py --seconds 30 --output runs/v3-batch-001/A06-user-status-001 --module scripts.resume_transfer -- status
```

The supervisor refuses a reused directory or owned-job lock. Read `runs/v3-batch-001/budget.json` before another job; the saved release allowance is a historical snapshot. The cap remains one numerical thread, 2 GiB committed memory and at most 900 seconds per job within the original aggregate budget. Do not clear another process's lock or expand the allowance implicitly.

Every learning cell is complete. Do not restart its batch script or reinterpret its final bank as new development data. Selected and intermediate checkpoints retain their protocol, immutable parent reference, optimizer, random states and best-development state. The interrupted-optimizer continuation contract was tested. Completed checkpoints are rejected as restart points; a new experiment must have a new protocol, output directory and independent evaluation bank.

The [next research specification](../NEXT_EXPERIMENT_PROTOCOL.md) starts from this exact owner for continuing hidden-state interaction. It remains proposed. No background job or future execution is scheduled by this document.

The release verifier checks saved source/result identities without training or sampling. Add `--local-checkpoints` on this machine to include every preserved local checkpoint and predecessor artifact:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/verify_transfer_release.py --local-checkpoints
```
