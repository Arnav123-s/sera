# Resume the continuing experimental learner

Run from the SERA repository in the recorded Python 3.12 / PyTorch 2.10 CPU environment:

```powershell
.venv/Scripts/python.exe -X utf8 -m experiments.continuing_control.integration status
```

This restores the real A06 base, the registered action-dynamics workspace, all 79 new interaction observations, the original 16 geometric observations and correction, and the re-proved finite library. The integrated solver identity is `e90f53cfde2f6c588a41aae69103daf53d8776d76f5f4b59c5a8242a76dbf48b`. Its shared-owner identity is `e7df91070866b346edc5379d9dbb8c1d4b887070dcd87894bac283c957d78b43`.

The compact owner bundle is in `integration/`: `owner.json` binds source and state hashes, `interaction-session.json` holds the new factual state, and the unchanged base checkpoint remains under ignored `runs/A06-integrated-001/solver`. The original A06 owner, old operational store, development candidates and all final controls remain preserved. A fresh clone without the local base checkpoint can inspect and verify published evidence but cannot restore trained weights.

The solver has already resumed the preselected lifetime and taken four more paid actions/observations. Its current world clock is 78; the original two coasting transitions are included. The complete simulated continuation state and its public goal schedule are in `integration/four-new-interactions.json`. Hidden simulator fields remain assessor-only; the controller receives observations and goals.

Read `runs/v3-batch-001/budget.json` before another numerical job. To supervise a fresh read-only status call within that same allowance:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/run_continuing_bounded.py --seconds 25 --output runs/v3-batch-001/A08-user-status-001 --module experiments.continuing_control.integration -- status
```

Use a fresh output path. Do not clear an owned-job lock. Limits remain one numerical thread, 2 GiB committed memory and the original aggregate allowance. There is no background job scheduled by these files.

Completed cohorts refuse restart. The study supports explicit `run --resume` only for an interrupted cohort: completed cells are reused after source/protocol checks, and `runs/<experiment>/in-progress.json` saves the last completed transition, full factual session, accumulated records and exact world state. Restoration charges factual replay without double-charging earlier observations. Both interrupted-state future-decision equivalence and ordinary full restoration were tested. These completed studies have no pending partial cell.

For new live observations, restore through `integration.restore()`, use the continuing session's paid `sensor`/`admit` contract and obtain a fresh conditional plan with `control.decide()`. An observation or any owner mutation invalidates an older plan. A new owner revision also invalidates whole-owner finite bindings, which require explicit reproof before reuse. Predictions and their narrow uncertainty rectangles are conditional, not certified answers.
