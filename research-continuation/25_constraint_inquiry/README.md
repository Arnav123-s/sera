# Use the saved constraint learner

Run from `D:/ai/projects/sera` using the existing environment. This is a local
conditional research tool for the taught sentence forms and acquired circular
motion model. It does not perform arbitrary tasks or certify physical feasibility.

Read the completed request and its assumptions:

```powershell
.venv/Scripts/python.exe scripts/sera_constraints.py result --id reach
```

Continue the already saved unfinished investigation:

```powershell
.venv/Scripts/python.exe scripts/sera_constraints.py solve --id unfinished
```

Ask another task using a tested production. The `marker` location is initially
missing from this saved learner; it can request and learn three simulated readings:

```powershell
.venv/Scripts/python.exe scripts/sera_constraints.py ask --id marker-task --text "can orbit reach marker"
.venv/Scripts/python.exe scripts/sera_constraints.py acquire --entity marker
.venv/Scripts/python.exe scripts/sera_constraints.py solve --id marker-task
```

Each invocation uses the existing resource supervisor, normally reserving at most
20 seconds and refunding unused time. Read the live budget before a new batch.
Restoration and validation are real work; no new allowance is assumed here.

Supported forms include `can orbit reach beacon`, `imagine orbit reaching beacon`,
`did orbit reach beacon`, `orbit must not reach beacon`, and their supplied passive
forms such as `please beacon can be reached by orbit`. Historical occurrence asks
for event evidence. Reversing the actor to a beacon requests its missing dynamics.
Unqualified wording such as `please can orbit reach beacon` is deliberately unresolved.

`work --id NAME --ticks N` performs explicit bounded refinement for inspection.
`solve` uses the fixed residual stopping rule. `observe --action 0` takes one
authorized local simulator step, updates the acquired dynamics, and makes dependent
predictions stale. Use `rebase --id NAME` before solving from the new situation.
Proposed continuous controls are imagined; this interface does not execute them.

The saved store is `runs/sera-constraints`, revision `000014-dd392f7936e0.json`
at release. It preserves 15 revisions. `reach` is complete; `unfinished` retains
two refinement steps. Earlier stores and their original pointers remain unchanged.
The selected checkpoint is `runs/CI-fits-final-001/selected.pt`. It contains the
new interfaces, with the separately pinned predecessor needed for exact restoration.

For results and limitations read [report.md](report.md); for source compliance read
[architecture-audit.md](architecture-audit.md); for the next dependencies read
[checklist.md](checklist.md). Static release verification is available without
running a new experiment through `scripts/verify_constraint_inquiry_release.py`.
