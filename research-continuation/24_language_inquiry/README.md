# Use the persistent language investigator

The new local store is `runs/sera-inquiry`. It contains a descendant of the existing task-bearing SERA owner, its learned language interface, continuing observed situation and persistent investigations. The original workbench at port 8765 and `runs/sera-task-transfer` remain intact.

From `D:/ai/projects/sera`, inspect a completed prediction:

```powershell
.venv/Scripts/python.exe scripts/sera_inquiry.py result --id ready
```

Ask a new question, allocate work and inspect its answer:

```powershell
.venv/Scripts/python.exe scripts/sera_inquiry.py ask --id trial1 --text "predict the position of orbit after push right then wait then push left"
.venv/Scripts/python.exe scripts/sera_inquiry.py work --ticks 3
.venv/Scripts/python.exe scripts/sera_inquiry.py result --id trial1
```

Every new inquiry needs a new ID. Repeated computations can reuse exact prefixes; the stored record still identifies its own question and dependencies. All commands run through the existing one-thread, 2 GiB resource supervisor and consume the shared local allowance. They do not start a background training service.

To perform the first control in a completed current inquiry:

```powershell
.venv/Scripts/python.exe scripts/sera_inquiry.py execute --id trial1
```

This acts only in the local circle simulator, obtains a new position measurement and updates the same owner's dynamics. It does not operate external equipment. Current predictions become stale when relevant evidence changes. Rebase explicitly before further work:

```powershell
.venv/Scripts/python.exe scripts/sera_inquiry.py rebase --id trial1
.venv/Scripts/python.exe scripts/sera_inquiry.py work --ticks 3
```

An interrupted investigation called `unfinished` is saved at one of three steps. Resume it with `resume --id unfinished`, then `work --ticks 2`. `status` lists all work. Use `pause --id ID` to preserve an active inquiry, or `--anchor historical` when asking a question that should remain tied to its original situation. Historical answers cannot execute current actions.

Supported sentence patterns are:

- `predict the position of orbit after push left then wait`
- `if orbit does push right then report its velocity`
- `after wait then push left report the position of orbit`

The supplied grammar accepts one to three ordered control phrases. Fields are `position`, `location`, `velocity` and `angular speed`. Position is a two-coordinate result in metres; velocity means angular displacement in radians per simulator tick. The active entity name defaults to `orbit`; an explicit `--entity comet` permits that name in the sentence through supplied binding.

Initially taught control phrases are `push left`, `apply negative control`, `not right but left`; `wait`, `apply zero control`, `remain still`; and `push right`, `apply positive control`, `not left but right`. They denote controls -1, 0 and +1. A positive control is not a guarantee of rightward physical motion: the learned response can reverse.

`do not wait`, `do not push right` and `do not push left` retain multiple possibilities. For example:

```powershell
.venv/Scripts/python.exe scripts/sera_inquiry.py ask --id ambiguity1 --text "predict the velocity of orbit after do not wait"
.venv/Scripts/python.exe scripts/sera_inquiry.py clarify --id ambiguity1 --actions 1
.venv/Scripts/python.exe scripts/sera_inquiry.py work --ticks 1
```

The saved owner has already learned one extra phrase, `coast`, from a supplied zero-control example and replay of the earlier vocabulary. Its pre-lesson rejection is preserved. `teach --phrase coast --action 0` is the finite lesson command for a fresh store; this release intentionally does not claim an arbitrary-vocabulary tutor or unlimited continuing lessons. `init --store runs/another-inquiry` starts a separate, explicitly named experimental descendant if needed; do not reinitialize the existing store.

All forecasts are **CONDITIONAL_UNCALIBRATED**. Unsupported wording is retained as unresolved/rejected work. The interface does not understand unrestricted English, read papers on its own or certify physical applicability. See the [results](report.md), [architecture audit](architecture-audit.md), [protocol](protocol.md) and [next actions](checklist.md).

The trained parent and checkpoint chain are local artifacts. A plain source checkout does not include those ignored model files. `scripts/verify_language_inquiry_release.py` checks the published evidence without running experiments; add `--local-state` to verify local checkpoint identities. Raw evaluation JSON is also preserved in `evaluation/raw-records.zip` to keep repository diffs compact.
