# Stage 48 — owner-connected language acquisition

This stage continues the actual SERA learner. Its starting point is the Stage 47
owner `62ea82cff82b88d1eb3848f7246494979c63894d869db603007e3d06111063ec`, the
same object that holds the retained mathematics, the four language interfaces,
the physical fields and the intervention models. Nothing here replaces that
owner or initialises a substitute model.

The work is organised in two parts. The first part repairs the machinery that a
static review found to be reporting more than it measured. The second part uses
the repaired machinery to teach the owner from human-authored books and then to
measure what actually changed.

## Part 1 — what was repaired first, and what it now shows

**Isolated imports (`experiments/owner_language/isolation.py`).** The reused
interpreter carries an editable install that appends the production `src`
directory to `sys.path`, so an unqualified `import sera` silently resolves to
production. `activate()` gives this checkout precedence, refuses to continue if a
non-laboratory module was imported first, and `identity()` records the
interpreter, its SHA-256, and the resolved file and hash of every laboratory
package. Every laboratory entry point calls it before importing anything else,
and the record is written into each attempt directory from inside the running
process.

**An explicit lineage, not a default (`experiments/owner_language/labstore.py`).**
Only the final Stage 47 revision had been copied into this checkout, and it was
copied flat, so no loader could have read it: `workbench.storage.Store` requires
`<root>/revisions/<name>.json` beside `current.json`. The copy is now given that
layout and re-hashed against the recorded
`b29d8cccea6a7b8b9e0f0bbc28432d146a6388bca7716a828c446b3b42036d95`, and the 27
ancestor stores plus 20 auxiliary artefacts that the restore chain actually reads
were copied from the read-only production reference with every file hashed on
both sides. The auxiliary list is not a guess: each entry was added because the
restore, executed against the laboratory root, asked for that exact file.
`lineage-summary.json` records all of it.

**A restore that names its store (`experiments/owner_language/owner.py`).** The
earlier verifier called `restore()` with no laboratory path and then checked only
that a file existed, so it reported production as if it were the laboratory
baseline. `restore_owner()` refuses any store outside `runs/`, and fails unless
the restored identity is the recorded owner.

**Retention measured by running the tasks (`scripts/owner_language_baseline.py`).**
The earlier retention check compared one subject's parameter lists against copies
taken from the same dictionary and returned three route names without calling
them; a report then read that as zero catastrophic forgetting. The declared
33-task retained suite and the 39 declared practical examples are now executed
through the restored owner, one case at a time, with per-case results and errors
preserved. A case that raises is recorded as a failure. The first run failed four
`gap_*` cases outright because `runs/KG-study-001/development.json` had not been
copied; that failure is preserved in the attempt directory rather than smoothed
over, and the missing artefact was added to the lineage.

**Supervision at the launch, not in a docstring (`scripts/run_owner_supervised.py`).**
The parent acquires the shared exclusive lease at
`D:/ai/projects/sera/runs/v3-batch-001/active.lock` — the single production-path
write this workstream is authorised to make — creates the child *suspended*,
assigns it to a Windows Job Object carrying the 2 GiB committed process-tree
limit, queries the job to confirm the limit was accepted and is active, and only
then resumes it. The lease is released only while this process still owns it; a
foreign lease is left untouched. Seconds and peak committed bytes are charged for
failed attempts exactly as for successful ones.

### Measured result of Part 1

`baseline-verification.json` is the published summary of attempt
`baseline-verification-002`. Restoring the owner from the laboratory store took
32.5 s and returned identity
`62ea82cff82b88d1eb3848f7246494979c63894d869db603007e3d06111063ec` with its 144
subjects and 2,439 owner tensors. All 33 retained tasks and all 39 practical
examples answered, none failed, and the session state was unchanged afterwards.
Under supervision the two attempts charged 54.11 s and 54.12 s, each peaking at
about 853 MB of committed memory against the 2 GiB cap, with the lease acquired
and released cleanly in both. Both are in the cost ledger, the failed one
included.

This establishes a trustworthy starting line. It is not yet a capability gain,
and nothing in Part 1 trained the owner.

## Part 2 — teaching

Part 2 is specified in `PROTOCOL.md`, which is frozen before any teaching.
