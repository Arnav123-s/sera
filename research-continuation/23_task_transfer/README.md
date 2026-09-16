# Use the numerical task learner

I added a persistent numerical learner that can acquire a three-input scalar rule from labeled CSV examples, reuse structure from earlier related tasks, make empirical predictions, withdraw a rule after contradictory observations, and learn a fresh version. The supplied function vocabulary contains 20 degree-three Legendre terms. It suits small numerical calibration and response-surface tasks; it does not read arbitrary tutorials or discover arbitrary programs.

The prepared descendant is `D:/ai/projects/sera/runs/sera-task-transfer`. It shares the existing SERA architecture and inherited trained state, with added owned numerical readouts. It has four retained example calibration tasks, a corrected second version of `calibration_estimator`, and a withheld out-of-family task. The separate live workbench at port 8765 remains running with its original state.

Run from `D:/ai/projects/sera` in PowerShell:

```powershell
.venv/Scripts/python.exe scripts/sera_tasks.py status
.venv/Scripts/python.exe scripts/sera_tasks.py predict --task calibration_estimator --csv research-continuation/23_task_transfer/TT-OWNER-001/examples/queries.csv
```

The four prepared outputs are approximately `0.3412, 0.3500, 1.24924, -0.30236`. They come from the restored owned parameters, not an example-answer lookup. The full [integration record](TT-OWNER-001/result.json) preserves teaching, drift, correction and restart results.

## Teach a task from your examples

Use `x1,x2,x3,y,role` as the exact CSV header. Inputs must be finite numbers normalized into `[-1,1]`; keep that same normalization for later queries and observations. Output `y` is one finite numerical value. Give rows the roles `fit`, `selection` or `calibration`, with no repeated input triples across roles. Use 4–256 fitting examples, 8–128 selection examples, and 64–512 fresh calibration examples. For a task without useful prior structure, start with at least 20 fitting examples.

```powershell
.venv/Scripts/python.exe scripts/sera_tasks.py teach --task my_calibration --group my_instruments --csv D:/ai/my_examples.csv --tolerance 0.05
```

The model selects a candidate using fitting and selection data, then checks that fixed candidate on calibration data. A shared `--group` allows reuse of previously admitted task structure. Group names are supplied by the user; the learner does not infer semantic relatedness from task names. A withheld or withdrawn task is excluded from future prior construction. `--noise 0.005` declares an absolute bound for the sparse fitter when appropriate; it is not an automatically inferred noise estimate.

The output is explicitly empirical. The reported 95% binomial risk limit assumes fresh independent calibration samples and the same stationary deployment distribution. It bounds the distributional probability of exceeding the declared absolute tolerance, not each individual answer. Rare exceptions, selected calibration examples, changing sensors and correlated data can invalidate an intended interpretation. No formal-answer badge is granted to these numerical fits.

## Predict, observe, correct

Prediction CSV uses `x1,x2,x3`; observation CSV uses `x1,x2,x3,y`.

```powershell
.venv/Scripts/python.exe scripts/sera_tasks.py predict --task my_calibration --csv D:/ai/new_inputs.csv
.venv/Scripts/python.exe scripts/sera_tasks.py observe --task my_calibration --csv D:/ai/new_measurements.csv
.venv/Scripts/python.exe scripts/sera_tasks.py teach --task my_calibration --group my_instruments --csv D:/ai/fresh_changed_examples.csv --tolerance 0.05
```

An observed error above tolerance withdraws the current task version. Predictions then return no values until a new version passes fresh calibration. Teaching preserves previous revisions; input triples already used for any task or observation in this session cannot be recycled as fresh calibration, even under a new name. Observations are not silently converted into independent validation. Changes that have not been observed cannot automatically be detected.

This is task-time learning with newly supplied observations; it has no date cutoff on numerical examples. It does not run an unattended web researcher or assume that elapsed time supplies new evidence. Existing workbench CSV watching remains available separately.

## State and limits

`scripts/sera_tasks.py` routes every operation through the existing cumulative resource ledger, one numerical thread and a 2 GiB committed-memory cap. Its default cap is 30 seconds per command. Each command prints its supervision receipt. It never grants itself more compute. A resource rejection preserves the current state.

The descendant supports sixteen named task readouts. Each new version is an immutable JSON revision under `runs/sera-task-transfer/revisions`; `current.json` points to the active revision. `parent.json` retains the exact inherited session, whose original neural checkpoint chain is still required in this workspace. The [session-record archive](TT-OWNER-001/session-records.zip) preserves all nine integration revisions, not a standalone distribution of every inherited neural checkpoint.

The interpreter and parent are checked by hash at restore. Changing their source requires an explicit migration. The prepared descendant is already at revision ten after the [audited calibration-scope migration](TT-MIGRATION-001/result.json); all nine earlier revisions remain intact. `migrate` accepts only the explicitly registered predecessor source and preserves learned tensors. A fresh independent workspace can be initialized using `init --directory runs/my_task_session`; initialization refuses an existing directory. No command resets the live workbench or silently replaces earlier histories.

[Results](report.md) · [Architecture audit](architecture-audit.md) · [Primary research](literature.md) · [Checklist](PLAN.md) · [Resume](resume.md)
