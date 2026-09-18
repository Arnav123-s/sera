# Use SERA

[Home](../README.md) · [Status](STATUS.md) · [Repository map](REPOSITORY_MAP.md) · [Archive](RESEARCH_ARCHIVE.md)

These commands use the prepared workspace at `D:/ai/projects/sera`, its environment and retained checkpoints. Numerical tasks use one CPU thread and the existing 2 GiB process-tree memory cap. The wrapper charges the live resource ledger; every output folder must be fresh.

## Predict motion from a learned history

```powershell
.venv/Scripts/python.exe scripts/run_concept_bounded.py --seconds 60 --output runs/CR-my-forecast-001 --module experiments.concept_refinement.runtime -- predict --subject body-1 --commands "0.8,0.8,0.8,0.8"
```

The answer appears in the output folder's `process.log`, including the future position/velocity sequence, model identity, source kind and assumptions. [Supply your own observations, acquire a model and revise it](../research-continuation/28_concept_refinement/README.md).

## Calculate polynomial motion

```powershell
.venv/Scripts/python.exe scripts/sera_study.py motion --coefficients "2,3,1" --time 3 --position 5 --velocity=-1
```

This evaluates the supplied acceleration polynomial with the retained integral operator and independently checks the algebra. [Study and source-acquisition guide](../research-continuation/27_self_study/README.md).

## Interpret a request

```powershell
.venv/Scripts/python.exe scripts/sera_study.py interpret --text "schalte das licht aus"
.venv/Scripts/python.exe scripts/sera_study.py interpret --text "enciende las luces"
```

These return the learned intent and entity interpretation. [Four-language results](../research-continuation/27_self_study/report.md).

## Annotate a local request file

```powershell
.venv/Scripts/python.exe scripts/sera_annotate.py --input requests.jsonl --output runs/my-inbox/annotations.jsonl
```

[Input format, limits and worked examples](../research-continuation/26_stream_curriculum/README.md).

For other tasks, use the [capability table](../README.md#what-the-saved-learner-can-do). For a fresh clone, begin with [setup and fixture restoration](REPOSITORY_MAP.md#setup-and-verification); a clone does not contain your ignored live stores.
