# Use SERA

The [continuing learner](../research-continuation/44_solution_portfolios/README.md) retains language, reading, mathematics, physical meanings and checked discoveries in one owner. Its latest interface solves questions through alternative learned routes and normally investigates a missing route before returning. Every predecessor store remains available.

[Home](../README.md) · [Status](STATUS.md) · [Repository map](REPOSITORY_MAP.md) · [Archive](RESEARCH_ARCHIVE.md)

These commands use the prepared workspace at `D:/ai/projects/sera`, its environment and retained checkpoints. Numerical tasks use one CPU thread and the existing 2 GiB process-tree memory cap. The wrapper charges the live resource ledger; every output folder must be fresh.

## Run several tasks through the continuing learner

```powershell
.venv/Scripts/python.exe scripts/run_solution_persistent.py --output runs/my-continuing-batch-001 --module scripts.solution_use -- --input research-continuation/44_solution_portfolios/example-tasks.json --output runs/my-continuing-results-001.json
```

The 11-task example batch returns multiple checked answers in four mathematical domains, then exercises the earlier empirical investigation, integral, English request and physical route. The model loads once for the batch. Edit a copy of the JSON to supply your own tasks; results retain source attribution and the exact owner identity. The persistent runner keeps Windows awake during its owned job, with unrestricted authorized local time and the existing memory protection. [Solution task fields and complete examples](../research-continuation/44_solution_portfolios/README.md).

Use `solve_solution` for all applicable answers, `curiosity` to complete a new imagination/proof/reward cycle, and `solution_findings` to inspect retained questions and evidence. `--read-only` prevents new investigation. The [earlier batch](../research-continuation/42_composed_discovery/example-tasks.json) exercises four-language requests and more learned equations.

New task kinds are `gap_findings`, `gap_predict`, `gap_consequences` and `gap_next_observation`. [Input fields and examples](../research-continuation/43_knowledge_gaps/README.md).

Supported `kind` values are `request`, `read`, `sum`, `integral`, `imagine`, `discovered_route`, `discoveries`, `solve_discovery` and `observed_motion`. A `read` task supplies a question, attributed source and up to sixteen source sentences. Polynomial coefficients are ordered from the constant term upward. `discoveries` returns the retained portfolio with its independent evaluation; `solve_discovery` executes applicable learned coefficients and reports each route's assumptions. `observed_motion` preserves the source and measured range of the fitted model. Request interpretation returns an intent and entity tags; external actions have their own execution interfaces.

## Practice investigation and learn from checked progress

Begin with [self-chosen and composed discovery](../research-continuation/42_composed_discovery/README.md). Three elementary process examples precede a sustained investigation study. SERA selects missing-input questions, proposes fitted programs or combines owned rules, and retains independently checked coverage. Its [earlier internal discovery](../research-continuation/35_operator_discovery/README.md) and [learning-progress portfolio](../research-continuation/34_learning_progress/README.md) remain preserved. The completion command below uses its predecessor store.

```powershell
.venv/Scripts/python.exe scripts/run_completion_bounded.py --seconds 60 --output runs/my-completion-001 --module experiments.verified_completion.runtime -- practice --id my-response-1 --seed 32510
```

This starts a reproducible conditional task: propose a missing acceleration response, imagine consistent motion, choose evidence, independently check the revised prediction and retain the rewarded investigation weights. Reusing the task identifier resumes its saved stages. [Full interface, status and analytic recommendation](../research-continuation/32_verified_completion/README.md) · [Training and measured results](../research-continuation/32_verified_completion/report.md).

## Apply a taught definition to a situation

```powershell
.venv/Scripts/python.exe scripts/run_books_bounded.py --seconds 40 --output runs/my-grounded-example-001 --module experiments.book_learning.runtime -- imagine --request research-continuation/30_grounded_books/example-request.json
```

The saved example supplies an original force definition, 6 N net force, 3 kg mass and two seconds from rest. SERA checks its learned type against the taught source, applies the declared physical premises and returns **4 m position and 4 m/s velocity**. It also checks opposite-force and doubled-mass alternatives. These results certify conditional algebra, not an observed event.

[Choose entries, change the request, preserve sessions and investigate missing meanings](../research-continuation/30_grounded_books/README.md). The supported route is explicitly labeled taught-source reuse. [Full training and transfer results](../research-continuation/30_grounded_books/report.md).

## Read a source or investigate a gap

```powershell
.venv/Scripts/python.exe scripts/run_reading_bounded.py --seconds 60 --output runs/my-source-reading-001 --module experiments.human_reading.runtime -- read --question "What does physics study?" --source "my-attributed-document" --text "Physics studies matter, energy and their interactions. Botany studies plants."
.venv/Scripts/python.exe scripts/run_reading_bounded.py --seconds 80 --output runs/my-research-001 --module experiments.human_reading.runtime -- ask --id sparse-recovery-002 --question "What assumptions allow sparse recovery?" --gap "compressed sensing"
```

`read` returns ranked original sentences with offsets and source hashes. `ask` reads a supplied passage or, when no local source is supplied, retains the question and searches arXiv for the specified gap. The result contains source identities and candidate evidence for verification. A completed goal keeps its pinned evidence; use a new ID for a new investigation. State is preserved in `runs/sera-reading-live` and the output is in the bounded job's `process.log`.

`match` compares two to sixteen attributed alternatives using the learned semantic weights; repeat `--candidate "text"` for each alternative and supply `--source`. [Measured tasks, controls and source provenance](../research-continuation/29_human_reading/report.md).

For a fresh checkout, run `scripts/restore_test_artifacts.py`, `scripts/reading_artifacts.py --restore`, then `scripts/books_artifacts.py --restore`. These verify hashes and preserve divergent local files. The artifact restore does not create a new compute allowance.

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
