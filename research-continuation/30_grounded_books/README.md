# Human books and grounded definitions

[Use SERA](../../docs/START_HERE.md) · [Results](report.md) · [Architecture audit](architecture-audit.md) · [Checklist](checklist.md)

I connected taught physical definitions to the continuing learner's certified motion operator. The saved interface can apply an attributed, taught definition to a numerical situation, compare conditional alternatives, preserve the task and investigate missing concepts on arXiv. Source interpretation, learned computation and algebraic checks remain separately identified.

Run the supplied example in the prepared workspace:

```powershell
.venv/Scripts/python.exe scripts/run_books_bounded.py --seconds 40 --output runs/my-grounded-example-001 --module experiments.book_learning.runtime -- imagine --request research-continuation/30_grounded_books/example-request.json
```

The example supplies 6 N constant net force, 3 kg mass, two seconds and explicit one-dimensional Newtonian premises. SERA returns **4 m position and 4 m/s velocity** from rest. The opposite-force branch gives −4 m; doubling the mass gives 2 m. These are conditional consequences, not observations.

`library` lists original taught physical entries and their source identities. Copy the example request to a new local file to change the quantities and original task. Source term, definition and URL must remain attributable; units and assumptions are explicit. `--store runs/my-grounded-session` creates a separate retained session without touching another one.

```powershell
.venv/Scripts/python.exe scripts/run_books_bounded.py --seconds 40 --output runs/my-definition-library-001 --module experiments.book_learning.runtime -- library
.venv/Scripts/python.exe scripts/run_books_bounded.py --seconds 100 --output runs/my-grounded-research-001 --module experiments.book_learning.runtime -- imagine --request research-continuation/30_grounded_books/missing-concept-request.json --acquire
```

The second request retains a momentum task and searches arXiv for its public missing topic. Papers remain references to verify, outside the human-only foundation curriculum. Use a new task ID for a new goal. Complete tasks, attempts and concepts persist in `runs/sera-grounded-live`; results appear in each bounded job's `process.log`.

The interface also supports `read --question ... --text ... --source ...` and `research --id ... --question ... --gap ...` on this same extended owner. Earlier language, source-reading and empirical tools remain available.

For a fresh checkout, restore dependencies in order:

```powershell
.venv/Scripts/python.exe scripts/restore_test_artifacts.py
.venv/Scripts/python.exe scripts/reading_artifacts.py --restore
.venv/Scripts/python.exe scripts/books_artifacts.py --restore
```

Restoration verifies hashes and refuses divergent local files. It grants no numerical allowance. The bounded wrapper is the Windows research-workspace supervisor; the Python module can also run directly in the qualified Linux environment with one thread.

Review the frozen [book protocol](PROTOCOL.md), [definition protocol](GROUNDING_PROTOCOL.md), [named-entry repair](GD002_PROTOCOL.md), [admission rule](ADMISSION.md), [sources](sources.json), [attribution](ATTRIBUTION.md), [integration audit](integration-audit.json) and [costs](costs.json). Exact taught-source reuse is labeled explicitly and is not counted as novel-definition transfer.
