# Checked answers from self-chosen questions

The continuing owner has **217 qualified rational solution routes** across polynomial mathematics and three motion models. It returns all applicable alternatives, their exact expressions, assumptions and proof records. The new routes make **58 saved questions executable beyond the previously retained routes and forward procedures**.

[Results and examples](report.md) · [Protocol](PROTOCOL.md) · [Original audit](audit.json) · [Current owner and credit audit](coverage-audit.json) · [Interruption and reward replay](runtime-audit.json) · [Checklist](checklist.md)

## Run the saved learner

From the prepared repository:

```powershell
.venv/Scripts/python.exe scripts/run_solution_persistent.py --output runs/my-solutions-001 --module scripts.solution_use -- --input research-continuation/44_solution_portfolios/example-tasks.json --output runs/my-solutions-001/answers.json
```

The batch exercises four solution portfolios plus the earlier empirical investigation, integral, language and physical route. Each result identifies its actual parameter owner. The wrapper uses one CPU thread, a 2 GiB process-tree memory ceiling, checkpointing and a Windows keep-awake request. The user-authorized time allowance is unrestricted; the request is released when the owned job ends.

To ask for a missing quantity, create a JSON list containing a task such as:

```json
[{"id":"infer-mass","kind":"solve_solution","domain":"motion_1","target":"m","observations":{"a1":"1","f":"12","t":"2","v":"7","v0":"1"}}]
```

Use `motion_0`, `motion_1` or `motion_2` for constant, linear or quadratic acceleration in time; `polynomials` covers the retained quadratic polynomial, its integral and finite sum. Input names, units and assumptions follow the retained model. A solution is conditional on those premises and its explicit denominator guard.

`solve_solution` returns every applicable retained route. If the task needs a new route, it normally completes an investigation before returning: imagine, exhaust the declared alternatives, commit, independently prove/check, answer and reinforce verified progress. `curiosity` explicitly runs that cycle even when an answer is already available. `solution_findings` returns the saved questions, proofs and credit. `--read-only` inspects the existing portfolio without initiating a new investigation.

Unfinished proposals keep their original observations, exact predictor and commitment. Resuming the same request completes that state. A changed request preserves the unfinished goal. Repeated completed questions add no credit or weight update. The earlier task kinds remain available through the same owner.

## Restore and preserve

The current store is `runs/sera-solution-progress-live`. The original completed study remains at `runs/sera-solutions-live`; earlier owners are unchanged. The [release manifest](release-manifest.json) pins the complete proposal history, original and corrected checkpoints, failed fits and source identities. On a checkout with the [parent dependencies](../43_knowledge_gaps/release-manifest.json) restored, run `python -m scripts.solution_artifacts --restore`. Every restored byte is verified; divergent local work is preserved.

The [repository workflow](../../.github/workflows/ci.yml) gives the complete fresh-checkout restoration order. Public historical assessment receipts are verified through the [read-only replay adapter](../43_knowledge_gaps/publication/portability.md).
