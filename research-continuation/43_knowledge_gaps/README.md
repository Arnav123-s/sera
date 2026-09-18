# Use SERA's self-identified gap investigation

SERA examined its retained evidence, selected an incomplete observation record, detected a mismatch and searched 12,288 candidate programs. The continuing owner retains eight qualified empirical explanations alongside all earlier capabilities.

[Detailed findings](report.md) · [Frozen protocol](PROTOCOL.md) · [Independent evaluation](final.json) · [Owner audit](audit.json) · [Costs](publication/costs.json)

From the repository root, run the seven-task example using the same bounded local supervisor:

```powershell
.venv/Scripts/python.exe scripts/run_gap_bounded.py --seconds 120 --output runs/KG-user-job-001 --module scripts.gap_use -- --input research-continuation/43_knowledge_gaps/example-tasks.json --output runs/KG-user-results-001.json
```

Use fresh output paths. The batch includes the learner's chosen gap, alternative predictions, a proposed discriminating observation, the missing acceleration implied by each explanation, an exact polynomial integral, English intent interpretation and an earlier learned physical route.

Available new request kinds:

| Kind | Input | Result |
|---|---|---|
| `gap_findings` | No assigned problem | Saved self-selected question, source identity, learned program weights, assessments and open follow-up |
| `gap_predict` | `times`, a list of source-clock seconds | All qualified predictions, disagreement and the applicable earlier route |
| `gap_consequences` | `time`, in source-clock seconds | Each learned program's position, velocity and acceleration; difference from the retained acceleration |
| `gap_next_observation` | No target supplied | The coordinate where the surviving explanations disagree most |

The examples use absolute source-clock seconds. The earlier `observed_motion` request retains its elapsed-time convention. The new interface routes queries inside the original verified interval through that preserved model. A query beyond the investigated source range remains an explicit acquisition goal.

The latest store is `runs/sera-gap-inquiry-live`. Its parent reference resolves the existing `runs/sera-observed-discovery-live`; the parent is preserved exactly. To restore a fresh checkout, first restore the parent archives using `scripts.self_chosen_artifacts` and `scripts.frontier_artifacts` and their documented dependencies, then run:

```powershell
.venv/Scripts/python.exe scripts/run_gap_bounded.py --seconds 120 --output runs/KG-restore-job-001 --module scripts.gap_artifacts -- --restore
```

[Exact resumable state and next action](NEXT_ACTION.md) · [Example input](example-tasks.json) · [Example output](example-results.json)
