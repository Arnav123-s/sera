# Learn a body's dynamics from observations and predict conditional motion

The prepared local continuation uses the same parameter owner as Stage 27. Give it an ordered observation history, let it fit and validate a force model, then ask for a trajectory under specified future commands. Learned coefficients remain registered weights; subject histories, assumptions and predecessor models remain separate records.

The examples below run in `D:/ai/projects/sera`. They use the existing one-thread, 2 GiB, charged execution wrapper. Each `--output` must be fresh. A numerical reservation is refunded for unused time; inspect `runs/v3-batch-001/budget.json` before new work. No new allowance is implied by these commands.

```powershell
# Use the saved body-1 model, including its latest observation and correction.
.venv/Scripts/python.exe scripts/run_concept_bounded.py --seconds 60 --output runs/CR-my-forecast-001 --module experiments.concept_refinement.runtime -- predict --subject body-1 --commands "0.8,0.8,0.8,0.8"

# Learn from the supplied synthetic example in a separate new persistent session.
.venv/Scripts/python.exe scripts/run_concept_bounded.py --seconds 60 --output runs/CR-my-task-001 --module experiments.concept_refinement.runtime -- task --store runs/my-refinement --subject body-1 --goal my-motion-task --input research-continuation/28_concept_refinement/example-observations.json --commands "0.8,0.8,0.8,0.8"

# The original learned integral operators still provide exact polynomial algebra.
.venv/Scripts/python.exe scripts/run_concept_bounded.py --seconds 60 --output runs/CR-my-algebra-001 --module experiments.concept_refinement.runtime -- motion --coefficients "2,3,1" --time 3 --position 5 --velocity -1

# The four-language request interface uses its retained weights.
.venv/Scripts/python.exe scripts/run_concept_bounded.py --seconds 60 --output runs/CR-my-language-001 --module experiments.concept_refinement.runtime -- interpret --text "enciende las luces"
```

The wrapper writes the answer to the job directory's `process.log` and records wall time and committed memory in `state.json`. Direct Python callers can use `RefinementSession.autonomous_refine`, `observe`, `refine`, `predict`, `exact_motion` and `study.interpret` under the same resource supervision. Default persistent store: `runs/sera-refinement-live`; the Stage 27 store remains unchanged.

## Supply observations

Use a JSON list like [example-observations.json](example-observations.json). Each entry has `subject`, `time`, `position`, `velocity`, `command`, `available`, `units`, `source` and `evidence`. Units are metres, metres/second and a dimensionless command; observations arrive on a .05-second clock. A past command applies from its observation until the next observation. The fitting window contains 81 readings / 80 transitions. Missing sensor values are `null` with a false availability flag; first and current readings must anchor the required state. Source identities stay with every reading. Mark simulator readings SYNTHETIC; mark actual sensor readings OBSERVED. The runtime preserves that distinction.

Predictions cover one to twelve command steps under the supplied airborne one-dimensional model family, normalized gravity=1 and command bounds [-8,8]. Input acceptance is a schema/range check, not evidence that the physical family applies. Held-out observed residual, identifiability and the development/final gate govern admission. Broader dynamics require new empirical qualification.

`task` detects a missing model, retains supplied observations, fits candidate coefficients, independently checks their executable recurrence, updates registered weights and returns to the unchanged goal. `observe` adds later readings; `refine` creates a new supported version. A changed observation history invalidates stale state. `predict` creates conditional branches without altering evidence or weights. Previously learned models and their support remain available for audit.

An EMPIRICAL_PREDICTION is conditional on its premises and is not an occurrence certificate. A CERTIFIED_ALGEBRA result checks the supplied polynomial calculation; it does not certify that a polynomial describes a body. MISSING_KNOWLEDGE, NEEDS_REFINEMENT and UNIDENTIFIABLE_CAUSE preserve the concrete next information need. Residual-based uncertainty is not calibrated coverage against omitted mechanisms.

## Evidence and continuation

Read [report.md](report.md), [the frozen protocol](PROTOCOL.md), [F01 repair](F01-simple-initial-condition.md), [architecture audit](architecture-audit.md), [checklist](checklist.md) and [continuation state](continuation.json). The corrected scientific results are in `r2/`; top-level development records preserve the first attempt. Checkpoint/result archives preserve local relative paths and source hashes. Completed final results are evidence to inspect, not a tuning set.
