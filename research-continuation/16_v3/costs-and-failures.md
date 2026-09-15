# Costs, failures and interpretation

The [supervisor ledger](worker-budget.json) records all bounded jobs from the first tests through full regression. It charges actual elapsed worker time, reserves the whole cap before launch, tracks owned descendants and preserves failed jobs. No pre-existing process was stopped. An intentionally short deadline terminates only its test-owned job and remains labeled TIMEOUT.

| Item | Supervised seconds | Result |
|---|---:|---|
| GG-GUARD-002 cohort | 134.374 | Complete; conditional gate passes |
| GG-GUARD-002 independent audit | 22.727 | Pass |
| GC-001 cohort | 5.219 | Correct; full-cost proxy fails |
| GC-002 cohort | 5.185 | Correct; per-candidate proxy passes |
| GC-002 explicit migration/repair checks | 2.728 | Pass on 1,024 fresh cases |
| Full regression | 32.864 | 199 passed |

Other targeted tests, intake verification, metadata collection, source review and archive writing remain distinct from these rows. The collector records 13.167 seconds; the separate intake compatibility diagnosis records 1.387 seconds. Complete assistant/orchestration time, energy and development effort were not measured. The CPU worker ledger is not a claim about total project cost or all device activity.

The supervised maximum during current regression was 915,619,840 reported committed job bytes, including deliberate failed-allocation testing. These counters are not RSS. The actual cohort peaks are separately recorded. The 2 GiB configured job ceiling was tested by reading its settings and checking allocation rejection before launching research jobs.

## Preserved unsuccessful work

- **Strict packet portability:** certificate upper bounds differ by tiny floating-point amounts and change canonical hashes. Alias least-squares parameters also fail exact equality. Original strict results remain FAIL; the separate compatibility pass requires exact decisions/counts and narrowly bounded numerical discrepancies.
- **Resource-test assertion:** the first test incorrectly assumed a high-water counter could never report above the configured cap after an allocation failure. Allocation rejection worked. The original source and failed log remain.
- **Preflight import:** a direct script launch could not import the repository package. No protocol or research world had been generated. Module invocation fixed the launcher; the empty failed run is retained.
- **GC-001 scientific failure:** 732,614 versus 621,064 operations at the planned scope. It is not relabeled as a success because individual calls are faster.
- **GC-002 proof-domain defect:** a reused forward table is invalid if the relation depends on the right-hand-side input. A counterexample fails the old checker; the repaired checker explicitly rejects that domain.
- **GC-002 interpreter defect:** the initial facade checked graph-internal hashes without requiring the actual running interpreter. The current facade does both; old graphs require explicit reproof migration.

The later repairs do not erase either frozen cohort. The current usable graph and the benchmarked graph have different identities. The frozen GC-002 15.1% advantage includes its own proof and restore costs but excludes the unsuccessful earlier development candidate and later engineering repair. Those costs are all retained and prevent a claim of net project-wide savings at this stage.

## Evidence boundary

The neural guard had already been trained in GG-GUARD-001; this batch adds zero neural optimizer steps. Threshold calibration consumes 1,800 observed query labels. The full guard cohort consumes 41,932 observed coordinate pairs including its prefixes. Each consolidation candidate has 18 teaching traces, plus all counted exhaustive finite proof evaluations; calling that simply an 18-label learning achievement would omit the supplied semantics and proof work.

The working state is compact, while audit archives and inherited checkpoints are larger and remain external. No infinite arbitrary memory, physical quantum computation, general intelligence or independently improved investigator is claimed.
