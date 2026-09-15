# SERA / Kavi — calibrated scope and guarded computation

I continued from SERA `9104839400d5053152ca423be6da1efe355aa71a` and Kavi `50f743cc44794b67bc1d30b927e25991197edce2`. Both worktrees were clean. I read the updated continuation brief, reconciled the actual local evidence, and completed three candidate experiments plus an implementation repair. Existing training, checkpoints, failed candidates and source archives remain preserved.

## Results

| Work | Executed result | Decision |
|---|---|---|
| [GG-GUARD-002](GG-GUARD-002/report.md) | 1,800 independent calibration worlds; 2,012 assessment worlds. Grouped policy answers 664/1,000 matched queries with zero observed errors. | Passes the predefined conditional calibration gate. No arbitrary-shift or shared-neural-transfer claim. |
| [GC-001](GC-001/summary.json) | Correct finite guarded programs, but 732,614 total counted operations versus 621,064 for local reasoning. | Reject the efficiency claim; preserve the candidate. |
| [GC-002](GC-002/report.md) | Shares forward work during exact proof. A fresh stream uses 527,156 counted operations versus 621,064 for local reasoning, including its acquisition, proof, correction and restore. | 15.1% per-candidate work-proxy reduction within supplied finite semantics. |
| [Contract repair](GC-002/repair/result.json) | Rejects an invalid proof domain and stale interpreters. Explicitly migrates two saved graphs, reproves their programs and checks 1,024 fresh cases. | Current usable checkpoints are in `GC-002/repair/`; original graphs remain unchanged. |

The grouped guard is more conservative than necessary for some tasks. The global certificate and simple 0.90 threshold have higher matched utility. Abrupt jumps receive **zero answers**, random-value worlds receive **zero answers**, and only 3/169 local-exception worlds receive answers. These limitations are part of the result, not hidden failures removed from the denominator.

The consolidation learner receives 18 successful finite reasoning traces per candidate lifetime: six for an affine equation, six for an independent offset equation, and six after changing the affine relation. It parameterizes the concrete values into executable inputs. Field arithmetic, formal role mapping and inverse reasoning rules are supplied. The numerical R1 weights are not retrained. This is a bounded executable-knowledge contribution, not discovery of general mathematics or improvement of the learning algorithm.

## Evidence and verification

- [Reconciliation](reconciliation.json): initial HEADs, clean status, pack identity, collector scope and job inspection limits. The private collector remains local.
- [Intake diagnosis](intake-replay-diagnosis.json): original strict calibration and alias replay failures remain FAIL. A separate compatibility check recovers identical decisions/counts with tiny numerical differences.
- Pack manifest: **133 entries / 22,604,808 bytes** verified after restoring the omitted baseline ZIP from the existing local download. Standard tests: **44 passed, 10 optional old-demo tests skipped**.
- [Current regressions](tests.xml): **199 passed**. The earlier completed 174-test release and its CI evidence were reused as historical evidence.
- Independent audits: [all 3,812 guard worlds](GG-GUARD-002/independent.json), [GC-001](GC-001/independent.json), [GC-002](GC-002/independent.json). Both consolidation audits check 56,320 policy answers across 11,264 case occurrences; growth stress deliberately repeats paired cases.
- [Preservation](preservation.json): all **45 shared-owner source files** and all **108 explicitly collected predecessor run files** retain their bytes. The earlier broad repository collector stopped at its 3,000-file cap; it was not a complete disk inventory.
- [Architecture comparison](architecture-audit.md), [research basis](research-notes.md), [failure and cost notes](costs-and-failures.md), [current state](state.json).

The frozen source archives preserve each candidate's exact code. GC-001 and the original GC-002 predate the later proof/interpreter repair. Their saved results remain associated with those versions; their old interpreter identities are deliberately refused by the current loader. The repaired graphs have new identities and a separate migration record.

## Resources and costs

The recorded conservative allowance is one active CPU worker with one numerical thread, at most 900 seconds per job and 3,600 aggregate supervised worker seconds. The native Windows job limit is 2 GiB committed memory. Launchers and owned child processes are accounted for together. [Every job state and log](supervision/) and the [remaining allowance](worker-budget.json) are retained, including failed tests, the failed preflight and the intentional timeout control.

The per-candidate 15.1% consolidation result does **not** amortize the earlier rejected candidate or the later repair into that candidate's fixed test stream. Those costs remain in the project ledger; including all research, audit and repair work does not establish a net project-wide speedup. Operation counts are a declared heterogeneous proxy, never FLOPs. Guard training is inherited from GG-GUARD-001 and was not repeated. Controller training and new neural optimizer steps are zero.

## Current scope and next research dependency

One existing trained owner carries the finite library by identity, and its weights remain unchanged. Positive transfer into a different learned route has not been demonstrated. The guard remains a conditional research component. The original probability-cloud experiments, strong generator results and unsuccessful branches remain available in their existing archives.

The next uncompleted architecture item is **A06**: no-program, detached-program, integrated-learning and equal-capacity controls on a different learned task at matched exposure and full cost, with retention. A08 then needs continuing hidden-state interaction and useful intervention-sensitive imagination. Learned latent migration, sustained plasticity, new investigator eta, crossed K × eta tests and actual improved successor generations remain open. The architecture is not complete, and M2–M4 are not established.

## Use the repaired finite library

From the repository root, using the existing local trained parent:

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.resume_v3_library --a 4 --b 5 --c 6
.venv/Scripts/python.exe -X utf8 scripts/verify_v3_release.py
```

The first command restores the corrected relation `a*x - b = c (mod 11)` and returns its conditional solution. A zero multiplier is unsupported. It does not modify the historical parent or acquire new observations. These commands do not restart training or the completed studies.
