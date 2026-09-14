# Connected-build development log

I preserve development failures separately from the final study. Final evaluation pools are not used for architecture decisions.

1. The new persistent solver passed fresh-process skill use, a second acquired skill, rejection and behavioral rollback. A historical test then found a missing top-level dataset identifier in the new admission result; I restored that output contract.
2. The first R1 probe trained for 150 updates on fully observed eight-step trajectories. On a development pool of 64 partially observed twelve-step trajectories, accuracy changed from 28.26% to 25.26%. This did not demonstrate useful world learning. I identified an architectural omission: the aggregation used only memory readout, whereas R1's forward equation also includes the current event embedding. I added that direct event-to-aggregation connection and introduced masked observations in subsequent training probes.
3. A 20-update controlled-instrument probe reduced its batch loss from 1.3441 to 0.6488, with channel/instrument completeness residuals below 3e-7. This is an optimization probe, not a final performance estimate.

4. After fusing current events with memory and using masked training, a development run reached 83.53% on its separate masked probe after 600 updates. This used four 150-step calls with optimizer resets; it is excluded from the final study.
5. The seed-901 end-to-end smoke run used intentionally small budgets. It exercised both symbolic promotions, a rejected feedback update, learned method selection, promotion/rejection, saved component reload and rollback. Its scores are development observations, not reported final estimates.
6. Lint caught an omitted `control_table` import in the new study controls. I corrected it before starting the first full run.
7. A provenance review during the first full run found that known-world meta-training and validation episodes could reuse support samples because the support seed omitted the split. Final reset-family tests had not run or been read. I stopped the partial run after meta-training episode 5, retained `runs/connected-v2`, derived support/diagnostic/update seeds from each complete episode identity, and added support-record identifiers to the artifacts. I restarted all three seeds in `runs/connected-v2-final`. No learning hyperparameters or evaluation thresholds changed because of this correction.

I retain these development failures separately from the final three-seed evidence.

## Original-document audit after publication

I reread the supplied originals and audited commit `00d61b8` against their contracts. Two counterexamples justified a 0.2.1 repair: query-marked traces could enter instrument program-credit updates, and fixed search constructed the full action enumeration before applying its execution cap. Program credit now passes the existing admission validator before any update. Fixed search validates length and budget and materializes only its bounded prefix. Valid-input training and 216 search outcomes agree with the baseline; cost counters intentionally reflect less construction.

I also identified incomplete live-session persistence, automatic library composition, general instrument memory, selective routing, approximation diagnostics and the full failure/curriculum loop. The [source audit](../reports/original-source-alignment-audit.md) keeps these open. No learning hyperparameters, original evidence tables or promotion thresholds changed in this repair. Post-publication library-removal probes are diagnostic evidence, excluded from the original generalization estimates.


## SERA 0.3 audit follow-through

I completed C01–C11 of the new architecture checklist with owned live state, general event instruments, library composition, typed encoders and procedures, active evidence, persistent failure/budget history, adapters, reference routing, useful lineage retrieval and successive policy updates. Self-review also caught duplicate final-event scoring, cross-method evidence contamination and all-missing-target averaging; regression tests cover the fixes.

The final source-frozen cohort contains three seeds, 558 intervention outcomes, nine policy versions and fifteen persistent proposals (three promoted, twelve rejected). Arithmetic procedures reach 100% on declared structural draws while neural arithmetic remains near chance. The classical belief model reaches 100% on aliased history, ahead of the general instrument. These are controlled findings within supplied finite tasks, not general intelligence or quantum advantage. The original benchmark was separately reproduced, including 36 fresh training runs.

The artifact audit, isolated installation, fresh-process behavior, continued CLI learning and report figure are verified. Public CI remains the final publication check. The new report retains weak binding, full-reference undertraining, retention failures, policy regressions and incomplete development-cost observations.
