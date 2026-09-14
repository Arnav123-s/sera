# Shared-learner workspace records

I keep the initial attempts, numerical failures, development pilots, repaired comparisons and active solver in distinct locations. Earlier work is neither deleted nor moved. The [variant registry](variants.md) and [machine-readable results](../reports/shared-learner-data.json) link the relevant hashes and measurements.

| Location | Role |
|---|---|
| `runs/shared-learner-development` | Validation-only addressing, numerical-stability and scoped-adapter pilots, plus small plumbing tests. Early failures and weak validation results remain here. |
| `runs/shared-learner-complete` | First full attempt at source commit `4a530c9`: three completed delta trials and three failed spectral-gradient reference trials. This name does not imply every trial completed; inspect each `status`. |
| `runs/shared-learner-repaired` | Six paired follow-up trials at source commit `47429eb`, using fresh query cases. Each retains pretraining evidence, validation, candidate patches, raw scores and complete measured invocation costs. |
| `runs/shared-learner-assembly` | Initial failed assembly attempt. Reusing deep macros exposed an overdepth generated candidate. Its missing timing information is explicitly recorded in the review. |
| `runs/shared-learner-assembly-repaired` | Fresh R2 instrument/library, measured controller episodes, frozen controller, initial shared solver and ordinary corrective-learning result. A separate R2 snapshot preserves work before controller training. |
| `runs/sera-0.5-current` | Active shared solver and cumulative admission ledger. Initial version, promoted version, evidence revisions and learning records remain immutable. This is a fresh shared lineage, distinct from the historical 0.4 lineage. |
| `runs/shared-learner-review` | Original-file preservation baseline, initial-attempt disposition, source identities, exact checkpoint replays, live admission replay, examples and isolated package verification. |
| `runs/shared-learner-review/package-check/dist` | Installable 0.5.0 wheel built from the frozen source. Its isolated installation restored shared ownership and matched the executable source hash. |

The two original main attempts use the same three seed identities; they are not six independent delta seeds. Separate-model controls reuse the corresponding full-update typed weights with a frozen independent owner. Parent-referenced checkpoint files store only changed tensors and retain a hash-checked link to their base. These choices preserve actual work without duplicating every unchanged tensor.

The previous `runs/sera-0.4-current`, standalone typed cohorts, HMM and recurrent belief controls, early sequence cores, source reproductions and raw reference material remain at their original paths. Their earlier catalogs stay available. The public repository includes source, compact results, provenance and reproduction commands; ignored full models and original attachments are local artifacts.
