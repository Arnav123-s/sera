# Preserved SERA research variants

I retain useful alternatives as named research variants with their original checkpoints, results and limits. A variant can be a component or an evaluation route; it is not automatically a complete architecture. No experiment is discarded because it diverged from the handbook.

The [machine-readable registry](variants.json) records source identities, exact artifact hashes and measured results. The [shared-learner report](../reports/shared-learner-study.md) adds three named variants to the 22 preserved earlier variants. The [0.4 performance report](../reports/evaluation-v2-study.md) and [earlier workspace index](workspace-index.md) remain historical records. [New workspace roles](shared-workspace.md) identify both initial failures and the repaired shared runs.

| Stable variant | Role | Preservation boundary |
|---|---|---|
| `sequence-delta-v1` — SERA Sequence / delta | sequence-core alternative | Historical four-task sequence benchmark. Parameter counts differ. The hybrid is a useful retained hypothesis; it is not the same model as the full R1 reference core. |
| `sequence-gru-v1` — SERA Sequence / gru | sequence-core alternative | Historical four-task sequence benchmark. Parameter counts differ. The hybrid is a useful retained hypothesis; it is not the same model as the full R1 reference core. |
| `sequence-rotor-v1` — SERA Sequence / rotor | sequence-core alternative | Historical four-task sequence benchmark. Parameter counts differ. The hybrid is a useful retained hypothesis; it is not the same model as the full R1 reference core. |
| `sequence-real-v1` — SERA Sequence / real | sequence-core alternative | Historical four-task sequence benchmark. Parameter counts differ. The hybrid is a useful retained hypothesis; it is not the same model as the full R1 reference core. |
| `sequence-hybrid-v1` — SERA Sequence / hybrid | sequence-core alternative | Historical four-task sequence benchmark. Parameter counts differ. The hybrid is a useful retained hypothesis; it is not the same model as the full R1 reference core. |
| `sequence-collision-v1` — SERA Sequence / collision | sequence-core alternative | Historical four-task sequence benchmark. Parameter counts differ. The hybrid is a useful retained hypothesis; it is not the same model as the full R1 reference core. |
| `sequence-collision_no_cross-v1` — SERA Sequence / collision_no_cross | sequence-core alternative | Historical four-task sequence benchmark. Parameter counts differ. The hybrid is a useful retained hypothesis; it is not the same model as the full R1 reference core. |
| `sequence-density-v1` — SERA Sequence / density | sequence-core alternative | Historical four-task sequence benchmark. Parameter counts differ. The hybrid is a useful retained hypothesis; it is not the same model as the full R1 reference core. |
| `delta-world-v1` — SERA Delta World | compact R1 world model | Select component r1 from the v0 solver bundle. Four identifying sensors and supplied world identity; deterministic control, not an aliased stochastic belief planner. |
| `reference-world-all-v1` — SERA Reference / all | handbook-size R1 experiment | Full rotor/delta/density state. All-routing checkpoint was independently reconstructed in the audit. Trained with fewer updates and smaller batches than Delta World; this is not a matched ranking. |
| `reference-world-top1-v1` — SERA Reference / top1 | handbook-size R1 experiment | Full rotor/delta/density state. All-routing checkpoint was independently reconstructed in the audit. Trained with fewer updates and smaller batches than Delta World; this is not a matched ranking. |
| `aliased-projective-v1` — SERA Control / Projective Instrument | aliased-event predictor comparison | Matched likelihood parameter counts and 24-second training budget in a bounded 8-state/4-event world. HMM is the strongest measured control. Proposal-head parameters are accounted separately; model tuning was limited. |
| `aliased-complex-v1` — SERA Control / Complex Instrument | aliased-event predictor comparison | Matched likelihood parameter counts and 24-second training budget in a bounded 8-state/4-event world. HMM is the strongest measured control. Proposal-head parameters are accounted separately; model tuning was limited. |
| `aliased-real-v1` — SERA Control / Real Instrument | aliased-event predictor comparison | Matched likelihood parameter counts and 24-second training budget in a bounded 8-state/4-event world. HMM is the strongest measured control. Proposal-head parameters are accounted separately; model tuning was limited. |
| `aliased-hmm-v1` — SERA Control / HMM-22 | aliased-event predictor comparison | Matched likelihood parameter counts and 24-second training budget in a bounded 8-state/4-event world. HMM is the strongest measured control. Proposal-head parameters are accounted separately; model tuning was limited. |
| `aliased-gru-v1` — SERA Control / GRU-21 | aliased-event predictor comparison | Matched likelihood parameter counts and 24-second training budget in a bounded 8-state/4-event world. HMM is the strongest measured control. Proposal-head parameters are accounted separately; model tuning was limited. |
| `typed-neural-v1` — SERA Typed / v1 | typed capability route | Select component typed. Same neural weights with/without two selected arithmetic procedures. Original extended/structure sets duplicate three task families; use evaluation-v2 for corrected transfer evidence. |
| `typed-procedural-v1` — SERA Typed + Rules / v1 | typed capability route | Select component typed. Same neural weights with/without two selected arithmetic procedures. Original extended/structure sets duplicate three task families; use evaluation-v2 for corrected transfer evidence. |
| `program-reuse-v1` — SERA Program Reuse | world-specific guide and macro-library experiment | Verified finite transformations. Macro reuse expands primitive action length under the same token budget. The classical program-study guide was not saved as a standalone checkpoint; its results, seed and training recipe remain preserved. |
| `intervention-selector-v1` — SERA Intervention Selector | learned eight-method improvement selector | Nine saved policies, three seeds. No sustained advantage over the strongest fixed procedure. Policies trained against a fixed base; retained as a finite selector experiment, not evidence of recursive improvement. |
| `typed-neural-v2` — SERA Typed / v2 | typed capability route | Fresh training on semantic-v2 support/validation; withheld composition cases. Same architecture as v1. This cohort remains separate from the admitted world solver. |
| `typed-procedural-v2` — SERA Typed + Rules / v2 | typed capability route | Fresh training on semantic-v2 support/validation; withheld composition cases. Same architecture as v1. This cohort remains separate from the admitted world solver. |
| `shared-delta-v1` — SERA Shared Associative | common R1 owner | Three jointly trained seeds; world, typed and sequence routes share parameters. Simpler associative core, with full/replay/adapter/scratch/scoped and separate-model controls. |
| `shared-reference-v1` — SERA Shared Reference | handbook-sized common R1 owner | Three trained seeds with the 35,840-byte core state and explicit frozen-projector gradient approximation. Earlier spectral-gradient failures remain preserved. Equal update/example budget to Shared Associative; different parameters and runtime. |
| `shared-scoped-v1` — SERA Shared Adaptation | persistent shared R1/R2 solver | Preselected seed-0 delta owner, verified programs, freshly trained bounded controller and admitted first-binding weight residuals. Supplied applicability scope; all older solver lineages remain separate and intact. |

The five aliased predictors share one comparison protocol. The eight early sequence cores share another. Reference-size versus compact R1 uses different training budgets. Neural and procedure-assisted results are separate routes. I do not pool these into an overall accuracy ranking.

The new shared variants implement the next integration step. The HMM and hybrid remain strong controls to revisit at the appropriate bottleneck. The standalone v2 typed checkpoint remains a research variant; it has not replaced an admitted solver. The shared continuation starts its own declared lineage and keeps every older lineage available.

## New workspace roles

| Path | Role |
|---|---|
| `runs/evaluation-v2-complete` | Canonical three-seed comparison, semantic datasets, checkpoints, score vectors and all 15 retrospective proposal checks. |
| `runs/evaluation-v2-review` | Preservation baseline, independent replay verification and fresh live-continuation report. |
| `runs/evaluation-v2-smoke` | Four-update development plumbing check with frozen variants; excluded from performance evidence. |
| `runs/evaluation-v2-fresh-smoke` | Four-update from-scratch entry-point check; excluded from performance evidence. |
| `runs/sera-0.4-current` | Active continuation fork with the prior ledger and one new rejected candidate; old 0.3 parent unchanged. |

Checkpoint and raw-data paths in the registry refer to preserved local artifacts. They are not implied to be included in a fresh Git checkout. The public repository contains code, compact evidence, hashes and reproducible recipes.
