# Shared R1 learner: source contract and experiment

I implement the handbook's R1 task mixture through one learned event/state core, with typed modality adapters and separate output heads. World, typed and legacy sequence interfaces refer to the same registered parameter owner. R2 remains a small instrument and verified program interface, as the handbook permits. Existing variants remain immutable research records.

## Source mapping

| Source handbook | Implementation obligation |
|---|---|
| Lines 85–105: nine-entry block and three time scales | Define input, state, output, parameters, transition, readout, update, validity and cost. Distinguish working-state transitions from retained parameter updates and improvement-policy updates. |
| 175–225: typed events | Preserve modality, units, position, availability and provenance. Keep targets outside observation features. |
| 422–446: R1 state and forward equations | Share the event core and aggregation; condition world predictions on actions. Train the 256-wide, 128-complex-rotor, 8×32² associative, 4×16×4-complex-factor reference as an explicit condition. |
| 450–475: learning and decisive tests | Mix tasks through shared parameters, inspect key/addressing failures, compare useful simpler memory against optional branches. |
| 477–547: R2 | Preserve likelihood/conditioning consistency, bounded execution and a verified library. No vocabulary-sized dense operator expansion. |
| 941–947: Recipe 2 | Joint supervised learning from scratch; development, validation and composition tests separated. |
| 951–957: Recipe 3 | No update, full update, adapter-only, replay and scratch comparisons; show support sizes, transfer, retention, calibration and actual costs. |
| 977–997: Recipes 6–8 | Later improvement-policy claims require actual sequential histories, a fixed anchor, cumulative frontier and a full-budget baseline. A successful shared-core implementation does not alone close these claims. |

## Block contract

Inputs are the existing typed observations, explicitly instructed 20-feature sequence events, and action-conditioned sensor/reward trajectories. A working state is owned by one episode and one model version. The core is reset between independent episodes. Parameters live once in `SharedR1`; typed and sequence views own no duplicate tensors. Output heads return categorical predictions, meter coordinates, next-observation distributions and reward predictions. Verified arithmetic procedures remain a separately measured route.

Learning uses admitted simulator examples, optimizes the shared owner, and mixes retained evidence when replay is selected. Validation selects a checkpoint; query/promotion examples never update it. Serialization must reconstruct object sharing, and resumed predictions must match before-save behavior. A frozen parent survives every candidate. Per-capability admission from 0.4 remains in force.

## Planned controlled study

The main comparison uses shared delta and the full reference state, both with event width 256 and eight 32-dimensional associative heads. The reference adds the specified rotor and low-rank workspaces. Equal example/update budgets are recorded together with unequal parameter and runtime costs; this is not a claim of exact compute matching. Three independent seeds are required for a final comparison. Development pilots are labeled and retained.

Joint pretraining covers typed modalities, the existing sequence tasks and one action-conditioned world. A novel binding rule carries an explicit instruction and uses disjoint support, validation and test cases. Query-key overwrites must change the correct result, so success cannot come from treating first and last binding as equivalent. Longer sequences and additional overwrites form transfer tests.

Compare no update, full adaptation, adapter-only adaptation, replay, scratch and a separate-model control at declared support/update budgets. The separate control preserves the world's owner and adapts an independent typed core; its extra parameters are reported. The shared condition must prove that typed gradients reach the same memory parameters used by world and sequence inference. This does not imply that arbitrary new rules or general intelligence have been learned.

The frozen main budget is 1,600 joint updates, followed by 192 updates per adaptation condition at 32, 128 and 512 support cases, with batch size 32. Replay splits the batch equally between new evidence and one retained stream; it receives half the novel-example exposure of full updating. All candidates freeze before scoring 256 cases in each novel/retained instructed-binding family, 256 retained world/legacy episodes, and 128 cases per typed task and partition. Neural and verified-procedure routes are reported separately. Seeds 0, 1 and 2 are separate from development seed 41. Each process uses one CPU thread; concurrently executed seeds share the machine, so process CPU costs and concurrency must accompany wall times.

All controls select checkpoints using only novel-binding validation accuracy, with the earliest maximum retained. The separate-model control reuses the full-update typed weights and a frozen pretraining owner for world/sequence tasks; it is a paired architectural counterfactual with twice the model parameters, not an additional independent training run. Fixed study comparisons use the same sealed cases, so their admission arithmetic is descriptive. Only the ordinary solver's separately frozen proposal and freshly drawn evaluation seed can produce a durable promotion.

Development exposed unstable binding addresses. The correction gives eligible symbolic binding events a learned key-only address projection shared across typed and sequence interfaces; query and write keys use the same learned projection. Payload values, rule-dependent update gates, fusion and outputs remain learned. The implementation does not execute an earliest/latest binding algorithm. Categorical features use unit scale and instruction embeddings initialize at standard deviation 0.05. Earlier unsuccessful pilots remain labeled as development evidence.

## Amendment after the first full run

The first three delta trials completed, but all three reference trials stopped on non-finite gradients. The delta trials also exposed severe old-capability regression under unrestricted binding adaptation. Those trials remain in `runs/shared-learner-complete` at source commit `4a530c98d37493d5b8fef7dc0d048476de6144f9`; they are not silently replaced with successful runs.

The repaired study uses the same pretraining/update/support budgets and fresh query seed namespace 1,300,000. It adds a scoped rank-at-most-16 weight adapter, enabled only for explicitly instructed first-binding episodes. The base parameters stay frozen during this adaptation; learned low-rank residuals alter the existing core projections and readout. It is a supplied applicability boundary, not a discovered task router or a separate complete model. All unrelated input routes use the unchanged base exactly. The separate-model control remains in the comparison.

The reference keeps the same spectral rank truncation in its forward computation, but freezes the selected right-singular subspace during each backward pass and clamps the mixture gate to `[1e-6, 1-1e-6]`. This is an explicit approximate gradient, not the exact derivative of spectral truncation. Repeated/small singular values make singular-vector derivatives unstable; the implementation also avoids saturated square-root mixture derivatives. The mathematical limitation is documented in [PyTorch's SVD reference](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html). Tests cover degenerate spectra, saturated gates and valid normalized state.

The repaired run is a follow-up after observing failures, not a preregistered claim covering all development decisions. Repeated seeds are paired recreations, not six independent delta seeds. The final record must include the earlier failure costs and any unmeasured development cost. A longer ordinary correction may use 1,024 updates; its result must be kept separate from the 192-update comparison.

## Acceptance checklist

- [ ] One registered learned core serves all three interfaces before and after serialization.
- [ ] Reference state matches 35,840 core bytes with valid normalized workspaces.
- [ ] All task gradients reach the shared memory; no target enters the input.
- [ ] Support/validation/query cases and rule/composition boundaries are audited.
- [ ] Models train from scratch with explicit complete budgets and saved checkpoints.
- [ ] All adaptation controls run on paired data; retention and calibration are reported.
- [ ] A failed query can trigger admitted corrective evidence and a persistent candidate through the ordinary interface.
- [ ] Candidate evaluation remains independent; parents, rejected candidates and previous studies remain available.
- [ ] Independent checkpoint replay, source comparison and Windows/Linux checks pass.

The final report will record which behavioral criteria passed. Negative learning results are retained and do not become completion claims.
