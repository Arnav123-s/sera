# SERA audit against the original research documents

I rechecked the original ZIP, both supplied physics PDFs, the executable implementation and the recorded study. **SERA follows the recommended first R1/R2 research direction, but it does not yet implement the full architecture or complete the training plan.** The connected 0.2 build is a narrow, working experiment. Treating its completed engineering checklist as completion of the handbook would be inaccurate.

This audit found two reproducible implementation defects, repaired in 0.2.1, and several substantive architecture and evaluation gaps. I preserve the original 0.2.0 results as historical measurements. I do not assign a compliance percentage: a missing connection matters more than the number of matching names or dimensions.

## What I checked

The baseline is public commit `00d61b8ce4de4c27afabb743cb54b5ddae7ea682`, with executable source SHA-256 `59c0835fba356932d945dc67c04b781aa5549702ee40389b09b9e03c73016594`. The 0.2.1 repair has executable source SHA-256 `69a7e361cae0bb337491647bafda862b645814dbcf2537d8d56d636b8bb12d35`. This hash covers the Python implementation, not every documentation or test file.

I freshly hashed all three original files, reread the handbook directly from the ZIP, compared its editable manuscript byte-for-byte with the local extraction, extracted the PDF text again, and visually inspected the relevant architecture and instrument diagrams/equations. The evidence record contains the complete hashes and page counts. Original scripts and checkpoints were not executed or loaded.

| Original source | Audit anchors | Identity check |
|---|---|---|
| `Quantum_AGI_Research_Package.zip` | `Quantum_AGI_Architecture_Handbook.md` and its 82-page PDF; source graph and R1/R2 diagrams | SHA-256 starts `7e9268f8a0281190`; editable handbook starts `2371a18a074680d1` |
| `Quantum_Physics_Deep_Knowledge_Atlas.pdf` | 86 PDF pages; chapters 1, 10 and 24; concept registry | SHA-256 starts `d72d68f196a01d3a` |
| `Quantum_Physics_Layered_Maps.pdf` | 14 maps; especially maps 1, 2, 4 and 7 | SHA-256 starts `62278878de0d41d7` |

All bytes and hashes match the original [source manifest](../research/source_manifest.json). I compared the actual graph identifier sets, not just their sizes: the ZIP's graph and SERA's physics map contain exactly the same 154 unique identifiers. This verifies traceability, not implementation of 154 cognitive functions.

Below, **H** means the original handbook; page numbers are one-based PDF pages and match its printed pages. Editable manuscript line numbers identify the exact source passages. **A** means the atlas; its printed page numbers are eight below its PDF page numbers. **M** means a numbered layered map. These references distinguish the source's requirements, explicit small prototypes and future hypotheses.

## Requirement-by-requirement findings

| Requirement and original anchor | Implementation evidence | Finding |
|---|---|---|
| Begin with narrow R1 plus one R2 program mechanism; H p. 52, lines 1256-1272 | Connected world prediction, action execution, instrument-guided search, solver and admission paths | **Aligned direction.** R3-R8 are alternatives/future branches; implementing all eight immediately is not required by this recommendation. |
| Distinguish representation, reduction, approximation and empirical evidence; A chapter 1, PDF pp. 9-10; M1, M2 and M4 | Independent state workspaces, explicit low-rank approximation, classical execution and separate measured reports | **Aligned claim boundary.** A bank of independent density factors is not a joint entangled register. Physics terminology and a preserved concept map do not demonstrate general intelligence or quantum advantage. |
| Declare input/state/output, transition/read/learning, validity and cost contracts; three update rates; H p. 7, lines 85-105 | R1/R2 methods, replay, numerical validity checks, `Work`, learned policy | **Partial.** Concrete narrow interfaces exist, but the common typed event/session contract is incomplete. |
| Preserve modality, position, scale/units, mask and provenance in typed events; H pp. 10-11, lines 175-225 | 22 supplied symbolic R1 features, sensor mask, trajectory provenance | **Narrowed.** No general multimodal encoder/decoder, unit-aware event adapter or learned representation discovery. |
| Reference R1 dimensions; H p. 18, lines 420-426 | `ReferenceMemory`: 256-wide embedding, 128 complex rotor coordinates, eight 32x32 memories, four 16x4 complex factors | **Dimensions implemented and tested.** The 35,840-byte number counts retained core tensors only. The trained model uses the compact delta core. |
| Route events to selected components and request relevant readouts; H p. 18, line 426 | `ReferenceMemory.step()` evaluates all three branches and mixes their outputs | **Narrowed.** This is readout weighting. It does not implement selective event delivery or conditional compute. |
| Condition observation/reward prediction on retained state and actions; H pp. 18-19, lines 430-459 | `RecurrentWorldModel.observe/predict`, `WorldSession.plan`, `planned_evidence`, replay updates | **Implemented for symbolic worlds.** The connected path exists; argmax imagined observations do not constitute a calibrated stochastic belief planner. |
| Track controlled low-rank approximation and its error; H p. 14, lines 308-320; A pp. 69-71 | Factor expansion, SVD truncation, renormalization; full-density reference test | **Partial.** `last_discarded_mass` keeps only the most recent batch maximum. There is no persisted per-trajectory error history, rank sweep or full error/cost curve. |
| Shared event instrument for likelihood, conditioning and generation; H pp. 20-22, lines 483-537; A pp. 31-32; M7 | Learned action CPTP maps; shared learned projectors; ignored events sum branches; independent likelihood check | **Valid restricted case.** Rank-one projective observations reset visible-state memory. This is much narrower than the general multi-Kraus-per-event instrument. |
| Predictive state guides executable programs; real traces teach the model; H p. 21, lines 519-523 | `ranked_programs`, `search_program`, verification, instrument likelihood/proposal credit | **Implemented for a finite action grammar.** Source dimensions and vocabulary are supplied; no general program synthesis claim follows. |
| New tasks compose acquired skills; library changes future search; H p. 21, lines 519-523; Recipe 5, p. 41 | Interpreter and proposal API accept library calls, ordinary inference uses saved skills | **Integration missing.** Both program-learning search calls in `connected.intervene()` omit `library=`. Automatic acquisition does not search with earlier skills as primitives. |
| Locate competence with library removal and frozen weights; H p. 21, line 523 | New post-publication ablation below; acquired goal skills execute with R1 weights unchanged during program acquisition | **Partly checked after publication.** Library contribution is real in two saved worlds; this does not establish held-out abstraction transfer. |
| Own and serialize session state with model/encoder versions and provenance; H pp. 35-36, lines 828-832; p. 58, lines 1394-1400 | Durable `SolverStore` is versioned; older symbolic `Session` has persistence | **Missing for live R1 sessions.** `WorldSession` has fast tensors but no owner/schema/model/encoder version fields or save/load interface. Durable solver reload is a different contract. |
| Admitted evidence only; separate support from sealed query/evaluation data; H p. 58, lines 1396-1400 | `EvidenceReplay` validates training trajectories; main learning uses it | **Defect repaired.** Program-credit `plans` previously bypassed the split admission check. Version 0.2.1 validates them before any optimizer update. |
| Failure classification, targeted curriculum, learner, verifier, consolidator and useful archive; H p. 37, lines 840-871 | Learned selector over six measured interventions, verifier, replay, immutable versions | **Partial.** No explicit missing-evidence/procedure/world-model/search/specification diagnosis, targeted exercise generator, or selection among useful archived lineages. |
| Policy uses previous failures and budget; useful active evidence; H pp. 53-54, lines 1288-1304 | Diagnostic accuracy/reward/entropy/support features; extra random evidence intervention | **Missing connections.** `previous_score` remains its default at callers; no remaining-budget feature or learned choice of informative observations. |
| Fresh paired admission, cumulative error spending, retention and cost; H pp. 38-39, lines 897-911 | Hoeffding gain bound, fresh candidate evaluation, empirical retention gates, cumulative ledger, rollback | **Implemented with stated limits.** Conditional gain bound is not a retention certificate. A local journal is not external evaluator process isolation. |
| Bounded executable search and complete cost disclosure; H pp. 38-42, lines 893-997 | Fixed interpreter, finite grammar, execution caps and category counters | **Search defect repaired; accounting partial.** Fixed candidate construction is now capped before materialization. Total operation counts are heterogeneous proxies, not complete FLOPs or whole-development cost. |
| Joint training and meaningful holdouts; H p. 19, lines 467-473; Recipe 2, pp. 40-41 | Retrieval/binding/renaming/temporal tasks, action effects, masked/long trajectories, family and goal-pair holdouts | **Narrowed.** No arithmetic/spatial/general modality curriculum or held-out program-structure evaluation. |
| Compare adaptation, world planning, curriculum and program learning against prescribed controls; Recipes 3-6, pp. 41-42 | No-update/update/replay controls, reactive planner, fixed program search, fixed intervention policies | **Incomplete controls.** See the experiment audit below; success of available controls does not substitute for missing ones. |
| Generate, screen and learn proposals for architecture changes; Recipe 7, p. 42, lines 983-989 | Registered components and versioned candidates; bounded choice of existing interventions | **Not implemented as architecture search.** There is no learned insertion/replacement/factorization grammar, staged mutation experiment or expansion outside the supplied operations. |
| Improve the improver over generations at full cost, fixed hidden anchor and expanding coverage; H p. 3, lines 19-29; Recipe 8, p. 42 | Three development-trained policies; two final policy-selected rounds per seed | **Not established.** The policy stays frozen during final rounds. No successive improvement in the improvement procedure or sustained gain-per-full-cost acceleration is demonstrated. |

Code anchors: [R1 and live sessions](../src/sera/r1.py), [reference memory](../src/sera/lowrank.py), [R2 and program search](../src/sera/r2.py), [connected acquisition and diagnostics](../src/sera/connected.py), [policy](../src/sera/curriculum.py), [admission](../src/sera/experience.py), [durable solver](../src/sera/solver.py). The behavioral and structural probes are in [the audit script](../scripts/audit_architecture_contracts.py), with committed [before/after evidence](original-source-audit-evidence.json).

## The R2 restriction is mathematically important

The source explicitly permits a compact exact rank-one submodel (H p. 22, lines 525-535). SERA's learned projective basis generalizes that starting example through learned action channels, but does not implement the general observation instrument described earlier in R2.

For the current projectors `P_y = |u_y><u_y|`, the selective update is

```text
P_y rho P_y / trace(P_y rho) = P_y              when the event has positive probability.
```

Thus a visible color replaces the predictive state with its projector, regardless of earlier history. Ignoring an event applies `sum_y P_y rho P_y`, which removes off-diagonal entries in this basis. Starting from `I/4`, every post-event state has the form `rho = sum_i b_i P_i`. The action/observation dynamics therefore admit the exact classical transition representation

```text
T_a[i,j] = trace(P_j C_a(P_i))
b_predicted = b @ T_a
b_after_visible_y = one_hot(y)
b_after_missing = b_predicted.
```

This also represents the post-event state read by the program proposal head. Its real/imaginary density features can be reconstructed from the four-component belief. Complex coordinates alone do not establish an additional memory capacity here.

I checked three random seeds, both real and complex variants, and 128 action/event updates per case. Maximum errors were `7.153e-7` for posterior equality to the observed projector, `3.278e-7` for the classical filter's next-event probabilities, and `7.165e-7` for density reconstruction. The algebra supplies the reduction; these probes check its implementation numerically.

This result agrees with A chapter 10 (PDF pp. 31-32, printed pp. 23-24) and M7: a POVM fixes probabilities but does not uniquely determine the conditional instrument. A broader R2 could retain history inside nontrivial event branches. It would need a separate implementation, aliased-history tasks and matched trained classical controls. The present reduction is a boundary on the current model, not a claim about all quantum instruments.

## Concrete defects and their repairs

**Program-credit admission.** A synthetic `query-audit` trajectory was rejected by `EvidenceReplay` but accepted through `fit_instrument(..., plans=...)`, and one update changed parameters. I found no evidence that the published run used sealed records in that path; its stored supports and queries remain separated. In 0.2.1, the same admission validator checks all program-credit traces before optimizer construction. The probe now rejects the trace without changing parameters. Repeated valid traces retain their original multiplicity and sampling order.

**Fixed-search construction budget.** With an execution budget of one, length six created 5,460 candidates before executing one. The fixed path lacked the guided path's maximum-length validation. Even valid length five constructed 1,364 candidates for that one attempt. Both paths now require integer lengths in 1-5 and execution budgets in 1-4,096; fixed search materializes only the bounded prefix it can execute. The length-six probe performs zero counted work, and valid length five constructs and executes one candidate. Guided search retains its separately counted proposal cap; execution caps are not equal-compute guarantees.

The 0.2.1 suite passes **40 tests** with Ruff checks passing. A deterministic valid-input comparison against the actual 0.2.0 source reproduced the same two-step training history and final parameter hash, including repeated credit traces. All **216** fixed-search cases had identical success, attempt, program and trace outcomes. The fixed-search construction counts intentionally change. These regression checks do not replace a new learning study.

I built the 0.2.1 wheel, installed it in a separate target directory and confirmed imports resolve there. It restored the original v5 solver and executed its admitted reset-world program successfully (`start=0`, `goal=1`, actions `[1]`). The wheel hash and runtime check are recorded with the audit evidence. All nine committed evidence artifacts pass checksum verification.

## What the training results do and do not establish

The [connected study](connected-study.md) remains the teaching and performance record for 0.2.0: three independently trained seeds; 48 policy episodes and 276 measured intervention outcomes; 15 persistent candidate evaluations. R1 reached 94.35% mean next-observation accuracy with 40% missing sensors at length 12, compared with 80.60% after resetting its trained memory. Learned R2 search solved 12/12 held-out goal pairs, fixed search 10/12, and the untrained instrument 8/12. Only two of six final policy-selected candidates passed promotion; four failed their gates.

These results support narrow learned behavior and a functioning acceptance lifecycle. The remaining controls constrain stronger interpretations:

- The memory-reset condition ablates a trained model. It is not a separately trained memoryless model with a matched budget.
- Base-world goals are reachable in one action. Perfect base planning success does not establish useful multi-step imagination. A matched oracle-world-model comparison and harder aliased control tasks are absent.
- R2 holds out start/goal pairs inside learned dynamics, not program structures or newly composed reusable abstractions. The connected study lacks trained classical hidden-state/recurrent controls matched for parameters and total compute. Historical real/complex cyclic experiments do not supply that comparison.
- Current connected adaptation lacks the complete adapter and scratch controls, calibration study and example-efficiency curves prescribed by Recipe 3. The older symbolic scratch experiment is narrower evidence.
- The learned selector chooses among supplied methods. Its small reset-family test and reported mean utility difference do not establish general decision superiority. Uniform/difficulty/fixed curriculum controls with easy, hard, ambiguous and noisy tasks remain incomplete.
- The formal generational run replays its initial 512-trajectory world buffer. Later support is archived, but this is not complete cumulative replay coverage. A prepared continued-learning copy merges actual records afterward; it is not an additional formal training result.
- Wall time and many operation categories are retained, including failed proposals. Baseline meta-evaluation counters, complete phase timing, peak memory and full development/search cost are incomplete. There is no evidence of the source's sustained improvement-per-full-cost objective.
- I did not reproduce the ZIP's released experiment suite. SERA's experiments are independent measurements, not a reproduction of the handbook's twelve-core results or an execution of its 36,900 schematic registry configurations.

I also removed the saved goal-skill library from the final seed-0 solver while keeping its weights fixed. In `reset-90000`, success changed from **12/12 to 5/12**; in `reset-90001`, from **12/12 to 4/12**. These are actual re-executions on already exposed worlds. They show that saved executable skills contribute to ordinary inference. They do not show that future program search composes those skills, that R1 absorbed the acquired procedure, or that the result transfers to unseen program structures. Other seeds had no corresponding admitted world programs to ablate.

## Corrections to earlier descriptions

The [0.2 audit resolution](architecture-audit-resolution.md) addressed specified connection failures from 0.1. Its wording about closing engineering gaps was too broad if read as an audit of the entire original blueprint. In particular:

- Versioned solver reload does not complete the live world-session ownership/resume requirement.
- Executable `call` syntax does not mean autonomous acquisition uses earlier skills as a search vocabulary.
- A shared instrument can be mathematically valid while representing a restricted classical belief filter.
- Storing the last discarded-mass value is not a persisted approximation-error audit.
- A learned intervention selector is only part of the full failure diagnosis, curriculum and improving-the-improver loop.

I corrected the architecture, README and checklist to make these boundaries visible. Historical evidence files and their measured numbers are unchanged.

## Open acceptance checklist

I leave these requirements open until implementation and the specified evidence both exist:

- [x] Reconcile the original file identities, handbook manuscript and all 154 graph identifiers.
- [x] Inspect the original R1/R2 and physics instrument diagrams/equations; audit the current code against them.
- [x] Reproduce and repair program-credit admission and fixed-search construction defects.
- [x] Compare valid behavior with the published baseline and document the post-publication library ablation.
- [ ] Add owner/schema/model/encoder identities and validated save/resume for live `WorldSession`; verify continuation equivalence and incompatible-state rejection.
- [ ] Connect acquired libraries to automatic program proposals; retain dependency versions, domain checks, failure examples and executable tests; demonstrate shorter search on held-out compositions with no-library controls.
- [ ] Implement a history-preserving event-instrument option and evaluate aliasing, likelihood and planning against trained classical filters/RNNs at matched parameter and compute budgets.
- [ ] Implement and evaluate selective event routing; train the full reference R1 preset; record per-trajectory truncation diagnostics and rank/accuracy/memory/runtime curves.
- [ ] Add the common typed-event adapters and a declared next curriculum, including arithmetic/spatial tasks and learned observation representations, with generator/family/structure holdouts.
- [ ] Connect explicit failure categories, previous attempts and remaining budget to curriculum/intervention choice; compare targeted evidence requests with random evidence and fixed curricula.
- [ ] Complete adaptation and planning controls: adapter/scratch/no-update/replay, reactive/learned-model/oracle planning, calibration and sample-efficiency curves.
- [ ] Evaluate cumulative replay/consolidation and useful lineage selection, including worst-task retention and archive/retrieval cost.
- [ ] Record all evaluation/search/development cost categories and separate hardware or external assistance before testing improvement efficiency.
- [ ] Implement and evaluate a typed architecture-mutation grammar with staged validity, transfer and retention checks before extending the operation set.
- [ ] Improve the learned improvement procedure across outer generations, with a fixed hidden anchor and expanding independent coverage; report unsuccessful generations as well as gains.

R3-R8, unrestricted multimodality and broad general intelligence remain separate future architectures or capability objectives. They are not silently marked complete by this first integration.

## Reproduce this audit

On 0.2.1, after installing SERA:

```text
python scripts/audit_architecture_contracts.py --output runs/my-architecture-audit.json
python -m pytest
python -m ruff check src tests scripts
python scripts/verify_release.py
```

Add `--study runs/connected-v2-final` only when the original local checkpoints are present to repeat the saved-library ablations. Without it, the mathematical, admission, budget, structural and valid-input probes run from scratch. The audit deliberately reports missing architecture contracts; those entries are not evidence of a failed execution.

To compare the baseline, run this same probe script with `PYTHONPATH` pointing to `src` from commit `00d61b8ce4de4c27afabb743cb54b5ddae7ea682`. This audit did so using a source-only archive of that repository commit. The original three-seed admission reproduction requires that historical source hash; a new 0.2.1 study needs its own output directory and evidence. See the [reproduction guide](../research/reproducing-connected-study.md).

The committed [audit evidence](original-source-audit-evidence.json) and [checksum manifest](original-source-audit-manifest.json) preserve source identities, before/after probes, compatibility results and the exact graph identifier set. Raw source text, rendered pages and private local checkpoint paths are kept outside the published evidence.
