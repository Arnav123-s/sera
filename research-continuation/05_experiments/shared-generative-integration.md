# Shared persistent generator integration

## Frozen protocol before implementation

This is a bounded software and numerical integration experiment. The supplied phase is t and the trigonometric phase is pi*t. One observed two-coordinate trajectory is explained by four declared linear Gaussian alternatives: circle (4 coefficients), ellipse (6), line (4), and radial-growth orbit (6). The radial alternative permits an affine complex amplitude; collinearity of its base and growth vectors is not enforced. Observation noise is known and isotropic. Coefficient priors are independent zero-mean Gaussians. Class priors are uniform. Neither the grammar nor semantic applicability nor the noise model is learned.

Implement a GenerativeSharedR1 descendant with only registered float64 sufficient-statistic, posterior and class buffers added to the existing numeric owner. A from_shared conversion must clone all existing parameter/buffer bytes, configurations, typed programs and gradient flags. Existing sequence, typed and world views keep referring to that same owner; the ordinary routes receive no analytic update. Full Solver checkpoint restore must reconstruct this descendant and reconnect views. Preserve original trained stores.

The live facade uses immutable Event records, ExecutableGraph definitions and complete MechanismRef guards. Only fully available, dimensionless (t,x,y) observations for a single entity enter the observed posterior. Independently verified corrections may replace effective coordinates at an admitted position while preserving original observations and correction records. Staged copies and same-schema event replay precede owner replacement. Old identities reject live continuation. Conditional branches query the same registered posterior without updating sufficient statistics. Rejected work still consumes the declared shared operation budget; acquisition and replay accounting stay separate. Preserve an immutable serialized predecessor with an explicit restoration path.

Meaningful checks: byte-preserving owner conversion, independent observation-space Gaussian posterior/evidence comparison, posterior and class change after observations, no second trainable model, exact prior route retention, full SolverStore save/load with one owner, observation/correction replay agreement, stale-session rejection, cross-entity/schema/availability checks, imagination/query contamination rejection, failed validation and exhausted replay without partial owner replacement. These fixtures establish ownership, persistent acquisition and conditional retention only. They are not learned semantic grounding, distilled transfer, latent migration, autonomous grammar discovery or a capability benchmark.

Run targeted new and existing lifecycle checks with a 60-second hard cap per invocation. Verify the actual trained-parent descendant and broader regression suite separately. The executed resource boundary below supersedes the proposed RSS metric: the supervisor measures peak committed job memory, which is not RSS. Preserve failed checks and their costs.

## Executed evidence

I implemented the registered generator route, conversion, full solver restoration, transactional observations and corrections, immutable predecessors and same-owner conditional prediction. Fourteen numerical and lifecycle regressions pass; the broader current suite passes all 156 tests.

The initial checks exposed three implementation defects. Conditional queries passed through a float32 conversion; the likelihood's integer count multiplied by a Python scalar created a float32 intermediate; and restore compared evidence hashes without independently rebuilding the stored sufficient statistics. I removed both precision losses, reconciled numerical state against the preserved observations/corrections on restore, and added explicit gradient-flag persistence and float64 validation. I did not relax numerical tolerances. The [failed and repaired test records](../09_failures/shared-generative/) retain the two scientific failures, the invalid test-path invocation, and the final focused pass. One initial fixture also passed NumPy scalars to the strict Event codec; it now converts them explicitly.

## Actual trained parent

[SHARED-GG-001](../04_protocols/SHARED-GG-001.json) froze the integration before execution. I loaded the existing trained SERA 0.5 parent, solver identity `6e45a89679f0c2706b6af3561557b3e87226b577579894842159856e6f6b6289`, and created four separate experimental descendants. The parent remains the active historical solver. These descendants are fixed architecture fixtures, not independent repeated training benchmarks or a promoted multi-world learner.

Each fixture receives 16 observed pairs over phase [-0.6, 0.6], with known noise standard deviation 0.05. One pair is deliberately corrupted by (+3, -2). A later independently sourced sensor correction replaces its effective coordinates while retaining the original observation and correction record. The evaluator alone knows the generating family and checks 81 positions across [-1.8, 1.8].

| Supplied family | Before acquisition MSE | After acquisition, corrupted evidence | After verified correction |
|---|---:|---:|---:|
| Circle | 0.468850 | 0.0611522 | 0.0000934580 |
| Ellipse | 0.501901 | 0.0618274 | 0.00111833 |
| Line | 0.515305 | 1.10743 | 0.00196184 |
| Growing orbit | 0.605363 | 0.140529 | 0.00301370 |

The model learns coefficient posteriors and class weights. After correction it favors the matching family in every fixture. The menu, phase coordinate, noise model, update rule and class prior are supplied. The line result also shows that learning from a corrupted observation can worsen an initially uninformed predictor.

All 92 original state tensors remain bitwise equal. Sequence and typed views resolve the same R1 object after acquisition, correction and save/load. The new generator adds **zero trainable parameters and 2,880 registered buffer bytes**. The existing controller and R2 instrument remain explicitly separate inherited components; the entire solver is not being relabeled as a single undifferentiated tensor. Configuration, interpreter, working events, full checkpoints and archive bytes are additional to the 2,880-byte numeric workspace.

On the circle descendant I reran all **40 existing retained capability groups**, comparing **944 score elements** and the complete reports with the parent. They match exactly. This is conditional preservation because generator updates leave old numerical routes unchanged. Positive transfer from generator learning into the old recurrent core remains untested.

Every fixture also restores its pre-correction predecessor, resumes the final factual session after full SolverStore reload, and gives bitwise identical direct and hypothetical predictions. Hypothetical queries change only their assumptions and the shared cost ledger. This is conditional coordinate completion; it does not demonstrate causal intervention or decision-improving imagination.

The experiment took 27.058 seconds inside its worker, 29.770 supervised wall seconds, and 288,817,152 peak committed job bytes. All 15 original parent files retained their hashes. Initial/final solvers, pre-correction predecessors and all intermediate event/update records remain under `runs/SHARED-GG-001/`. [Compact evidence](../14_release/SHARED-GG-001-result.json) and [resource record](../14_release/SHARED-GG-001-resources.json) are published separately from the full local checkpoints.
