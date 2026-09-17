# C01/C02: history-sensitive empirical refinement on the Stage 27 owner

Frozen before new fits, 17 September 2026. Parent: 9da1b21b06da4504d7d2b83777d69e874d4caeff. No new resource grant. Existing ledger and one-thread/2 GiB process-tree limits apply. The v12/v13 banks are reference evidence; none is this study's final evaluation.

## Question and contracts

Can the actual StudyR1 learn/use observed history to forecast a one-dimensional system, retain a simpler adequate model, and preserve its exact polynomial and four-language routes? Shared registration is not positive transfer. Features, system families, probing curriculum, verifiers and model-selection policy are supplied engineering. Coefficients and neural changes are learned from observations.

The supplied simulator has h'=v, v'=-1+b*u+w-d*v+F-q*v*abs(v), F' relaxation toward -c*v. Discrete integration is semi-implicit at dt=.05. Instantaneous worlds have c=0; delayed worlds have tau in [.15,.9], c in [.25,1.3]. An omitted quadratic-drag family has q>0. The learner sees h,v,u,dt and availability only. Hidden force, coefficients, family and future outcomes remain assessor-only. Synthetic provenance is explicit. These are airborne trajectories, with no inferred landing/collision guarantee.

## Finite design

Training: 64 independent parameter draws, four smooth, varied-command trajectories per draw, 80 transitions. Two training seeds 2801 and 2802. Development: 8 fresh worlds per family, two histories ending at identical position and velocity, three observation conditions (clean; Gaussian h/v noise .002/.01; that noise plus 10% missing observations). Final: 12 fresh worlds per family with the same predeclared conditions and wholly separate random seeds. No final generation or access before selection is saved.

Histories are supplied excitation probes: smooth velocity paths plus independently generated hidden-force dynamics determine physically consistent past controls. Their endpoint positions are translated to 10 and velocities fixed to -.3; future commands are identical within a pair. This construction and its variables are supplied, not discovered. Current state is observed at query time. No future state is input. Evaluate 1, 4 and 12 steps ahead, h/v RMSE separately and a combined squared-error score; history-pair velocity separation and sign on non-negligible true differences; errors by family/condition, not only pooled totals.

Neural controls: actual R1 with frozen memory and trained input/readout; same R1 with conditional rank-4 memory/fusion updates; memory reset every step; a 32-state GRU; an 8-step MLP. All receive the same observable channels. Two seeds; 240 Adam updates, batch 8, 40-step training windows, first 12 steps are warm-up. Save optimizer/RNG state every 40 updates. An 8-update development-only pilot measures cost and detects implementation errors; it is not a final fit. Maximum 500 seconds per individual training run. If necessary the complete finite run is resumed, never silently shortened or discarded. No hyperparameter sweep.

Physical controls: coarse force (u and constant), instantaneous drag (+v), and relaxing-memory identification on the fixed tau grid [.1,.15,.25,.4,.65,1.,1.6]. These learn bounded coefficients from the first 48 observed transitions, select on the next 16, then refit using the 80 observed transitions. Missing transitions are excluded, noisy velocities are not represented as exact derivatives. Penalize added complexity only through a predeclared simplicity rule: choose the smallest model whose validation MSE is within 10% + .0025 of the best. Record all candidate fits. These controls have supplied equations; they do not independently discover physical variables.

Admission: choose on development only. A physical candidate may become the runtime route if its delayed clean 12-step MSE is at least 20% below no-memory instantaneous identification, its paired-history separation is correct on at least 75% of differences >.002, and its simple-system MSE is no more than instantaneous +.0025. Noisy/missing conditions must each remain below combined MSE .25. Neural controls are admitted instead only if they also meet these gates and improve the selected physical candidate by at least 5%. Omitted-mechanism errors are diagnostic, not a target for tuning. Runtime model adequacy is conditional on observed held-out residual; uncertainty is an empirical residual diagnostic, never a calibrated guarantee against omitted mechanisms. Final evaluates the fixed choice once; failure rejects promotion and preserves a resumable candidate.

## Integration and independent verification

Register empirical parameters on the actual StudyR1 descendant, preserve all parent tensors byte-for-byte, rebind existing guarded knowledge, and maintain one owner across interfaces. Per-subject history and ephemeral recurrent state are outside weights. Observations carry source, units, timestamps, availability and subject identity. Imagined branches cannot write factual history or weights. Report EMPIRICAL_PREDICTION separately from CERTIFIED_ALGEBRA; insufficient history yields MISSING_KNOWLEDGE. Refined models retain prior model records and coefficients.

Persist source/checkpoint hashes, owner identity, original goals, every fit and cost, subject records, branch provenance and exact replay state. Independent matrix recurrence verifies scalar forecasts. Test changed source/weights, subject separation, branch isolation, restore/replay, past-route tensor equality and four-language development probes. Completed Stage 27 final sets remain sealed. Preserve failures and engineering repairs; any scientific protocol revision precedes new final access.

## Research basis

- [Mori–Zwanzig learning](https://arxiv.org/abs/2101.05873): reduced observable states can require history and unresolved forcing; supplied kernels are candidates, not guaranteed identified truth.
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464): test the existing gated delta memory before adding a recurrent architecture.
- [Sparse identification](https://arxiv.org/abs/1509.03580): fit finite candidate mechanisms; supplied feature libraries are not independently discovered concepts.
- [GRU encoder–decoder](https://arxiv.org/abs/1406.1078): classical recurrent control.
- [Output-error identification](https://arxiv.org/abs/2502.14432): noisy output/state estimation motivates explicit noisy and missing-observation tests; this experiment does not implement that paper's full method.
