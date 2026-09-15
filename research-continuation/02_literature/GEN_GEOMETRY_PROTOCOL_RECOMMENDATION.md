# Generative Geometric Memory: staged experimental protocol

Status: **G — PROPOSED**. Prepared 2026-09-14 from the user's complete handoff and the primary-source review in this directory. This document records an experimental recommendation; it contains no newly executed performance result. The experiment records determine actual implementation status.

The first experiment should test whether a small generator predicts held-out observations at a lower *total* retained cost, while detecting observations its class cannot explain. It should explicitly separate parameter estimation, class selection, representation discovery, acquisition-policy learning, and preservation under revision.

## 1. Retained object and estimand

Retain a versioned object
`K = (generator AST or model, quantized parameters, input/output interface, applicability, residuals, uncertainty, provenance, dependencies, decoder version)`.
A prediction must identify its K version, whether its inputs were observed or hypothetical, and which uncertainty components it includes.

The principal estimand is a predictive risk–retained-bytes Pareto curve at fixed acquired observations and total computation. Report interpolation, extrapolation, completion, exception detection and post-revision risk separately. A lower in-sample residual or a smaller core parameter vector alone is not success.

Distinguish three epistemic objects:

- Parameter uncertainty conditional on a class: `p(theta | M,D)`.
- Class uncertainty: `p(M | D)`, or a finite unweighted version space if a coherent likelihood/prior is unavailable.
- Class inadequacy: a diagnostic that *every* candidate can be wrong. A normalized posterior over an incomplete menu does not measure this.

Bayesian predictive mixing uses `p(y|x,D) = sum_M int p(y|x,theta,M) p(theta,M|D) dtheta`. Scores such as residual plus complexity do not become probabilities merely by applying a softmax; identify that construction as a heuristic and calibrate it externally.

## 2. Two geometry tracks

**Track P: supplied-coordinate prediction.** Observe `(t, x(t), y(t))`, with t supplied as a scalar coordinate. Fit generators of t. Start here because all eight controls can be made small and inspectable. Trigonometric primitives and phase conventions are supplied inductive bias. Success here establishes coefficient fitting, bounded expression search or class selection; it does not establish discovery of geometry from unordered points.

**Track U: unordered geometry.** Observe an unordered, noisy point cloud without t or family labels. Queries concern set membership, distance, completion of missing arcs and predictions under interventions. Use one specified point-generation/measurement model shared across classes; nearest-curve distance alone is not a normalized likelihood. Include uncertain correspondence, partial arcs, rotated ellipses, nuisance transformations and sampling-density changes. A learned encoder must be invariant to point permutation, or its order sensitivity must be measured.

For both tracks, reserve ambiguous prefixes: three noncollinear points can identify a circle *within the circle family*, but do not determine that circles are the correct family. A general interpolating curve and many ellipses can agree with limited data. Calibration and abstention are required on underdetermined completions.

## 3. All eight requested control families

Each family is an independently serialized model with an explicit query contract. Shared primitives and pretraining are disclosed.

| ID | Control | Minimum concrete implementation | Main comparison obligation |
|---|---|---|---|
| A | Raw episodic memory | Quantized observations and exact lookup; unfamiliar queries return an explicit unknown distribution or abstain under a fixed scoring policy | Count observation coordinates, values, IDs, index and query code. Report both selective risk and coverage; abstention cannot silently remove errors. |
| B | Nearest neighbor/interpolation | Euclidean kNN plus linear interpolation where the input order makes that meaningful; k/tie rules fixed on development data | Count retained support points and index. Compare on the same query set and budget. |
| C | Fixed circle generator | Fit center/radius or the equivalent phase-conditioned constrained model, with noise estimate | Supplied circle class is a control. Test failure on ellipse, spiral and exceptions. |
| D | Circle/ellipse/model-family selection | Finite menu of circle, ellipse, line/low-order curve candidates, fitted separately; complexity selection and predictive mixture both reported | Freeze menu before confirmation; charge class search and family code. Include a misspecified-menu condition. |
| E | Symbolic regression | Bounded expression grammar with learned constants and a frozen candidate-evaluation budget; optional pinned PySR/AI Feynman follow-up | A library basis regression is labelled library regression, not unconstrained symbolic discovery. Count supplied operators, simplifier, fitting/search work and failed candidates. |
| F | Neural latent representation | A trained encoder/decoder or explicitly specified neural predictor with a latent layer; fixed architecture/precision and optimizer | Count *all* weights and any per-episode latent. If using an episode encoder, charge meta-training data/compute separately; a generic MLP is labelled as such. |
| G | Programmatic generator | Bounded DSL search over compositional procedures, transforms or recurrences; retain program and learned constants | Supplied named circle/ellipse constructors are supplied primitives. Composition and guard learning must be measured separately. |
| H | Mixed generator + residual memory | Winning generator plus indexed, quantized residual/exception store and a residual applicability rule | Count residual coordinates and retrieval/guard cost. Stored exceptions only support their tested domain; local correction cannot imply unseen exception generalization. |

A–H are the minimum requested family coverage. An initial subset is useful exploratory progress, but must be marked incomplete rather than described as the full suite. Optional later implementations should remain in the same families instead of replacing a weak result without a new experiment ID.

## 4. All ten requested task families

| Task | Generative/negative condition | Held-out or intervention test | Likely discriminating controls | Stage |
|---|---|---|---|---|
| 1. True circles | Shifted/scaled circles; full and partial arcs; noise sweep | New angles, centers/radii, missing arc, nuisance coordinate transforms | C vs D/E/F/G/H and A/B | P0; U1 |
| 2. Ellipses | Unequal axes and rotated ellipses, with near-circle eccentricities | Missing extremes, unseen rotation and eccentricity | C vs D; E/G; near-circle ambiguity | P0; U1 |
| 3. Periodic functions | Sinusoids plus harmonic mixtures with held-out frequencies | Outside observed periods, phase wrap, changed frequency | Fourier/phase vs polynomial/Euclidean controls | P1 |
| 4. Spirals | Archimedean/logarithmic examples and held-out variants | Additional turns, changing radial growth | Fixed circle failure vs D/E/G | P1; U1 |
| 5. Piecewise curves | Continuous and discontinuous segments; hidden breakpoints | Unseen breakpoint placement and boundary queries | H or guarded programs vs one global smooth model | P1 |
| 6. Hierarchical relations | Trees and DAGs with renamed nodes; hierarchy violations | Ancestor/composition queries, new branches, graph rewiring | Explicit graph/program vs hyperbolic and Euclidean embeddings | R2 |
| 7. Order-sensitive transforms | Noncommuting affine maps or finite permutations | AB vs BA; longer words; inverse and composition queries | Exact matrix/permutation baseline vs phase, DeltaProduct-style candidates | R2 |
| 8. Dynamical systems | Linear systems then simple nonlinear dynamics, partial observations | New initial states, parameters, interventions and long rollouts | Linear identification/Kalman/SINDy/program vs neural dynamics | D2 |
| 9. Random labels | Independent labels or coordinate values indexed by held-out input IDs | Exact recall of acquired labels and prediction of unseen independent labels | A/H storage scaling; calibrated chance on unseen labels | P0; every stage |
| 10. Almost-circles with systematic exceptions | Circle plus localized deterministic deformation, branch or exception rule | Missing exception region, shifted exception, late counterexample | C failure; H storage/repair; D/E/G structural revision | P0; U1 |

Task 9 must not expose the assessor's pseudorandom seed, family ID, answer table, generating code or other compact side channel to the learner. A reusable PRNG seed is structured data; it is not the arbitrary independent-label control. There is no universal exact compression of all N-bit strings into fewer than N bits: the encoder would need to inject `2^N` messages into only `2^B` codewords. This is **E — MATHEMATICALLY-DERIVED** for fixed B-bit storage and a shared fixed decoder.

Pure componentwise complex rotations commute. Consequently they cannot generally preserve AB versus BA when the target operations are noncommuting. This is a separate **E — MATHEMATICALLY-DERIVED** negative control; complex arithmetic itself is not evidence of order sensitivity. For `A = B+iC`, the realification `[[B,-C],[C,B]]` acting on concatenated real and imaginary coordinates produces exactly the same linear map in exact arithmetic. Match scalar state count, numerical precision and implementation cost.

## 5. First executable protocol: GG-P0

Freeze before evaluation:

1. Use Track P and tasks 1, 2, 9, 10. At least A–H, if feasible; otherwise preserve the missing-family checklist. Include clean and noisy regimes and a partial-coverage prefix.
2. Use separate development seeds to settle ranges, grid density, numerical tolerances, grammar, model capacity, selection penalty, noise treatment and budget. Do not tune them on confirmatory failures.
3. Freeze train, calibration and test split generation, then hash the config and implementation. Hold out both parameter instances and input regions. New random seeds from the same generator are not held-out task families.
4. Recommended small design: 8, 16, 32 and 64 acquired observations; at least 30 paired independent instances per task/noise stratum; 128 independent test queries per instance, divided between interpolation and deliberately omitted/extrapolated regions. These are proposed defaults, not executed counts or an advance power guarantee.
5. Freeze float32 as the first serialized format and repeat selected models at 8-/16-/64-bit parameter precision using declared quantization rules. Score the *decoded* artifact, including quantization error and exceptions. Store calibration metadata in the artifact.
6. Choose a fixed development-set predictive error tolerance and report the smallest total artifact satisfying it; also report the complete Pareto frontier. Do not let each method choose its favorable tolerance after results.
7. Evaluate fixed circle misspecification, finite-menu selection, mixture calibration, and H's exception-memory growth. Report cases where A/B dominate at small N because their fixed overhead is lower.
8. Preserve all per-instance records and timeout/rejection reasons. Bootstrap *paired independent instances*, not individual correlated points; report confidence intervals for effect sizes and the distribution of failures. This first run is an exploratory feasibility study unless its frozen sample size and success criteria were chosen independently.

Predictive error is family-appropriate: coordinate RMSE/normalized MSE for curves, Brier/log loss for categorical labels, graph-query accuracy for relations and horizon-stratified error for dynamics. Do not average incompatible raw units into one score. For curves, expected calibration is defined through an explicit predictive distribution, not a geometric-distance confidence score.

## 6. Full cost accounting

Publish both deployment cost and acquisition cost.

`L_total = L(generator) + L(parameters,precision) + L(decoder/interpreter) + L(guard) + L(residuals) + L(uncertainty) + L(provenance) + L(dependencies) + L(index) + L(shared grammar)`.

Use actual deterministic serialized bytes as the principal retained-size metric. Where reporting a theoretical MDL bit score, specify the coding convention, quantization, likelihood and prefix/universal code; do not equate JSON bytes or a BIC-style penalty with a rigorous code length. Shared infrastructure may be amortized only across a declared deployment count N, with both standalone and amortized values shown. Training data retained for replay count as retained bytes.

Acquisition accounting records observations, oracle cost, simulator calls, gradient steps, candidate models evaluated, expression nodes expanded, rejected/invalid candidates, CPU/GPU seconds and peak memory. Inference includes guard evaluation, retrieval, decoding and internal simulation. Report wall-clock cost and algorithmic work because kernels can change their relationship.

An exact replay audit must launch a fresh process from the serialized artifact and declared decoder dependencies. It must reconstruct predictions without process-resident training arrays, uncounted closures or hidden cached coefficients.

## 7. Active discrimination and imagination

Start with a small finite action grid and an explicit likelihood. The reference acquisition objective is
`a* = argmax_a I(Y_a; Q | D) - lambda cost(a)`,
where Q is a declared downstream quantity, candidate class or model parameter. Unknown-noise observations can have high predictive entropy and low information about Q. Compare:

- Random sampling and uniform space-filling sampling.
- Maximum predictive disagreement.
- Exact/enumerated one-step information gain where feasible.
- A fixed cost-aware rule.
- The learned policy after its offline cost is included.

Use a nuisance-noise trap: an available action yields unpredictable noise but does not help the target query. Compare information objectives on the same candidate grid, observations, posterior update and budget. After this works, use short-horizon policy search or DAD-style amortization.

Imagined outcomes come from the same versioned dynamics/likelihood used to make predictions. Every branch stores parent factual state, assumed intervention, model version, uncertainty and depth, with `provenance=hypothetical`. Imagination can change an action choice; it cannot add an observed training label or count as an acquired experiment. Measure decision improvement against zero-rollout and equal-compute extra-data/extra-fit controls.

## 8. Guarded consolidation, revision and retention

Compile only a successful computation with explicit interfaces, dependencies and claimed domain. The deterministic obligation is `g(x) => P(x)=F(x)`. For small finite domains, exhaustive checking or a sound solver can establish that bounded claim. Random tests establish empirical reliability only; record number and distribution. For noisy targets, freeze a risk bound/coverage criterion rather than assert equality.

Compare always-run-source reasoning, unguarded compilation, supplied guards, learned guards and learned guards with dependency invalidation. Include adversarial near-matches. Measure false acceptance separately from false rejection.

A changed dependency must invalidate downstream cached conclusions or trigger revalidation. A live-state migration maps `(K_t,z_t)->(K_(t+1),z'_t)` while retaining meaning or explicitly marking unavailable information. Exercise it mid-episode and compare to a fresh replay using the revised model. Version names alone are not a migration mechanism.

Retention and plasticity are separate endpoints: old *still-valid* capability accuracy; time/data to learn a fresh task after a long stream; successful correction of obsolete assertions. Compare replay, fresh initialization, small local adapters, regularization and any unit-replacement mechanism at matched storage and update cost. An old error being corrected must not be scored as harmful forgetting.

## 9. Investigator generation experiment

Freeze independent `K_g` and `eta_g` artifacts at each generation. Eta includes every component that changes the acquisition/update procedure: policy weights, optimizer state, scheduler, feature transforms and any learned thresholds. K includes learned programs and priors. Make the boundary explicit; a MAML initialization can be prior knowledge, so cannot automatically stand in for a changed learning rule.

Evaluate four crossed cells on paired unseen environments:

| | eta_old | eta_new |
|---|---|---|
| K_old | Y_00 | Y_01 |
| K_new | Y_10 | Y_11 |

Use higher-is-better Y such as negative acquisition cost to a fixed accuracy target, or negative area under the learning-curve loss versus total cost. Report:
`Delta_eta(K_old)=Y_01-Y_00`;
`Delta_eta(K_new)=Y_11-Y_10`;
`Delta_K(eta_old)=Y_10-Y_00`;
`interaction=Y_11-Y_10-Y_01+Y_00`.

A better Y_11 than Y_00 alone does not isolate learning-procedure improvement. Compare learned eta to strong fixed/random/greedy-information policies plus equal-compute hyperparameter search. Account for training trajectories and meta-optimization, reporting the deployment-count break-even curve.

Use outer train tasks for policy updates, validation tasks for promotion and sealed confirmatory task *families* for the scientific claim. Freeze several disjoint future-generation panels in advance; repeatedly inspecting the same held-out results makes them validation data. Test new mechanism families, changed noise/cost schedules, renamed entities and changed representation inputs. Compatible serialization and identical starting evidence are prerequisites to crossing K and eta.

A first trained policy may be a small cost-aware investigator learned from actual acquisition outcomes. Do not call a hand-designed rule or a manually chosen hyperparameter improved by the system. One accepted eta update establishes at most that local effect. Successive useful generations require multiple fresh crossed comparisons with no loss of still-valid capabilities. The meta-updater V remains supplied unless its own improvement is separately tested.

## 10. Promotion and failure record

Advance P0 -> P1/U1 -> R2/D2 -> acquisition -> consolidation/revision -> learned investigator -> successive generations only after preserving the current level's counterexamples and explaining the central failures. No numerical promotion threshold is inferred from literature effect sizes; set a task-relevant minimum before confirmation.

Retain a machine-readable status per family and mechanism: proposed, implemented, exploratory executed, confirmatory executed, failed, rejected, deferred. A failure updates the uncertainty/model menu; it does not authorize editing the frozen confirmatory protocol in place. Any changed hypothesis starts a new experiment ID with the previous result retained.

