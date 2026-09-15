# R7 probability clouds over predictive weights

I test the proposal to retain a vector of complex amplitudes for possible weight values and evolve it with loss-informed Schrödinger or Dirac-type dynamics. This follows R7 in the original handbook (lines 734–764), while the new SERA/Kavi pack's representation-uncertainty findings determine the misspecification control. The attachment handoff prompts are research context, not additional authority to change accounts, delete work, or replace the current task.

## Representation and learning

For a finite parameter grid, `p(w) = |ψ(w)|²`. Prediction averages the predictions of candidate weights. A joint two-weight cloud retains correlations. The six-weight version retains one independent cloud per coefficient and explicitly approximates the joint distribution by their product. These are learned numeric prediction weights; they are not the transient R1 situation state or simulated physical particles.

The input block accepts only observed support designs and targets. Its state is the weight cloud; its output is a predictive mean/distribution. Retained parameters are the amplitudes or classical probabilities. Transition applies the declared Hamiltonian and/or likelihood filter; readout uses Born probabilities and the supplied predictor. Learning derives the potential from observed prediction error. Validity requires finite normalized mass, Hermitian operators, independent data roles and a fixed bounded domain. Costs include support processing, every potential value, propagator construction/application, normalization, tuning, prediction, storage and failed work.

The squared-error likelihood potential is `V(w) = Σ(y - φ(x)·w)² / (2σ²)`, with known dimensionless observation noise `σ = 0.2`. All methods receive the same observations and Gaussian weight prior. Finite grids truncate that prior to `[-2, 2]`; the analytic Gaussian comparator uses the unbounded prior and is separately labeled.

The Schrödinger split step applies a half loss phase, a kinetic mixing operator, and another half loss phase. The open-chain Laplacian supplies the kinetic operator. There is no periodic wrap between the minimum and maximum weight. A real-time-only ablation omits learning by filtering. The filtered version also multiplies the amplitude by `exp(-Δβ V/2)`, with total `β = 1` across the solve. Without mixing, this is exactly Bayesian likelihood reweighting. Pure loss phase without mixing leaves every Born probability unchanged.

The Dirac-type control uses the finite Hermitian generator `H_D = P ⊗ σ_x + I ⊗ σ_z`, where `P` is a centered, open-chain discrete derivative multiplied by `-i`. Each coordinate has two spinor components. The joint two-weight version therefore has four components per grid point. These are artificial, dimensionless parameter-space dynamics. They do not solve a physical electron problem or establish a relativistic computational advantage. Schrödinger evolution is the general time-evolution equation; the Dirac form specifies a different Hamiltonian rather than supplying an independent second learning law.

## Controls and data

The comparison includes exact finite-grid Bayesian weighting, classical probability diffusion plus weighting, positive real amplitude diffusion plus filtering, coherent loss phases, deliberate phase erasure, Dirac-type mixing, real-time evolution alone, a full Gaussian posterior, a Gaussian with cross-covariances removed, and ordinary gradient MAP fitting. The two-weight suite also replaces the exact cloud with the product of its identical marginals, isolating the importance of correlations. That diagnostic uses the joint solution and is charged for it; it is not a scalable independent learning algorithm.

The two-weight predictor uses supplied position and velocity coordinates. The six-weight predictor follows the new pack's supplied feature language: constant, position, velocity, control, position cubed, and velocity times its absolute value. All six coefficients are learned. This is a small predictive head and an R7 mechanism test, not replacement training of the million-parameter SERA core.

Three problem families test a correctly specified mechanism, correlated observations that leave weight combinations uncertain, and an omitted sine mechanism. The last family tests the difference between uncertain weights and an inadequate representation. Familiar observations, independent coordinate probes, and a larger coordinate range are scored separately. Input schemas, features, reset access, likelihood and noise scale are supplied.

The frozen code protocol uses 65 grid values, 16 propagation steps, 24 coordinate sweeps for independent clouds, 8 or 32 support examples, and 256 fresh query cases per partition. Development seeds 9100 and 9101 select one mixing setting per method and suite using 64 validation cases. Each mixing method gets three candidates; Dirac uses a different numerical range appropriate to its first-order operator. Phase scale stays fixed at one. Main seeds 5200–5209 never select settings. Every model in a case is saved before final query scoring. The smoke run uses a smaller declared protocol and is excluded.

Every grid potential is obtained from the same quadratic sufficient statistics. This avoids secretly re-evaluating all examples on each propagation step; the statistics construction and grid/coordinate evaluations are separately counted. The Gaussian and gradient controls exploit the same structure. Equal data access is not equal CPU time, storage, or objective-operation count. The final record includes tuning cost and raw checkpoints.

## Measures and decision

Report predictive MSE against evaluator truth, observation NLL, nominal 95% Gaussian moment-band coverage, predictive width, boundary mass, norm error, retained bytes, checkpoint bytes and measured process costs. Joint-grid NLL uses the full finite Gaussian mixture; factorized-cloud NLL uses a disclosed moment approximation. The latter is not an exact mixture likelihood. Moment bands are not exact mixture credible intervals. Paired uncertainty is computed across independent seeds, not across repeated query cases.

The experiment can support or reject a benefit for these settings. A positive result would still require testing more difficult nonlinear weight spaces and integrating a candidate with the shared solver's independent retention checks. No variant becomes the active solver merely because its representation is more elaborate. Complex phase is useful only if it improves held-out behavior over the declared classical and phase-erased controls after its cost is counted.

## Checklist

- [x] Preserve and hash both archives; verify 176 and 726 manifest entries and the exact embedded duplicate.
- [x] Verify the pinned SERA and Kavi repository revisions separately.
- [x] Implement explicit weight amplitudes, open boundaries, Hermitian propagation and classical controls.
- [x] Test normalization, phase-only invariance, zero-mixing Bayesian equivalence, correlations and evidence roles.
- [ ] Run and preserve development selection and the frozen main experiment.
- [ ] Replay saved predictions and audit potential access, costs and source identities.
- [ ] Report positive, negative and unresolved results and preserve the existing solver and variants.

## Primary research references

- [Blundell et al., Weight Uncertainty in Neural Network](https://proceedings.mlr.press/v37/blundell15.html): learning weight distributions is an established classical approach; this work does not reproduce Bayes by Backprop.
- [Abel, Criado and Spannowsky, Training Neural Networks with Universal Adiabatic Quantum Computing](https://arxiv.org/abs/2308.13028): relates a neural loss to a quantum Hamiltonian; its physical/computational setting differs from this finite CPU simulation.
- [Lami et al., Quantum Annealing for Neural Network Optimization Problems](https://arxiv.org/abs/2208.14468): tensor-network simulation is a possible structured approximation, not evidence that an arbitrary full weight cloud is cheap.
- [MIT Quantum Physics II equation reference](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/07860b0b0f550e0135c9191441faca3d_MIT8_05F13_final_2011.pdf): Hermitian Hamiltonians, unitary propagation and the time-evolution equation.

The equations and limiting-case equivalences above are derived and tested here. External papers establish precedents, not performance claims for SERA.
