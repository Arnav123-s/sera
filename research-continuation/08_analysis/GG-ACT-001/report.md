# GG-ACT-001: fixed active inquiry

I completed 300 policy runs across 60 paired latent worlds: 12 frozen instances in each of five families. Every run paid for 10 observed pairs, including the same two initial observations. The learner received a supplied phase coordinate, known noise, a finite action grid and four generator classes. It learned coefficients and class weights by Bayesian updating. The five inquiry policies were fixed; no investigator was trained.

The grid offered 15 target measurements and 15 known zero-design nuisance measurements. Nuisance noise had standard deviation 20, versus 0.05 on target measurements. The reference space-filling policy used the supplied channel metadata. This tests raw entropy against epistemic usefulness under known noise, not discovery of which sensor is informative.

## External prediction after ten paid observations

Clean MSE per coordinate, averaged over 12 worlds and 41 independent query positions on [-1.8, 1.8]. Lower is better. Each policy sees the same potential observations when it chooses the same action.

| Family | Random | Space filling | Raw entropy | Disagreement | Information gain |
|---|---:|---:|---:|---:|---:|
| circle | 0.00104073 | 0.000395231 | 0.884534 | 0.000472325 | 0.00047934 |
| ellipse | 0.00132355 | 0.000825921 | 3.48862 | 0.000727108 | 0.00082877 |
| line | 0.00120754 | 0.000417007 | 2.2429 | 0.000538443 | 0.000509533 |
| radial_orbit | 0.00290303 | 0.00075882 | 3.4034 | 0.00116137 | 0.000857328 |
| menu_exception | 0.111693 | 0.088737 | 1.25384 | 0.0686113 | 0.066968 |

Information gain reduced mean MSE relative to random sampling on all five families. Its exploratory paired bootstrap intervals excluded zero on the four in-menu families, but included zero on the omitted-pattern family. All four in-menu information-gain versus space-filling intervals included zero, and space filling had lower mean error in each. On omitted patterns, information gain reduced MSE versus space filling, while uncertainty remained badly miscalibrated. I do not promote an overall information-gain advantage.

| Family | IG minus space-filling MSE | Exploratory 95% interval |
|---|---:|---:|
| circle | 8.41096e-05 | [-0.000111773, 0.000273803] |
| ellipse | 2.84921e-06 | [-0.000305224, 0.000310284] |
| line | 9.2526e-05 | [-7.24785e-05, 0.000242547] |
| menu_exception | -0.021769 | [-0.0320855, -0.0117572] |
| radial_orbit | 9.85076e-05 | [-0.000145945, 0.000406742] |

The bootstrap resamples paired worlds with 1,000 frozen-seed draws. These are small exploratory comparisons without multiplicity correction.

## Confidence and model inadequacy

| Family, information gain | Joint query NLL | Marginal 95% coverage | Mean best-class weight | Any inadequacy alarm |
|---|---:|---:|---:|---:|
| circle | -2.9762 | 95.3% | 99.890% | 0.0% |
| ellipse | -2.9727 | 95.9% | 100.000% | 0.0% |
| line | -3.034 | 96.1% | 100.000% | 0.0% |
| radial_orbit | -2.8537 | 95.7% | 100.000% | 0.0% |
| menu_exception | 17.421 | 51.3% | 99.738% | 75.0% |

The omitted family is a circle with a localized oscillatory exception absent from the menu. Information gain ended with 99.738% mean best-class weight but only 51.3% marginal coverage. The prequential alarm fired in 9 of 12 runs and missed 3. Confidence among supplied alternatives is not confidence that an adequate alternative exists. The alarm threshold is uncorrected across adaptive repeated tests and is not an applicability guarantee.

Raw predictive entropy chose nuisance measurements on all eight discretionary steps in every run. Random sampling used 3.083 nuisance observations on average; the other policies used zero. High entropy alone rewarded irreducible noise. Broad intervals from the raw-entropy policy often covered values despite poor point predictions.

## Verification and cost

Every action, full candidate-score vector, hypothetical branch, observation, posterior, external prediction, joint likelihood, marginal interval vector and summary replayed exactly. A separate implementation checked all 300 expected policy records, all 3,000 observations and 12,000 batch Gaussian posteriors. Maximum independent mean difference was 1.14e-12 and maximum log-class-weight difference was 3.22e-9.

Primary process time was 98.379 seconds wall and 95.109 seconds CPU; supervised wall time was 101.149 seconds, with 257,298,432 peak committed job bytes. Exact replay used 62.346 supervised seconds. These local measurements include scientific evaluation and writing; they are not a speed comparison against another implementation. The primary manifest covers 184,009,917 bytes of raw audit artifacts. Decoder source occupies 70,262 bytes; final retained side state is about 15.7–16.0 KB per run. Python, Torch, NumPy and SciPy installations are additional shared runtime costs. No complete packaged compression claim is made.

Order-nine bivariate Gauss-Hermite quadrature approximates mixture entropy/EIG. Information gain used 98,496 density evaluations per run; raw entropy used 134,784. Space filling is substantially cheaper here. Selection effort and observations must both count.

This panel keeps the analytic posterior as explicit side state. Its owner adapter is contract-tested, but the full fixed-policy panel does not train SERA's neural owner. The separate [trained-parent integration](../../05_experiments/shared-generative-integration.md) places a matching four-class generator in registered R1 buffers.

[Frozen protocol](../../04_protocols/GG-ACT-001.json) · [Full compact statistics](../../14_release/GG-ACT-001-summary.json) · [Exact replay](../../14_release/GG-ACT-001-replay.json) · [Independent audit](../../01_audit/GG-ACT-001-independent.json)
