# GG-P0-001: compact generative memory results

I trained and evaluated eight controls on four supplied-phase task families. This is a component study; it does not establish discovery from unordered points, learned semantic applicability, shared neural transfer or improved investigation.

The complete cohort contains 640 paired cases and 5120 serialized models. Source identity: `b83aab04331d39eadb59db4d6bff183ad69a84324adecadc09d4eee444dcc590`. All methods used the same observations within a case. The neural control used 160 gradient steps; symbolic search evaluated 130 supplied expression candidates.

The following tables show means over 20 instances at 64 support observations and noise standard deviation 0.02. All other support/noise strata and paired intervals are retained in `summary.json`. MSE is lower-is-better; withheld-arc coverage is a diagnostic for 95% Gaussian-moment marginal intervals, not a guarantee.

## circle

| Control | In-region MSE | Withheld-arc MSE | Extrapolation MSE | Arc coverage | Artifact bytes | Alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| A_episodic | 0.46072 | 0.44456 | 0.445062 | 100.0% | 4733 | 100% |
| B_interpolation | 0.000365745 | 0.229895 | 1.1424 | 8.7% | 4760 | 0% |
| C_circle | 9.75575e-06 | 1.81845e-05 | 1.60304e-05 | 96.3% | 1433 | 0% |
| D_family | 9.75583e-06 | 1.81863e-05 | 1.60318e-05 | 96.3% | 2707 | 0% |
| E_symbolic | 1.63045e-05 | 6.79073e-05 | 4.95502e-05 | 96.2% | 1655 | 0% |
| F_neural | 0.00151098 | 0.254513 | 1.45552 | 14.0% | 13720 | 10% |
| G_program | 9.75575e-06 | 1.81845e-05 | 1.60304e-05 | 96.3% | 1419 | 0% |
| H_residual | 2.5752e-05 | 1.81845e-05 | 1.60304e-05 | 96.4% | 5595 | 0% |

## ellipse

| Control | In-region MSE | Withheld-arc MSE | Extrapolation MSE | Arc coverage | Artifact bytes | Alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| A_episodic | 0.323576 | 0.366563 | 0.353035 | 100.0% | 4731 | 100% |
| B_interpolation | 0.000321272 | 0.155903 | 0.832085 | 8.0% | 4758 | 0% |
| C_circle | 0.00976914 | 0.0809635 | 0.0516957 | 33.0% | 1452 | 70% |
| D_family | 1.88205e-05 | 7.36867e-05 | 5.39261e-05 | 96.6% | 2701 | 0% |
| E_symbolic | 2.15183e-05 | 0.00337758 | 0.0897444 | 92.0% | 1655 | 0% |
| F_neural | 0.00128382 | 0.17937 | 1.09105 | 9.5% | 13732 | 10% |
| G_program | 1.88205e-05 | 7.36867e-05 | 5.39261e-05 | 96.6% | 1648 | 0% |
| H_residual | 2.9549e-05 | 7.36867e-05 | 5.39261e-05 | 96.7% | 5826 | 0% |

## exception

| Control | In-region MSE | Withheld-arc MSE | Extrapolation MSE | Arc coverage | Artifact bytes | Alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| A_episodic | 0.435939 | 0.328267 | 0.34073 | 100.0% | 4730 | 100% |
| B_interpolation | 0.000295978 | 0.165735 | 0.862986 | 8.0% | 4757 | 0% |
| C_circle | 0.00519599 | 0.00537283 | 0.00529313 | 94.0% | 1446 | 55% |
| D_family | 0.00435132 | 0.00799862 | 0.00622151 | 76.5% | 2706 | 45% |
| E_symbolic | 0.0027771 | 0.424363 | 8.24993 | 25.9% | 1992 | 30% |
| F_neural | 0.00240076 | 0.173205 | 0.987152 | 12.0% | 13716 | 20% |
| G_program | 0.00242094 | 0.100103 | 0.376554 | 34.1% | 2334 | 30% |
| H_residual | 0.000806102 | 0.100103 | 0.376554 | 23.2% | 6461 | 0% |

## random

| Control | In-region MSE | Withheld-arc MSE | Extrapolation MSE | Arc coverage | Artifact bytes | Alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| A_episodic | 0.998498 | 0.999825 | 1.02052 | 99.4% | 4717 | 100% |
| B_interpolation | 1.67585 | 1.85143 | 1.87567 | 93.0% | 4736 | 100% |
| C_circle | 1.03742 | 1.06778 | 1.07873 | 93.8% | 1444 | 100% |
| D_family | 1.05682 | 1.21826 | 1.24432 | 92.1% | 2703 | 100% |
| E_symbolic | 1.04138 | 16.968 | 772.484 | 78.1% | 1619 | 100% |
| F_neural | 1.04315 | 1.16341 | 1.30552 | 92.7% | 13760 | 100% |
| G_program | 1.04566 | 1.42387 | 1.84681 | 89.9% | 1752 | 100% |
| H_residual | 1.08466 | 1.42387 | 1.84681 | 90.7% | 5706 | 100% |

## Cost and claim boundary

Case execution charged 207.27 seconds, including serialization and scoring. The methods used 94,080 candidate fits and 102,400 neural gradient steps. All model objects total 19,401,037 bytes; the whole preserved run is 169,104,365 bytes. The shared decoder source adds 14,584 bytes to a standalone deployment, or 145.84 bytes per object at an explicitly declared 100-object deployment. Runtime dependencies remain additional to these figures.

The circle and other program primitives, phase convention, noise level, expression grammar, fitting algorithms and neural architecture were supplied. Learned quantities are coefficients, class weights or expression choices, network weights, local residuals and calibration variance. A class posterior remains conditional on its menu. A residual alarm is empirical and may miss an unseen exception.

A stores exact observations and marks unfamiliar queries unavailable; its reported unseen MSE uses the fixed zero-mean fallback so abstention does not remove hard cases. Random unseen values are independent facts and remain unpredictable. B clamps beyond the observed interval. H's local residual applies only inside its stored support range. E/G select a single candidate using a fixed selection criterion; their uncertainty omits selection uncertainty. D uses coherent model averaging but may still miss an absent class.

All intervals are exploratory paired-instance intervals, without multiplicity correction or a claim of confirmed superiority. The complete model records, candidate outcomes, source copies and query vectors remain in the run directory. Fresh-process replay is reported separately.
