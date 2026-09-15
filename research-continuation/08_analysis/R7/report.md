# R7: preserved parameter-distribution controls

I preserve this optional experiment as a reference branch. The main SERA/Kavi roadmap remains the priority. I fitted 1,260 finite-cloud and classical controls and replayed 3,780 query conditions. The development panel is separate and is not counted as additional final evidence.

Weights live on a 65-point grid over [-2, 2]. The joint two-weight and factorized six-weight controls compare Gaussian updating, likelihood filtering, phase evolution, dephasing, imaginary-time evolution, classical diffusion, a Dirac-like spinor and a gradient point estimate. Every fitted loss is quadratic in the weights. These are classical simulations of supplied dynamics, not quantum hardware or a demonstration of general neural training.

The table gives independent-query MSE for the six-weight control at 32 observed examples, averaged over ten seeds. All strata and storage costs remain in summary.json.

| Family | Method | MSE | Numeric state bytes | Operator bytes |
|---|---|---:|---:|---:|
| correlated | dirac | 0.252796 | 13000 | 270400 |
| correlated | filter | 0.211703 | 3640 | 0 |
| correlated | gaussian | 0.237798 | 336 | 0 |
| correlated | schrodinger | 0.982948 | 6760 | 67600 |
| misspecified | dirac | 0.117624 | 13000 | 270400 |
| misspecified | filter | 0.11656 | 3640 | 0 |
| misspecified | gaussian | 0.0320386 | 336 | 0 |
| misspecified | schrodinger | 0.124026 | 6760 | 67600 |
| well_specified | dirac | 0.00702303 | 13000 | 270400 |
| well_specified | filter | 0.00651929 | 3640 | 0 |
| well_specified | gaussian | 0.00654005 | 336 | 0 |
| well_specified | schrodinger | 0.00779289 | 6760 | 67600 |

The matched filtering controls do not establish a reliable benefit from coherent phase evolution. Correlated designs and omitted features expose poor confidence and large prediction errors in some factorized variants. The cheap exact Gaussian reference is particularly strong because the loss and prior assumptions make it appropriate. No quantum advantage is claimed, and no SERA architecture replacement is justified.

Primary execution took 146.154 seconds wall and 127.922 seconds CPU. Numeric arrays, operators, serialized checkpoints, runtime and development costs are separate; the table is not a complete deployment-size comparison. Full checkpoints and queries remain under runs/parameter-cloud-main; exact verification is [published here](../../14_release/R7-replay.json). No nonlinear follow-up was executed.
