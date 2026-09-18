# VC-001: a learned investigation procedure on the continuing owner

I connected a learned evidence-selection policy and joint completion weights to the existing GroundR1 owner. A conditional acceleration gap now leads to coherent imagined motion, a committed investigation, an independently checked prediction and a persistent policy update. The original task, predictor versions and evidence identities bind the credit; repeated or stale credit is rejected.

## Teaching and evaluation

The v14 prior is reused unchanged: 48 supplied numerical mechanisms and 1,152 synthetic observations. New policy teaching uses 4,096 conditional worlds, two seeds, 800 minibatches of 64 per seed and **102,400 repeated reward presentations**. These are not 102,400 independent worlds. The eight public features, response family `a=b*u-d*v+c`, observation mechanism, costs and Gaussian log-score checker are engineered. Each policy update receives only its selected action's reward. Neither hidden coefficients nor the full reward table enter the policy.

The selected seed is 3212, chosen by mean reward on 512 development worlds: 0.507964 versus 0.494594 for seed 3211. Both candidates and their full costs remain preserved. Selection was frozen before final evaluation: 256 matched, 256 shifted and 256 omitted-mechanism worlds, crossed with six methods for 4,608 evaluations. Every final world is independent of the teaching and development seeds.

Reward is the independently checked Gaussian log-score improvement minus 0.1 times acquisition price. Error is mean squared error against the assessor's clean response. Coverage measures the observed fraction inside the nominal 95% predictive interval. STOP costs zero and supplies no observation.

| Matched method | Mean reward | Prediction MSE | Coverage | Mean cost | STOP |
|---|---:|---:|---:|---:|---:|
| Learned policy | 0.542925 | 0.034829 | 95.31% | 0.148085 | 15.63% |
| Initial policy | -0.428880 | 0.177717 | 94.92% | 0.428880 | 0.00% |
| Random choice | -0.133740 | 0.133974 | 95.70% | 0.375176 | 5.86% |
| Always STOP | 0.000000 | 0.177717 | 94.92% | 0.000000 | 100.00% |
| Goal information | 0.637109 | 0.028282 | 96.48% | 0.177659 | 13.67% |
| Parameter information | 0.285174 | 0.061838 | 95.70% | 0.275770 | 0.00% |

The learned policy reduces matched error by **80.40% against its initial policy** and **74.00% against random choice**, meeting the frozen qualification for explicit conditional practice. Goal-information selection earns the strongest matched reward and retains the default recommendation. No model or threshold was adjusted using final results.

| Fresh family | Learned reward | Learned MSE | Learned coverage | Goal-information reward | Goal-information MSE |
|---|---:|---:|---:|---:|---:|
| Shifted parameters | 1.033851 | 0.084737 | 83.98% | 1.122203 | 0.073321 |
| Omitted nonlinear mechanism | 0.152240 | 0.123840 | 78.91% | 0.248896 | 0.093267 |

The shifted and omitted cohorts retain useful improvements against initial and random choices. Their coverage also identifies where the supplied model's uncertainty is incomplete. This diagnostic is preserved as evidence for future mechanism refinement; its final cases remain sealed. The measurements concern this conditional family and make no general ability or LLM ranking claim.

## Connection and credit

`CompletionR1` extends the actual owner in place. `completion_policy` is a learned 8â†’32â†’1 candidate scorer with 321 parameters. Each task's `completion_clouds` entry registers a three-coefficient mean and covariance. One mechanism sample supplies each branch's complete rollout; the actual retained StudyR1 integral operator executes the motion, and independent rational arithmetic checks its consequences.

The lifecycle is **original goal â†’ candidate â†’ committed decision â†’ acquisition â†’ frozen prediction â†’ independent audit â†’ credit**. Credit updates only the policy; acquisition changes the cloud; imagination changes neither evidence store. A per-source audit identity prevents renamed tasks from receiving the same reward again. Decisions are sampled from the current policy for on-policy credit. Final comparisons use deterministic highest-scoring choices. Both behaviors are explicit in the records.

I implemented the infrastructure; SERA learned the policy weights from checked practice. Supplied source flags and numerical representations are recorded as supplied. The existing applicability and novel-definition gates are preserved.

## Verification and preserved repairs

The adopted v14 packet's 253 manifest identities and scientific jobs were checked once. Its strict archived lifecycle hash comparison failed across environments. Local interrupted and uninterrupted replays agree exactly; portable comparison found at most 8.89e-16 numeric difference and 1.12e-16 policy-tensor difference. Original failures and all hash differences are retained in [packet qualification](packet-verification.json).

The first integration tests exposed a denominator mismatch in imagined values: the retained exact operator reconstructs rational coefficients with denominator at most 120. The new branch sampler initially passed much finer fractions. [VC-F01](failures/VC-F01/DIAGNOSIS.md) preserves the original implementation and failing output. The repair declares a 1/60 acceleration grid for execution, retaining continuous cloud values for acquisition and scoring. Earlier mathematical weights and completed evaluations were untouched. The repaired completion suite passed 12 targeted tests.

The [independent audit](integration-audit.json) passed: both 400â†’800 training resumptions reproduced weights, optimizer, random state, baseline and per-step traces exactly. A separate NumPy network implementation reproduced every final action; precision-form conditioning and log-score calculation independently checked all 4,608 decisions. The largest network-logit difference was 3.56e-15.

The actual owner retained all **186 predecessor tensors, 128 four-language probes, 26 taught physical bindings**, reading, exact mathematics and empirical forecasts. Three sequential practice tasks earned rewards 0.795066, 0.393433 and 0.867035, with exact interrupted continuation and five verified storage revisions. All earlier task records, including the open momentum investigation, remained unchanged. The saved live command completed a further task with reward 1.017918 and one retained policy update.

Full regression, exact artifact identities and publication receipts accompany this report. All resource reservationsâ€”including failed checks, replay and source intakeâ€”are recorded in [costs](costs.json). [Architecture comparison](architecture-audit.md) distinguishes learned procedure changes, fixed supervision and retained knowledge.

The first full local regression reached its 420-second job cap after 432 passing checks, with no reported assertion failure. The exact log and full wall cost are retained. The remaining modules passed 52/52 in a second bounded run, including nine overlapping checks: the union covers all 475 collected tests with zero skips. This is a completed partitioned regression, not a single uninterrupted run. The public Linux/Windows workflow independently checks a clean checkout.
