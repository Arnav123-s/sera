# GG-GUARD-002 — independent-world selective calibration

I kept the complete GG-GUARD-001 predictor unchanged: 291 learned scalars with artifact identity `8e5f72ef11676b660848d4ff094a25736fdeb73213ba7f01bb758cb12ac8ab2a`. This study fits acceptance thresholds, not new neural weights. It tests the v3 packet's statistical correction on the actual SERA feature/predictor implementation.

## Prospective contract

The [frozen protocol](protocol.json) specifies 1,800 calibration worlds, 1,000 matched assessment worlds, 500 shifted-mixture worlds, and 256 worlds each from two newly defined mechanisms: abrupt translated jumps and decaying spirals. Each world supplies ten fixed support observations and **one** independently selected query. Query selection uses a separate RNG stream before its outcome. The 33 available query positions are not counted as 33 independent examples.

Four groups use only public observed features: mean late prequential minimum component residual at most four versus larger, crossed with near/in-support versus edge queries. The threshold grid is 0.90, 0.925, 0.95, 0.975 and 0.99. A one-sided Clopper–Pearson upper error bound must be at most 0.10 with at least 40 accepted calibration worlds. The first candidate spends 0.025 of a declared 0.05 sequential confidence budget; global and grouped policies each receive 0.0125, then split it across their finite group/threshold families.

These guarantees require the fixed predictor, features, query/acquisition policy, loss, groups and conditional IID population. Every dependency is fingerprinted. The probability of an error refers to a randomly selected eligible query under that contract. It is not a guarantee for every point or an arbitrary unobserved mechanism.

## Assessment

| Panel | Old empirical policy: answered / wrong | Global certificate: answered / wrong | Group certificate: answered / wrong |
|---|---:|---:|---:|
| Matched, 1,000 worlds | 840 / 85 | 682 / 1 | 664 / 0 |
| Mechanism-mixture shift, 500 | 364 / 110 | 146 / 3 | 119 / 1 |
| Abrupt jump, 256 | 160 / 130 | 0 / 0 | 0 / 0 |
| Decaying spiral, 256 | 256 / 10 | 251 / 9 | 212 / 8 |

The matched grouped-policy error estimate is 0/664. Its individual one-sided 95% upper bound is **0.450%**, not zero true risk. In-menu coverage is 661/671, or **98.51%**. The matched consistent-near and consistent-edge groups both pass the stricter prespecified simultaneous assessment bound. Contradicted groups do not obtain a certified threshold.

The grouped policy accepts only **3/169 local-exception worlds** and **0/160 random-value worlds**. Zero wrong answers on three exception examples gives a 63.16% individual upper bound; it does not establish mastery of exceptions. Abrupt-jump coverage is zero. Decaying-spiral accepted error is **8/212 = 3.77%**, with an individual one-sided 95% bound of 6.71%. Newly exposed mechanisms are no longer protected material for later research.

The global certificate chooses 0.90 and exactly matches the simple probability-floor control here. On matched worlds its utility is 0.677 versus 0.664 for the grouped policy. The paired grouped-minus-floor utility difference is **−0.013**, with the predefined descriptive bootstrap interval **[−0.023, −0.001]**. This meets the −0.05 noninferiority margin but provides no superiority result. I retain both certificates and favor the simpler global rule as the performance reference; the grouped certificate supplies more explicit conditional risk boundaries at a coverage cost.

The shifted-mixture panel is a stress test, not a theorem about arbitrary shifts: changing mechanism proportions can change risk within an observable group. The supplied pack's hidden-within-group failure remains valid negative evidence.

## Audit and costs

The [independent audit](independent.json) reconstructs every query's features from observed events using batch Gaussian algebra, evaluates the network independently, reproduces query selection and labels, and inverts binomial tails separately. Maximum discrepancies: features 2.20e−10, means 1.85e−10, probability 2.23e−16 and risk bounds 1.37e−15; all are below the prospectively declared tolerances.

The cohort acquired **38,120 support observations and 3,812 query outcomes**. No extra inquiries or optimizer steps were added. Execution used **134.374 supervised seconds** and **662,003,712 peak committed job bytes**; independent audit used another **22.727 seconds**. Timing fixtures, failed preflight, regression tests and earlier model training have separate records. The entire raw cohort, certificates, pre-final identities and summary are retained here.

The predefined component gate passes. This does not integrate the guard with the shared learning route, resolve missing mechanisms, demonstrate positive transfer or establish an improved investigator.
