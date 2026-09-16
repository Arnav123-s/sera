# Useful imagination and online correction in a continuing learner

I completed two development candidates (42 continuing lifetimes) and a separately frozen 168-lifetime study in 24 paired simulated worlds. The revised component passes its declared gate and is integrated into a persistent experimental descendant of the actual A06 learner. The original operational model and all rejected work remain preserved.

## What it learned and did

The learner observes noisy positions and its own actions, infers motion without receiving phase or velocity, and learns three action-response coefficients from real transitions. It imagines four-action sequences, executes the first action, observes the result and revises its dynamics. An observed-innovation detector handles a reversal of actuator response without discarding factual history. Every target change happens within the same continuing world; there is no reset between questions.

The acquired circle coefficients remain useful input, while the action grammar, linear estimator, detector and planning procedure are supplied. The same shared R1 gains 248 bytes of registered statistics/state; no new neural parameters are trained. This is K acquisition under an engineered eta, not learned learning-to-learn.

## Prospective performance

Lower physical cost is better. It combines squared target-position error, action effort and velocity penalty. Each nominal comparison uses 12 paired worlds; the other 12 worlds add a force omitted from the learned dynamics grammar.

| Controller | Nominal physical cost | Omitted-force cost | Nominal priced work proxy | Mean nominal lifetime seconds |
|---|---:|---:|---:|---:|
| adaptive_mpc | 0.388328 | 0.481698 | 0.413064 | 1.314 |
| full_history_mpc | 1.287953 | 1.368207 | 1.312688 | 1.307 |
| frozen_mpc | 1.405445 | 1.491746 | 1.429167 | 1.284 |
| adaptive_one_step | 0.698697 | 0.783478 | 0.700963 | 1.125 |
| adaptive_reactive | 0.683151 | 0.782566 | 0.685207 | 1.085 |
| incorrect_geometry | 0.455099 | 0.563767 | 0.479835 | 1.318 |
| analytic_mpc | 0.358854 | 0.429026 | 0.383589 | 1.311 |

Four-step imagination reduces nominal physical cost by **43.16%** versus reactive control and **44.42%** versus one-step prediction. Every one of the 12 nominal worlds improves both comparisons.

The paired absolute-gain 97.5% bootstrap interval against `adaptive_reactive` is [0.211546, 0.379298].
The paired absolute-gain 97.5% bootstrap interval against `adaptive_one_step` is [0.215733, 0.409079].

The prespecified priced proxy improves **39.72%** over reactive control. Post-reversal physical cost falls **82.80%** versus frozen dynamics. Actual computation is higher: nominal lifetime time is 1.314s versus 1.085s. Equal measurement counts and maximum decision allowances are not exact machine-cost equality. The proxy's explicit prices are in the frozen protocol.

Using corrected rather than the actual incorrect predecessor geometry reduces nominal mean cost by 14.67%; two individual worlds go the other way. This secondary contrast is not a uniform transfer guarantee. The privileged linear-coefficient reference is stronger on average and still lacks hidden state and the omitted force.

## What failed or remains uncertain

The first development candidate barely beats one-step prediction (0.70260 versus 0.70289). Retaining every old transition in one stationary fit is harmful after response reversal. Those results and source versions remain preserved. The revised detector was tested on new development seeds, then frozen before prospective worlds; no final-bank tuning occurred.

Conditional uncertainty is incomplete. The table below reports actual coverage of the diagnostic coordinatewise 1.96-standard-deviation rectangles, alongside rollout error. These rectangles are not jointly calibrated 95% regions and are not an answer certificate.

| World family | One-step rollout RMSE | Four-step rollout RMSE | One-step rectangle coverage | Four-step rectangle coverage |
|---|---:|---:|---:|---:|
| gain_reversal | 0.01584 | 0.10335 | 49.07% | 89.70% |
| omitted_torque | 0.02195 | 0.19468 | 34.72% | 47.69% |

Unpropagated geometry/state error and omitted dynamics limit these conditional intervals. Higher decision utility does not cure that uncertainty failure. Fresh policy-specific qualification is required before any supported-answer claim. All earlier statistical certificates remain attached to their original policies/owners.

## Audit, retention and actual continuation

The independent auditor reconstructs all **12,096 decisions**, checks noisy sensor provenance, uses complex-plane physical dynamics and QR-based coefficient fitting, and verifies action choices and counterfactual predictions. Maximum coefficient disagreement is below 6e-14. Each of the 168 final owners restores from its real history with original tensors unchanged. The 12,096 repeated decisions are not 12,096 independent worlds.

The predeclared first nominal stream becomes the new experimental descendant. Its 75-observation session reloads and takes four further paid actions/observations, reaching 79 observations and world clock 78. Original facts and the sensor correction replay under the new owner; stale factual/library bindings are rejected and finite programs fully re-proved. The finite example `4*x-5=6 modulo 11` still returns 0.

A separate fresh retention bank gives **exactly identical group and case scores across 61 groups**. **219 regressions pass**, including evidence isolation, mutation-bound stale plans, genuine correction, future-decision equivalence after restore, and migration-scope rejection. This does not establish long-term neural plasticity: old neural representations are held fixed.

## Costs, preservation and next state

This continuation charges **365.54 supervised worker seconds (6.09 minutes)**, including development, all verification and fresh retention. Peak job committed memory is **1,270,173,696 bytes** under the original 2 GiB ceiling. The cumulative allowance has **278.87 seconds remaining**; it is not exhausted or expanded. There are 15,250 newly acquired sensor measurements across development, prospective controls and four continuation steps. Historical source-acquisition costs remain in the original lineage. Interactive research/engineering time is outside the measured worker boundary.

The previous 451-file release and 108 original predecessor artifacts remain byte-identical, including all 45 shared interpreter sources. The full local ledger, failed candidates, per-lifetime raw records, frozen source archives and checkpoint inventory are retained. No background experiment is left running and no remote publication was performed.

See [the source-architecture audit](architecture-audit.md), [complete independent outcomes](A08-FINAL-001/independent-audit.json), [costs](costs.json), [failures](failures.md), [source research](research-notes.md) and [resumption commands](resume.md). The next major requirement is independently learned eta on unfamiliar continuing tasks, with K and eta crossed separately; uncertainty qualification and retained shared-representation learning remain open. M2–M4 are not declared complete.
