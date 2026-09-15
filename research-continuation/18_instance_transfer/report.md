# Did corrected acquired knowledge help the neural route?

The prospectively frozen acquisition-dependent component **passes its gate**. I completed 20 trained descendants, five new learning streams and four controls. Every source/checkpoint, raw prediction, failed result and cost is preserved. This follows the [50-run normalized-teaching study](../17_transfer/report.md); its useful readout result remains separate from its failed acquired-coefficient attribution.

## What changed and what the model learned

I removed the normalization that canceled the source instance's fitted coefficients. The conditional teacher now generates trajectories in the original simulated circle's coordinates. Its actual pre-correction and post-correction checkpoints provide the K intervention; their initial neural weights are bitwise equal. A preflight verifies a nonzero teaching-target change before training.

Each stream receives 32 new simulator trajectories, each with three coordinate observations and a next-position target. Conditional arms add 512 trajectories from either corrected or incorrect K; the oracle-exposure control instead receives independent simulator trajectories at the matched phase schedules. The learner never sees those phases, center, radius or angular speed. The circle law and noiseless future simulator are supplied. This is one simulated physical instance, not a collection of independently acquired physical laws.

The same 514 existing readout parameters are trained for 256 updates, using a target batch of 32, an old-replay batch of 32 and 32 parent-output anchors. All other neural parameters and acquired generator buffers remain fixed. A separate 128-trajectory development set selects among four immutable steps. The final 256 ordinary and 128 faster-turning trajectories per stream are not used for training or selection.

The original K was acquired from 16 observations. The corrected source additionally has 1 independent sensor recheck; its cost remains in the original study. I do not describe that extra information as free.

## Performance

| Teaching source | Ordinary RMSE | Faster-turning RMSE | Worst old-score drop | Training seconds |
|---|---:|---:|---:|---:|
| observed_only | 0.168353 | 0.435256 | 0.000 pp | 85.77 |
| corrected_teacher | 0.155238 | 0.423901 | 0.000 pp | 87.10 |
| uncorrected_teacher | 0.177209 | 0.435726 | 0.000 pp | 89.05 |
| oracle_exposure | 0.155119 | 0.423533 | 0.000 pp | 88.59 |

The unadapted neural parent's ordinary RMSE was **0.319510**. Scores above pool five paired learning streams from one pretrained parent. The 20 descendants make 7,680 final predictions on 1,920 unique final trajectories; repeated control predictions are not independent worlds.

Against **observed_only**, corrected teaching changes mean MSE by **14.97%** in the favorable direction. The paired absolute-gain interval is **[0.0030743, 0.0054226]**. This planned contrast **passes**.

Against **uncorrected_teacher**, corrected teaching changes mean MSE by **23.26%** in the favorable direction. The paired absolute-gain interval is **[0.0044230, 0.0100277]**. This planned contrast **passes**.

The gate requires at least 10% mean MSE gain against both observed-only and incorrect-K controls, positive paired 97.5% bootstrap intervals, improvement in every learning stream, and at most two points of loss in any of 61 retained groups. The intervals are descriptive estimates over five learning streams, not a universal or finite-sample guarantee.

Correcting the acquired instance helped this separate neural prediction route under the stated controls. The result concerns a protected readout within a supplied family. It does not show improved shared representations, learned grammar, hidden-state online investigation or an independently improved learning procedure.

## Audit, persistence and next step

The independent auditor restored every selected checkpoint and checked all **7,680** saved predictions exactly. Its separate geometric recurrence verified final targets with maximum error **1.78e-15**. Retention summaries were reconstructed from the saved paired controls. Original owner/checkpoint identities remain unchanged.

The predeclared integration rule selected `research-continuation/18_instance_transfer/seed-307/corrected_teacher`. The [migration report](../17_transfer/integration/result.json) records a new experimental shared solver, explicit factual replay, full finite reproof, stale-record rejection and exact reload. Its original cumulative costs and knowledge remain preserved. The existing operational store was not replaced.

Full costs across both cycles, including failed checkers, development and all three resumption checks, are in [final release accounting](final-costs.json). Its boundary distinguishes supervised worker time from interactive research, engineering and file verification. All 210 local regressions passed. A [separate tensor audit](parameter-preservation-audit.json) verifies that every selected raw-coordinate model preserved all generator buffers and every numerical parameter outside the readout. See the [architecture audit](../17_transfer/architecture-audit.md), [actual K intervention](../17_transfer/acquisition-intervention/result.json), [frozen protocol](protocol.json), [independent results](audit-summary.json) and [saved learner commands](resume.md).

The unresolved architectural work is retained shared-representation learning, continuing hidden-state interaction and a separate test of eta. A readout improvement or accumulated K does not establish learning-to-learn. M2–M4 remain open.
