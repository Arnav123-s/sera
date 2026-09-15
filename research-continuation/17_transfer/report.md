# Neural motion transfer and acquired-knowledge audit

I completed two prospectively frozen scopes, five independent teaching/optimizer streams per scope, and five controls per stream: **50 trained descendants**. Three development candidates, one failed mechanism test and every checkpoint/cost remain preserved.

The numerical question is whether conditional circle teaching improves a different neural route. The stronger acquired-knowledge question has a separate answer: normalizing the source circle cancels all its fitted coefficients. I therefore reject a claim of causal transfer from those acquired coefficients, regardless of a numerical component gate.

## What was taught

Each learning stream has 32 independently simulated observed circle trajectories. Each trajectory supplies three equally spaced coordinate observations and one next-position target. Conditional arms add 512 generated trajectories; the simulator-exposure arm receives independent ground-truth targets for the same 512 worlds. Observed-only arms resample their original 32 worlds. The neural route never receives center, radius, angular velocity or phase and never calls the analytic generator at assessment. The grammar, transforms, noiseless-circle assumption and optimizer are supplied.

Every arm takes 256 AdamW updates with 32 target and 32 old-replay examples per update, plus 32 parent-output anchor examples. A separate 128-world development bank selects among four saved steps. Final assessment uses 256 new ordinary and 128 faster-turning worlds per stream. The two scopes share those sealed banks for a paired comparison, so they comprise 1,920 unique final worlds, not 19,200 independent worlds. The 19,200 descendant predictions reuse those worlds across arms/scopes.

## Prospective results

| Scope | Control | Final RMSE | Faster-turning RMSE | Worst old-score drop | Training seconds | Extra parameters |
|---|---|---:|---:|---:|---:|---:|
| Shared path (478,483) | observed_only | 0.21268 | 0.41921 | 5.469 pp | 185.22 | 0 |
| Shared path (478,483) | detached | 0.18209 | 0.38704 | 4.688 pp | 183.20 | 1,070,354 |
| Shared path (478,483) | integrated | 0.18209 | 0.38704 | 4.688 pp | 184.67 | 0 |
| Shared path (478,483) | extra_capacity | 0.21268 | 0.41921 | 5.469 pp | 185.22 | 1,070,354 |
| Shared path (478,483) | oracle_exposure | 0.18209 | 0.38704 | 4.688 pp | 191.98 | 0 |
| Readout (514) | observed_only | 0.20739 | 0.40868 | 0.000 pp | 88.23 | 0 |
| Readout (514) | detached | 0.19368 | 0.39754 | 0.000 pp | 88.40 | 1,070,354 |
| Readout (514) | integrated | 0.19368 | 0.39754 | 0.000 pp | 89.30 | 0 |
| Readout (514) | extra_capacity | 0.20739 | 0.40868 | 0.000 pp | 89.23 | 1,070,354 |
| Readout (514) | oracle_exposure | 0.19368 | 0.39754 | 0.000 pp | 89.06 | 0 |

RMSE is in the simulator's coordinate units. The worst retained drop covers all five streams and 61 groups: 40 existing output capabilities plus 21 neural-only typed groups. It is an empirical gate, not a population-wide retention guarantee.

**A06-TRANSFER-001:** conditional teaching changes MSE by 26.70% relative to observed-only learning. The paired absolute MSE-gain interval is [0.007513, 0.017607]. The predefined conditional component gate **fails**. Gate details: `{'all_streams_improve': True, 'relative_gain_at_least_10_percent': True, 'paired_stream_interval_positive': True, 'retention': False, 'matched_detached_prediction': True, 'conditional_component_pass': False}`.

**A06-TRANSFER-002:** conditional teaching changes MSE by 12.79% relative to observed-only learning. The paired absolute MSE-gain interval is [0.002635, 0.008133]. The predefined conditional component gate **passes**. Gate details: `{'all_streams_improve': True, 'relative_gain_at_least_10_percent': True, 'paired_stream_interval_positive': True, 'retention': True, 'matched_detached_prediction': True, 'conditional_component_pass': True}`.

Detached and integrated arms must produce exactly the same trained state; their distinction is which owner is used by old routes and whether an additional full owner is retained. The extra-capacity observed-only control likewise matches observed-only training. All controls have the same optimizer starts, target/replay/anchor schedules and event shapes. Wall time and parameter/memory cost are measured separately; I do not turn matching update counts into a claim of exact hardware-cost equality.

## Acquired knowledge: a real intervention

I restored the original circle model before and after its independent sensor correction. Their neural parameters are bitwise identical, while their acquired coefficients differ. Across 1,024 transformed teaching worlds, the largest coordinate change was **1.22e-15**. The teaching transformation erased the intervention. On 257 raw-coordinate queries, the maximum prediction change was **0.47901**; MSE changed from **0.070327** to **0.000100**. The original learned correction is useful; this teaching representation does not identify its contribution.

The new executable preflight rejects acquisition-attribution claims when an otherwise matched K intervention leaves the teaching channel unchanged. A detectable change is only necessary: future work must also show beneficial transfer, matched exposure/cost and retained competence.

## Persistence and integration

The predeclared selection rule chose stream 307 from `research-continuation/18_instance_transfer/seed-307/corrected_teacher` for an **experimental persistent descendant**, `d774aa5175488634043f75711ffee4f943313aa92a396351dcdcb57690ba8285`. I replayed 16 original observations and one correction, retained cumulative operation costs, rejected both stale state/library bindings, re-proved the finite programs under the new owner, and restored the complete solver exactly. The old operational solver and all original libraries/checkpoints remain unchanged. No statistical answer certificate was transferred, and this is not learned latent migration.

## Costs and evidence

At this report snapshot, bounded jobs charged **2708.23 seconds**, including failures, pilots, source freezes and verification. Peak owned-job committed memory was **1,268,051,968 bytes** under a 2 GiB ceiling. The remaining cumulative allowance was **651.52 seconds**. Final release verification may have a later ledger snapshot. Event counts, per-arm timing, all immutable checkpoints and exact source/runtime hashes are recorded separately. No paid compute or remote publication was used.

See [full costs](costs.json), [failed candidates](failures.md), [architecture comparison](architecture-audit.md), [primary research](research-notes.md), [shared-path audit](A06-TRANSFER-001/audit-summary.json), [readout audit](A06-TRANSFER-002/audit-summary.json), and [knowledge-intervention audit](acquisition-intervention/result.json).

The acquisition-dependent follow-up is now complete in [its separate report](../18_instance_transfer/report.md). The original report cost snapshot above is preserved; [final release accounting](../18_instance_transfer/final-costs.json) additionally includes the three resumption checks. Retained shared-representation learning, hidden-state continuing interaction and independently improved eta remain open. I have not demonstrated M2–M4 or a self-improving general learner.
