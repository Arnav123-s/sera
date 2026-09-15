# Learned applicability research

[Results and limitations](report.md) · [Architecture audit](architecture-audit.md) · [Research basis](literature.md) · [Frozen protocol](protocol.json)

GG-GUARD-001 is complete and independently audited. The candidate improves the declared score over fixed guards but **fails the conditional-risk gate**. It is preserved for further research and is not promoted into SERA's operational shared owner.

| Artifact | Purpose |
|---|---|
| `guard-model.json` | All learned weights, normalization, probability calibration and teaching provenance |
| `records.json.gz` | Complete 1,344-world raw cohort: observations, outcome witnesses, hidden assessment values, features, posteriors and final decisions |
| `summary.json`, `thresholds.json`, `before-final.json` | Per-family metrics, risk/coverage curves, paired uncertainty and operating point frozen before final observations |
| `frozen-sources.zip`, `protocol.json` | All 53 original source files and exact configuration |
| `replay.json`, `frozen-workspace-replay.json` | Exact retraining, decision and metric replay; the second uses an isolated frozen source workspace |
| `independent.json`, `audit-repair.json` | Independent mathematics and the declared checker repairs |
| `failures/` | Both failed audit logs/states and the intermediate checker source; original checker is in the frozen ZIP |
| `predecessor-inventory.json.gz`, `preservation.json` | Exact byte preservation of 16,556 older local model/result files |
| `supervision/`, `timing*.json` | Bounded runtime, actual time charged, process-tree committed memory and failed-attempt costs |
| `current-tests.xml`, `release-manifest.json` | Release test evidence and artifact checksums |

The package contains its complete compact cohort; older large cohorts remain in their original local directories. No checkpoint or failed result was deleted. The previous release inventory stays immutable: evolving navigation files are checked against their preserved bytes at `71512a6`.

Use [the reproduction commands](report.md#reproduce). Run the original decoder through `scripts/replay_guard_release.py`; the live independent auditor has the explicitly recorded repair and therefore differs from the original frozen auditor.
