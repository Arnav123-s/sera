# Connected SERA release verification

I verified the executable implementation associated with source SHA-256 `59c0835fba356932d945dc67c04b781aa5549702ee40389b09b9e03c73016594` (implementation commit `71707ee`). Report and verification scripts are separate from this implementation hash.

| Check | Observed result |
|---|---|
| Python tests | 36 passed in 22.05 seconds on the local CPU environment |
| Ruff | All selected checks pass for source, tests and scripts |
| Full study | Seeds 0, 1, 2 complete; 276 actual intervention outcomes; 15 persistent proposals |
| Independent artifact reproduction | All 15 recorded paired decisions reproduced; final symbolic behavior and accepted program execution agree after reload |
| Evidence separation | 1,920 nonoverlapping meta-support identifiers; 48 unique meta-query datasets |
| Published evidence integrity | Eight selected artifacts verify against their manifests; 154 unique source concepts retained |
| Reference low-rank state | 35,840 core bytes; full-density spectral projection agrees with factor update; differentiation and probability contracts pass |
| Wheel build | `sera_learning-0.2.0-py3-none-any.whl`, SHA-256 `adf36e373db6c38227c04ddfd89b8d154c0f19568a04c3df98078877ad41b858` |
| Isolated wheel installation | Imported from a fresh target directory; restored v5 and executed its accepted reset-world skill successfully |
| Continued-learning copy | Prepared v5 with 608 actual support trajectories across four known worlds; no model/skill changes |
| Real CLI learning | Separate copy chose replay for a new reset world, consumed round 5, rejected the candidate for gain/retention, preserved current state and retained 640 actual support records |
| Solver archive | Local trained-solver ZIP includes the current solver, parent versions, manifests, journals and prepared experience |
| Clean checkout | 36 tests pass in 28.13 seconds; imports resolve to the clean checkout rather than the development tree |
| Clean report rebuild | Rebuilt every connected table, plot and evidence manifest from committed JSON; no report bytes changed |
| Public artifact check | Downloaded the three primary JSON evidence files from the published commit; all SHA-256 hashes match local evidence |
| GitHub Ubuntu CI | Passed tests, lint, evidence verification and CLI check |
| GitHub Windows CI | Passed tests, lint, evidence verification and CLI check |

The CLI check used four update steps and 256 paired samples per world on a disposable copy. It is a runtime verification, excluded from the formal learning tables. The prepared solver retains seed 0 by a fixed example convention, not by choosing a checkpoint within a seed using its final test score.

Both jobs completed successfully in [GitHub run 34800392826](https://github.com/Arnav123-s/sera/actions/runs/34800392826) for published commit `7dd6b23cd651209faa4b2cfcc8e1eb447dcaa089`. The original learning run and its source hash are unchanged by subsequent release documentation.
