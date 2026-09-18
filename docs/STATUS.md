# Current status

[Home](../README.md) · [Use SERA](START_HERE.md) · [Archive](RESEARCH_ARCHIVE.md)

## Verified release

| Item | Evidence |
|---|---|
| Published release | [`85a8bdd`](https://github.com/Arnav123-s/sera/commit/85a8bdde6f73cf82415e52c2afffae6eaa38f9ab) · [Receipt](../research-continuation/28_concept_refinement/publication.json) |
| Verified implementation | [`3cdc379`](https://github.com/Arnav123-s/sera/commit/3cdc37948f39745585cf7112fe8290f361784035) |
| Completed main verification | [Successful Linux and Windows run](https://github.com/Arnav123-s/sera/actions/runs/35288526321) |
| Full local regression | 436 passed, no skips |
| Clean Linux checkout | 390 passed, 46 existing optional-fixture skips |
| Clean Windows checkout | 391 passed, 45 existing optional-fixture skips |
| Scientific archive | 188 artifacts and 270 archive members verified |

The receipt commit adds records to the tested implementation. [Check the newest main run here](https://github.com/Arnav123-s/sera/actions/workflows/ci.yml?query=branch%3Amain).

## Earlier failures in Actions

Earlier clean-checkout runs exposed missing ignored fixtures, an interpreter-version mismatch and floating-point replay differences. These were diagnosed and repaired before the successful main run. [Failure explanations and retained logs](../research-continuation/28_concept_refinement/publication-hardening/README.md) show what changed. Historical runs remain preserved; current status belongs to the verified commit.

## Measured C01/C02 outcomes

- 95.90% lower forecast MSE than instantaneous identification on the defined clean delayed-system cohort.
- 69.69% lower delayed-system error for adapted existing R1 memory than its reset-memory control.
- 256 teaching trajectories and 216 held-out evaluation views.
- 162 predecessor tensors unchanged, with polynomial checks and four-language retention probes.

[Full report](../research-continuation/28_concept_refinement/report.md) · [Architecture audit](../research-continuation/28_concept_refinement/architecture-audit.md) · [Checklist](../research-continuation/28_concept_refinement/checklist.md).

## Current work

Repository navigation and real-text learning are being extended. The new curriculum uses attributed human-authored questions and source passages, with article-disjoint development and evaluation. Results are admitted only after comparison, retention and integration checks.

The last recorded balance is 512.037 numerical seconds. The live `runs/v3-batch-001/budget.json` governs new reservations; this page grants no additional compute.
