# Current status

[Home](../README.md) · [Use SERA](START_HERE.md) · [Archive](RESEARCH_ARCHIVE.md)

## Preserved Stage 28 verified release

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

## Current human-text integration

Stage 29 completed the ordered WordNet/example, SQuAD sentence, AMI manual-transcript and OpenStax physics-prose curriculum. The selected semantic weights improved source-association accuracy over the starting weights in all four final domains. The selected sentence reader scored 82.6% on 1,000 unseen human questions; BM25 scored 83.1%. All controls and failed candidates remain in the [full report](../research-continuation/29_human_reading/report.md).

The integration audit passed: 165 predecessor tensors, 128 four-language probes, exact algebra and empirical forecasts were retained; checkpoint resumption and persistence passed. Live arXiv search retrieved and ranked sources while preserving the original question. It recorded zero unverified weight updates. [Use the interface](START_HERE.md#read-a-source-or-investigate-a-gap).

Full local regression reached its 120-second cap; the partial log is retained. Targeted reader/source tests passed. The latest commit's full Linux/Windows result is linked through the main verification badge; see the stage costs for exact local charges.

Current work follows the user's full-book and grounded-definition request. [Next finite protocol](../research-continuation/30_grounded_books/PROTOCOL.md). Numerical time is governed by the live ledger; source intake grants no additional resources.
