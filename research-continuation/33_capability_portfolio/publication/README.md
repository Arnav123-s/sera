# Release follow-up

The sealed research report and checkpoints describe the completed experiment. Current live placement is recorded in [live-state.json](live-state.json): the assessor was copied into **runs/sera-quests-live/assessor**, with its original key and ledger verified locally. Future live updates now leave the completed audit directory unchanged. The copy appends a learner revision and preserves weights, optimizer, evidence and qualification exactly.

This placement supersedes the assessor path printed in the sealed NEXT_ACTION.md. The commands in that file still use the same live owner. The earlier audit, its private local key and all revisions remain preserved. Fresh-clone learning already creates an assessor inside its own live store.

Verified implementation **9ca50a7aab26fa4b75e7adbcd22de38026344343** passed [GitHub run 42](https://github.com/Arnav123-s/sera/actions/runs/35317484025): 440 tests on Linux and 441 on Windows, with 46/45 existing optional-fixture skips. Every archive and release check passed. [Successful receipt](github-run-42.json) and its compressed original logs preserve the exact result.

[Run 41](github-run-41.json) and [its diagnosis](CI_REPAIR.md) preserve the missing-fixture failure and repair. The model and frozen research results are unchanged. Locally, the 485-test full regression and the subsequent 11-check targeted run together cover 486 distinct checks, with ten overlapping checks and zero local skips.

[Continuation](continuation.json) records the exact live revision, remaining numerical allowance and unfinished work. [Costs](costs.json) records 829.70 seconds of charged local numerical work in this cycle, leaving 1166.07 seconds from the previously approved allowance. File-only assessor relocation is separately timed in [its record](live-state.json). Publication receipts and post-seal resource costs are recorded here without changing the sealed experiment.
