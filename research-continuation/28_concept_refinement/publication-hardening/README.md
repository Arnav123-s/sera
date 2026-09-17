# Clean-checkout verification

The scientific release is commit `2c51d2338d6be5c6e0f560fb4cf5b261fd0429b2`. Its first GitHub run exposed five older tests that assumed ignored local checkpoints or dataset files already existed. I preserved that failure in `first-ci.log` and repaired artifact restoration without changing the learned models, completed evaluations or test assertions.

I traced the five affected tests and all 18 concept-refinement tests. Their dependency closure contains 17 files (29,925,203 uncompressed bytes). Nine files already exist in four published archives; eight remaining witnesses are in `additional-fixtures.zip` (4,639,926 bytes). `fixtures.json` identifies every source archive, member, byte count and SHA-256. The existing dataset archive retains its source and license records. No new training data was collected for this repair.

The restoration helper checks archive and member identities, accepts an existing identical file, creates absent files exclusively, and stops before overwriting divergent local work. Eight additional contract tests cover exact/idempotent restoration, preservation of changed local files, path escapes and corrupted identities. All eight passed. A separate detached checkout of the scientific commit restored the prerequisites and passed all 23 affected integration tests; its receipts and logs are retained here. The full local research suite had already passed 377 tests before these packaging changes.

For a fresh checkout, install the repository's documented dependencies, then run:

```text
python scripts/restore_test_artifacts.py
python -m pytest
python scripts/verify_concept_release.py
```

The GitHub workflow performs the same restoration before testing on Linux and Windows. Historical tests that explicitly require other optional local archives retain their existing skip conditions. Restoration does not restart completed experiments. The completed final bank is copied because an integration fixture checks that it exists; these tests consume development observations rather than scoring or tuning final outcomes.

`local-verification.json` accounts for the additional local jobs separately from the already sealed scientific cost record. Each job's original state and log are in `jobs/`. The scientific release manifest remains unchanged. Remote workflow outcomes and the final allowance balance are recorded separately in the publication receipt after verification completes.

## Completed platform qualification

The successor readers distinguish executable semantics and evidence from uncalibrated neural confidence and the last bits of a numerical energy sum. [F03](F03-platform-replay.md), [F04](F04-execution-profile.md), [F05](F05-constraint-proposal-replay.md) and [F06](F06-archived-result-and-request-replay.md) retain the measured failures, reviewed changes and source migrations. Original records, weights, archived branches and evaluation results remain unchanged. The complete diagnostic includes stale branches and compares the actual owner byte identity before interpreting saved requests.

`qualified-local-verification.json` is the final local receipt: 436 tests passed with no skips, the saved forecast ran successfully, all protected artifact hashes matched, and all owned numerical jobs settled. `local-verification.json` and `final-local-verification.json` remain historical intermediate snapshots; their earlier balances are not the current allowance. The final balance is 512.0372116000653 seconds (8.533953526667755 minutes), with no new grant.

The [publication receipt](../publication.json) identifies the verified code commit and both platform runs. Remote workers retain their existing optional local-artifact skips; the full local suite exercises those cases. `remote-verification.zip` preserves decoded logs and complete run/job timestamps, with line endings normalized to LF. Its manifest records every archived byte identity. Remote elapsed job time is reported separately from local supervised numerical charges.
