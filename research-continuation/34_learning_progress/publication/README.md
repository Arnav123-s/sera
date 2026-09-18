# Publication and resource accounting

The sealed research is unchanged. [Latest costs](costs.json) include archive creation and verification; the stage's original cost snapshot remains preserved. Every supervised job retains its state and log under `checks/`. The live local resource ledger is authoritative; published balances are snapshots.

Local verification covers 493 full-suite tests and three subsequent discovery tests, with no skips. Learning artifacts passed 56 source-file and 83 archive-member checks. Discovery artifacts passed 25 source-file checks, nine exact checkpoint reconstructions and five independent algebra certificates.

Implementation `5195a2ea75eb085f60e13bae98f9acaa0e681aa3` passed [GitHub run 45](https://github.com/Arnav123-s/sera/actions/runs/35339284292): **455 tests on Linux and 456 on Windows**, with 46/45 established optional-fixture skips. All release, artifact and mathematical verification steps passed. [Receipt and original logs](../../37_constraint_acquisition/publication/github-run-45.json) retain the exact commit, jobs, steps and counts. The receipt commit only adds publication records and navigation.
