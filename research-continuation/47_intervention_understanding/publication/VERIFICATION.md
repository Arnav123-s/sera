# Publication verification

The actual local predecessor was `54e64945f3de4280387cfba5fa296225a568ccd3`. This is the uninterrupted SHA printed with spacing in the frozen report. The source and final protocol for this successor are pinned by [the release manifest](../release-manifest.json).

Local verification completed:

- All **613 regression tests passed**, with no skips or failures.
- Whole-tree lint, all seven existing release verifiers and the CLI help check passed.
- The current interface executed all **39 example requests** through the qualified successor.
- The same batch restored and ran under the CPU profile used by GitHub: AVX2, Haswell OpenBLAS and disabled X86_V4 NumPy features.
- Independent scientific replay checked **144 investigations**, source outcomes, actual control histories, qualification and all trained R2 comparisons.
- Exact restoration and both interrupted-acquisition boundaries passed on the actual continuing owner.
- The sealed archive passed source, member, size and SHA-256 verification.

[Command results](local-checks.json) · [JUnit receipt](local-junit.xml) · [Full costs and failed attempts](costs.json) · [Owner audit](../audit.json).

The [published archive](https://github.com/Arnav123-s/sera/releases/tag/research-2026-09-18-interventions) contains 1,308 members and is 20,460,076 bytes. Its SHA-256 is `150ea830fa57f8ee3cf44550c6943a92b5d28d74a556d96d7143e5002e02de4f`, independently matched against GitHub's asset digest. [Publication receipt](release.json).

The scientific implementation is [`e9b4cf2`](https://github.com/Arnav123-s/sera/commit/e9b4cf2760b93afafd23d93005b104c715a981e4). The additive restoration repair is [`94fb2d9`](https://github.com/Arnav123-s/sera/commit/94fb2d9e98a4cc92ea68c9a16617e8a4124be97f). [GitHub run 54](https://github.com/Arnav123-s/sera/actions/runs/35414873944) passed from clean checkouts on both platforms:

| Platform | Passed | Established optional-fixture skips | Result |
| --- | ---: | ---: | --- |
| Linux | 579 | 46 | All workflow steps passed |
| Windows | 580 | 45 | All workflow steps passed |

Each platform restored the release chain, executed the 39-task current interface, ran the regression suite, and passed lint and all seven historical release verifiers. The 625 collected checks comprise the original 613 plus 12 new portability tests. Locally, the complete 613-test suite and all 12 targeted additions passed without skips; the targeted additions were rerun after a numerical assertion was corrected. These are 625 distinct local checks, not one new full-suite run. [Exact matrix receipt](github-run-54.json) · [Original complete logs](github-run-54-logs.zip).

The first publication run's Linux failure remains preserved. Native diagnostics traced it to one independent optimizer's termination description and a maximum refit probability difference of `1.0978373765624383e-9`. The separately pinned repair preserves all scientific decisions and learned weights. [Diagnosis, comparison contract and negative tests](PORTABILITY.md).

Local supervised work across 25 jobs, including six failed attempts, consumed **1,507.7326902 seconds (25.13 minutes)**. Peak process-tree committed memory was **1,562,042,368 bytes**, within the 2 GiB cap. Jobs used one CPU thread and no paid service. The archive upload consumed another **20.2431143 seconds**; hosted CI step times are separately retained in both run receipts and original logs. Source reading and orchestration are outside the supervised numerical timer. [Full costs and resource ledger](costs.json).

The saved owner remains `62ea82cff82b88d1eb3848f7246494979c63894d869db603007e3d06111063ec` in `runs/sera-intervention-live`. Subsequent publication receipts and navigation updates change no scientific source, saved parameter, admission or final evaluation. The original strict implementation and all predecessor releases remain preserved.

[Run the saved learner](../README.md) · [Follow one complete investigation](WORKED_EXAMPLE.md) · [Next prospective research](../report.md#next-executable-research).
