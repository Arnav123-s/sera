# Publication verification

Published source: `a93293d2be171d9ab40e24d50684ded37babe72d`. The [GitHub run](https://github.com/Arnav123-s/sera/actions/runs/35409326749) passed on both platforms. Both jobs restored the predecessor chain and current owner from the published artifacts, executed the practical batches, and ran regression and release checks from clean checkouts.

| Platform | Passed tests | Explicit optional-fixture skips |
|---|---:|---:|
| Linux | 547 | 46 |
| Windows | 548 | 45 |

The local workspace passed **593 tests with no skips**, lint, all seven release verifiers and the base CLI check. The current-owner batch completed **33 requests**. The documented direct-script invocation was verified by both remote jobs; its earlier import-path repair is retained in the Stage 46 publication records.

[Release](https://github.com/Arnav123-s/sera/releases/tag/research-2026-09-18-structural) · [Exact verification receipt](github-run-52.json) · [Verbatim runner logs](github-run-52-logs.zip) · [Local checks](local-checks.json).

The archive contains **383 records**, **55,595,874 compressed bytes**. GitHub reports the matching SHA-256 `ed7d0cd6c3e1410861e523524de647ec73c93a434ae63206d9c7c4057ae248c8`. Its source/evidence manifest separately pins 27 files. [Archive identity](release.json) · [Manifest](../release-manifest.json).

This stage records **2,491.495 supervised local seconds**, including failed attempts and archive verification. Together the two stages record **10,128.584 seconds (168.81 minutes)**. The local policy remains one CPU thread, a 2 GiB worker-tree cap and no paid services. GitHub job times are preserved in the verification receipt, and upload time is recorded separately in the release receipt. [Full local costs](costs.json).

The scientific source, final cohorts, original failures and acquired owner remain sealed. Publication receipts and navigation are added afterward without rewriting those records.
