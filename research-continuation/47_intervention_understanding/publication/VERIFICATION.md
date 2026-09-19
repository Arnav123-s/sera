# Publication verification

The actual local predecessor was `54e64945f3de4280387cfba5fa296225a568ccd3`. The source and final protocol for this successor are pinned by [the release manifest](../release-manifest.json).

Local verification completed:

- All **613 regression tests passed**, with no skips or failures.
- Whole-tree lint, all seven existing release verifiers and the CLI help check passed.
- The current interface executed all **39 example requests** through the qualified successor.
- The same batch restored and ran under the CPU profile used by GitHub: AVX2, Haswell OpenBLAS and disabled X86_V4 NumPy features.
- Independent scientific replay checked **144 investigations**, source outcomes, actual control histories, qualification and all trained R2 comparisons.
- Exact restoration and both interrupted-acquisition boundaries passed on the actual continuing owner.
- The sealed archive passed source, member, size and SHA-256 verification.

[Command results](local-checks.json) · [JUnit receipt](local-junit.xml) · [Full costs and failed attempts](costs.json) · [Owner audit](../audit.json).

The public release and clean-checkout platform receipts are added after the corresponding GitHub operations complete. Scientific results and the sealed final evaluation remain unchanged during publication.
