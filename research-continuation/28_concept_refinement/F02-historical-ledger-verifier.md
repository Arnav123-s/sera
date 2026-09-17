# F02: historical literature-ledger verification

The full377-test suite passed. The subsequent CI integrity pass found a pre-existing verifier mismatch: `verify_continuation.py` treated the shared literature ledger as immutable at the Stage14 release, although later valid commits8ae0b3e and73dd66c had appended research. The file was clean at this session's starting HEAD and I did not modify it.

I added that exact maintenance path to the verifier's existing evolving-file list. The verifier still requires the Stage14 manifest hash and checks the original ledger bytes from the pinned historical commit71512a612961f3bd6d3e58a779e65dc8d2dec352. Frozen scientific results and source archives still require exact working-tree equality. This preserves both the old literature record and the newer valid additions.

The old verifier source, failed integrity receipt and full cost remain preserved. Resume verification from the failed historical check; the preceding release check is already complete. This changes no learner, checkpoint, scientific protocol or evaluation result.

The Stage15 applicability verifier also inventories the continuation verifier. Its maintenance-path list now admits this exact verifier path, while authenticating its original bytes from9104839400d5053152ca423be6da1efe355aa71a. The actual applicability protocol, model, thresholds, calibration evidence and gate checks remain unchanged. The second failed integrity receipt and original applicability-verifier source are retained.

The v3 verifier subsequently found only `pyproject.toml` different from its inventory. I verified that its exact expected bytes are already present in v3's authenticated `current-sources.zip`, and made that one metadata path use that frozen witness. A read-only inventory comparison found no changed scientific files in v3, instance-transfer or continuing releases. This permits current package/lint metadata to evolve while retaining exact v3 source checks. Its old verifier and failed receipt are also preserved.

Because v3's inventory includes its own verifier, the amended verifier also uses its exact original archived source as that entry's witness. The first self-inventory rejection is preserved. Both witnesses match the original manifest, whose SHA-256 is now explicitly pinned in the verifier. All scientific evidence still requires byte equality; this does not waive or rerun any applicability evaluation.
