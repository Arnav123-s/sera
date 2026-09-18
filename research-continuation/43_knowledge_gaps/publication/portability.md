# Published checkpoint restoration

Run 48 at commit 3724e35 failed on both GitHub platforms because the inherited Stage 33 quest snapshot retained an absolute Windows assessor directory. The first local fresh-tree smoke test did not catch this: its subprocess could still access that original directory. Both failure logs are preserved here.

The additive `scripts.portable_gap_use` entry point restores the same model through an explicit read-only historical receipt adapter. It checks the already-published Stage 33 archive's fixed SHA-256, matches complete receipts, and independently regenerates and grades their original cases. It never opens the embedded host path, publishes a private issuer key, issues a new receipt, or awards fresh assessment authority. Live HMAC verification remains unchanged in the preserved research source.

This separates public verification of historical published evidence from permission to issue new assessments. The old owner, optimizer, scientific source hashes, scores, and sealed tests remain unchanged. The portable entry point replaces the original entry point in cross-platform CI. Focused regression checks reject altered receipts, corrupted predictions, and attempts to create new credit through the read-only adapter.
