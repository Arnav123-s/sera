# Clean-checkout retention fixture repair

GitHub run 41 reached the tests on both platforms. Linux reported 436 passes, 46 optional-fixture skips and three setup errors; Windows reported 437 passes, 45 skips and the same three errors. The original logs and failed receipt are preserved here.

The new owner computes its existing 128 language-retention probes. The local workspace had the historical English development file and three multilingual development files; the clean-checkout restorers did not install them. The failure was a missing runs/SC-data-002/dev.jsonl, before a model update or test assertion.

The repair restores the exact four files and their existing multilingual license from the published Stage 26/27 data archives. Each archive and member is fingerprinted. Existing divergent files are preserved. No new data, trained weights, source contracts, reward formula or completed evaluation was changed.

An empty-directory regression checks fresh restoration, exact bytes, idempotence and refusal to overwrite divergent local work. The original ten quest tests are rerun alongside it. The workflow and fresh-clone setup instructions now include this prerequisite explicitly.
