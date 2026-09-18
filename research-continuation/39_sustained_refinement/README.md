# Sustained retained practice

This continuation compares learned optimizer/strand choice with balanced Adam while rehearsing earlier abilities. Both seeds run 2,048 decisions per arm. [Protocol](PROTOCOL.md) · [Selection](selection.json) · [Final results](final.json) · [Independent audit](audit.json) · [Report](report.md).

The source is `experiments/sustained_refinement.py`. Owned jobs use `scripts/run_refinement_bounded.py`. To resume genuinely incomplete training, run the module's `train` action under that wrapper with a fresh job output directory. Completed selections/finals reject accidental reruns. Source hashes, optimizer state, RNG and all accepted/rejected events remain attached to the saved continuation.

Use the [batch task interface](../../docs/START_HERE.md#run-several-tasks-through-the-continuing-learner) to load the latest independently admitted owner. The original Stage 38 owner remains at `runs/sera-growth-live`.

`python -m scripts.refinement_artifacts --stage 39` verifies the publication archive. Repeated tensors are stored once; every original local checkpoint is retained, and compact restoration is checked for exact tensors, optimizer values, RNG and events. The source provenance and parent corpus exposure are explicit in the protocol and archived manifest.
