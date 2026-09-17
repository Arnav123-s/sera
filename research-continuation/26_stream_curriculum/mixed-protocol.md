# SC-003: training-order intervention

I freeze this intervention after inspecting SC-002 seed 2641's automatic early-row
development probe (12/128 intent decisions correct), before any official final
test access. Source-order interference is a hypothesis to test. I preserve all
SC-002 fits and their fixed controls, without relabelling their results.

Use precisely the same 11,514 training rows and supplied schema, source parent,
interface, 1,800 updates, 32-example batches, Adam 0.003, gradient clipping and
seeds 2641/2642 as SC-002. Cross both shared and pooled routes. The intervention
is an independently saved full permutation of the training row offsets, seeded
by the model's declared seed. Write a source-identified permuted JSONL stream,
then use the same bounded 128-row shuffle reader during optimization. Repeated
epochs use that same fixed permutation. Retain index/ordering provenance and
count storage and preparation costs. This mixes known teaching cases; it does
not add labels or alter the evaluation data.

Save model/optimizer/RNG/buffered-cursor checkpoints every 200 updates, rather
than every 20 in SC-002. This changes persistence granularity and I/O cost, not
the optimizer's sequence or number of updates. Preserve all earlier checkpoints.
Measure actual wall time separately; do not attribute this I/O saving to learning.

Run all four fixed descendants, report them alongside the SC-002 controls, and
use the already registered development gate and tie-break for annotation-aid
selection over the complete eight-checkpoint cohort. Freeze those eight identities
and the development selection before a single official final evaluation.
The amendment expands the candidate cohort before final access; it does not
alter the final labels, metric or acceptance threshold.

The same two seeds are crossed across schedules and routes, so there are two
optimization seed replicates, not eight independent populations. Inspect per-domain
development results and assess the order hypothesis through paired differences.
Any later sequential replay study receives its own frozen protocol and fresh seeds.
