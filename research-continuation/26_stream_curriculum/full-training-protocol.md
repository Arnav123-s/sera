# SC-002: full English request acquisition, pending resource extension

I froze this next acquisition batch after SC-001's development results and before
running any SC-002 fit or accessing the official test for model assessment.
This work is prepared; numerical execution awaits an additional approved allowance.

Use the exact original MASSIVE English train partition, all 11,514 eligible rows,
and the existing 60-intent/111-BIO-tag schema. Training source hash:
`0adff67b32be7d384e6405f9c709432c02b1b0f81fe91c04d910dc10e52d1e56`.
All 2,033 development rows reconstructed exactly. The 2,974 test rows remain
reserved for the later frozen evaluation; their source-role quarantine is intact.

First acquisition run: fresh seed 2641, shared route, batch 32, 1,800 updates,
Adam 0.003, clipping 1.0, original sum of intent/slot cross entropy. The existing
128-row buffer produces a bounded shuffled source stream, not a global IID shuffle.
Read all teaching rows as the stream advances and record actual unique coverage.
57,600 presentations are planned; save exact stream/optimizer state every 20 steps.
The saved SC-001 pilot stays independently preserved and usable.

Use seed 2642 for a second independent optimization replicate. Run pooled controls
at both seeds with the same prepared data, update count, embedding/heads, batching,
stream seeds and learning rate. The pooled control omits recurrent computation,
so actual wall and computation differ and must be reported separately.
Training remains on the new interface; the complete prior owner is inherited.
No pretrained language model or its generated labels enters this batch.

The first 128 development rows already inspected in SC-001 remain labelled
development evidence. The current trainer records this same probe for continuity.
Before any official test assessment, implement and validate a separate evaluator
for all 2,033 development rows and the official test. Lock all candidate checkpoint
identities before reading test labels. Use all four fixed final checkpoints rather
than selecting a winner on test. Report complete per-row results, class/domain
breakdowns, exact frames, intent accuracy, span F1, and split text duplication.
If source duplicates occur, include the full official score and a separately
identified novel-text subset. Do not remove errors after inspecting predictions.

Report prior-language preservation, model/dataset/optimizer bytes, source
identities, unique examples, presentations and all failed costs. Compare the
trained controls; do not rank SERA against standard LLMs absent a matched study.

Pilot measured ~0.06 seconds per update at batch 32, ~5 seconds setup and ~4 seconds
other worker/import overhead; larger source lengths can change that rate. The
first full fit receives a 240-second worker cap and 200-second internal checkpoint
deadline. Resume partial fits, do not restart prefixes. Four fits, checks, evaluator
engineering tests and integration are provisionally allocated 1,800 supervised
seconds of a requested 3,600-second extension. The remaining half is reserved for
diagnosis and a separately frozen sequential-learning/replay experiment. The
continuation must validate this estimate rather than consume the whole allowance.

The next exact command after the extension is granted is:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/run_stream_curriculum_bounded.py --seconds 240 --output runs/SC-full-shared-2641-worker --module experiments.stream_curriculum.study -- train --output runs/SC-full-shared-2641 --seed 2641 --kind shared --limit 0 --steps 1800 --work-seconds 200
```

This acquisition study precedes a source-disjoint sequential curriculum and bounded
replay comparison. That later protocol must match both novel-example exposure and
total presentations, measure retention after each scenario block, and retain
interruptions and contradictory-evidence corrections. Counts from one repeated
request classification task are not relabelled as independent problem types.
