# SC-005: fixed additional exposure from exact saved optimizers

I register this cycle after the complete SC-002/003 development measurement and
before opening official test rows for assessment. SC-003 seed 2642 qualified under
the original gate: 74.9631% intent accuracy, 52.0701% span F1 and 40.5804% exact
frames on 2,033 development requests. The fixed extra-exposure question is whether
more optimization of the acquired interface improves those measured abilities.

Continue all four SC-003 checkpoints, seeds 2641/2642 crossed with shared/pooled,
from step 1,800 to step 5,400. Load their exact Adam state, parameters, torch RNG,
buffered source cursor and training-history IDs. Do not repeat the first 1,800
updates. No data, label, architecture, optimizer, rate or loss change is allowed.
This adds 3,600 updates and 115,200 presentations per continuation, for 172,800
lifetime presentations per resulting model. Unique source examples remain 11,514.

Use the unchanged SC-003 trainer and its original source/protocol validation.
The new per-run freeze records the parent checkpoint hash and target step; this
addendum supplies the authority and interpretation for that extended target.
Save every 200 updates, preserving all earlier checkpoints and costs. Each shared
continuation receives at most 430 supervised seconds with a 395-second internal
checkpoint deadline; each pooled continuation receives at most 100/75 seconds.
If a job stops at its internal deadline, resume only its unfinished optimizer.

Evaluate the four new checkpoints on the same development partition. Reuse the
already saved eight earlier development prediction records rather than re-running
those models. Select the annotation aid among these twelve fixed checkpoint
identities using the unchanged development exact-frame/tie-break rule and the
unchanged 0.70 intent / 0.50 span-F1 gate. This expands the development candidate
set before final access. Freeze the complete set and selection before the one
official test evaluation; include the four SC-004 lifetimes as supplementary
models without using them to reselect the annotation aid.

Additional exposure is a continuation of existing learners, not four new random
seeds, a new independently sampled dataset, or evidence of learned meta-updating.
This is a fixed finite exposure comparison; no outcome-dependent step search is
registered. Keep an unsuccessful continuation's earlier qualified checkpoint.
