# Next executable work and exact continuation boundary

SC-001 through SC-005 are finite completed experiments when release-manifest.json
is present. Their training, development selection, final assessment and qualified
annotation integration are preserved. Do not restart their completed optimizers or
automatically assess a newly tuned model on the now-exposed MASSIVE test partition.

For immediate use, process a real request file with the command in README.md.
For research, first read the live `runs/v3-batch-001/budget.json` and active lock.
The balance at sealing is historical, not a new allowance. Each terminal model's
exact resumable path/hash is in `checkpoint-archives.json`; every intermediate
remains locally indexed by `local-state-manifest.json`.

The next architectural dependency is a tested connection from learned request
entities to persistent, reversible local tasks. Freeze that task stream before
teaching. A useful initial task is maintaining a local structured task inbox:
add/list/change entries, ask for a missing field, preserve interrupted work, then
correct the interpretation from verified user labels. Keep the old numerical
motion route's evidence/applicability gate distinct. Public dataset requests are
examples, not authorization for external actions.

Use the currently admitted shared checkpoint as one starting point. Compare
verified correction plus 256-record replay with correction alone and an unchanged
checkpoint under equal new-example and total-update budgets. Match replay storage
and count source retrieval, verification, retries and failed attempts. Freeze new
domain-disjoint task records and expected state transitions before training;
keep their final labels outside any learner-accessible source. Test consequence
correctness separately from intent/entity accuracy, including negation, ambiguous
roles, missing information and a corrected stale plan.

The existing trainers already provide exact continuation. A future mixed-data
continuation must use `experiments.stream_curriculum.mixed --resume <checkpoint>`
with a fresh output directory and an explicitly registered larger target step;
the current 5,400-update cycle is finished. A new correction curriculum should
preserve its source identity separately rather than overwrite that trainer's
frozen files. Append a new protocol and experiment ID.

Evaluate acquisition, already-taught retention and reacquisition on the same
persistent lifetime, plus fresh held-out tasks after freezing all parameters.
SERA's learned weights and any acquired task entries are evidence of learning;
the supplied correction scheduler/replay algorithm must remain attributed to
engineering. Test a learned controller later through the cookbook's controlled
old/new knowledge × old/new learning-procedure comparison.
