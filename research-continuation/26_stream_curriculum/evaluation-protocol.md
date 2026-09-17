# SC-002 evaluation and useful integration gate

I register these details before inspecting the SC-002 development outcomes or
opening official test rows for assessment. The first fit may be running or have
written its automatic 128-row development probe; those results have not been read.
This supplements the pre-fit full-training protocol without changing training.

Evaluate the four fixed 1,800-update checkpoints on all 2,033 development rows.
Record source-annotation coverage, intent accuracy, exact intent-plus-span frames,
micro span F1, per-domain metrics and source-text duplication. The control is the
most frequent full-training intent with all tokens outside slots. Preserve full
records, including token tags and declared softmax scores.

Select the checkpoint with greatest development exact-frame rate; break ties by
intent accuracy, then lexical checkpoint path. For use as a local annotation aid,
require development intent accuracy >=0.70, span F1 >=0.50, 145 exactly unchanged
predecessor tensors and 96 exactly retained finite-language outputs. This gate
qualifies request classification/entity annotation, with predicted fields exposed.
Executing an external action or admitting a factual observation retains its own
existing authorization/evidence requirements. The numerical applicability gate
is not transferred to request classification.

Before official test access, freeze the four checkpoint paths/hashes, their
development selection and evaluator source. Evaluate all four once at the fixed
weights, with raw results retained. Report the selected checkpoint's final result
without reselection or retuning. The official test is 2,974 source rows; inspect
parser eligibility automatically and retain any exclusions with source IDs.
Also report novel-text performance relative to the full training source, alongside
the complete official result. This subset is a declared second measurement, not
an edited headline score. Source requests describe desired tasks, not observed
events or authorization to execute them.

The separate sequential curriculum uses official development labels for learning
diagnostics only. It will never teach from development/test rows. Its final model
identities and protocol must be frozen before its one final evaluation. Once the
official test is exposed for SC-002, a later sequential test result on it is a
shared benchmark measurement rather than an independent new confirmation cohort.
