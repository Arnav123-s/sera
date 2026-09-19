# OLA-001 — connected language acquisition on the current owner

Prospective and finite. This document, `protocol.json`, the frozen corpus
manifest and the frozen evaluation sets are fixed before any weight is updated.
Development results may change the next protocol; they do not change this one.
The final split stays sealed until the schedule and every repair are frozen, and
is opened exactly once.

## The question

Stage 30 gave the owner a human-book route, but `BookR1.book_features` is
decorated `@torch.no_grad()`: the features are computed under the inherited
core and only a linear readout is trained. No earlier stage has let human
language change the shared substrate at all.

So the question is not "can a language model be trained here" — it is:

> Does routing learning from human text through the owner's own shared
> recurrent core, instead of through a readout on frozen features, produce a
> larger and more transferable language gain at equal exposure?

This is falsifiable in the direction that matters. If a probe on frozen
inherited features matches the connected arm, then the inherited core carries
nothing useful for language and the connection is the wrong idea; that result
will be reported as a rejection, not buried.

## The connection, and what it cannot touch

`SharedR1.shared_step` is the one function every recurrent route of this owner
passes through — the typed route, the sequence route, the world route, the
request route, the reading and book routes. `SharedR1.add_adapter()` inserts a
residual adapter there. The descendant uses that existing insertion point:
new language modules encode tokens, the owner's own recurrent memory carries the
context, and the adapter is the trainable path by which what is learned changes
what the shared core emits.

The adapter's output projection is zero-initialised, so at step 0 the descendant
reproduces the parent exactly and every later difference is attributable to
teaching.

Trainable: `lex_embedding`, `lex_projection`, `lex_norm`, `lex_head`,
`adapter.0.weight`, `adapter.2.weight`. Frozen: every one of the 1,236 other
inherited parameter tensors.

Protected by construction, because they never read the shared recurrent core:
the exact rational mathematics routes (`solve_solution`, `what_if`,
`apply_inquiry_rule`), the fitted physical field weights (`field_what_if`,
`field_trajectory`, `field_plan`) and the fitted intervention coefficients
(`intervention_what_if`, `intervention_explain`, `intervention_plan`). That is a
claim about the code; the run also measures it by replaying the declared 33-task
retained suite through the trained descendant and comparing case by case against
the baseline record.

## Arms

All three arms share the identical corpus, vocabulary, seed, batch order,
schedule, optimiser and step budget. They differ in exactly one thing.

| Arm | Trainable | Inherited core | Question it answers |
| --- | --- | --- | --- |
| `connected` | language modules + adapter | inherited, frozen | the proposal |
| `blocked` | language modules only, adapter held at its zero initialisation | inherited, frozen | is reaching into the shared core worth anything over a probe on frozen inherited features? |
| `disconnected` | language modules + adapter | re-randomised with a fixed seed | does the inherited memory itself carry anything, or would any recurrent core of this shape do? |

The `blocked` arm is the disconnected/blocked-update control the review asked
for, and it is also exactly the Stage 30 contract. The `disconnected` arm is a
control only and is never saved as a descendant.

The no-update parent is the step-0 checkpoint of the `connected` arm, which is
the restored owner with a zero-output adapter. Its measurements are reported as
the before-teaching row of every curve. Where the parent has no route for a
question at all, that is reported as "no route", never as a score.

## Sources

Five human-authored books, all with recorded provenance in
`runs/HB-corpus/acquisition.json` of the read-only reference, re-hashed on both
sides into this laboratory and never modified:

| id | work | role |
| --- | --- | --- |
| `webster` | Webster's Unabridged Dictionary | train / dev / final |
| `grammar` | Baskervill & Sewell, *An English Grammar* | train / dev / final |
| `plato` | Plato, *The Republic*, Jowett translation | train / dev / final |
| `calculus` | Thompson, *Calculus Made Easy* | train / dev / final |
| `descartes` | Descartes, *Discourse on the Method*, Veitch translation | **transfer only, never trained** |

`dailydialog` stays quarantined: its recorded licence is research-only with an
archive README still to check, and the shared inventory already marks it
quarantined. The absence of a conversational corpus is a real limitation of this
run; Plato's dialogues are the nearest human-authored substitute present and are
not a replacement for ordinary conversation.

No generated, paraphrased or translated-by-machine text is admitted anywhere.
Tokenisation is a training view of the authors' bytes, not a new dataset.

## Partitions

The split is a deterministic function of the *group*:
`sha256("OLA-48:<source>:<group>") mod 10`, with 0–7 train, 8 dev, 9 final. A
group is one dictionary headword entry, or one contiguous block of four
paragraphs of prose. A group is therefore wholly inside one split and cannot
appear on both sides of an evaluation. `descartes` is withheld from all three.

Vocabulary: the 8,192 most frequent word tokens of the **training** split alone,
plus five control tokens. Dev, final and transfer never vote on it. Recorded
digest `78cb571df98221c676f2168adb2616e35aded70ad0c1b5f02c3efd833dec443c`,
covering 88.5% of training tokens; everything else is `<unk>`.

## Evaluation

Three families, all cut mechanically out of the authors' own text, frozen and
hashed before teaching:

* **next_token** — mean negative log-likelihood per predicted token on held-out
  passages. The direct learning curve.
* **cloze** — a content word the author wrote is removed; the learner chooses
  between it and seven deterministic same-frequency-band distractors. Chance is
  exactly 0.125.
* **definition** — given a lexicographer's definition, choose the headword it
  belongs to among eight candidates. Chance is exactly 0.125.

An evaluation that cannot run returns an explicit `UNAVAILABLE` record with
counts and a reason. It never returns loss 0 or perplexity 1, and it never
qualifies or selects. A family with any failed batch has `qualifies: false`.

Transfer is measured on `descartes` only: an author, translator, century and
subject the descendant never trained on.

Retention is measured by replaying the declared 33-task retained suite and the
39 practical examples through the reloaded descendant and comparing every case
against the baseline record, reporting which routes are identical and which
changed.

## Schedule

Four stages, each mixing its new source with everything taught before it by
explicit round-robin, so exposure is spread rather than exhausted from the first
source. Actual examples, tokens and distinct groups per source per stage are
recorded from the cursors, not assumed. A stage that consumes zero examples from
a declared source is recorded as `DECLARED_SOURCE_NOT_EXPOSED`; three
consecutive batches with no usable example is a recorded `NO_PROGRESS` failure
rather than an endless loop.

Step budgets are in `protocol.json` and were chosen from the measured
throughput in `calibration.json`, not guessed.

## Checkpoints and resumption

A checkpoint carries the complete baseline contract (parent owner identity,
Stage 47 contract hashes, subject count and store pointer), a distinct
descendant identity, the trainable tensors, the Adam state, all three RNG
streams, every source cursor and the metric history. It is written weights-first
with an fsync, then metadata, then an atomically replaced pointer, into an
append-only revision series. The baseline store is never written to.

Resumption is proven, not asserted: a run interrupted mid-stage and resumed must
reach the same development measurement as the same run executed without
interruption.

## What would make this fail

* `connected` not better than `blocked` on held-out NLL → the shared-core
  connection is rejected for language.
* Any protected route's answer changing after teaching → the protection claim is
  false and the run is reported as a failure of the design.
* Any evaluation family returning `UNAVAILABLE` at a decision point → no
  selection is made from it.
* A declared source not actually exposed → the stage is marked and the curve is
  not described as covering that material.
