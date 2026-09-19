# OLA-002 — teaching the frame the question is asked in

Prospective and finite, frozen 2026-09-19T20:40Z, before any weight moves and
before the OLA-001 control arms had reported. It reuses OLA-001's corpus bytes,
OLA-001's frozen vocabulary and OLA-001's frozen evaluation sets unchanged. No
new source is acquired and no new licence question is opened.

The evidence that motivates it is in
[`DEFINITION-FAMILY-DIAGNOSIS.md`](../48_owner_language_acquisition/DEFINITION-FAMILY-DIAGNOSIS.md):
OLA-001's `definition` family never left chance because the frame the question
is asked in — `<def> … <is> ?` — never occurs during training, and the two frame
tokens' input embeddings therefore hold their initialisation in every OLA-001
checkpoint.

## The question

> Is OLA-001's chance-level `definition` result a fact about the learner, or a
> fact about the harness — and if the frame is taught, does the lexical
> knowledge the learner has already read turn out to be there?

## What changes, and what does not

Changed: one additional *training view* of dictionary entries that are already
in the training split and already read in OLA-001, rendered directly as token
ids from the existing frozen vocabulary:

```
<def>  <definition words, capped at 40>  <is>  <headword>  <eos>
```

Unchanged, and re-verified by digest before the run starts:

- the corpus bytes and `runs/owner-learning-001/corpus/manifest.json`
- the frozen vocabulary, digest `78cb571df98221c676f2168adb2616e35aded70ad0c1b5f02c3efd833dec443c`
- the frozen `dev`, `final` and `transfer` evaluation sets
- the parent owner `62ea82cf…` and the laboratory baseline store

The new view builds ids from the existing vocabulary and adds no word to it, so
the vocabulary digest cannot move and the evaluation sets stay valid. Only
entries whose split is `train` are framed; `dev` and `final` entries are never
framed and never trained on, so the measured family stays held out.

`source_digest()` will change, because `corpus.py` and `training.py` change.
That is expected: OLA-001's descendants remain restorable because each
checkpoint carries its own `language_config` and `restore_descendant` reinstates
it, and the restore report states `code_unchanged_since_training: false` rather
than hiding it.

## Starting point, and why it is a continuation

OLA-002 starts from OLA-001's step-6000 descendant — **the arm that OLA-001's
own frozen decision rule selects**, not an arm chosen after seeing results. The
rule is `protocol.json` of OLA-001: `connected` if it beats `blocked` on
held-out dev mean next-token NLL at equal steps and on `descartes` transfer,
otherwise `blocked`. Naming the rule rather than the arm is what keeps this
pre-registered.

A continuation rather than a fresh 6,000-step run is chosen because the question
is specifically whether knowledge *already read* becomes reachable once the
frame is taught. Starting from the same weights makes the before/after a
comparison on one set of parameters rather than across two training runs, and it
fits the one-worker, no-paid-compute budget.

## Arms

Both arms start from the identical selected OLA-001 checkpoint and differ in
exactly one thing: the view of the same webster training entries.

| Arm | Training view | Question it answers |
| --- | --- | --- |
| `framed` | `<def> definition <is> headword <eos>` | does teaching the frame recover the ability? |
| `unframed` | the definition body alone, exactly as OLA-001 | is any gain just more webster training? |

Same 1,500 steps, same sources, same optimiser, same batch, same sequence
length, same seed, same entries. The `unframed` arm is the format control: if
definition accuracy rises there too, the frame is not the cause and the
diagnosis is wrong.

Exposure is matched to one OLA-001 stage (1,500 steps), so the additional
teaching is bounded and comparable to the original `definitions` stage.

## Pre-conditions, checked before training and recorded either way

1. The `<def>` and `<is>` rows of `lex_embedding` in the selected OLA-001
   checkpoint are bit-identical to the values `LanguageAcquisitionR1.attach`
   produces at the recorded seed. If they are not, the diagnosis is wrong and
   this protocol is withdrawn rather than adjusted.
2. No framed item's entry has split `dev` or `final`.
3. The vocabulary digest after adding the view equals the digest above.

## Decision rule

`dev` has 589 definition questions over eight candidates, so chance is 0.125
with a standard deviation of 0.013627 on accuracy, and 0.019272 on a difference
between two arms.

- **Frame hypothesis supported** if `framed` dev definition accuracy exceeds
  **0.1659** (chance + 3 sd) *and* exceeds `unframed` by more than **0.0385**
  (2 sd of the difference).
- **Frame hypothesis rejected** if `framed` does not exceed 0.1659. The learner
  then cannot bind definitions to headwords at this capacity even when taught
  the frame, and that is reported as a negative result about the architecture,
  not about the harness.
- **Diagnosis rejected** if `unframed` also exceeds 0.1659. The gain would then
  be extra webster exposure, not the frame.
- **Retention guard**: dev cloze must not fall by more than 0.02 and dev mean
  next-token NLL must not rise by more than 2% against the starting checkpoint,
  and every case of the declared 33-task retained suite must still answer. A
  breach is reported as a failed repair even if the definition number rises.
- **Generalisation check**: `descartes` transfer cloze is measured before and
  after and reported either way. It has never been trained on in any protocol.

## What OLA-002 deliberately does not do

It does **not** open the sealed `final` split. OLA-001 opens `final` exactly
once; opening it again would make it a split that has been looked at twice, and
quietly spending a sealed resource is the kind of erosion this campaign exists
to stop. OLA-002's evidence is therefore developmental: `dev`, which was never
trained on but was used for OLA-001's curve, and `descartes` transfer, which was
never used for anything. Confirming OLA-002 on a genuinely sealed split needs a
split that does not yet exist, and that is stated as a limitation rather than
worked around.

It also does not touch the OLA-001 descendants. The corrected weights are
written to a separate store and OLA-001's checkpoints stay byte-identical.

## Amendments

Amendments are appended with a date. The frozen text above is never rewritten,
so a reader can always see what was predicted and what actually happened.

### 2026-09-19T21:05Z — the source digest does not have to change after all

The frozen text says `source_digest()` will change, because `corpus.py` and
`training.py` would change. While drafting the implementation it turned out that
neither has to.

`source_digest()` hashes an explicit tuple,
`SOURCE_FILES = ("descendant.py", "corpus.py", "training.py")`
([`descendant.py:48`](../../experiments/owner_language/descendant.py#L48)). The
framed view can live in a *new* module, `experiments/owner_language/frame.py`,
holding a `FramedStream` that subclasses `SourceStream` and overrides only how
an entry is rendered to ids. `Trainer` builds `self.streams` in its constructor
and never rebuilds them, so the OLA-002 driver substitutes the webster stream
after construction. The cursor, the seeded per-epoch permutation, the resume
contract and the exposure counters are all inherited unchanged.

Nothing in `SOURCE_FILES` is touched, so:

- `source_digest()` is unchanged;
- every OLA-001 descendant keeps reporting `code_unchanged_since_training: true`
  rather than merely staying restorable through its saved `language_config`;
- OLA-001 and OLA-002 descendants are directly comparable at the code level.

This is strictly better than what was frozen and changes no question, arm,
threshold or decision rule. It is recorded here rather than silently corrected
above.

### 2026-09-19T21:05Z — the format control drills the same entries

The frozen text describes the `unframed` arm as "the definition body alone,
exactly as OLA-001" on "the same webster training entries". Taken literally that
would have let the control stream over *all* webster training entries while the
`framed` arm streams only entries with a single in-vocabulary headword — the
only entries that can carry a frame. A gain in the framed arm could then have
come from drilling a narrower, easier population rather than from the frame.

So the eligibility filter is applied in **both** arms and only the rendering
differs. The driver records `matched_entry_population` as a pre-condition with
both counts, so the reader can check it rather than take it on trust.

### 2026-09-19T21:05Z — restoring a descendant does not restore its arm

`restore_descendant` rebuilds through `LanguageAcquisitionR1.attach`, which
makes every `lex_*` and `adapter.*` tensor trainable. If the OLA-001 decision
rule selects `blocked`, continuing from that checkpoint without re-freezing the
adapter would silently turn OLA-002 into a connected run and the comparison
would be against a different architecture than the one it started from. The
driver re-applies the arm's freeze after restore and records which tensors it
re-froze.
