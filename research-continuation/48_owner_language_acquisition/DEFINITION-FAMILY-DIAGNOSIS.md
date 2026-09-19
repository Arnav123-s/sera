# Why the `definition` family sits at chance, and what that does and does not mean

Written while the `blocked` and `disconnected` arms were still running, from the
frozen protocol, the frozen evaluation manifest and the `connected` arm's
development curve. It is a diagnosis of the *measuring instrument*, not a result
about the connection hypothesis, and it is deliberately written before the
control arms finish so that it cannot be shaped by them.

## The observation

The `connected` arm's development curve moves on two of the three frozen
families and not at all on the third:

| evaluation | step | dev next-token NLL | dev cloze | dev definition |
| --- | --- | --- | --- | --- |
| before-teaching | 0 | 9.5815 | 0.1325 | 0.1341 |
| after-definitions | 1500 | 4.5556 | 0.4483 | 0.1002 |
| after-grammar-and-sentences | 3000 | 4.6517 | 0.4633 | 0.1171 |
| after-argument-and-dialogue | 4500 | 4.7631 | 0.4558 | 0.1188 |
| after-mathematics-and-science | 6000 | 4.9150 | 0.4550 | 0.1273 |

The `definition` set has 589 questions and eight candidates, so chance is
0.125: 73.6 correct with a standard deviation of 8.03. Converting the column
to counts and to standard deviations from chance:

| step | correct | accuracy | z from chance |
| --- | --- | --- | --- |
| 0 | 79 | 0.1341 | +0.67 |
| 1500 | 59 | 0.1002 | −1.82 |
| 3000 | 69 | 0.1171 | −0.58 |
| 4500 | 70 | 0.1188 | −0.45 |
| 6000 | 75 | 0.1273 | +0.17 |

Every point of the trajectory lies inside ±2 standard deviations of chance, at
five looks. The family is at chance before teaching and at chance after 6,000
steps. The dip at step 1500 is the largest excursion and is still not
significant once five looks are accounted for. Nothing was learned here and
nothing was unlearned here; the needle never left the peg.

## The cause, in the code

The cause is not the connection and not the optimiser. The training stream and
the evaluation prompt are in different formats, and the evaluation format never
occurs during training.

A Webster entry becomes exactly one training record, and the record's text is
the definition body alone — the headword is carried beside it as metadata and is
never placed in the sequence:

- [`corpus.py:213`](../../experiments/owner_language/corpus.py#L213) —
  `{"kind": "definition", ..., "headword": entry["headword"], "text": entry["definition"]}`
- [`training.py:127`](../../experiments/owner_language/training.py#L127) —
  `ids = self.tokens.encode(item["text"], limit=self.length)`

So the only thing the learner ever predicts from a dictionary entry is the next
word *inside* the definition. It is never once asked to produce the headword,
and the headword never appears adjacent to its own definition in any training
sequence.

The evaluation asks a different question in a different frame:

- [`training.py:215`](../../experiments/owner_language/training.py#L215) —
  `ids = [tokens.define, *tokens.encode_words(row["definition"]), tokens.is_token]`,
  scored at the final position.

`<def>` and `<is>` are reserved vocabulary entries
([`tasks.py:21`](../../experiments/owner_language/tasks.py#L21),
[`corpus.py:238`](../../experiments/owner_language/corpus.py#L238)) and the
tokeniser cannot produce them from text: `TOKEN_PATTERN`
([`corpus.py:80`](../../experiments/owner_language/corpus.py#L80)) splits
`<def>` into `<`, `def`, `>`. So neither id is ever an *input* during training,
and their two rows of `lex_embedding` receive an exactly zero gradient at every
one of the 6,000 steps. The optimiser is `Adam` with no weight decay
([`training.py:372`](../../experiments/owner_language/training.py#L372)), for
which a row with an exactly zero gradient has an exactly zero update, so those
two rows still hold their initialisation in every OLA-001 checkpoint of every
arm.

Their rows in `lex_head` *are* trained, as negative classes of the softmax — but
the definition question is scored only over the eight candidate headwords, so
those output rows never enter the answer. What enters the answer is the hidden
state produced from a prefix that begins with an untrained embedding and ends
with an untrained embedding, around a body whose continuation the learner was
never trained to make a headword.

The headwords themselves *are* in the vocabulary — `build_vocabulary` counts
them explicitly ([`corpus.py:236-237`](../../experiments/owner_language/corpus.py#L236))
— so they are scoreable. They are simply never a target in this frame.

## What this does and does not affect

It does **not** affect the protocol's decision rule. `protocol.json` rests the
connection claim on held-out dev mean next-token NLL at equal steps and on
`descartes` transfer, not on the definition family. And the defect is a property
of the corpus renderer and the evaluation renderer, which are byte-identical
across all three arms — the arms share one corpus, one vocabulary and one frozen
evaluation set. A measurement that is at chance for the same mechanical reason in
every arm cannot bias a comparison between arms.

It does mean one of the three headline numbers in OLA-001 is uninformative, and
the report must say so in those words rather than presenting 0.127 beside 0.455
as if both were measuring learning that happened.

It also means the natural reading — "the learner failed to acquire lexical
knowledge" — is not supported. The run does not show that. It shows that this
protocol never tested it. The honest claim is *untested*, not *failed*.

## The falsifiable prediction for OLA-002

If the diagnosis is right, the fix is a second *training view* of the same
already-hashed dictionary bytes — no new source, no new licence question, no new
acquisition — rendering each training-split entry as

```
<def> <definition words> <is> <headword> <eos>
```

so that the frame the evaluation uses is a frame the learner has been taught in,
and `<def>`/`<is>` acquire trained embeddings.

The prediction is sharp and can fail:

1. **Definition accuracy rises well above chance** on the untouched dev set.
   If the diagnosis is right this should be a large move, because the knowledge
   needed is already in the entries the learner has read; only the frame is new.
2. **Cloze and next-token do not degrade** beyond the retention tolerance —
   the second view adds material, it does not remove the first view.
3. The `<def>` and `<is>` embedding rows, currently at initialisation in every
   OLA-001 checkpoint, become non-zero-gradient parameters. This is checkable
   directly against the OLA-001 checkpoints and is recorded as a pre-condition.

If (1) fails while the frame is demonstrably trained, then the diagnosis is
wrong and the learner genuinely cannot bind definitions to headwords at this
capacity — which would be a real and reportable negative result about the
architecture rather than about the harness.

## Why this is not fixed inside OLA-001

OLA-001 is frozen. Its corpus manifest, vocabulary, evaluation sets, schedule
and step budget were fixed before any weight moved, and its three arms have been
trained against one byte-identical renderer. Changing the renderer now would
change `source_digest()`
([`descendant.py:48-56`](../../experiments/owner_language/descendant.py#L48)),
which is part of every descendant's exported identity, and would leave the arms
non-comparable and the published identities unreproducible.

So OLA-001 reports the definition family as untested and OLA-002 is frozen
separately, against the same corpus bytes, with the prediction above written
down first.
