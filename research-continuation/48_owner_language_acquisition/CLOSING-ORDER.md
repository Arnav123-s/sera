# The order OLA-001 is closed in, fixed before any held-out number is read

Written 2026-09-19T21:20Z, while the `blocked` arm was at step 2,825 of 6,000
and the `disconnected` arm had not started.

What was visible at that moment, stated exactly: the in-training development
curves printed at stage boundaries — the `connected` arm's complete curve
through step 6,000, and the `blocked` arm's curve through step 3,000. Those are
`dev` only. What was **not** visible, and still is not: any `transfer`
measurement for any arm, any `final` measurement for any arm, and any of the
`evaluation-<arm>.json` records the decision rule actually reads.

The point of writing it down is the sealed split. `protocol.json` says `final`
is opened once, "after the schedule and every repair are frozen and after all
development selection is complete". Deciding *when* that moment is, after
already seeing the development numbers, would make the seal decorative. So the
order is fixed here first.

## Order

1. **Evaluate all three arms on `dev` and `transfer` only.** The `connected`
   arm additionally runs the retained-suite comparison and the practical
   demonstration. `--open-final` is not passed.
2. **Correction episode** on the arm the frozen decision rule selects. It cuts
   `dev` into a diagnosis half and a check half by group, repairs the weakest
   source, and measures retention of the others. It writes to a separate
   `<arm>-corrected` store; OLA-001's own checkpoints stay byte-identical.
3. **Independent replay** — a second process rebuilds the descendants from the
   laboratory baseline and the checkpoints and re-derives the claimed numbers.
4. **Open `final`, once.** One declared evaluation event, covering the three
   arms' step-6000 descendants — the subjects the protocol is about, not the
   corrected weights. Every repair is frozen by this point and no selection
   remains. After this, `final` is spent and no later protocol reopens it.
5. **OLA-002** (`framed` and `unframed`), on `dev` and `transfer` only. It never
   touches `final`; see its own protocol for why, and for the limit that puts on
   its evidence.
6. **The repository regression suite**, once training is idle, under the single
   numerical worker rule.
7. **`RESULTS.json` and `RESULTS.md`**, assembled from the retained artefacts.

## What is fixed by writing this down

- The corrected descendant is **not** measured on `final`. Its evidence is
  developmental, and the report says so rather than borrowing the seal.
- Step 4 happens before OLA-002 trains, so the `final` numbers are about the
  descendants OLA-001 produced and are not conflated with later teaching.
- If a step fails, the failure is recorded and charged and the order does not
  change to route around it.

## Correction, 2026-09-19T21:26Z

As first committed this document said "No arm has been evaluated on any held-out
split yet, so nothing below can have been arranged around a result." That was
false. The training script evaluates on `dev` at every stage boundary and prints
the result, so the `connected` arm's full development curve and the `blocked`
arm's first three points were already in the run logs and already read.

The opening paragraph now states what was visible instead. The order itself is
unchanged: it follows from `protocol.json`, not from a comparison, and `final`
has never been opened. But "nothing could have been arranged around a result"
was a stronger claim than the facts support, and the fix is to say what was
known rather than to leave the stronger sentence standing.
