# SC-001: acquisition from a real request stream

I will teach the current SERA owner from source-labelled, real task requests and
report measured acquisition and retention. Untested abilities are open research
questions. An unsuccessful finite result is an observation about that run, not a
general inability. Comparisons require explicitly matched information and costs.

## Source and partitions, frozen before fitting

Use original MASSIVE 1.1 English JSONL, CC BY 4.0, recording exact bytes and source
headers. MASSIVE is attributed to FitzGerald et al.; English seed data come from
SLURP, Bastianelli et al. The public corpus has train/dev/test roles. Stream only
train into the optimizer. Derive vocabulary of intent/slot labels from train.
Source text and labels are data, never executable instructions or observed events.

Pilot uses a fixed first 256 training records for development and the first 128
development records for a separate measurement. These are development results,
not a final benchmark. Official test labels stay out of pilot/training/selection.
Record every rejected or overlength row rather than silently hiding exclusions.
Token limit is 40 whitespace tokens; exact annotation/text alignment is required.

## Mechanism and controls

Attach a new jointly trained intent/slot interface to the exact saved constraint
owner, retaining the current acquired dynamics, older language weights, facts and
programs. A stable 8,192-bucket word encoding feeds a learned 48-dimensional
embedding, a projection into the existing recurrent core, and learned output
heads. The existing recurrent core is the actual shared owner; new interface
parameters are trained. Preserve the complete predecessor and all old tensors.

The recurrent route uses the existing memory/fusion operation at each valid token.
A pooled embedding control bypasses recurrence while using the same data and
declared interface. No advantage of ownership or memory is assumed in advance.
Use intent cross entropy plus token BIO slot cross entropy. Report intent accuracy,
slot span F1, joint exact frames, examples, presentations, steps, storage and wall.
Also retain the all-outside slot control and majority-intent control.

## Development, persistence and continuation

Pilot seed 2631; Adam learning rate 0.003; gradient norm at most 1; batch 32;
at most 80 updates in the initial pilot. Inspect only teaching and development
outcomes. Save optimizer, model delta, RNG, stream cursor and buffered examples,
source identities, predecessor identity and cumulative costs at every 20 updates.
Resume a saved optimizer; never rerun its prefix as a new experiment.

Check actual owner aliases, byte preservation of predecessor tensors, positive
derivative through the shared fusion, annotation reconstruction, split isolation,
and exact interrupted/uninterrupted update equivalence. A development checkpoint
can produce an explicitly provisional request frame; integration must keep these
predictions distinct from factual observations and executed actions.

Before a full study, use measured pilot cost to freeze its complete finite budget,
seeds, controls, curriculum ordering, stopping/selection rules and test access.
The next intended study crosses shared versus pooled interfaces and sequential
learning with versus without bounded replay across source scenarios. Measure
retention after each block and learning efficiency on later blocks. An IID
request classifier alone is not counted as that sequential study.

The current numerical allowance is 67.82661569984339 seconds, one CPU thread and
2 GiB process-tree committed memory. All numerical tests and fits use the existing
supervisor. Source collection, implementation and static archiving are separately
identified engineering costs. A larger batch requires an explicit allowance;
its exact protocol and resumable commands will be prepared before that request.
Earlier datasets, sealed results, applicability gates and live stores are preserved.
