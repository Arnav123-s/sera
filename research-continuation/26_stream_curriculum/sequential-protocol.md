# SC-004: one learner across six successive teaching blocks

I freeze this protocol before any SC-004 training and before official test access.
The scope is successive acquisition of request intents and entity spans across
source domains. Earlier finite component experiments remain separate evidence.

Use fresh seeds 2651 and 2652, crossed with no replay and a 256-record reservoir.
Each independent lifetime starts from the same preserved constraint owner with
a fresh shared request interface. Within a lifetime, carry every learned interface
weight and Adam moment across all six blocks; never reset at a block boundary.
The existing predecessor's 145 tensors stay protected. Output schema is supplied
from the training partition, exactly as in SC-001/002/003.

Sort the 18 source scenarios alphabetically and group consecutive triples into
six blocks. Within each block use a fixed seed-specific full permutation of its
training records, then the existing bounded 128-row shuffle reader. Each block
receives 300 updates: 16 current-stream examples plus 16 additional examples per
update. No-replay repeats those same 16 current examples. Replay samples 16 rows
with replacement from the reservoir before admitting the current rows; its first
update repeats current examples when memory is empty. Thus current-stream exposure
is matched (28,800 presentations per lifetime), and total presentations are matched
(57,600, including repeats/replay). Count unique records as well.

Update the reservoir once for each newly seen source ID, using standard uniform
reservoir replacement with a separate fixed RNG. Both arms maintain the same
reservoir/RNG state for audit; the no-replay arm never trains on its contents.
Use a second RNG for replay sampling. Persist all IDs needed to avoid counting
a repeated pass as a new source example; count that bookkeeping storage too.
The optimizer/interface/loss/clip match the shared SC-002/003 routes.

After each block, measure all 2,033 development requests and save their raw
predictions. These labels are measurements only and never enter either optimizer
or reservoir. Report current-block accuracy, average accuracy on already taught
blocks, all-domain accuracy, exact frames and slot span F1. For each earlier block,
forgetting is its best measured post-teaching intent accuracy minus final intent
accuracy; also show the full matrix so improvements remain visible.

Primary paired outcome: final mean intent accuracy over the six equally weighted
blocks. Secondary outcomes: mean forgetting, final span F1, acquisition accuracy
after each block, and per-block learning curves from training loss. Compare both
seeds individually as well as their mean. A measured replay improvement remains
scoped to this request-domain curriculum. The replay procedure is supplied code,
not a learning algorithm independently invented by SERA.

Save at each 100 updates and each block boundary, including weights, optimizer,
source cursor, buffered rows, both reservoir RNGs and unique-ID history. Preserve
all intermediate/failed states. Check exact pause/restore behavior on a small
development fixture before canonical lifetimes. Keep final official assessment
of these four models supplementary to the eight fixed SC-002/003 candidates;
it must not reselect the annotation aid or retune those candidates.

Budget: at most 900 supervised seconds for four lifetimes, based on measured
shared-route fitting cost; revise only from timing/checkpoint evidence if needed.
Per-job cap 240 seconds with an internal checkpoint deadline at 205 seconds.
Resume partial jobs with their exact optimizer rather than restarting any prefix.
Numerical checks, independent audits and the whole-project regression use the
remaining approved extension and are charged separately.
