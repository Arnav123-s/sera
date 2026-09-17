# Use the continuing SERA study learner

This release continues the trained English owner with Spanish, French and German
request interpretation, source-learned polynomial operators and checked motion
calculations. It can acquire the supported mathematical examples from a versioned
arXiv paper, retain the learned weights, correct a rejected equation and resume an
interrupted study. [Full results](report.md), [architecture audit](architecture-audit.md),
[checklist](checklist.md), [sources](literature.md), [exact continuation](next-cycle.md).

The admitted local descendant is `runs/sera-study-live`. The earlier request,
constraint and workbench stores remain available with their original identities.
Run from `D:/ai/projects/sera` using the prepared environment:

```powershell
.venv/Scripts/python.exe scripts/sera_study.py interpret --text "set an alarm for nine am"
.venv/Scripts/python.exe scripts/sera_study.py interpret --text "schalte das licht aus"
.venv/Scripts/python.exe scripts/sera_study.py solve --domain sum --coefficients "0,2,0,3,0,1"
.venv/Scripts/python.exe scripts/sera_study.py solve --domain integral --coefficients "1,2,3"
.venv/Scripts/python.exe scripts/sera_study.py motion --coefficients "2,3,1" --time 3 --position 5 --velocity=-1
.venv/Scripts/python.exe scripts/sera_study.py status
```

Coefficients are exact integers or fractions, constant first. For example,
`0,2,0,3,0,1` denotes 2x+3x³+x⁵. A `sum` result is q(n)=Σ p(k), k=0..n−1.
An `integral` result is q′(x)=p(x), q(0)=0. Every proposal carries its checked
scope. The motion example uses acceleration 2+3t+t², initial position5,
initial velocity−1 and t=3. It returns position125/4 and velocity55/2 under
the stated one-dimensional, consistent-unit assumptions.

The current retained sum curriculum covers odd powers through11; the integral
curriculum covers powers0..5. Inputs can contain up to13 coefficients. The
certificate determines whether the particular proposed answer is admitted.
This is a useful executable mathematical contract; its representation, exact
checker and physical assumptions are supplied explicitly.

For a new independent study episode, choose a fresh store name. This creates a
descendant of the preserved English parent and lets the numerical study maps
acquire their examples. It does not reset any existing store:

```powershell
.venv/Scripts/python.exe scripts/sera_study.py study --store runs/my-new-study --id power-sum --domain sum --coefficients "0,2,0,3,0,1"
```

The existing live store already retains the verified lessons and the corrected
eleventh-power identity. Calling `study` for an already executable task preserves
the original goal and returns its checked answer without another source read.
Interrupted goals retain their source budget and deterministic remaining order.
Up to32 named study goals are supported per store. Completed experiments should
be inspected through their saved evidence rather than repeated as demonstrations.

Read current arXiv metadata or a versioned paper separately:

```powershell
.venv/Scripts/python.exe scripts/sera_sources.py search 'ti:"Faulhaber" AND au:"Knuth"'
.venv/Scripts/python.exe scripts/sera_sources.py search 'cat:math.NT' --limit 3 --refresh
.venv/Scripts/python.exe scripts/sera_sources.py paper math/9207222v1
```

`--refresh` explicitly obtains a current response; older content-addressed bodies
and acquisition logs remain preserved. The adapter uses public HTTPS research
endpoints, bounded downloads and rate limiting. Source retrieval is distinct from
learning: the tested mathematical reader recognizes the specified power-sum
table, proposes examples and submits them to independent checks. A cached page
by itself does not change the learner's weights.

Request interpretation returns intent and entity predictions for review, with
checkpoint identity and training lineage. It records zero external actions.
For existing English file annotation use [the preserved batch interface](../26_stream_curriculum/README.md).
The four-language single-request route uses the new continuing owner. The
[report](report.md) provides intent accuracy and stricter complete-frame accuracy.

Numerical commands use the existing supervisor: one CPU thread,2GiB process-tree
committed memory and a recorded wall allowance. Receipts appear under
`runs/study-command-*`. There is no required background process. A current budget
read, rather than the historical balance in this release, controls each invocation.

`data.zip` contains the exact prepared MASSIVE records, license and manifests.
`checkpoints.zip` contains all terminal comparisons, the failed100-step checkpoint
and the admitted continuation. `sources.zip` and the failure/source-history
archives preserve implementation identities. `live-state.zip` holds the
append-only descendant. See [archive-manifest.json](archive-manifest.json) for
member hashes and [next-cycle.md](next-cycle.md) for restoration prerequisites.

Data attribution: FitzGerald et al., [MASSIVE](https://github.com/alexa/massive),
CC BY4.0. Mathematical source receipts and derived lessons are published; paper
bodies and the user's unrelated local corpora remain in their original locations.
