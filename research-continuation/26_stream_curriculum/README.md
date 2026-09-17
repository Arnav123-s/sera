# Use the trained SERA request learner

SERA can label English task requests and extract their entities into a local file.
The model learned from 11,514 annotated MASSIVE requests across 18 domains and 60
intent categories. The saved checkpoint was selected on development exact frames;
[the report](report.md) records the complete held-out results, successive learning,
controls and costs. [Architecture audit](architecture-audit.md).

Run these commands from `D:/ai/projects/sera` with the existing environment.

```powershell
.venv/Scripts/python.exe scripts/sera_requests.py ask --id my-request --text "set an alarm for nine am"
.venv/Scripts/python.exe scripts/sera_requests.py result --id my-request --json
```

`ask` saves a prediction under a new identifier. `result` restores that exact saved
prediction. The default store is `runs/sera-requests-live`; the earlier pilot at
`runs/sera-requests` remains separate and preserved. The session holds up to 32
named requests. For larger files, use the batch command below, which leaves the
model and saved session unchanged.

Create a UTF-8 `.txt` file with one request per line, or a `.jsonl` file such as:

```jsonl
{"id":"alarm-1","text":"set an alarm for nine am"}
{"id":"weather-1","text":"what is the weather in london"}
{"id":"list-1","text":"add milk to my shopping list"}
```

```powershell
.venv/Scripts/python.exe scripts/sera_annotate.py --input requests.jsonl --output runs/my-inbox/annotations.jsonl
```

The output contains the original text, learned intent, softmax score, entity spans,
checkpoint identity and a per-row status. Invalid input receives an error record;
duplicate IDs are rejected. Use a fresh output path for every batch. A companion
receipt records input/output hashes and counts. A chunk can contain at most 10,000
rows and 2 MiB; each request supports 1–40 whitespace tokens. The default supervised
cap is 60 seconds, so use smaller chunks if the full file exceeds it; completed
output rows remain preserved. Softmax is a model score, not calibrated correctness.

The example input and result are saved at `runs/SC-annotation-demo/requests.txt`
and `runs/SC-annotation-demo/annotations.jsonl`. These three illustrations were
fixed at the pilot and all their predictions are retained in `integration.json`.
Annotation is the performed file task. Predicted alarm/list/weather fields are
exposed for review; the annotation command records `actions_executed: 0`.

All model operations run under the existing local allowance: one numerical CPU
thread, 2 GiB committed process-tree memory, and a wall-time cap. Each invocation
has a receipt under `runs/request-command-*` or `runs/annotation-command-*`.
No service has to remain running for these commands. The old numerical and
constraint tools remain available with their own saved stores and qualifications.

The immutable terminal checkpoints contain Adam state, torch RNG, stream cursor,
buffered examples, source identity and unique-example history. SC-004 checkpoints
also contain the replay reservoir and both RNGs. [The continuation instructions](next-cycle.md)
identify the next new experiment; completed fits should not be restarted.

Data attribution: FitzGerald et al., [MASSIVE](https://github.com/alexa/massive),
CC BY 4.0, using English source material from Bastianelli et al.,
[SLURP](https://aclanthology.org/2020.emnlp-main.588/). `data.zip` preserves the
original license, source identities and exact prepared records. This implementation
uses the supplied labels to teach a new interface on the preserved SERA owner.
