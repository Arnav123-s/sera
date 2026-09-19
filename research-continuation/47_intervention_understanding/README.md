# Acquired models, checked interventions and returned answers

SERA can retain competing explanations of observed decay, choose a distinguishing pulse experiment, learn from independently simulated outcomes, check which pulses actually occurred, and return to its original prediction question. The acquired coefficients execute on the same continuing owner as the earlier mathematics, language, reading and physical-model routes.

```powershell
.venv/Scripts/python.exe scripts/sera_current.py --input research-continuation/47_intervention_understanding/example-tasks.json --output runs/my-intervention-results.json
```

Use a fresh output path. The batch includes 39 examples: six new intervention requests and 33 earlier practical tasks. These queries read the saved learner without adding observations or rewards.

For one question, save this as a JSON list and pass its path to `--input`:

```json
[
  {
    "id": "predict-an-echo",
    "kind": "intervention_what_if",
    "subject": "mixed-47101-information",
    "program": {"ticks": 10, "pulses": [5]}
  }
]
```

Each tick is 0.25 seconds. A pulse boundary reverses the subsequent phase accumulation in the installed physical model. The response includes the original goal, learned probability, source, evidence revision, competing descriptions, model-conditional uncertainty and independent adequacy result. Its pulse history describes an actual applied control; command execution requires a monitor receipt.

| Request | Meaning |
| --- | --- |
| `intervention_what_if` | Predict a supplied applied pulse history; omit `program` to return to the original question. |
| `intervention_explain` | Inspect the initial ambiguity, acquired coefficients, actual pulse application and independent evidence. |
| `intervention_plan` | Compare a finite `programs` list by predicted plus-X probability. Returns a conditional plan; it does not execute a device. |
| `intervention_findings` | List retained investigations, qualification and next actions. |

New empirical acquisition uses `InterventionSession.start`, `commit_probe`, `observe`, `fit` and `qualify`. See [the study driver](../../scripts/intervention_study.py) for a complete source-bound cycle. Every acquisition decision is saved before its outcome; independent selection and adequacy remain outside weight fitting. Use a fresh successor store for additional research so completed final evaluations stay preserved.

[Results and comparisons](report.md) · [Prospective protocol](PROTOCOL.md) · [Architecture audit](architecture-audit.md) · [Packet verification](packet-verification.json) · [Source reading](sources.md) · [Checklist](checklist.md)

For a clean clone, run `python scripts/restore_current.py` after installing the project's pinned environment. It verifies and restores predecessor archives and this successor, preserving any existing differing files. The older Stage 46 interface remains available through `python -m scripts.structural_use`.
