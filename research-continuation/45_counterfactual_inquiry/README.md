# Counterfactual investigations

SERA can take an executable quantity it already understands, imagine changing it under several different assumptions, and investigate what follows. It preserves competing explanations, chooses another observable when they predict the same initial answer, and broadens its mechanism basis after a contradiction. Checked relationships can become compact reusable response rules on its continuing owner.

[Results](report.md) · [Engineering](ARCHITECTURE.md) · [Frozen protocol](PROTOCOL.md) · [Reconciliation](RECONCILIATION.md) · [Checklist](checklist.md) · [Current project status](../../docs/STATUS.md)

## Try the saved learner

From the prepared project directory:

```powershell
.venv/Scripts/python.exe scripts/run_counterfactual_persistent.py --output runs/my-counterfactual-001 --module scripts.counterfactual_live -- --input research-continuation/45_counterfactual_inquiry/example-tasks.json --output runs/my-counterfactual-results-001.json
```

The resource wrapper uses one CPU thread, the 2 GiB process-tree cap, unrestricted authorized local time and a scoped keep-awake request. Every output directory must be fresh. The example batch includes smaller/larger mass, changing time, polynomial accumulation and eleven earlier tasks through the same owner. On a restored Linux checkout use `.venv/bin/python -m scripts.counterfactual_live` for the task interface. The current store is `runs/sera-counterfactual-qualified-live`; its experimental predecessor remains separate.

Copy the example JSON and change a request:

```json
[
  {
    "id": "explore-mass",
    "kind": "what_if",
    "domain": "motion_0",
    "axis": "m",
    "target": "a0",
    "multipliers": ["1/4", "1/2", "1", "2", "4"]
  }
]
```

This compares acceleration consequences when mass changes. Every alternative states which quantities are controlled and which may change together. The result includes the exact conditional response, values at the requested multipliers, guards, stationary points, limits, a proof receipt and the saved conjecture file. The complete conjecture set is saved before checking and explanatory naming.

Available retained domains are constant, linear and quadratic acceleration (`motion_0`, `motion_1`, `motion_2`) and degree-two polynomial value/integral/finite-sum relations (`polynomials`). The axis and target are retained symbols; unambiguous taught word labels also resolve through the saved bindings. A custom `context` must provide all quantities in a consistent baseline world. A branch that fails its domain guard returns an explicit unadmitted prediction.

| Request | Result |
|---|---|
| `what_if` | All checked conditional alternatives for the requested change |
| `inquiry_findings` | Retained investigations, compact rules, points, procedure choice and provenance |
| `apply_inquiry_rule` | A normalized rule's conditional value from `rule`, `initial`, `multiplier`; its mechanism and applicability contract accompany the result |
| `inquiry_next` | An unexplored dependency, or an exhausted-frontier record that prevents repeating the same completed work |
| Earlier task kinds | Existing solution, language, reading, integral, summation, motion and evidence interfaces |

## What is being learned and verified

Retained operator/inverse weights supply the imagined worlds. New evidence updates compatible-world weights. Verified practice changes an owned procedure-value head. A surviving response that generalizes across symbolic backgrounds becomes a compact coefficient vector and a guarded reusable rule. Full comparisons determine which procedure is used in deployment.

The question grammar, measurement candidates, mechanism family, verification code and reward contract are engineering. Scientific records distinguish conditional identities, explicitly simulated observations and attributed physical data. The study keeps all hypotheses and failed attempts; reward is reserved for independently checked new nonconstant response shapes of derived quantities. Additional valid routes remain available even when they receive no novelty points.

The current archive preserves complete practice/validation/final trajectories, independent checks, exact checkpoints and costs. Completed phases refuse reruns. After the represented frontier is exhausted, the next research action must add a justified representation or observation source; repeatedly asking the same closed questions does not create more discovery credit.

## Reproduce and inspect

Restore predecessor artifacts in the order documented in the [repository map](../../docs/REPOSITORY_MAP.md), then restore this stage:

```powershell
.venv/Scripts/python.exe -m scripts.counterfactual_artifacts --restore
.venv/Scripts/python.exe -m pytest tests/test_counterfactual_inquiry.py
```

Archive restoration verifies source identities and refuses to overwrite divergent local work. The original solution owner, every earlier checkpoint and all completed evaluations remain preserved.
