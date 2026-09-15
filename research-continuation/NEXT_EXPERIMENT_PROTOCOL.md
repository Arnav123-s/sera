# Next experiment: GG-GUARD-001

Status: **designed, not executed and not yet source-frozen**. This replaces the completed GG-P0 and GG-ACT execution queue; their frozen protocols and original outcomes remain preserved. No future-family data have been drawn for this experiment.

The target is the demonstrated applicability failure: a model can assign almost all probability to an inadequate explanation. The guard must estimate whether a proposed prediction is supported by actual evidence. It must preserve useful acceptance on valid models, rather than improving error merely by rejecting everything.

## Implementation and frozen comparison

Use a single supplied-phase Gaussian generator with the current four-class menu. Train a small calibrated guard from prior teaching episodes with actual observed prediction outcomes. Allowed inputs are public query coordinates, distance to observed support, coefficient uncertainty, class disagreement and prequential residual summaries. True family IDs, simulator parameters, future measurements and query answers remain evaluator-only. Freeze the feature extractor, learner, calibration procedure, action policy, labels, budgets and family partitions before final execution.

Compare always-predict, fixed residual/uncertainty threshold, distance-to-support guard and learned guard. Include always-abstain only as a coverage-zero reference. All controls receive identical observations; any distinguishing inquiry paid for by one method must be charged and included in matched controls. Labels used to teach the guard are actual observed outcomes from separate teaching episodes; imagined outputs cannot supply factual labels.

Use separate environment and optimizer seeds. Split episodes by underlying mechanism instance, not by query rows. Keep all final instances and at least two mechanism families excluded from guard training and threshold selection. Include in-menu mechanisms, localized exceptions, nonstationary or piecewise mechanisms, and random-value negatives. Disjoint noise draws from the same mechanism are not independent knowledge-discovery tasks.

Measure false acceptance on wrong predictions, false rejection on valid predictions, accepted-set MSE, acceptance fraction, coverage-versus-risk curves, proper scores and calibration by family. Report paired instance differences and uncertainty; the outcome unit is a world, not thousands of correlated query points. Compare costs for observation acquisition, guard fitting/calibration, inference, added parameters, retained witnesses and the full decoder.

Before selecting final counts, run a disjoint timing fixture and freeze a bounded local CPU allowance. Preserve capped or failed cohorts in fresh named directories. Do not retrospectively widen a cap and call it the original run. Publish no learned-applicability claim until fixed controls, future-family leakage checks and independent replay pass.

## Connect to the requested architecture

After the component gate succeeds, place the accepted guard and generator references in a versioned executable definition inside the declared shared-owner path. Preserve the current generator implementation as a source-pinned predecessor. Reject stale learned guards when their feature schema, generator, dependencies or interpreter changes. Explicitly distinguish supported, conditional, ambiguous and unsupported answers.

Then test guarded consolidation: remove incidental episode constants from reusable definitions, retain necessary exceptions/witnesses, invalidate affected shortcuts after a correction and migrate the active state under an explicit mapping. Recheck all old capabilities and whether the acquired representation benefits another neural route under matched exposure. Equal old outputs alone establish retention, not transfer.

Only after these gates train an investigator eta. Keep K fixed for the first investigator comparison, then cross old/new K with old/new eta and test protected future families. Reproduce the packet's null-policy result as a control. Continue to actual generations only if policy effects survive that separation. Do not promote M2–M4 from the guard or the fixed inquiry panel.

## Exact current verification commands

The completed work can be checked without starting this proposed experiment:

```powershell
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m experiments.generative_memory.acquisition --output runs/GG-ACT-001-final --replay
.venv/Scripts/python.exe -m experiments.generative_memory.acquisition_audit --run runs/GG-ACT-001-final --output runs/GG-ACT-001-independent-recheck.json
.venv/Scripts/python.exe scripts/verify_release.py
```

Use fresh output paths. For future code revisions, use the [frozen protocol sources](14_release/README.md) to reproduce old cohorts. The next executable development task is to implement the guard and its episode-level separation tests, then freeze the machine-readable GG-GUARD-001 protocol before drawing final observations.
