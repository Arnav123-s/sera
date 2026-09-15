# Next experiment: GG-GUARD-002

Status: **prospective design, not implemented or source-frozen**. GG-GUARD-001 is completed and preserved in [its release directory](15_applicability/). Its learned ranking improved utility, but the pooled calibration threshold admitted unacceptable error on local exceptions and changing mechanisms. No shared-owner guard was promoted.

## Diagnose calibration and information separately

The next component must separate two problems. First, a pooled calibration target let easy worlds subsidize poor predictions: the selected minimum validity probability was 0.084. Second, an unobserved exception can be compatible with every acquired observation. More conservative calibration addresses the first issue; a guarantee about arbitrary unobserved exceptions requires additional evidence or explicit assumptions.

Before final data, freeze a minimum validity-probability rule and calibration strata defined only by observed features, such as prequential residual severity, within/outside observed support and effective local evidence count. Include sufficient calibration worlds per stratum; otherwise return unresolved. Compare a globally calibrated guard, a probability-floor guard, stratified calibration, a fixed residual guard and always-abstain. Do not use mechanism names as runtime scope labels.

Retain the four-class generator and the first trained guard as versioned predecessors. Training, probability calibration, threshold selection and final assessment must remain episode-disjoint. Use new environment and optimizer seeds. Both formerly protected families—piecewise drift and chirp—have now been evaluated and cannot be described as unseen research material again. Reserve at least two new mechanisms, for example an abrupt jump and a decaying spiral, exclusively for a new final bank. Fix their definitions before collecting data. No such bank has been drawn yet.

Use actual outcomes for correctness labels. Do not label a model-supported answer correct merely because its own generator agrees with it. Report the calibrated probability and the acceptance decision separately. If a model rejects almost everything, report that coverage collapse explicitly. Keep false acceptance, false rejection, accepted error, proper scores and uncertainty by world and declared observed-feature stratum.

## Paid evidence when the initial prefix is inadequate

Then compare zero, one and two additional observations at fixed costs. Use the same additional observation allocation for learned and fixed calibration controls when attributing gains to calibration. For policy comparisons, hold total observation counts fixed and compare a local counterexample probe with random, space-filling and information-gain controls.

Commit each query's prediction before acquiring its assessment outcome. Acquiring the query itself can supply a verified measurement at that point, but it does not establish a reusable mechanism or count as unobserved prediction. Include a matched observation-only control. Preserve all action histories and costs, and keep imagined continuations ineligible as labels.

A finite prefix cannot certify all possible local exceptions. Explicit outputs should distinguish conditional prediction, ambiguity, unsupported prediction and a claim supported by the declared evidence/assumptions. A guard must not upgrade a restricted model-class posterior into global correctness.

## Gate and subsequent architecture work

Run a disjoint timing fixture, choose cohort sizes and hard caps from that fixture, then freeze code, runtime, split definitions, primary scores, thresholds and promotion criteria. Do not retrofit the GG-GUARD-001 protocol or tune on its final outcomes and call them a fresh test. Preserve failed attempts, resource records and both versions of every changed interpreter.

Only after prospective reliability and useful-acceptance gates pass, integrate the accepted guard through a versioned executable definition and the declared common owner. Test stale feature/generator/schema/dependency rejection, corrections, transitive invalidation, active-state replay and old-capability retention. Then measure shared positive transfer under matched exposure. The packet's W08 is not completed by attaching a guard or preserving old outputs alone.

Investigator eta training, old/new K × old/new eta, successive generations and language acquisition remain later work. M2 is open; M3 and M4 are not established.

## Verify the completed predecessor

```powershell
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/verify_release.py
.venv/Scripts/python.exe scripts/verify_continuation.py
.venv/Scripts/python.exe scripts/verify_applicability.py
.venv/Scripts/python.exe scripts/replay_guard_release.py --output runs/my-guard-replay
```

Use a fresh output directory. The replay helper restores the original frozen source; the current independent auditor includes the explicitly documented repairs. Exact replay requires the recorded numerical runtime.
