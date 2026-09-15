# Preserved development failures

1. The first mechanism test retained a live `SharedOwnerRef` object and compared its identity after mutation with another live reference. Both correctly resolved the current state. The test now snapshots the identity string before the update. The old test source, failed XML and resource log are retained. This was a test error, not a failure of dependency invalidation. The source owner and its generator buffers remained unchanged; gradients reached memory, fusion, the numeric encoder and decoder.

2. A06-PILOT-001 improved development RMSE from 0.2605 to 0.1272 but its worst retained score dropped 0.0625. It fails the proposed 0.02 retention margin. Ordinary replay excludes the legacy earliest-binding probe and does not cover every held-out typed structure.

3. A06-PILOT-002 added conditional parent-output anchors on support-only typed inputs and all five legacy sequence tasks, with two sequence lengths. It also used a smaller step size. On a new development stream, RMSE improved from 0.2429 to 0.1494, but the largest retained drop was still 0.03125. This is a different pilot cohort; it is not a controlled estimate of the improvement over pilot 001.

4. A06-PILOT-003 restricted numerical updates to the existing 514-parameter motion output head. Its new development stream improved from 0.2348 to 0.1691 with no measured old-score drop. This is a candidate that protects the recurrent core, not evidence of improved shared representations. Both anchored full-core and readout-only scopes will be frozen before any confirmatory outcomes, using the same five future streams and the same five controls. No hyperparameters will be changed between those final cohorts.

The original source snapshots, all three pilot checkpoints, detailed predictions and charged resource jobs remain separate. One unused-import lint finding was removed before source freeze. Exact interrupted-optimizer continuation passes its mechanism test; completed checkpoints are refused as restart points.

5. The original frozen final checker completed its comparisons but failed to serialize a `numpy.bool_` in the gate summary. Its failed resource job and traceback are retained. An isolated adapter converts only NumPy Boolean output values to Python Boolean values; it does not change numerical comparisons, gates, source data or model checkpoints. The frozen checker source stays unchanged. The repeated read-only checks are charged to separate resource jobs.

6. The first output adapter exposed `dumps` but omitted `loads`, causing an immediate read failure. Its source and charged failed job are preserved. The corrected adapter retains the unchanged standard JSON reader as well as its output conversion.
