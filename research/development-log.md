# Connected-build development log

I preserve development failures separately from the final study. Final evaluation pools are not used for architecture decisions.

1. The new persistent solver passed fresh-process skill use, a second acquired skill, rejection and behavioral rollback. A historical test then found a missing top-level dataset identifier in the new admission result; I restored that output contract.
2. The first R1 probe trained for 150 updates on fully observed eight-step trajectories. On a development pool of 64 partially observed twelve-step trajectories, accuracy changed from 28.26% to 25.26%. This did not demonstrate useful world learning. I identified an architectural omission: the aggregation used only memory readout, whereas R1's forward equation also includes the current event embedding. I added that direct event-to-aggregation connection and introduced masked observations in subsequent training probes.
3. A 20-update controlled-instrument probe reduced its batch loss from 1.3441 to 0.6488, with channel/instrument completeness residuals below 3e-7. This is an optimization probe, not a final performance estimate.

I continue recording material fixes and negative results here as they occur.
