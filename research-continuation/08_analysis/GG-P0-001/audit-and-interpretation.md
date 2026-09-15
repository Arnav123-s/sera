# Interpretation and audit corrections

I retain GG-P0-001 as an exploratory comparison of supplied representations. All 5,120 decoded models and 20,480 partitions replay exactly. Repeating every fit under the frozen source reproduces all 5,120 complete artifact hashes. Independent checks cover every posterior mean/covariance, marginal likelihood and marginal NLL; a stable SVD reference confirms all 640 class-weight vectors.

At 64 support observations and noise 0.02, the program generator has withheld-circle-arc MSE **0.00001818**, compared with **0.229895** for interpolation. Its mean complete JSON object is **1,419 bytes**, versus **4,760 bytes** for interpolation. A standalone deployment also needs the **14,584-byte shared decoder** and the disclosed Python/NumPy/Torch runtime. The learned ellipse alternatives achieve **0.00007369** arc MSE, whereas forcing a circle gives **0.0809635**. These results support fitting and selecting reusable generators when their supplied language matches the world.

The strongest failure concerns exceptions. Residual memory lowers in-region MSE from **0.002421 to 0.000806**, but both methods retain **0.100103** error on unseen arcs. Calibration then narrows the residual method's interval: arc coverage falls from **34.1% to 23.2%**, despite the nominal 95% label. Its in-region inadequacy alarm does not fire. A local repair therefore does not justify broader applicability or confidence. Neither generator selection nor residual memory is promoted as a general solution.

Random unseen values remain unpredictable. Exact episodic lookup recalls recorded values and explicitly returns unavailable on unfamiliar coordinates; its scored zero-mean fallback has unseen MSE near one. The bounded symbolic search can extrapolate catastrophically on these negative controls. A compact formula cannot recover missing independent facts.

## Corrections to the initial protocol labels

The original source and artifacts are preserved, including these limitations:

- The actual codec is mixed: coefficient/covariance arrays, neural weights and stored points/residuals are float32-representable, while calibration values, class weights and other scalars are float64. NumPy decoding uses float64. The full JSON byte counts already charge those actual representations. A hypothetical packed floating payload is 3,199,856 bytes across the cohort, excluding integer/schema/decoder/runtime costs; it is not the deployed 19,401,037-byte model collection.
- The neural initializer received `seed + 50000`. Inspected fitting code uses it only for Torch initialization; no simulator oracle was used. The literal API-isolation claim was too strong. Future source uses an independently declared fixed initializer.
- The original 900-second mechanism is a soft start-admission budget, not a hard timeout. The cohort finished below that limit, at 207.27 charged case seconds. Subsequent work uses an external process-tree supervisor with a durably reserved deadline.
- Original peak memory is unavailable. The first refit supervisor measured only the Windows launcher and is retained as a failed telemetry approach. A corrected Job Object supervisor includes its Python descendants and reports **committed memory**, explicitly distinct from resident memory.
- The original fit source/protocol were frozen; the auditor, mathematical references and report generator have separate later identities. This is reproducible trusted research code, not a fully isolated adversarial evaluator. A follow-up auditor adds an explicit four-partition completeness check.

These corrections narrow the claims without editing a saved coefficient, score, failed result or protocol. Current-source fixes are separate from the frozen GG-P0-001 source. Reproduction must use the frozen copy or the release's recorded commit, rather than silently substituting the current implementation.

## Next decision

Keep the compact generator and its classical controls. Continue with several explanations of one world, active distinguishing observations, and explicit applicability checks before treating any local repair as broadly valid. Integrate through the existing shared owner with full evidence and dependency identities. Learned shared transfer, learned guards, unordered geometry and investigator-policy improvement remain separate unproven stages.
