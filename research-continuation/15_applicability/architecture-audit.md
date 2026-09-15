# Architecture audit after GG-GUARD-001

I reread the second packet's architecture/findings and autonomous cookbook sections on uncertainty, applicability, compilation, consolidation and retention, and compared this revision with the adopted handoff. The original R1/R2 source comparison and the previous W00–W15 audit remain preserved. This revision addresses one missing component; it does not complete the requested architecture.

## Contract and implementation comparison

| Source requirement | What this revision actually implements | Evidence and remaining limitation |
|---|---|---|
| W05: do not equate class confidence with adequacy | A separate learned validity probability using observed residuals, uncertainty and support geometry | Eight-family panel and explicit alias case. The selected operating policy fails conditional-risk requirements. |
| W08: train applicability on positive examples and counterexamples | A 291-parameter guard trained on 576 distinct observed teaching episodes | Separate 192-world probability calibration and 192-world threshold selection. Family IDs, coefficients, clean targets and future observations cannot enter the fitting interface. |
| W08: compare learned and engineered scopes | Residual/uncertainty and support-distance controls with identical prefix observations and matched threshold labels | Learned utility improves over the stronger residual control, but the distance control rejects everything and is a weak reference in this cohort. |
| W08: guarded common-owner consolidation and positive transfer | No promotion after the frozen risk gate failed | The guard remains under `experiments/`, explicitly outside the operational shared owner. Calling that integrated understanding would exceed the evidence. |
| W01/W07: complete executable/dependency identity | Guard artifacts pin feature names/schema, generator/feature code, relevant event dependencies and numerical runtime | Changed contracts are rejected even if an artifact's outer checksum is recomputed. The accepted object owns its canonical serialized bytes. This is a component identity contract, not a new common graph executable. |
| W09: preserve past knowledge while revising | All 45 shared-owner source files and 16,556 listed predecessor files are unchanged | There was no shared-owner update. The previous 40-group retention comparison remains historical evidence; no new transfer or lifelong-learning claim is made. |
| W15: independent audit and failed-attempt preservation | Exact source-workspace replay, independent batch algebra, hand-evaluated network, episode partition audit and checker regressions | Two checker errors were corrected explicitly. Their original source/failed logs remain recoverable; no model result or tolerance was changed. |

## Where the experiment falls short

The calibrated threshold controlled a pooled empirical error fraction. Easy in-menu cases dominated that aggregate, allowing local exceptions and changing mechanisms to have unacceptable accepted-answer error. The network's low Brier scores on some negative families did not prevent a poor acceptance policy. The learned representation, the calibration procedure and the policy gate need separate judgments.

The alias construction exposes an access limitation independent of training quality. If two mechanisms produce the same acquired observations and disagree elsewhere, neither the current guard nor a more expressive classifier can identify which mechanism is present without additional evidence or assumptions. It must not return “supported” solely from a high probability in the restricted model class.

The fitted guard does not acquire an open grammar, discover phase from unordered observations, train the investigator eta, create an autonomous compiler, improve a neural route, or establish new knowledge × policy factorial gains. Existing optional parameter-cloud experiments remain separate and preserved. M2 is open; M3 and M4 are not established.

## Source references and preservation

- Delivered architecture bundle SHA-256: `4e9b829f67aca7bbb543b0c03eff022b019b67698a0f27f899fbd7a8186e814f`. Relevant members: `docs/ARCHITECTURE_AND_FINDINGS.md`, `docs/AUTONOMOUS_COOKBOOK.md` W05/W08/W09/W15, and `patches/INTEGRATION_MAP.md`.
- Original handbook bundle SHA-256: `7e9268f8a0281190551ecfdadf8270eab989190e62e4621c03b93590c3932cce`. Its R1/R2 and broader hypotheses remain documented in the [shared source audit](../../reports/shared-source-audit.md).
- The [previous full architecture comparison](../01_audit/final-architecture-comparison.md) is unchanged historical evidence. The common source/model parent is commit `71512a612961f3bd6d3e58a779e65dc8d2dec352`.
- [Preservation report](preservation.json), [checker amendments](audit-repair.json), [frozen protocol](protocol.json), [results](report.md).

The packet's W08 failure response applies: retain the candidate as an explicit conditional research tool, inspect its applicability errors, and establish corrected boundaries and transfer before integration. That remains the next integration gate.
