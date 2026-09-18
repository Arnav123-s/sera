# Stage 39: sustained refinement

**Deployment decision: retain the independently admitted Stage 38 owner at `runs/sera-growth-live`.** This experiment and all its checkpoints remain available for research.

I ran **8,192 decisions and 1,048,576 main practice presentations** across twelve retained strands. The supplied alternatives were SGD and Adam, with independent progress rewards, interleaved rehearsal and an original-development-accuracy floor. The learned controller chose primary strands and update procedures. Balanced Adam received the same minibatch budget.

Development selected `balanced_adam-3902`. Long rejection plateaus and cumulative small accepted losses were visible before final access. The prospective step-resolution successor was registered while the final cohort was still closed; it preserves these runs unchanged.

## Frozen final comparison

The final used prospectively held-back groups from the preceding training pool. The parent had earlier exposure to these records. Stage 39 and the prospectively registered Stage 40 successor share this one reserved cohort; neither final informed another training or selection pass.

| Strand | Stage 38 correct | Selected successor correct | Cases | Score change |
|---|---:|---:|---:|---:|
| dictionary | 44 | 44 | 128 | +0.002314 |
| conversation | 19 | 17 | 128 | +0.002012 |
| grammar | 29 | 29 | 128 | +0.002495 |
| philosophy | 6 | 5 | 54 | +0.003260 |
| mathematics | 26 | 22 | 69 | -0.002532 |
| science | 48 | 38 | 128 | -0.010420 |
| reading | 101 | 102 | 128 | -0.000593 |
| en-US | 123 | 122 | 128 | -0.011693 |
| es-ES | 111 | 112 | 119 | +0.002552 |
| fr-FR | 116 | 115 | 119 | +0.007556 |
| de-DE | 113 | 114 | 119 | +0.006854 |
| methods | 24 | 24 | 128 | -0.000101 |

The selected candidate did not pass the predeclared admission gate. Mathematics and science retention account for material losses in this cohort; the model already in service is preserved. The table also retains improvements in other strands. These results concern this finite continuation, its representations and practice procedures.

## All candidates

| Candidate | Accepted decisions | Development score | Final score | Admitted |
|---|---:|---:|---:|---|
| balanced_adam-3901 | 4 | 0.457309 | 0.469154 | False |
| autonomous-3901 | 81 | 0.456460 | 0.466943 | False |
| balanced_adam-3902 | 10 | 0.458850 | 0.468811 | False |
| autonomous-3902 | 1459 | 0.452032 | 0.458810 | False |

## Future acquisition procedure

The separate future comparison crosses old/new knowledge with old/new procedure and balanced Adam at both seeds, 96 decisions per trial. Its twelve trials add 147,456 practice presentations. The full table is in [procedure.json](procedure.json). Results support retaining the fixed control as default; individual positive comparisons remain in the record. Knowledge accumulation, procedure learning and robust improvement of the procedure are scored separately.

## Verification and preservation

[Independent audit](audit.json) · [Finite protocol](PROTOCOL.md) · [Frozen selection](selection.json) · [Full final](final.json) · [Costs and worker logs](costs.json) · [Compact checkpoint index](checkpoint-index.json).

Original local checkpoints remain untouched. Public tensor blobs are deduplicated, with exact semantic roundtrip checks for weights, optimizer state, RNG and complete event histories. Source identities and parent exposure are explicit. Failed updates, rejected candidates and every comparison remain preserved.
