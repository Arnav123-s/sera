# Stage 40: verified step resolution

**Deployment decision: retain the independently admitted Stage 38 owner at `runs/sera-growth-live`.** This experiment and all its checkpoints remain available for research.

I continued the development-selected Stage 39 checkpoint for **2,048 decisions and 262,144 retained main practice presentations**. Every proposed update was tried at full size, then smaller fractions down to 1/256 when needed. Acceptance retained the original independent checker and additionally required strictly positive macro progress.

Across the two runs, **94 proposals were recovered by reducing their step size**. Every attempted scale and checker result is retained. This line-search algorithm is my engineering contribution; SERA learned the continued weights and action preferences through checked outcomes.

The first worker was externally interrupted after the 640-decision checkpoint. Its full 900-second reservation is charged, without an estimated refund. The next job restored that checkpoint. Counts describe retained completed decisions; any uncheckpointed fragment lost at interruption has unknown presentation count and remains covered by the full charged reservation.

## Frozen final comparison

The final used prospectively held-back groups from the preceding training pool. The parent had earlier exposure to these records. Stage 39 and the prospectively registered Stage 40 successor share this one reserved cohort; neither final informed another training or selection pass.

| Strand | Stage 38 correct | Selected successor correct | Cases | Score change |
|---|---:|---:|---:|---:|
| dictionary | 44 | 44 | 128 | +0.003255 |
| conversation | 19 | 18 | 128 | +0.002234 |
| grammar | 29 | 31 | 128 | +0.002986 |
| philosophy | 6 | 5 | 54 | +0.002302 |
| mathematics | 26 | 22 | 69 | -0.004588 |
| science | 48 | 40 | 128 | -0.012346 |
| reading | 101 | 102 | 128 | -0.000524 |
| en-US | 123 | 123 | 128 | -0.003933 |
| es-ES | 111 | 112 | 119 | -0.000607 |
| fr-FR | 116 | 116 | 119 | +0.005680 |
| de-DE | 113 | 114 | 119 | +0.004487 |
| methods | 24 | 24 | 128 | -0.000552 |

The selected candidate did not pass the predeclared admission gate. Mathematics and science retention account for material losses in this cohort; the model already in service is preserved. The table also retains improvements in other strands. These results concern this finite continuation, its representations and practice procedures.

## All candidates

| Candidate | Accepted decisions | Development score | Final score | Admitted |
|---|---:|---:|---:|---|
| 4001 | 72 | 0.459644 | 0.468468 | False |
| 4002 | 97 | 0.460004 | 0.468535 | False |

## Verification and preservation

[Independent audit](audit.json) · [Finite protocol](PROTOCOL.md) · [Frozen selection](selection.json) · [Full final](final.json) · [Costs and worker logs](costs.json) · [Compact checkpoint index](checkpoint-index.json).

Original local checkpoints remain untouched. Public tensor blobs are deduplicated, with exact semantic roundtrip checks for weights, optimizer state, RNG and complete event histories. Source identities and parent exposure are explicit. Failed updates, rejected candidates and every comparison remain preserved.
