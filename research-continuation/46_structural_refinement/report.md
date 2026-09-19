# Acquiring and qualifying a physical model on the continuing owner

I integrated v16 C01/C02 into the independently qualified Stage 45 learner. SERA now acquires executable force coefficients, investigates whether its initial symmetry assumption fits the evidence, compares imagined controls and returns to the original prediction task. The final experiment and independent actual-owner replay passed. Release regression and publication checks have their own receipts.

## What SERA learned

The learner received typed position, velocity, input and acceleration observations. It fitted two candidate descriptions: a radial restoring force and a direction-dependent restoring force, each with nonlinear potential, damping and rotational velocity terms. Its coefficients were acquired from observations and registered on the actual R1 owner. The existing learned integral supplied the verified quadratic and quartic potential factors.

The numerical procedure, candidate representations, units, coordinates, observation interface and checker are supplied engineering. The investigation controllers are fixed. This study measures acquisition of physical coefficients and evidence-qualified representation selection. The paper mechanisms reviewed in the research notes remain separately identified research leads.

## A completed original task

For `directional-46101-disagreement`, the saved request was acceleration at position `(0.3, 0.7)` m, velocity `(-0.1, 0.2)` m/s and input acceleration `(0.1, 0)` m/s².

| Answer | Predicted acceleration, m/s² |
|---|---|
| Initial radial model, after 12 observations | `(-0.200666, -0.786473)` |
| Acquired directional model, after 12 further probes and independent assessment | `(-0.446365, -1.858206)` |
| Independent simulator assessment | `(-0.448836, -1.855095)` |

The original-task MSE fell from **0.601771 to 0.000007891**. The acquired stiffness coefficients were approximately `Kxx=1.0590`, `Kyy=2.4431` and `Kxy=0.3670`: direction matters, and the two axes interact. Those numbers were fitted from observations. The richer family was a supplied candidate. The original request, initial answer, chosen probes, supporting observations and final answer remain together in the saved record.

Across all twelve directional worlds, the disagreement procedure reduced mean original-task MSE from **0.593329 to 0.000003597**. On the twelve radial worlds it improved from 0.000010187 to 0.000002293. The balanced comparison reached 0.000006663 and 0.000004452, respectively. These are saved original goals, assessed after acquisition; the [goal-return analysis](analysis.json) aggregates existing records without further fitting.

## Frozen final evaluation

The final cohort contains **36 worlds**, organized as twelve seed triplets across radial, directional and omitted-mechanism families. Each world was investigated with two procedures, making **72 paired investigations**. Each procedure used 24 fitting pairs, 16 separate representation-selection pairs and 32 further adequacy pairs: **5,184 observation-pair exposures** in total. The paired controls reuse the same teacher and some observations; they are not 72 independent worlds.

No final target was used for fitting, selection or adequacy. There were 256 final states per world/procedure, **18,432 prediction exposures** over 9,216 distinct world/query pairs. Wider states, trajectories and the original task were assessed separately. [Frozen protocol](PROTOCOL.md) · [Source freeze](freeze.json) · [Final measurements](final.json).

| Teacher family | Disagreement-driven qualification | Selected model | Acceleration MSE |
|---|---:|---|---:|
| Radial | 12/12 accepted | Radial in all 12 | 0.000005411 |
| Directional | 12/12 accepted | Directional in all 12 | 0.000010887 |
| Omitted mechanism | 12/12 retained as open investigations | Both alternatives preserved | 0.123702, unqualified |

The balanced procedure made the same 36 admission decisions. Its respective MSEs were 0.000007028, 0.000011221 and 0.125361. Both fixed procedures worked in this experiment; the small error differences are reported without a broad scheduler-superiority claim.

## Comparison models and diagnosis

All entries below are mean acceleration MSE, in m²/s⁴, over the same fresh states. The always-directional control uses the same 24 acquisition pairs. The extra-data radial and current-R1 readout controls receive all 72 observed pairs. The readout uses 263 features from the actual inherited sequence encoder and recurrent memory, with a fitted ridge head. It receives no final query target.

| Model or procedure | Radial worlds | Directional worlds | Omitted worlds |
|---|---:|---:|---:|
| Initial radial, frozen after 12 pairs | 0.000078972 | 0.806441 | 0.828395 |
| Radial with extra data | 0.000002885 | 0.452007 | 0.485384 |
| Always-directional independent fit | 0.000008563 | 0.000010887 | 0.123702 |
| Current R1 features and fitted readout | 0.101294 | 0.094217 | 0.287234 |
| Owner-connected qualified selection | 0.000005411 | 0.000010887 | 0.123702, unqualified |

The directional refinement matched strong classical least squares. More data helped the restricted radial model but left its directional error much larger. On radial worlds, the extra-data simple model was the strongest mean-MSE control. These comparisons support the particular acquired representation and its qualification contract; ownership and retention are assessed separately.

The omitted family included influences outside both candidate descriptions. The residual adequacy check kept every such case open, with its original goal and competing conditional predictions preserved. Parameter agreement inside the available family did not qualify that family as complete. All failed alternatives and their costs remain in the archive.

## What-if prediction, planning and explanation

`field_what_if` returns conditional acceleration with model identity, evidence version, units and assumptions. `field_trajectory` keeps one fitted mechanism throughout a branch. `field_plan` compares five finite constant-input branches against a target. `field_explain` reports the candidate comparison, original answer and independent evidence. Planning returns a proposed action; it executes no external control.

In every qualified radial and directional world, both procedures selected the independently best of those five controls: **24/24 for each procedure**, with zero regret relative to that finite choice set. The choices include four axis-aligned inputs and zero input. The twelve omitted-mechanism worlds retained conditional branches without admitting a qualified plan. [Complete per-world planning records](analysis.json).

Across the disagreement procedure's one-second trajectories, mean position RMSE was **0.217 mm** for radial worlds and **0.255 mm** for directional worlds. Mean velocity RMSE was **0.511 mm/s** and **0.603 mm/s**, respectively. The unqualified omitted-family trajectories had 39.2 mm position RMSE and 90.7 mm/s velocity RMSE. Every trajectory is retained. An independent adaptive ODE solver provided the reference.

Wider-state acceleration MSEs were 0.0001166, 0.0001308 and 0.316340. These predictions retain their extrapolation status. Continuous energy balance and input derivatives have separate tests; numerical energy behavior is not used as a substitute for trajectory accuracy.

## Reconciliation, evidence and preservation

The starting public pin was `20e836e702102cf531d641c2142e578e76e2f65d`. Newer local Stage 45 work was completed and qualified before this integration. Its exact owner, all 855 tensor records, 217 route identities, 29 practical outputs, 26 taught physical meanings and source gate were pinned in [C01 reconciliation](LOCAL_RECONCILIATION.json).

The [independent audit](audit.json) replayed all 72 investigations and retained every inherited tensor, all 217 routes, all 29 practical outputs, all 26 taught meanings and the source gate. It checked duplicate rejection, read-only imagination, stale-view rejection and exact restoration. A deliberate interruption after a committed probe resumed the remaining 60 observations exactly, without repeating the first twelve. The saved current owner is `376b3d076294ac5b92f9360bb27553edbd96d1aa2b12ae22913b55c0b135fa94` in `runs/sera-structural-live`.

Across all 72 retained subjects and both candidates, the extension adds **720 learned coefficients**. Parameters and sufficient-statistic buffers add **44,928 bytes**, bringing complete owned tensor storage to **12,470,552 bytes**. R1's recurrent working state remains **32,768 bytes**. The full evidence, source and revision archive is separately preserved; these tensor counts do not stand in for total research storage.

The v16 packet's 43 tests passed. Its exact dataset regeneration check exposed last-bit differences on Windows. I preserved that strict failure, separately checked all regenerated arrays and replayed the unchanged scientific checks from the original archived inputs. All 54 saved models, 480 refinement episodes, 288 propagation records and 18 compiled models passed numerical replay. No packet model was retrained. [Verification scope](packet-verification.json) · [Primary research and decisions](RESEARCH_NOTES.md).

The original and repaired Stage 45 studies remain separate. This stage adds its own development/final records, source snapshots, comparison weights, observation traces and append-only owner revisions. The local runner keeps one CPU thread, the 2 GiB process-tree cap, full elapsed-cost accounting and a scoped Windows keep-awake request. The final physical study used **1,299.114 supervised seconds**. The [complete cost receipt](publication/costs.json) includes development, failed packet verification, independent replay, examples and verification; the earlier study has its separate receipt.

The full local suite passed **593 tests**, lint, all seven existing release verifiers and the base command-line check. All **33 current-owner example requests** completed. [Local check receipt](publication/local-checks.json) · [Engineering](ARCHITECTURE.md) · [Use the learner](README.md) · [Completion checklist](checklist.md).
