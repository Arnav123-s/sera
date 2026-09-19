# One complete investigation

This is the first predeclared pulse-loss world, `pulse_loss-47101-information`. The explanation below is my account of the saved numerical trace; the numbers come from SERA's acquired weights and independent assessor.

The original question was: **after a 2.5-second history with an applied pulse halfway through, what is the probability of a plus-X readout, and which retained explanation accounts for it?**

## 1. A curve leaves several explanations

After eight passive observations, SERA estimated a total decay rate of **0.465402 per second**. It retained a continuous family of splits between reversible static spreading and irreversible loss. Pulse-induced loss was still unidentified.

Three illustrative members of that family, assuming no extra pulse loss, predicted very different outcomes for the original intervention:

| Candidate | Original-question probability |
| --- | ---: |
| All fitted passive loss assigned to irreversible rate | 0.656195 |
| Equal split between irreversible and static rates | 0.779459 |
| All fitted passive loss assigned to static spreading | 1.000000 |

These are conditional examples of an unresolved class, not three established physical facts or a complete uncertainty interval. The recorded initial point prediction was 0.779459; the intervention question remained open.

## 2. Choose and check an intervention

The information policy committed twelve further controls before requesting outcomes. It selected histories including:

- Eight quarter-second ticks with pulses at boundaries 2, 4 and 6.
- Sixteen ticks with a pulse at boundary 8.
- Eight ticks with pulses at boundaries 2, 3, 5 and 6.

Repeated controls acquired new independent shots. They were not rewarded as new discoveries. Each control had 1,024 binary observations and a separate applied-pulse record. In this world every requested pulse was logged as applied. The unreliable-pulse worlds separately exercised failed commands.

## 3. Learn and compare the two installed descriptions

Counts taught these richer-model coefficients:

| Coefficient | Learned value |
| --- | ---: |
| Irreversible rate per second | 0.177887 |
| Static detuning width per second | 0.287324 |
| Extra attenuation per applied pulse | 0.076734 |

On separate selection evidence, negative log likelihood fell from **0.606274** for the two-rate model to **0.603969** for the pulse-loss model. That exceeded the predeclared improvement threshold. The later independent adequacy set also passed its finite diagnostic gate. These outcomes qualified the richer model for the assessed conditional predictions.

The model forms and controller were supplied. SERA learned the coefficients and their supported use from the observations; it was not given these fitted values or the final expectation.

## 4. Return to the unchanged question

| Quantity | Plus-X probability |
| --- | ---: |
| Initial point prediction | 0.779459 |
| Returned prediction from acquired weights | **0.796829** |
| Independent expectation, revealed only for final assessment | **0.795947** |

The returned probability is about **0.088 percentage points** from the independent expectation. The answer retains its original goal, source, pulse assumptions, alternative description, model identity and adequacy evidence. It is a qualified prediction for the simulated apparatus, not a proof that no other physical explanation exists.

Run this saved investigation through `intervention_explain` in the [39-task example batch](../example-tasks.json). The complete record is preserved in the release archive at `runs/IU-study-001/final/completed/pulse_loss-47101-information.json`. [Full cohort and all controls](../report.md) · [Independent audit](../audit.json).
