# Learning progress and verified procedure credit

I trained a procedure-reward predictor on the actual continuing SERA owner, alongside continued practice in six curriculum strands. The selected successor passed the predefined acquisition/retention gate and was saved as the latest learner. Its method portfolio was independently requalified on **256/256 fresh mechanics cases**. Balanced practice remains the default; both learned-policy seeds, every control, failed promotion and full costs remain recorded.

## What changed in the learner

The existing book head receives dictionary, human conversation, mathematics and science prose. The existing sentence head receives human source questions. The existing mechanics head receives independently checked conditional method-selection examples. The four prose strands share one head, so this is six curriculum strands across three readouts. The inherited encoders and 190 other tensor records remain unchanged.

A new 337-parameter candidate scorer learns from realized gains. It chooses a strand and one of three supplied SGD rates. Each update receives signed progress credit from external labels, a bonus for exceeding the persistent best score, and an accuracy component. Previous accurate-alternative credit remains in the retained portfolio. Repeated easy success earns no new high-water bonus, and a forget/relearn cycle cannot produce net new discovery credit. Points grant neither permissions nor correctness.

I supplied the update menu, representations, reward, evaluation contracts and data partitions. SERA learned the reward-prediction weights and changes to the existing reading, book and method heads. It practiced ordering and rate selection; the three numerical step sizes were supplied options.

## Teaching and exposure

Four controllers, two seeds and two consecutive 48-update transitions produced **768 knowledge updates and 192 procedure-predictor updates**. Each controller received the same eight batches of eight examples per strand in each transition. That is 6,144 example presentations across the eight training runs: 5,120 attributed human-text presentations and 1,024 supplied conditional mechanics presentations. The selected learner received 96 knowledge updates and 96 procedure updates, with 768 example presentations.

Human prose and SQuAD annotations are reused from the established Stage 29/30 sources. Whole source groups separate LP-001 teaching, reward probes and future acquisition/evaluation cohorts. The starting learner had historical training/development exposure to those corpora; this experiment measures continued acquisition and procedure transfer within those retained families. It does not relabel them as entirely new human knowledge. The old sealed final datasets were not used. Exact group IDs, row IDs, cache hashes and wrapping counts are recoverable from the data manifest and decision traces.

The future tests contain **662 probe items** across two cohorts. The mechanics contexts are newly seeded; the programs and checker are supplied. Four-way old/new knowledge by old/new procedure comparisons, plus three controls from each knowledge state, produced **40 acquisition trials**, each with 24 matched updates and 192 example presentations. Eta stayed frozen in these trials. Initial knowledge at the second transition is the actual first successor.

## What the selected weights did

The development-selected checkpoint was `learned-3401`; its aggregate score passed the frozen gate on the future probes. The gate requires an aggregate gain and no per-strand score loss greater than .02. The score is the macro mean of 1/(1+cross-entropy), with human reading credit supporting every supplied correct sentence. It is a numerical practice measure, not a percentage of general understanding.

| Strand | First cohort loss, before → after | Second cohort loss, before → after | Correct items across both cohorts, before → after |
|---|---:|---:|---:|
| Dictionary next word | 5.99389 → 5.97845 | 5.98742 → 5.97887 | 19/107 → 19/107 |
| Conversation next word | 6.39677 → 6.38171 | 6.25325 → 6.23412 | 10/107 → 10/107 |
| Mathematics prose next word | 5.65395 → 5.64225 | 5.95291 → 5.95571 | 14/98 → 15/98 |
| Science prose next word | 6.85264 → 6.85426 | 7.26065 → 7.24973 | 11/94 → 11/94 |
| Human source-sentence selection | 0.44904 → 0.45018 | 0.80057 → 0.80086 | 105/128 → 105/128 |
| Best next mechanics method | 2.46187 → 2.30237 | 2.43611 → 2.27617 | 25/128 → 27/128 |

The mechanics selector's mean loss fell **6.52%**, and best-next-method accuracy rose **1.56 percentage points**. This is a different task from executing the verified portfolio: the selector predicts the best next method under weighted unmet requirements, while the qualified portfolio executes a valid route for a task. The latter passed all 256 new cases. Prose metrics include the inherited unknown-token class; the raw final records also report known-vocabulary accuracy and unknown-token rate. Mathematics prose prediction and exact mathematical problem solving retain separate labels.

| Final checkpoint | Mean future-probe score |
|---|---:|
| Balanced, either seed | .244573 |
| Random, seed 3401 / 3402 | .244831 / .245141 |
| Measured progress, either seed | .244573 |
| Learned, seed 3401 / 3402 | .246338 / .244320 |

Both learned seeds remain visible. The selected checkpoint was chosen before these final values were opened.

## Did the learning procedure improve?

The table averages acquisition gain over both seeds and both starting knowledge states. Every comparison has the same examples and 24 updates. Each individual old/new-K comparison remains in `final.json`.

| Frozen procedure | Transition 1 mean score gain | Transition 2 mean score gain |
|---|---:|---:|
| Old eta | .0001413 | .0002950 |
| New eta | .0002679 | .0002931 |
| Balanced | .0001370 | .0001166 |
| Random | .0003232 | .0001676 |
| Measured progress | .0001370 | .0001166 |

The new procedure improved acquisition over the old one at the first transition; their second-transition gains were essentially tied, with the old procedure slightly higher. Random practice had the largest first-transition gain. The two-transition promotion rule therefore retained **balanced** as the default. Learned policies remain saved for explicit experimental use. These results distinguish accumulated knowledge from improved learning procedures and preserve the strongest observed controls.

## Independent verification and continued use

NumPy independently recomputed **23,832 prediction scores**, with maximum cross-entropy discrepancy **1.185e-6** from PyTorch. The exact step-40 checkpoint reproduced step 48, including weights, optimizer, RNG, decision traces and persistent reward state. Serialization and restoration reproduced the full selected learner exactly.

The audit retained **190 inherited tensor records**, **128 four-language probes**, the existing exact polynomial route, 26 taught physical meanings and earlier empirical/source state. Sixty-four fresh mechanics executions matched independent targets. Original tasks, including the open momentum investigation, remain recorded. The novel-definition gate stays under its earlier source contract.

After this audit, the current owner received a new retention anchor and a separate portfolio assessment. All **256/256** cases passed; the global fourth assessment spent alpha .00125 and obtained an exact lower bound of .974226. The existing assessor ledger tracks attempts across all quests. This new qualification belongs to the updated owner; old qualification is not silently inherited after a weight change.

The full local regression passed **493 tests**. The qualified work example returned **4 m/s**. The first human reading demonstration selected an unrelated revenue sentence for a question about a country; its prediction and the supplied human supporting sentence remain in [the correction record](reading-correction-open.json). This error is an identified acquisition task; no weights were retuned on the completed evaluation.

[Use the saved owner](README.md) · [All final records](final.json) · [Audit](audit.json) · [Costs at seal](costs.json) · [Latest costs](publication/costs.json) · [Source and mechanism notes](RESEARCH_NOTES.md) · [Checkpoint inventory](checkpoint-inventory.json)
