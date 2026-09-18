# Human books, definition learning and conditional execution

I completed a human-book curriculum, two definition-binding studies and the connection from a taught physical definition to the actual shared StudyR1 motion operator. The usable result is **guarded reuse of 26 taught physical entries**, with 300 numerical integration cases and 900 independently checked branches. The learned reader and the mathematical operator execute on one continuing owner.

## What was taught

Original source bodies came from Webster's Unabridged Dictionary; everyday DailyDialog conversations; Baskervill and Sewell's An English Grammar; Plato's Republic in Jowett's translation; Descartes' Discourse on Method in Veitch's translation; Thompson's Calculus Made Easy; and OpenStax Physics. [Identities and authors](sources.json), [attribution and license notes](ATTRIBUTION.md).

Complete books and dictionary entries are preserved locally. The bounded teaching run selected original windows from them; acquisition of a full book is not evidence that every page was trained or understood. Modern automatically generated Gutenberg catalogue summaries and new arXiv training text were excluded. Dictionary pronunciation, grammatical labels, usage and full senses remain in the original entries; the word tokenizer used a subset of this information.

| Domain | Training windows | Development | Final | Final source groups |
|---|---:|---:|---:|---:|
| Dictionary | 2,000 | 160 | 160 | 151 headwords |
| Everyday conversation | 2,000 | 160 | 160 | 145 dialogues |
| Grammar | 2,000 | 160 | 160 | 21 paragraph blocks |
| Philosophy | 2,000 | 160 | 160 | 17 paragraph blocks |
| Mathematics | 723 | 120 | 37 | 2 paragraph blocks |
| Science | 2,000 | 160 | 160 | 10 modules |

The 10,723 training windows contain 16 preceding words and one original next word each. The same two 2,048-word heads received 120 updates per domain, 720 updates each in total, in the listed order. Prior tensors stayed fixed. One readout used existing R1 ordered state; the control used pooled context. Vocabulary was selected only from teaching data. Checkpoints contain optimizer, order, cursor and RNG state.

## Frozen next-word results

Lower cross entropy is better; values are nats per target in the declared vocabulary, including its unknown bucket. These results do not compare SERA with a standard pretrained language model.

| Final domain | Unigram | Smoothed bigram | R1 ordered | Pooled | Unknown targets |
|---|---:|---:|---:|---:|---:|
| Dictionary | 4.623 | 5.071 | 5.771 | 6.292 | 33.75% |
| Conversation | 5.897 | 6.660 | 7.129 | 7.777 | 10.00% |
| Grammar | 5.223 | 6.030 | 6.249 | 6.734 | 22.50% |
| Philosophy | 4.980 | 5.476 | 5.693 | 6.264 | 18.75% |
| Mathematics | 6.018 | 7.247 | 7.762 | 7.596 | 10.81% |
| Science | 5.381 | 5.934 | 6.420 | 6.909 | 20.63% |

Development selected the ordered readout. It had lower final loss than pooled context in five of six domains, while unigram had lower loss than both trained heads in every domain. I preserve this head as experimental state. [Per-example predictions, covered-vocabulary accuracy and losses](final.json) include all controls; [development trajectories](selection.json) preserve sequential changes. The mathematics final has only 37 windows from two blocks, so it offers little evidence about broader mathematical prose.

## Definition-to-type learning and repair

The binding curriculum used 97 original human definition spans: 26 physical entries and 71 other senses/terms. Twenty-seven separate glossary terms formed development. The supplied label set was position, velocity, acceleration, force, mass and other. Full dictionary entries and XML/source hashes stay attached. No generated paraphrases were teaching targets.

GD-001 used definitions without their headwords. Its best development macro accuracy was 23.61%, and its gate rejected autonomous execution. I retained that study and froze GD-002: original term plus original definition, with the same finite comparison and a headword-only control. Development selected the learned lexical readout at update 160, with 38.06% macro accuracy. Its gate also remained closed. R1 and combined features were measured as alternatives; shared ownership alone is not evidence of beneficial transfer.

The final used 59 definitions from the first 60 modules of a separately pinned College Physics 2e source. Exact normalized definition duplicates with teaching/development were removed. All learned conditions below use the jointly saved update-160 checkpoint selected by development; the other heads were not reselected on final data. Both books are OpenStax publications; this is a new source/wording probe with residual editorial dependence, not an independent-author population estimate.

| Frozen final condition | Correct / 59 | Accuracy | Mean per-class accuracy |
|---|---:|---:|---:|
| Selected learned lexical head | 50 | 84.75% | 72.08% |
| Existing R1/semantic features | 36 | 61.02% | 54.17% |
| Combined features | 40 | 67.80% | 55.83% |
| Majority class | 40 | 67.80% | 16.67% |
| Supplied headword-only rule | 49 | 83.05% | 88.33% |

There were 40 other definitions and 19 physical definitions. The learned model scored 13/19 physical and 37/40 other correctly. The headword rule demonstrates substantial nomenclature information and had higher macro accuracy. These counts support a narrowly measured classification result; they do not establish unrestricted conceptual understanding. [All predictions and controls](named-final.json). The final did not change weights, selection or thresholds, and zero novel definitions were admitted for autonomous execution.

## Supported executable connection

For a known taught source, the runtime checks exact term/definition/source identity and agreement between the learned type and the human teaching label. Compatible SI units and explicit physical assumptions are then required. **All 26 physical teaching entries passed this reuse check**. Source matching is retained knowledge reuse; it is not scored as learned abstraction.

The admitted type determines where its value enters the actual StudyR1 motion calculation. Position and velocity set initial conditions; acceleration supplies the constant acceleration; force or mass uses the explicitly supplied Newtonian F/m relation. The retained learned integral operator produces the trajectory, and a separately implemented rational formula checks position and velocity. Opposite-drive and doubled-mass/half-acceleration branches do not modify observations or weights.

The [runnable example](example-request.json) supplies 6 N net force and 3 kg mass from rest for two seconds. The checked result is 4 m and 4 m/s; reversing force yields −4 m, while doubling mass yields 2 m. [Complete result and certificates](example-result.json).

The independent audit passed **300/300 cases and 900/900 branches** across all five quantity types. Numerical conditions are generated diagnostics, separately labeled from human language teaching. One markup-only unrelated glossary headword was correctly rejected by the runtime; its audit-accounting repair is preserved under [GD-F01](failures/GD-F01/DIAGNOSIS.md).

Unfamiliar meanings remain open and can invoke the existing arXiv acquisition interface while retaining the original task. Retrieved sources carry identities and dates and remain references pending independent verification. They do not silently update the human-only foundation weights. [Live route receipt](arxiv-live.json).

## Retention, ownership and reproducibility

- 172 predecessor tensors remained byte-exact.
- All 128 earlier English, Spanish, French and German probe outputs were retained.
- Earlier source-reading results, exact polynomial algebra and empirical forecasts were retained.
- The book checkpoint replay from update 80 to 120 and binding replay from 80 to 160 were exact.
- Snapshot restoration, invalid premises, nonphysical senses, source interruption and tamper rejection passed the targeted tests.
- Teaching, development and final book groups and exact windows were disjoint; final definition overlaps were excluded.
- No imagined branch was inserted as a physical observation.

[Independent audit](integration-audit.json) · [Taught-source results](teaching-reuse-audit.json) · [21 targeted tests](verification.json) · [Costs including failed work](costs.json) · [Complete local artifact index](local-artifacts.json).

The complete local suite subsequently passed **463 tests with no skips** in 308.77 seconds. One persistence test was then made independently runnable by creating its own concept record; it passed in isolation. Production code and all scientific evaluation results were unchanged by that test-fixture correction. The exact published revision receives the full Linux/Windows suite in CI.

I supplied preprocessing, class labels, units, the finite curriculum, source admission and independent checkers. SERA learned the new readout weights, reused its earlier learned mathematical operators and retained supported source associations. The new frozen heads do not themselves demonstrate independently improved learning procedures. I keep this distinction explicit when deciding the next experiment.
