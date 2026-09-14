# SERA 0.3 study protocol

I use the original architecture handbook, its source chapters and diagrams, the physics references and the supplemental project context as the design baseline. The research objective is failure-driven acquisition, retention and improvement of the learning procedure. This protocol develops the recommended R1/R2 starting architecture. It does not equate all eight architecture proposals with eight modules in one model.

The executable configuration is `StageConfig` in `src/sera/stage_three.py`. The run manifest records its values, seeds, runtime versions, source hash and the original archive hash before training. I keep aborted and development runs separate. Final runs use three independent initialization/data seeds. All models train locally on CPU from random initialization; no pretrained language model or external teacher supplies labels.

## Development exposure

I used an aliased-history pilot and an equal eight-second learning-rate search over 0.001, 0.003 and 0.015 for each of five predictors. Validation negative log likelihood selected 0.015 for all five. A separate typed-task pilot exposed weak arithmetic and binding. I then added a supplied, finite arithmetic grammar whose selected procedure is induced from admitted examples and checked on separate validation examples. The final structural draws differ from these pilots; the repeated-action family itself was development-visible. I claim withheld draws and exclusion from gradient training, not a blind discovery of that family.

The first integration smoke run exercised all stages, then deliberately failed its source-freeze check because development edits continued. A later interrupted run was stopped after discovering that the saved solver's legacy sequence decoder also needed fresh training. The next cohort exposed an empty-target likelihood average in a fully masked batch. I repaired it, added a no-invented-target regression and reexecuted the formerly failing episode. All earlier runs remain development costs. The release cohort starts fresh after these repairs.

The published cohort uses `scripts/parallel_stage_three.py`: three independent seeded processes, one Torch thread each, sharing the same host. Per-model budgets and task configuration match the serial study. Whole-cohort wall time includes overlap; CPU time sums worker process CPU. Reported peak RSS is the largest individual worker high-water mark, not simultaneous aggregate memory. Timings include host contention and do not imply exclusive cores or identical FLOPs.

## Teaching and controls

| Experiment | Teaching | Evaluation and controls |
|---|---|---|
| Original source reproduction | Original reviewed 12-core implementation, three seeds, 500 steps, four training tasks and adaptation | 45 original math/gradient checks; 36 supplied checkpoint rechecks; 36 independent fresh training runs; separate from new SERA results |
| Legacy sequence decoder | Five supplied symbolic tasks, 500 steps, generated batches | Existing independent evaluator; decoder saved in the same executable solver |
| Aliased event history | 256 length-12 trajectories; observations alias eight hidden states into four labels | Projective instrument, general real/complex instruments, trained HMM and GRU; same 24-second training cap including validation; likelihood parameter counts reported separately from unused proposal heads |
| Typed learning | 192 examples per task across seven tasks; 48 validation examples per task; 1,400 joint updates | 128 examples per task/family; ordinary, extended and structural draws; no-program and acquired-program conditions |
| World prediction | 256 length-8 trajectories with 30% missing sensors; 360 updates | Missing sensors, longer trajectories, memory reset; reactive, learned-model and oracle planning |
| Reference R1 | Full 256/128/8x32x32/4x16x4 dimensions; 120 updates per routing mode | All branches versus hard top-1 delivery; frozen learned weights evaluated at ranks 2, 4 and 8; retained bytes, actual executed rows, time and discarded mass |
| Adaptation | 8, 32 or 128 new support trajectories; 32 updates | No update, full update, adapter, replay and scratch; old/new prediction, NLL and Brier score |
| Program composition | Verified goal programs and whole two-action transformation demonstrations | Previously unseen whole transformations; acquired calls versus no library, fixed versus general-instrument versus HMM scoring; execution and construction caps |
| Outer policy | 12 measured training episodes, three validation episodes, eight available methods | Three actual parameter updates with cumulative training; one fixed four-world anchor and two additional frontier worlds per generation; fixed, uniform, difficulty and diagnosis controls |
| Persistent improvement | Three successive real learning attempts with a saved controller and cumulative replay | Fresh paired admission, rejected attempts, worst-world retention, typed mutation screening, development-selected archive retrieval and a subsequent lineage proposal |

The numeric/text/image/audio tasks use tiny supplied representations. Spatial relations use meters or centimeters; motion predicts two coordinates in meters; byte arithmetic uses short decimal strings; images are 4x4 patches; audio uses 16 samples. These tests establish functioning learned adapters and finite procedures. They are not natural-language, general vision or speech evaluations.

## Interpretation rules

I report ordinary accuracy, NLL, Brier score, motion MSE, per-seed dispersion, support sizes, actual executed outputs and retention. The scalar motion score `exp(-MSE)` only serves the declared mixed-task validation objective; it is not a classification accuracy. The arithmetic grammar, parser, task vocabulary and feedback access are engineered. Selecting `sum modulo four` from 21 candidates is procedure induction within that grammar, not spontaneous invention of arithmetic.

The general event instrument uses action maps and several Kraus operators per observed event, sharing likelihood and conditioning. The projective control remains classically reducible. Real/complex comparison counts complex entries as two real parameters. The QR implementation follows [PyTorch 2.10's reduced QR contract](https://docs.pytorch.org/docs/2.10/generated/torch.linalg.qr.html); the source handbook supplies the instrument construction and physical invariants. General measurement can change the retained state, as described in [IBM's measurement overview](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/general-measurements/introduction).

Rank sweeps reuse rank-four-trained weights. Changing rank also changes the initial factor; the sweep is a sensitivity experiment, not an isolated error bound on a fixed full-density trajectory. Discarded mass describes one normalized spectral truncation. Hard routing uses a declared straight-through surrogate gradient. The R1 planner uses most-probable imagined observations; it is not an exact belief-state planner.

Program comparisons share eight executed candidates and a 128 environment-action cap, with at most three program tokens and 128 constructed candidates. A library call can expand to several primitive actions. Any reuse benefit therefore applies to this token-budget interface; it does not establish a benefit at identical primitive sequence length. Scoring work and acquisition of the library remain charged.

Outer updates are actually trained and saved. All three policy checkpoints freeze before anchor/frontier scores are calculated; those outcomes never enter policy fitting. The inner solver stays fixed for this controlled policy comparison. Separate persistent rounds change task parameters and libraries. Positive acceleration would require robust improvement after counting the entire development and deployment cost; a new policy hash alone does not establish it.

## Costs and trust boundary

Each invocation and phase records wall time, process CPU time and the OS process RSS high-water mark. Phase records include failures. Operation categories cover acquisition, learning, planning, execution, verification and evaluation; byte counters remain separate from the heterogeneous operation-sum admission proxy. Meta-episode files retain baseline evaluation and each counterfactual's construction/evaluation work. Source reproduction, pilots, final runs and verification are separately identified. Process RSS is cumulative, not an isolated phase allocation. External research assistance, human labor, energy and unobserved service costs are not numerically measured, so I do not claim a complete research-efficiency denominator.

Externally interrupted development processes did not always flush their final wall/CPU accounting. The development record marks those costs unavailable, rather than assigning zero or combining a guessed duration with measured study costs. The normally failed empty-target run did flush its phase/failure record. The completed final cohort has its own full invocation accounting.

Admission uses fresh paired examples, cumulative error spending, empirical per-world retention and a finite cost cap. Frozen registered components and a bounded interpreter are the proposal boundary. The local journal and split checks do not provide hostile-process isolation or statistical retention certification. Scientific uncertainty and rejected improvements remain in the final report.
