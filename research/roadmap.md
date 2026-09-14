# SERA research roadmap

My long-term target is a learner that can acquire unfamiliar capabilities, transfer them, retain earlier competence and improve the methods it uses to learn. With release 0.1, I establish a measured experimental platform for this question. I do not interpret this first release as evidence of general intelligence or recursive acceleration.

## Completed first foundation

From-scratch neural training; selective binding and temporal tasks; interchangeable classical and quantum-inspired cores; stateful sessions and serialization; finite action prediction and planning; active program acquisition; continual-learning controls; fresh paired promotion; evidence ledger; rollback; repeatable local experiments and cross-platform CI configuration.

## Next experiment: isolate the hybrid's benefit

The first study includes a parameter-count control. Next, compare the current hybrid to delta+rotor without density, delta+density without rotor, and the same hybrid with phase disabled. Keep the training data, initialization rules, parameter counts and tuning budget declared. Add long-range binding, ordered transformations and interference-sensitive synthetic tasks with meaningful real-coordinate controls. Five seeds and confidence intervals should replace rankings based on three seed means when making a design decision.

**Acceptance:** a complementary benefit survives parameter and compute controls on withheld task generators and longer sequences. **Failure response:** remove unhelpful branches or route them only to the domain where a benefit is measured. A negative result is preserved.

## Next capability: learn when to acquire a program

Replace the fixed `ordered_control` router and diagnostic threshold with a learned controller. Construct support/query episodes where replay, new observations, program search and additional planning have different measured payoffs. Train a small controller from those payoffs. Hold out task families and use a fixed-budget uniform policy as a control.

**Acceptance:** better query improvement per total acquisition/update cost, with worst-task retention at least as good as the fixed controller. Report failure categories and rejected proposals. The first controller must not train on its own final evaluation pool.

## Remove privileged observable state IDs

Begin with deterministic finite worlds whose states are rendered as noisy feature vectors. The learner must infer a stable abstraction from observations. Then introduce perceptual aliasing and partial observability. Evaluate multiple independently generated worlds and withheld rendering conventions.

**Acceptance:** long-horizon prediction and plan execution succeed under new observation noise, while calibration detects aliasing. Compare to an exact observable-state upper bound and a conventional recurrent belief model. State-ID access must be excluded from learner inputs.

## Acquire and compose richer programs

Expand the finite interpreter to a typed DSL with bounded map, fold, arithmetic and binding operations. Synthesize from examples and verified simulator feedback. Learn reusable library abstractions only after fresh tests; retain interpreter fuel, shape/type bounds and provenance.

**Acceptance:** previously acquired abstractions reduce search cost on withheld compositions. Compare no library, fixed hand-written library and acquired library under the same feedback budget. No arbitrary generated host-language code is needed for this stage.

## Broader continual learning

Add task streams with overlapping and conflicting rules. Compare selective replay, adapters, parameter isolation and explicit program storage. Use several support sizes and count novel draws, replay draws, optimizer work, library growth and retrieval cost separately.

**Acceptance:** new-task gains survive strict per-task retention and calibration tests across a sequence of updates. A single successful related-task adaptation is not the milestone.

## Perception and language

Introduce byte-level text and small image patches only after symbolic composition is reliable. Keep units, position, scale and target masks explicit. Train small encoders from scratch as the primary condition; any pretrained teacher or encoder becomes a separately labeled condition with its external data and compute disclosed.

**Acceptance:** transfer across task descriptions and withheld compositions with enough controls to distinguish surface memorization from learned procedure. A chat interface is useful only when it exposes actual model capabilities.

## R3-R8 branches

Gauge/graph reasoning is motivated for geometry and relational tasks; tensor compression for controlled correlated state; equilibrium learning for iterative constraints; population search for bounded diverse discovery; parameter clouds for small optimizer spaces; joint registers for exact quantum-data experiments. Implement one contract and one decisive control per branch before combining them. The inherited 36,900 schematic configurations are an index, not a training agenda.

## Research-efficiency milestone

Only after the learned controller works should candidate policies modify an update procedure. Across multiple generations, retain a fixed evaluation anchor plus an expanding frontier, external evaluation ownership, total compute accounting and rollback. Measure improvement divided by the full cost of producing each next useful version, including failed candidates.

**Acceptance:** a changed improver produces better subsequent learners at equal budget on genuinely withheld families, with old capabilities retained. One successful program acquisition or one code change does not establish this.

## Local execution envelope

The initial workstation has a GTX 1650 Ti with 4 GB VRAM. Release experiments use CPU and one PyTorch thread per run. Larger experiments should start from measured memory/runtime estimates and explicit run budgets. Cloud GPU spending, external services, and open-ended unattended jobs are not configured in this release.

Every research run should end as completed, rejected, failed or unresolved with its cause and resumable artifacts. Unchanged repeated attempts do not count as additional evidence.
