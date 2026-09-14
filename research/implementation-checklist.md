# SERA connected learning implementation checklist

I use the architecture alignment audit as the acceptance baseline for this build. The target is the first connected R1/R2 research system with a learned improvement policy. The other six architectural families and unrestricted multimodal intelligence remain separate research programs; a completed engineering checklist does not establish those capabilities.

The checked items below describe the bounded 0.2 engineering milestone. The [audit against the original documents](../reports/original-source-alignment-audit.md) is the broader acceptance record and identifies additional gaps inside R1/R2 itself.

## Required work

- [x] P1. Store and load an executable solver version, including its neural checkpoint and admitted skills.
- [x] P2. Make ordinary inference use the current version automatically; test reload in another process.
- [x] P3. Support successive improvement rounds, cumulative evaluation accounting, rejection, and behavioral rollback.
- [x] E1. Make actual learning consume validated, serialized experience with source provenance and disjoint support/query records.
- [x] E2. Record acquisition, execution, verification, update, evaluation and wall-time costs, including failed proposals. Category counters and total wall time are retained; the report explicitly identifies incomplete phase timing/baseline counters and does not claim FLOP accounting.
- [x] R1. Connect retained recurrent state and action to next-observation and reward prediction.
- [x] R1. Plan through that same model, execute in the environment, admit actual feedback and update a candidate with replay.
- [x] R1. Evaluate longer trajectories, partially hidden observations and held-out dynamics with declared controls.
- [x] R2. Implement action-controlled event conditioning and a program proposal distribution from predictive state.
- [x] R2. Execute bounded typed programs, verify reusable skills, and learn from execution traces.
- [x] R2. Compare learned proposal search with fixed search under the same candidate-execution cap, with model-proposal costs reported separately.
- [x] M1. Train an improvement policy from measured outcomes of actual learning interventions, with disjoint development and evaluation episodes.
- [x] M2. Use that policy in the persistent solver and report its decision quality, cost and downstream retention against fixed controls.
- [x] V1. Run independent numerical and behavioral tests, including the lifecycle gaps missed by version 0.1.
- [x] V2. Run declared multi-seed studies from scratch, retain raw evidence, and document negative results.
- [x] D1. Produce a teaching record: tasks, available inputs, targets, examples, updates, outcomes, retention and limitations.
- [x] D2. Update architecture, CLI examples, audit disposition and reproducibility instructions.
- [x] D3. Build/install the package, verify a clean checkout, publish under Arnav123-s and check Windows/Linux CI. Both public jobs passed in [the release verification run](https://github.com/Arnav123-s/sera/actions/runs/34800392826).

## Completion rule

I mark an item complete only when implementation and its stated verification exist. I distinguish successful software behavior from successful learning. An experiment that fails to learn remains a measured negative result and does not justify claiming its capability was attained. All studies disclose supplied representations, action alphabets, interpreters and feedback access.

## Evidence

- The [connected report](../reports/connected-study.md) documents three independently trained seeds, 276 measured intervention outcomes, 15 persistent proposals and every admission/rejection. Two of six learned-policy generations were admitted; four were rejected.
- The [artifact audit](../reports/connected-verification.json) reproduces all 15 paired decisions, checks 1,920 unique meta-support identifiers and 48 unique query datasets, re-executes accepted goal skills and verifies final symbolic scores.
- The 0.2.0 test suite contained 36 passing cases, including fresh-process skill use and an independent full-density reference for the low-rank update. The 0.2.1 audit repair passes 40 cases, adding sealed program-credit and bounded-construction regression checks.
- A separately installed 0.2.0 wheel loads the accepted solver and executes its saved program. Continued CLI learning in a disposable prepared copy consumes the next round, retains the incumbent on rejection and archives its actual new support.
- The [audit disposition](../reports/architecture-audit-resolution.md) maps each original gap to code and evidence; the [release verification record](../reports/connected-release-verification.md) records packaging and publication checks.

The untrained reference preset, restricted replay coverage in the formal generational experiment, supplied symbolic representations, finite program grammar and small family holdout remain explicit limitations. These are future research acceptance criteria, not silently completed capability claims.

## Original-source audit follow-through

- [x] Recheck the original ZIP/PDF hashes, manuscript and exact graph identifier set.
- [x] Repair the program-credit admission bypass and excessive fixed candidate construction; compare valid behavior with 0.2.0.
- [x] Document the R2 classical-filter reduction and actual saved-library ablation.
- [ ] Implement validated live world-session ownership and save/resume.
- [ ] Use acquired libraries in automatic program search and test held-out composition.
- [ ] Evaluate a history-preserving R2 instrument against matched trained classical controls.
- [ ] Complete selective routing, full-reference training and persisted rank/error/cost diagnostics.
- [ ] Connect failure diagnosis, previous attempts, budget and targeted evidence to curriculum choice.
- [ ] Complete the missing teaching, adaptation, planning, retention and full-cost controls.
- [ ] Demonstrate improvements to the improvement policy across outer generations.

The [full open checklist](../reports/original-source-alignment-audit.md#open-acceptance-checklist) defines the evidence required for each item. These remain incomplete, including where syntax or a component API already exists.
