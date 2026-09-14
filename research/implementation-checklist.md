# SERA connected learning implementation checklist

I use the architecture alignment audit as the acceptance baseline for this build. The target is the first connected R1/R2 research system with a learned improvement policy. The other six architectural families and unrestricted multimodal intelligence remain separate research programs; a completed engineering checklist does not establish those capabilities.

## Required work

- [x] P1. Store and load an executable solver version, including its neural checkpoint and admitted skills.
- [x] P2. Make ordinary inference use the current version automatically; test reload in another process.
- [x] P3. Support successive improvement rounds, cumulative evaluation accounting, rejection, and behavioral rollback.
- [x] E1. Make actual learning consume validated, serialized experience with source provenance and disjoint support/query records.
- [ ] E2. Record acquisition, execution, verification, update, evaluation and wall-time costs, including failed proposals.
- [ ] R1. Connect retained recurrent state and action to next-observation and reward prediction.
- [ ] R1. Plan through that same model, execute in the environment, admit actual feedback and update a candidate with replay.
- [ ] R1. Evaluate longer trajectories, partially hidden observations and held-out dynamics with declared controls.
- [ ] R2. Implement action-controlled event conditioning and a program proposal distribution from predictive state.
- [ ] R2. Execute bounded typed programs, verify reusable skills, and learn from execution traces.
- [ ] R2. Compare learned proposal search with fixed search under the same execution budget.
- [ ] M1. Train an improvement policy from measured outcomes of actual learning interventions, with disjoint development and evaluation episodes.
- [ ] M2. Use that policy in the persistent solver and report its decision quality, cost and downstream retention against fixed controls.
- [ ] V1. Run independent numerical and behavioral tests, including the lifecycle gaps missed by version 0.1.
- [ ] V2. Run declared multi-seed studies from scratch, retain raw evidence, and document negative results.
- [ ] D1. Produce a teaching record: tasks, available inputs, targets, examples, updates, outcomes, retention and limitations.
- [ ] D2. Update architecture, CLI examples, audit disposition and reproducibility instructions.
- [ ] D3. Build/install the package, verify a clean checkout, publish under Arnav123-s and check Windows/Linux CI.

## Completion rule

I mark an item complete only when implementation and its stated verification exist. I distinguish successful software behavior from successful learning. An experiment that fails to learn remains a measured negative result and does not justify claiming its capability was attained. All studies disclose supplied representations, action alphabets, interpreters and feedback access.
