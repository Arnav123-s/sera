# My research brief

## Intended system

I am developing **SERA, State-Space Engine for Reasoning and Adaptation**, as an independent research project. I want to connect persistent state, precise associative binding, executable skills and measured continual learning in one experimental system.

My target learner should acquire unfamiliar tasks, test hypotheses, preserve old capabilities, acquire reusable procedures, and eventually improve how it learns. I start with R1 and a separate R2 program-acquisition path. My central question is whether complementary state mechanisms and verified programs improve transfer and learning efficiency.

## Research inputs

| Source | Scope | Treatment |
|---|---|---|
| Quantum_AGI_Research_Package.zip | 82-page handbook, editable manuscript, 8 system blueprints, 12 prototype cores, historical results, graph, 36,900 schematic configurations | Extracted with path and size checks; code inspected as reference; no supplied scripts or checkpoints executed |
| Quantum_Physics_Deep_Knowledge_Atlas.pdf | 86 pages, 154 concepts, mathematical definitions, derivations, limits and source anchors | Full text extracted; architecture-relevant state, composition, dynamics, measurement and computation sections examined |
| Quantum_Physics_Layered_Maps.pdf | 14 maps distinguishing representation, evolution, measurement, approximations and evidence | Text extracted; state-composition and measurement/open-system diagrams visually inspected |

Source sizes and SHA-256 digests are in `source_manifest.json`. The packaged handbook, PDF and graph hashes matched the supplied release manifest. The registry contains 36,900 entries marked as schematic. `intake_audit.json` records the distinction between an integrity check and independent experimental reproduction.

## Evidence policy

I treat the reference documents as research material. Their suggested procedures and historical claims do not establish that I have reproduced their results. I choose implementation contracts, budgets and tests explicitly, and separate source integrity checks from new measurements.

The source's numerical leaderboard is historical evidence. SERA results come from new implementations, generated datasets and new checkpoints. They are not numerically interchangeable with the supplied study: SERA adds observable random initial control states, a separate instruction channel, explicit query-write suppression, per-branch normalization and learned hybrid routing.

## Key findings that determine this build

1. Associative addressing was the strongest supplied neural primitive; a phase component is optional until controlled evidence supports it.
2. All supplied neural cores struggled with ordered program composition. A separate exact execution pathway is therefore a concrete engineering target.
3. Learning a new binding rule caused forgetting. Replay and per-task retention measurements are necessary to interpret improvement.
4. An instrument defines both event probability and conditional state update. Validity of that operation does not guarantee successful optimization.
5. Finite-register and low-rank representations have real storage and contraction costs. The initial density workspace is deliberately small and exact.
6. The source's active program learner receives richer feedback than its neural learners. The new integration records and preserves that distinction.
7. The source's full R1-R8 systems, representation discovery and learned improvement controller were unimplemented. SERA begins with a bounded R1/R2 path and records remaining work explicitly.

## Traceability

| Source requirement | SERA implementation | Verification |
|---|---|---|
| Input/state/output/learn/evaluate contracts | `contracts.py`, `models.py`, `training.py`, `evaluation.py` | Metadata, target admission, streaming equivalence, shape/dtype checks |
| Three learning time scales | Session state, optimizer checkpoints, explicit improvement engine | Parameter immutability during observation; resume; parent versions |
| B1/B3/B4/B5/B6/B7 mechanisms | `models.py`, `quantum.py` | Matrix reference, gradients, probability normalization, exact writes |
| R1 action-conditioned prediction | `world.py` | All finite transitions and executed plans |
| R2 acquired executable programs | `programs.py` | New environments, long sequences, budget and domain rejection |
| Failure-driven lifecycle | `engine.py` | Actual failure, discovery, fresh paired test, promotion, rollback |
| Provenance and independent admission | `storage.py`, `evaluation.py` | Data fingerprints, ledger integrity, reuse rejection, bounded scores |
| Retention | `adaptation_experiment` | No update/full/replay/scratch, separate support/query namespaces |
| Larger-scale generality | Roadmap, not an implemented capability | Future acceptance criteria |
