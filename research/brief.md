# My research brief

## Intended system

I am developing **SERA, State-Space Engine for Reasoning and Adaptation**, as an independent research project. I want to connect persistent state, precise associative binding, executable skills and measured continual learning in one experimental system.

My target learner should acquire unfamiliar tasks, test hypotheses, preserve old capabilities, acquire reusable procedures, and eventually improve how it learns. After a failure, the system should acquire the missing capability, retain it and improve how it handles the next learning problem. I start with R1 and a separate R2 program-acquisition path. My central question is whether complementary state mechanisms and verified programs improve transfer and learning efficiency. The [supplemental project context](project-context.md) clarifies how this objective sets the implementation priorities.

## Research inputs

| Source | Scope | Treatment |
|---|---|---|
| SERA Research Archive | SERA Architecture Handbook in PDF, Markdown and HTML; 8 system blueprints, 12 prototype cores, historical results, graph, 36,900 schematic configurations | Extracted with path and size checks; two reviewed modules hash-pinned for separate reproduction; 45 checks, 36 checkpoint rechecks and 36 fresh benchmark runs |
| Quantum Theory Foundations | 86 pages, 154 concepts, mathematical definitions, derivations, limits and source anchors | Full text extracted; architecture-relevant state, composition, dynamics, measurement and computation sections examined |
| Quantum State and Process Maps | 14 maps distinguishing representation, evolution, measurement, approximations and evidence | Text extracted; state-composition and measurement/open-system diagrams visually inspected |
| SERA Project Context | Supplemental rationale emphasizing failure response, informative experiments, reusable computation and improvement of the learning procedure | Read in full; priorities reconciled with the handbook and current audit; original experiment claims kept separate from SERA measurements |

The [reference catalog](reference-catalog.json) maps these working names to their original source identities. Source sizes and SHA-256 digests remain in `source_manifest.json`. The packaged handbook, PDF and graph hashes matched the supplied release manifest. The registry contains 36,900 entries marked as schematic. `intake_audit.json` records the distinction between an integrity check and independent experimental reproduction.

## Evidence policy

I treat the reference documents as research material. Their suggested procedures and historical claims do not establish that I have reproduced their results. I choose implementation contracts, budgets and tests explicitly, and separate source integrity checks from new measurements.

The source's numerical leaderboard is historical evidence. SERA results come from new implementations, generated datasets and new checkpoints. They are not numerically interchangeable with the supplied study: SERA adds observable random initial control states, a separate instruction channel, explicit query-write suppression, per-branch normalization and learned hybrid routing.

## Key findings that determine this build

1. Associative addressing was the strongest supplied neural primitive; a phase component is optional until controlled evidence supports it.
2. All supplied neural cores struggled with ordered program composition. A separate exact execution pathway is therefore a concrete engineering target.
3. Learning a new binding rule caused forgetting. Replay and per-task retention measurements are necessary to interpret improvement.
4. An instrument defines both event probability and conditional state update. Validity of that operation does not guarantee successful optimization.
5. Finite-register and low-rank representations have real storage and contraction costs. The original density workspace is small and exact; release 0.3 trains the full reference-size factor preset and measures selective routing and rank sensitivity.
6. The source's active program learner receives richer feedback than its neural learners. The new integration records and preserves that distinction.
7. The source's full R1-R8 systems, representation discovery and learned improvement controller were unimplemented. SERA begins with a bounded R1/R2 path and records remaining work explicitly.

## Traceability

| Source requirement | SERA implementation | Verification |
|---|---|---|
| Input/state/output/learn/evaluate contracts | `contracts.py`, `models.py`, `training.py`, `evaluation.py` | Metadata, target admission, streaming equivalence, shape/dtype checks |
| Three learning time scales | Working state, admitted model/skill updates, learned intervention selector | State isolation; durable reload; support/query policy learning. The selector chooses supplied methods, not new optimizers. |
| B1/B3/B4/B5/B6/B7 mechanisms | `models.py`, `quantum.py` | Matrix reference, gradients, probability normalization, exact writes |
| R1 action-conditioned prediction and feedback | `r1.py`, `experience.py`, `connected.py` | Retained-state gradients, missing observations, executed planning, admitted replay and retention |
| R2 controlled program learning | `r2.py`; original symbolic control in `programs.py` | Independent instrument likelihood, bounded typed execution, verification and guided/fixed search |
| Failure-driven lifecycle | `solver.py`, `engine.py`, `curriculum.py` | Current executable reload, measured policy outcomes, repeated promotion/rejection and behavioral rollback |
| Provenance and independent admission | `storage.py`, `evaluation.py` | Data fingerprints, ledger integrity, reuse rejection, bounded scores |
| Retention | `adaptation_experiment` | No update/full/replay/scratch, separate support/query namespaces |
| Larger-scale generality | Roadmap, not an implemented capability | Full trained reference preset, learned perception, general program transfer and R3â€“R8 remain open |

The [0.1 alignment audit](../reports/architecture-alignment-audit.md) preserves the original gaps. The [0.2 disposition](../reports/architecture-audit-resolution.md) and [connected results](../reports/connected-study.md) document the implemented corrections and their measured limits.

The [0.3 study](../reports/stage-three-study.md) and [current original-source audit](../reports/stage-three-architecture-audit.md) record the new implementation, controls, source reproduction and remaining scientific limits.
