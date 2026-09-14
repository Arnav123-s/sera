# SERA 0.3 bounded experiment checklist

I use the original SERA Research Archive as the design reference, including the handbook, editable source chapters and R1/R2/failure/time-scale diagrams. The archive SHA-256 is `7e9268f8a0281190551ecfdadf8270eab989190e62e4621c03b93590c3932cce`. This work starts from `f8e1cd064deecaf148a391b3bfc00722891b5212`; earlier study evidence stays tied to its own source version.

I mark implementation, verification and learning evidence separately. A negative controlled experiment is a completed experiment, not a demonstrated positive capability. R3-R8 remain separate architecture hypotheses under the handbook's narrow-first integration recommendation.

The [full source-packet review](../reports/source-packet-comparison.md) corrects the earlier completion interpretation. This table records the bounded 0.3 work, not completion of every source requirement. The complete next implementation sequence is in that review and the roadmap.

| Item | Original requirement | Acceptance | Status |
|---|---|---|---|
| C01 | Live session ownership and serialization; handbook lines 828-832, 1396 | Versioned owner/model/encoder, tensor shape/dtype checks, provenance, continuation and mismatch rejection | Bounded world-session contract implemented; no universal multimodal session |
| C02 | Acquired library composition; lines 519-523, 967-973, 1398 | Automatic calls, domain/dependency versions, examples/tests and no-library controls | Bounded within-world composition tested; abstraction and cross-domain transfer open |
| C03 | General event instrument; lines 483-537 | Multi-Kraus events, numerical references, aliased histories and classical controls | Core implemented; aliased model selection and expansion outside the automatic learner |
| C04 | Selective reference routing and approximation controls; lines 308-320, 420-426 | Selective execution, diagnostics, reference training and rank/cost curves | Separate reference experiment complete; independently trained rank and matched core studies open |
| C05 | Failure diagnosis and targeted curriculum; lines 840-880, 1288-1304 | Prior attempts, budgets and diagnosis affect actual interventions | Hybrid rules implemented; policy training on sequential failure histories and irreducible noise open |
| C06 | Typed events and broader teaching; lines 175-225, 941-949 | Units/scale/masks encoded; provenance checked at boundary; mixed teaching and true family holdouts | Separate typed model trained; shared world path missing; three extended/structure tests duplicate semantic examples |
| C07 | Adaptation and planning controls; lines 951-965 | No update/full/adapter/replay/scratch and reactive/learned/oracle controls | Small-world controls complete; stochastic belief planning and broader transfer open |
| C08 | Retention and useful archive; lines 848-850, 1294-1298 | Replay, lineages, critical-capability retention and retrieval/storage costs | Replay/archive implemented; gate covers composite world scores, not every capability |
| C09 | Complete experiment accounting; lines 19-29, 893-911 | Include run/phase cost, failures, hardware and evidence exposure | Final invocations measured; some development/external costs unavailable; full-cost acceleration unestablished |
| C10 | Controlled architecture changes; lines 983-989, 1398 | Bounded grammar, validity/transfer/retention screening, immutable parents and rollback | Adapter screening tested; instrument expansion is fresh replacement; broader growth/search open |
| C11 | Improve the improver; lines 991-997 | Successive useful learners and improvers under comparable full budgets | Separate eta-training stages complete; coupled sequential solver/improver learning and advantage open |
| C12 | Audit and release | Source comparison, regression checks, results, install/reload and CI | Release verified; later comprehensive review corrects scope and preserves all prior evidence |

Development started at 2026-09-14 03:44:27 UTC. Development and failed pilot work are recorded separately from confirmatory study costs. The complete reference catalog and supplemental project context remain part of the review baseline.

The final cohort uses source `0d28049174092ae48ff91af9ff37c2478eedf51c159c41e3ccdcec6f39ce72bd`. Three seeds measured 558 intervention outcomes across 75 episodes and nine actual policy versions. Fifteen persistent proposals produced three promotions and twelve rejections. The original 12-core benchmark was separately reproduced with 45 numerical checks, 36 exact checkpoint accuracy rechecks and 36 fresh training runs. The 55-test regression suite includes the all-missing-target defect found during development.

C09 covers measured final invocations and recorded development costs. Some externally interrupted development processes did not flush complete CPU/wall records; these are marked unavailable. Complete research-efficiency and sustained acceleration remain unestablished. C10 measures adapter screening within the finite grammar; it does not establish broad architecture search. C11 measures actual outer updates and their outcomes, including failures; a changed policy does not by itself demonstrate a better improver.

The bounded experiments and release checks were completed. The full architecture requirements remain partial as labeled above; this is not an all-twelve-contract sign-off. [Public Windows/Linux verification](https://github.com/Arnav123-s/sera/actions/runs/34810417587) passed for the implementation commit. The later audit adds the omitted original replay/instrument/discovery reproduction and a preservation catalog without changing that cohort.
