# SERA 0.3 architecture completion work

I use the original SERA Research Archive as the design reference, including the handbook, editable source chapters and R1/R2/failure/time-scale diagrams. The archive SHA-256 is `7e9268f8a0281190551ecfdadf8270eab989190e62e4621c03b93590c3932cce`. This work starts from `f8e1cd064deecaf148a391b3bfc00722891b5212`; earlier study evidence stays tied to its own source version.

I mark implementation, verification and learning evidence separately. A negative controlled experiment is a completed experiment, not a demonstrated positive capability. R3-R8 remain separate architecture hypotheses under the handbook's narrow-first integration recommendation.

| Item | Original requirement | Acceptance | Status |
|---|---|---|---|
| C01 | Live session ownership and serialization; handbook lines 828-832, 1396 | Versioned owner/model/encoder, tensor shape/dtype checks, admitted provenance, continuation equality and incompatible-state rejection | Implemented and evaluated; limits in the report |
| C02 | Acquired library composition; lines 519-523, 967-973, 1398 | Automatic search uses earlier skills, checks domains/dependency versions, retains examples and tests; held-out compositions and no-library controls | Implemented and evaluated; limits in the report |
| C03 | General event instrument; lines 483-537 | History-preserving multi-Kraus event branches, exact references, aliased-history evaluation against trained classical controls | Implemented and evaluated; limits in the report |
| C04 | Selective reference routing and approximation controls; lines 308-320, 420-426 | Selective event execution, persisted trajectory diagnostics, reference training and rank/error/memory/runtime comparisons | Implemented and evaluated; limits in the report |
| C05 | Failure diagnosis and targeted curriculum; lines 840-880, 1288-1304 | Failure categories, previous attempts and remaining budget affect the actual learning loop; targeted evidence versus random controls | Implemented and evaluated; limits in the report |
| C06 | Typed events and broader teaching; lines 175-225, 941-949 | Units, scale, masks and provenance reach learned encoders; arithmetic/spatial/representation tasks and disjoint generators | Implemented and evaluated; limits in the report |
| C07 | Adaptation and planning controls; lines 951-965 | No update/full/adapter/replay/scratch, sample-efficiency and calibration; reactive/learned/oracle planning on nontrivial tasks | Implemented and evaluated; limits in the report |
| C08 | Retention and useful archive; lines 848-850, 1294-1298 | Cumulative admitted replay, preserved specialized lineages, worst-task retention and retrieval/storage costs | Implemented and evaluated; limits in the report |
| C09 | Complete experiment accounting; lines 19-29, 893-911 | Record complete run/phase time, memory, acquisition/search/update/evaluation work, failures, hardware and evidence exposure | Implemented and evaluated; limits in the report |
| C10 | Controlled architecture changes; lines 983-989, 1398 | Typed bounded mutation, validity/transfer/retention screening, immutable parents and rollback | Implemented and evaluated; limits in the report |
| C11 | Improve the improver; lines 991-997 | Actual outer policy updates over multiple generations, fixed anchor plus new frontier and equal-budget fixed controls | Implemented and evaluated; limits in the report |
| C12 | Final independent audit and release | Fresh source comparison, meaningful regression tests, new results and limitations, install/reload checks and public CI | Pending |

Development started at 2026-09-14 03:44:27 UTC. Development and failed pilot work are recorded separately from confirmatory study costs. The complete reference catalog and supplemental project context remain part of the review baseline.

The final cohort uses source `0d28049174092ae48ff91af9ff37c2478eedf51c159c41e3ccdcec6f39ce72bd`. Three seeds measured 558 intervention outcomes across 75 episodes and nine actual policy versions. Fifteen persistent proposals produced three promotions and twelve rejections. The original 12-core benchmark was separately reproduced with 45 numerical checks, 36 exact checkpoint accuracy rechecks and 36 fresh training runs. The 55-test regression suite includes the all-missing-target defect found during development.

C09 covers measured final invocations and recorded development costs. Some externally interrupted development processes did not flush complete CPU/wall records; these are marked unavailable. Complete research-efficiency and sustained acceleration remain unestablished. C10 measures adapter screening within the finite grammar; it does not establish broad architecture search. C11 measures actual outer updates and their outcomes, including failures; a changed policy does not by itself demonstrate a better improver.
