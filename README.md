# SERA

**State-Space Engine for Reasoning and Adaptation**

[Repository](https://github.com/Arnav123-s/sera) · Research by [Arnav123-s](https://github.com/Arnav123-s)

I am building SERA to study how a learner maintains state, predicts action consequences, acquires executable skills, retains them and improves its learning procedure. The [project context](research/project-context.md) and [original research references](research/references.md) define that direction.

Version 0.5 implements one shared R1 parameter owner behind world prediction, typed tasks and legacy sequence inference. I train both a simpler associative core and the handbook-sized rotor/associative/low-rank reference from scratch, then compare corrective learning with replay, full updating, adapters, scratch learning and a separate-model control. Shared views remain connected after checkpoint restoration. The controller still selects from a finite supplied method set; ordinary continued learning does not train a new improvement policy.

**Research status:** a bounded research system trained locally from scratch. Task vocabularies, simulators, arithmetic grammar, action alphabet and feedback access are supplied. Classical controls, failed learning, retention losses and rejected candidates are part of the evidence. General intelligence, broad language/perception and sustained research acceleration have not been established. R3–R8 remain separate architecture hypotheses under the handbook's narrow-first recommendation.

Start with the [shared-learner results](reports/shared-learner-study.md), [source comparison](reports/shared-source-audit.md), [training protocol and checklist](research/shared-learner-protocol.md), and [preserved variants](research/variants.md). The [0.4 report](reports/evaluation-v2-study.md), [full 0.3 source-packet comparison](reports/source-packet-comparison.md), [0.3 study](reports/stage-three-study.md), [0.2 results](reports/connected-study.md) and [earlier workspace snapshot](research/workspace-index.md) remain historical evidence.

The [15 September continuation](research-continuation/RESULTS_2026-09-15.md) adds 5,120 geometric-memory artifacts, 300 fixed-inquiry runs and generator acquisition inside the actual trained R1 parent. That cohort passed 156 tests and preserved 40 checked capability groups exactly. The strongest unresolved failure is confident prediction outside the supplied model class.

The subsequent [learned-applicability study](research-continuation/15_applicability/report.md) trains and audits a guard on 1,344 distinct worlds. It improves the declared score over fixed controls, but fails the conditional-risk gate on exceptions and changing mechanisms. The complete model, raw cohort, failed checks and exact replay are preserved; it remains an experimental candidate. That release passed **174 tests** and preserved 16,556 prior model/result files byte-for-byte. Its [packet comparison](research-continuation/15_applicability/architecture-audit.md) remains historical evidence.

The [SERA/Kavi v3 continuation](research-continuation/16_v3/README.md) calibrates that frozen guard on independent worlds and develops guarded trace consolidation. Its matched grouped policy answers 664/1,000 queries with zero observed errors, while abstaining on all abrupt-jump cases. A finite compiler reduces its per-candidate full-work proxy by 15.1% after an unsuccessful first candidate; subsequent proof and interpreter repairs have explicit migrated checkpoints. That release passed 199 tests and its [architecture audit](research-continuation/16_v3/architecture-audit.md) remains preserved.

The [acquisition-dependent transfer study](research-continuation/18_instance_transfer/report.md) completes 70 prospective training runs and three development candidates. Corrected acquired knowledge lowers neural motion MSE by **15.0%** versus observed-only learning and **23.3%** versus incorrect acquired knowledge, with no measured loss in 61 retained groups. Only the 514 existing motion-readout parameters change. A larger shared-path update fails retention; an earlier normalization-based result remains useful but does not identify transfer of fitted coefficients. I preserve both failures and integrate the qualified readout through explicit factual replay and finite reproof. That release passed 210 tests.

The latest [continuing-control study](research-continuation/19_continuing/report.md) executes 168 prospective lifetimes in 24 paired simulated worlds. From positions and actions, the shared owner learns action-response coefficients, imagines consequences and corrects a reversed response without resetting between targets. Nominal physical cost falls **43.2%** versus reactive control and **44.4%** versus one-step prediction. A restored descendant continues to 79 observations, retains exact scores in a fresh 61-group bank and adds 248 bytes of registered state. **219 tests pass.** Its uncertainty intervals remain too narrow; the circle/action grammar and investigator procedure are supplied. Shared-representation plasticity, calibrated adaptive-policy uncertainty and learned eta remain open. [Resume the saved learner](research-continuation/19_continuing/resume.md).

## Run

Use Python 3.12 and CPU PyTorch 2.10. On Linux/macOS replace `.venv/Scripts/python.exe` with `.venv/bin/python`.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e ".[dev,reports,reference]"
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m sera shared-study --output runs/my-shared-study --seeds 0 1 2
```

This trains both shared cores from scratch: 1,600 joint updates, then 192 updates per adaptation control at 32, 128 and 512 support cases. It writes evidence, checkpoints, raw paired scores, validation histories and costs. Each seed uses one CPU thread. All candidate weights freeze before final scoring. Use `--pretrain-steps 2 --adapt-steps 2 --support-sizes 8 --samples 8 --seeds 19 --kinds delta` only for a plumbing check. A fresh output directory is required. The earlier standalone typed protocol remains available through `scripts/evaluate_variants.py`.

Assemble the first delta seed into a persistent solver, train its bounded R2/controller, and attempt corrective learning:

```text
python scripts/start_shared.py --study runs/my-shared-study/0/delta --output runs/my-shared-solver --review runs/my-shared-assembly
python -m sera status runs/my-shared-solver
python -m sera typed-solve runs/my-shared-solver examples/typed-addition.json
python -m sera typed-solve runs/my-shared-solver examples/typed-binding-first.json
python -m sera typed-solve runs/my-shared-solver examples/typed-binding-latest.json
python -m sera learn-binding runs/my-shared-solver --seed 1200001 --support 128 --steps 192
python scripts/audit_shared.py --root runs/my-shared-study --seed 0 --output runs/my-shared-audit-0.json
```

`learn-binding` teaches the explicitly instructed first-binding rule from simulator corrections. Its default trains scoped low-rank residuals while freezing the shared base; full/replay/global-adapter controls remain available. It freezes the candidate, checks fresh gain and 34 retained capabilities, and keeps rejected proposals. The supplied first/latest example pair has the same writes and query but different explicit instructions. Continued world learning on a shared solver also checks both instructed binding rules, for 40 retained capabilities with one world. These retention checks are empirical; the objective gain uses a separate conservative bound. Bounded Brier calibration is gated for instructed binding only.

The earlier R1/R2 experiment driver `scripts/study_stage_three.py` remains available and retains the v1 typed generator for historical work. Exact 0.3 reproduction uses the pinned source commit in [release source identities](research/release-sources.json). For an existing full solver, the ordinary commands are:

```text
python -m sera status runs/sera-0.4-current
python -m sera solve runs/sera-0.4-current --family permutation --world-seed 310000 --start 0 --goal 3
python -m sera typed-solve runs/sera-0.4-current examples/typed-addition.json
python -m sera evaluate runs/sera-0.4-current --output runs/current-evaluation
```

`typed-solve` routes a declared typed request to its trained component and applicable verified procedure. The example adds a sequence of integers modulo four. Returned values are labeled predictions. `solve` executes a saved skill or plans through the learned world model. The legacy sequence decoder is also trained and stored, so ordinary symbolic evaluation uses learned weights.

The active local shared solver is `runs/sera-0.5-current`. The earlier `runs/sera-0.4-current` retains its entire ledger, versions and replay. The new shared solver has its own explicit initial version; it does not rewrite that history or imply a promotion across different solver architectures. `sera learn` collects simulator feedback, reads attempt history and remaining budget, follows the saved controller and checks fresh paired world gain. Binding correction currently uses the explicitly selected CLI method. `sera rollback` preserves cumulative history. The standalone v1/v2 typed cohorts and useful older sequence, HMM and instrument variants remain available. [Workspace roles](research/shared-workspace.md) identify every new attempt and artifact family.

## Implemented paths

| Path | Behavior | Boundary |
|---|---|---|
| R1 | Learned recurrent state, action prediction, reward prediction and planning | Supplied symbolic worlds; heuristic imagined observations |
| R2 | General multi-Kraus events, shared likelihood/conditioning and execution-guided programs | Finite state and action grammar; matched classical controls |
| Typed tasks | Modality adapters and categorical/numeric heads using the same R1 core as world and sequence tasks | Small synthetic representations; supplied arithmetic grammar; explicit task instructions |
| Learning loop | Failure categories, targeted evidence, full/adapter/replay updates, saved attempt budgets | Learned selection among finite supplied methods |
| Library | Domain/version-checked calls, verification traces and future composition | Explicit domain applicability and bounded expansion |
| Persistence | Owned live sessions, immutable solver versions, cumulative replay and useful archive retrieval | Local ownership and integrity checks; no hostile-process isolation |
| Reference memory | Full-size rotor/delta/density shared learner, compared with equal update/example budgets | Unequal parameter/runtime costs; core bytes exclude weights, activations and optimizer state |

## Reproduce and inspect

```text
python scripts/reproduce_archive.py research/reference-materials/sera-research-archive.zip --output runs/source-reproduction --retrain
python scripts/review_source_models.py research/reference-materials/sera-research-archive.zip --output runs/source-model-comparison
python scripts/review_integration.py runs/stage-three-complete --output runs/integration-review.json
python scripts/build_stage_three_report.py
python scripts/verify_release.py
python -m ruff check src tests scripts
python -m pytest
```

Source reproduction requires the original local archive. The first command covers the 12-core benchmark; `review_source_models.py` adds the original replay, instrument and program-discovery experiments plus direct restricted-model comparisons. These commands verify reviewed source hashes, redirect output to fresh directories and load checkpoints with `weights_only=True`. Fresh training initializes independently. Source results and SERA measurements stay separate. The supplied 36,900 configuration entries are an index, not trained architectures.

`src/sera/` contains executable models and learning; `tests/` contains numerical and lifecycle checks; `research/` contains source provenance and protocols; `reports/` contains findings and checksummed evidence. Full checkpoints, journals and original materials remain under ignored `runs/` and `research/reference-materials/`. The [architecture](research/architecture.md) explains the contracts and the [roadmap](research/roadmap.md) records unresolved research objectives.

All cost categories remain separate. The admission operation sum is a declared proxy, not FLOPs or complete research cost. No quantum advantage is claimed. I have not assigned an open-source license to this research release; original references retain their source attribution.
