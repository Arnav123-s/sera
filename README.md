# SERA

**State-Space Engine for Reasoning and Adaptation**

[Repository](https://github.com/Arnav123-s/sera) · Research by [Arnav123-s](https://github.com/Arnav123-s)

I am building SERA to study how a learner maintains state, predicts action consequences, acquires executable skills, retains them and improves its learning procedure. The [project context](research/project-context.md) and [original research references](research/references.md) define that direction.

Version 0.3 contains a partially integrated R1/R2 research system: owned working state, event instruments, bounded program composition, typed encoders, targeted experiments, adapters and candidate admission. The world, typed and legacy sequence models have separate learned parameters. Controller training is a separate experiment; ordinary continued learning does not update that controller. Saving these components together does not make them one shared learner.

**Research status:** a bounded research system trained locally from scratch. Task vocabularies, simulators, arithmetic grammar, action alphabet and feedback access are supplied. Classical controls, failed learning, retention losses and rejected candidates are part of the evidence. General intelligence, broad language/perception and sustained research acceleration have not been established. R3–R8 remain separate architecture hypotheses under the handbook's narrow-first recommendation.

Start with the [full source-packet comparison](reports/source-packet-comparison.md) and [workspace index](research/workspace-index.md). The comparison identifies disconnected learning paths, duplicated extended/structure tests in three typed tasks, limited model growth and the composite retention gate's boundary. The [0.3 study](reports/stage-three-study.md), [bounded task checklist](research/stage-three-checklist.md) and [declared protocol](research/stage-three-protocol.md) retain the measured stage's scope. Historical [0.2 results](reports/connected-study.md) and [earlier audits](reports/stage-three-architecture-audit.md) remain available.

## Run

Use Python 3.12 and CPU PyTorch 2.10. On Linux/macOS replace `.venv/Scripts/python.exe` with `.venv/bin/python`.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e ".[dev,reports,reference]"
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/study_stage_three.py --output runs/my-study --seeds 0 1 2
```

The study uses explicit finite budgets and requires a fresh output directory. It writes checkpoints, teaching provenance, counterfactual policy episodes, live-session state, immutable candidates, replay and a complete run manifest. `--smoke --seeds 19` exercises the integration at short training budgets; it is not a learning result.

```text
python -m sera status runs/my-study/0/solver
python -m sera solve runs/my-study/0/solver --family permutation --world-seed 310000 --start 0 --goal 3
python -m sera typed-solve runs/my-study/0/solver examples/typed-addition.json
python -m sera evaluate runs/my-study/0/solver --output runs/current-evaluation
```

`typed-solve` routes a declared typed request to its trained component and applicable verified procedure. The example adds a sequence of integers modulo four. Returned values are labeled predictions. `solve` executes a saved skill or plans through the learned world model. The legacy sequence decoder is also trained and stored, so ordinary symbolic evaluation uses learned weights.

The designated local continuation workspace is `runs/sera-0.3-current`; the similarly named `runs/sera-current` is a preserved 0.2 workspace. For another experiment, use `scripts/prepare_solver.py` with an explicit completed seed and a fresh output before using `sera learn`. That command collects real simulator feedback, reads earlier attempt history and remaining budget, follows the saved policy, freezes a candidate and checks fresh paired world scores. Its retention gate applies to composite world scores, not every individual capability. Rejection preserves the incumbent and actual evidence. `sera rollback` restores an admitted parent while keeping the cumulative audit history.

## Implemented paths

| Path | Behavior | Boundary |
|---|---|---|
| R1 | Learned recurrent state, action prediction, reward prediction and planning | Supplied symbolic worlds; heuristic imagined observations |
| R2 | General multi-Kraus events, shared likelihood/conditioning and execution-guided programs | Finite state and action grammar; matched classical controls |
| Typed tasks | Separate numeric, symbolic, byte, patch and audio model; categorical/numeric outputs | Small synthetic representations; supplied arithmetic grammar; outside automatic world learning |
| Learning loop | Failure categories, targeted evidence, full/adapter/replay updates, saved attempt budgets | Learned selection among finite supplied methods |
| Library | Domain/version-checked calls, verification traces and future composition | Explicit domain applicability and bounded expansion |
| Persistence | Owned live sessions, immutable solver versions, cumulative replay and useful archive retrieval | Local ownership and integrity checks; no hostile-process isolation |
| Reference memory | Separately trained full-size rotor/delta/density preset and rank diagnostics | Not the admitted solver's core; core bytes exclude parameters, activations and optimizer state |

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
