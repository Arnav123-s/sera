# SERA

**State-Space Engine for Reasoning and Adaptation**

[Repository](https://github.com/Arnav123-s/sera) · Research by [Arnav123-s](https://github.com/Arnav123-s)

I am building SERA to study how a learner maintains state, predicts action consequences, acquires executable skills, retains them and improves its learning procedure. The [project context](research/project-context.md) and [original research references](research/references.md) define that direction.

Version 0.3 implements the next R1/R2 research stage: owned and serializable working state, general event instruments, acquired program composition, typed learned encoders, failure diagnosis, targeted experiments, adapters, useful lineage retrieval and successive improvement-policy updates. Models, learned procedures and the controller are saved together in an executable solver.

**Research status:** a bounded research system trained locally from scratch. Task vocabularies, simulators, arithmetic grammar, action alphabet and feedback access are supplied. Classical controls, failed learning, retention losses and rejected candidates are part of the evidence. General intelligence, broad language/perception and sustained research acceleration have not been established. R3–R8 remain separate architecture hypotheses under the handbook's narrow-first recommendation.

Read the [0.3 study and teaching results](reports/stage-three-study.md), [source alignment audit](reports/stage-three-architecture-audit.md), [task checklist](research/stage-three-checklist.md) and [declared protocol](research/stage-three-protocol.md). Historical [0.2 results](reports/connected-study.md) and the [earlier audit](reports/original-source-alignment-audit.md) retain their original measurements.

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

For further learning, copy a solver directory to a new working directory before using `sera learn`. That command collects real simulator feedback, reads earlier attempt history and remaining budget, follows the saved policy, freezes a candidate and checks fresh paired outcomes. Rejection preserves the incumbent and retains actual evidence. `sera rollback` restores an admitted parent while keeping the cumulative audit history.

## Implemented paths

| Path | Behavior | Boundary |
|---|---|---|
| R1 | Learned recurrent state, action prediction, reward prediction and planning | Supplied symbolic worlds; heuristic imagined observations |
| R2 | General multi-Kraus events, shared likelihood/conditioning and execution-guided programs | Finite state and action grammar; matched classical controls |
| Typed tasks | Numeric, symbolic, byte, patch and audio adapters; categorical/numeric outputs | Small synthetic representations; engineered arithmetic candidate grammar |
| Learning loop | Failure categories, targeted evidence, full/adapter/replay updates, saved attempt budgets | Learned selection among finite supplied methods |
| Library | Domain/version-checked calls, verification traces and future composition | Explicit domain applicability and bounded expansion |
| Persistence | Owned live sessions, immutable solver versions, cumulative replay and useful archive retrieval | Local ownership and integrity checks; no hostile-process isolation |
| Reference memory | Trained full-size rotor/delta/density preset, selective routing and rank diagnostics | Core bytes exclude parameters, activations and optimizer state |

## Reproduce and inspect

```text
python scripts/reproduce_archive.py research/reference-materials/sera-research-archive.zip --output runs/source-reproduction --retrain
python scripts/build_stage_three_report.py
python scripts/verify_release.py
python -m ruff check src tests scripts
python -m pytest
```

The source-reproduction command requires the original local archive. It verifies its identity, imports only two reviewed hash-pinned modules, redirects generated output and loads supplied checkpoints with `weights_only=True`. Its fresh training initializes independently. Source results and new SERA measurements are reported separately. The supplied 36,900 configuration entries are an index, not trained architectures.

`src/sera/` contains executable models and learning; `tests/` contains numerical and lifecycle checks; `research/` contains source provenance and protocols; `reports/` contains findings and checksummed evidence. Full checkpoints, journals and original materials remain under ignored `runs/` and `research/reference-materials/`. The [architecture](research/architecture.md) explains the contracts and the [roadmap](research/roadmap.md) records unresolved research objectives.

All cost categories remain separate. The admission operation sum is a declared proxy, not FLOPs or complete research cost. No quantum advantage is claimed. I have not assigned an open-source license to this research release; original references retain their source attribution.
