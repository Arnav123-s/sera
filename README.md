# SERA

**State-Space Engine for Reasoning and Adaptation**

[Repository](https://github.com/Arnav123-s/sera) · Research by [Arnav123-s](https://github.com/Arnav123-s)

I am building SERA to study how a learner maintains state, predicts action consequences, acquires executable skills, and learns which improvement procedure to try next.

Version 0.2 connects the first R1/R2 research path: recurrent memory drives observation and reward prediction; planning produces real execution feedback; admitted feedback trains candidates; a controlled event instrument guides bounded program search; and a trained intervention policy selects among six learning procedures. A persistent solver restores its models, policy and accepted skills together. Promotion, rejection and rollback affect ordinary inference and later learning rounds.

**Research status:** a working symbolic research system trained locally from scratch. Its four-color worlds, action vocabulary, task identifiers, program grammar and feedback access are supplied. General language, perception, invention of new learning algorithms, R3–R8 and general intelligence remain research objectives. I distinguish implemented connections from demonstrated learning in the [connected study](reports/connected-study.md) and [audit resolution](reports/architecture-audit-resolution.md).

## Run the connected study

Use Python 3.11 or newer. I verify Python 3.12 with CPU PyTorch 2.10.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[dev,reports]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m sera study --output runs/my-study --seeds 0 1 2
```

After activation, `sera` and `python -m sera` are equivalent. On Linux/macOS use `.venv/bin/python`. The default study trains three separate solvers, measures every available intervention during policy development, evaluates withheld reset-family worlds, and retains successes and failures. Budgets and assumptions are in the [study protocol](research/connected-study-protocol.md).

```text
sera solve runs/my-study/0/solver --family rotation --world-seed 50000 --start 0 --goal 3
sera evaluate runs/my-study/0/solver --output runs/current-evaluation
sera status runs/my-study/0/solver
python scripts/prepare_solver.py --input runs/my-study/0 --output runs/current-solver
sera learn runs/current-solver --family reset --world-seed 123456 --steps 32
sera rollback runs/current-solver
```

`solve` loads the current accepted solver and executes its acquired skill or neural plan. `learn` collects evidence, uses the saved learned policy, constructs a candidate and evaluates it against the incumbent on fresh paired problems. A rejection preserves the current solver. `rollback` restores the accepted parent and preserves the cumulative evaluation ledger. Learning mutates that solver directory; keep the study's original directory intact when doing additional experiments.

## What is implemented

| Path | Executable behavior | Evidence boundary |
|---|---|---|
| R1 | Current event plus recurrent state → action-conditioned observation/reward prediction → bounded planning → actual feedback → replay update | Compact associative model; symbolic sensors; most-probable imagined states |
| R2 | Learned action channels → shared event instrument → predictive-state program proposals → execution/verification → trace learning | Four-dimensional instrument; finite typed DSL; reset access required for search |
| Improvement policy | Fits measured intervention utilities and selects no change, update, replay, evidence, planning or program acquisition | Learns a selector over supplied methods; final test family withheld |
| Durable solver | Neural model, world model, controller and verified skills in immutable versions | Finite task/domain routing; validated component registry |
| Admission | Fresh paired scores, empirical retention, validity and counted-cost gates | Small finite generators; retention is not a statistical certificate |
| Reference R1 preset | 256-wide embedding, 128-complex rotor, eight 32×32 memories, four 16×4 density factors | 35,840-byte core state tested; full preset not trained in this study |
| Original core controls | Delta, GRU, rotor, no-phase rotor, collision and no-cross control, full density, hybrid | Historical mechanism study; eight cores do not mean eight architecture families |

The [first study](reports/first-study.md) remains historical evidence. Its larger hybrid did not outperform a comparable-parameter classical memory on longer sequences. I make no quantum advantage claim. The [0.1 audit](reports/architecture-alignment-audit.md) records the missing connections that motivated this build.

## Evidence and reproducibility

```text
src/sera/        solver, memory, R1/R2, experience, policy, evaluation and CLI
tests/           numerical references, provenance and executable lifecycle tests
scripts/         report generation and release verification
research/        original-source hashes, 154-concept map, protocol, checklist and roadmap
reports/         measured findings, raw selected evidence and checksums
runs/            full local studies, checkpoints, evidence, candidates and journals (ignored)
research/intake/ original reference package and extracted PDF text (ignored)
```

I keep a [task checklist](research/implementation-checklist.md), [development log](research/development-log.md), [architecture](research/architecture.md) and [reproduction guide](research/reproducing-connected-study.md). I inspected the supplied package as research material and did not execute its scripts or load its checkpoints. The source manifest distinguishes original material from independently measured SERA results. The 36,900 supplied registry entries are schematic configurations.

```text
python scripts/build_connected_report.py
python scripts/verify_release.py
python -m ruff check src tests scripts
python -m pytest
```

The connected report can be rebuilt from committed JSON evidence; its tables do not require rerunning training. Model checkpoints and full journals remain local. `requirements-lock.txt` records the verified Python 3.12 environment; install its CPU PyTorch entry using `--extra-index-url https://download.pytorch.org/whl/cpu`.

All operation categories are retained separately. Their sum is a declared cost proxy, not FLOPs. The local hash journal catches ordinary tampering; it does not isolate a malicious process. Generated programs are bounded data interpreted by fixed code. No arbitrary generated Python is executed.

Legacy `train`, `benchmark`, `world`, `instrument`, `adapt` and `run` commands remain available for the original mechanism experiments. Legacy neural checkpoints can be evaluated, but historical version-0.1 directories are not silently migrated into connected solvers.

I have not assigned an open-source license to this research release. Original reference materials retain their source attribution.
