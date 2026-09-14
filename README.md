# SERA

**State-Space Engine for Reasoning and Adaptation**

[Repository](https://github.com/Arnav123-s/sera) · Research by [Arnav123-s](https://github.com/Arnav123-s)

I am building SERA to study how a learner can maintain state, bind information in associative memory, predict action consequences, and acquire verified executable skills when its neural model fails.

In this first release, I implement a bounded R1/R2 experimental path from my architecture research. I include from-scratch training, eight interchangeable neural cores, a learned finite world model, planning, active program discovery, continual-learning controls, independent candidate evaluation, an evidence journal, and rollback. The system runs locally without a pretrained model, hosted LLM, or quantum device.

**Research status:** an executable symbolic research system. Open-ended language, perception, learned representations, autonomous invention of learning algorithms, and general intelligence remain research objectives. Results and limitations are recorded in [the first study](reports/first-study.md).

## Run it

Use Python 3.11 or newer. The verified local environment uses Python 3.12 and CPU PyTorch 2.10. Install the CPU wheel explicitly to keep the initial environment small.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m sera run --output runs/my-first-run --steps 500
```

On Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`. After activation, `sera` and `python -m sera` are equivalent.

An integrated run trains a neural memory model, learns action transitions, tests planning, compares adaptation with and without replay, diagnoses an ordered-control failure, acquires a transition program through experiments, checks the candidate on fresh tasks, and records whether it was promoted. All budgets are finite. A run can produce a rejection or an explicit unresolved result.

```text
sera train --kind delta --output runs/delta --steps 500
sera train --kind delta --output runs/delta --steps 1000 --resume
sera benchmark --output runs/comparison --kinds delta gru rotor real hybrid --seeds 0 1 2
sera evaluate runs/delta/checkpoint.pt --output runs/evaluation
sera adapt runs/delta/checkpoint.pt --output runs/adaptation
sera improve runs/delta/checkpoint.pt --output runs/improvement --max-queries 100
sera status runs/improvement
sera rollback runs/improvement
sera world --output runs/world
sera instrument --output runs/instrument --steps 300
```

Use a new output directory for a new experiment. Training resume preserves configuration and source identity; only the training-step budget may increase. The checkpoint contains the current optimizer trajectory and separately the best validation-selected weights. Evaluation loads the latter. Benchmark resume requires an identical manifest.

## Implemented components

| Component | Implementation | Evidence boundary |
|---|---|---|
| Associative binding | Gated rank-one delta writes | Classical memory; learned keys can interfere |
| Complex state | Input-controlled rotation, write and forgetting | Classical arithmetic with a real no-phase control |
| Collision state | Exact Bloch-coordinate partial swap | Independent local states; no arbitrary joint entanglement |
| Density workspace | Exact 4 x 4 controlled reset channel | Small full-rank workspace; no compression claim |
| Hybrid | Learned routing over delta, rotor and density branches | Larger parameter count is explicitly controlled |
| Event instrument | QR-normalized real or complex Kraus operators | Tiny cyclic vocabulary; conditional update and generation |
| World model | Learned action-conditioned categorical transitions | Fully observed finite simulator |
| Program acquisition | Active transition identification and finite interpreter | Resettable deterministic dynamics and observable state IDs |
| Continual learning | No update, full update, replay and scratch controls | One related withheld task; empirical retention |
| Improvement | Diagnose, acquire, verify, compare, promote or reject | Fixed diagnostic/routing policy in this release |

## Working state and durable learning

`StatefulModel.forward()` resets state for each independent batch. `Session` owns streaming state with a session ID, model version, encoder version, shape and dtype checks. Observing an event changes working state; it does not run an optimizer. `Experience` requires simulator or verified target provenance. Model predictions are not automatically accepted as training truth.

The current neural encoder processes the declared 20-feature symbolic format. Other modality names in the metadata contract reserve types; text, image and audio encoders are not implemented. The broader R1 dimensions in the handbook are scaling proposals, not hidden defaults in this repository.

## Evidence and project layout

```text
src/sera/        models, finite quantum operations, training, evaluator, engine, CLI
tests/           independent numerical references and end-to-end regression tests
scripts/         repeatable parameter-count control and instrument suite
research/        source hashes, concept graph, requirements, literature, roadmap
reports/         measured findings and selected JSON evidence
runs/            local checkpoints, full experiments, journals and verified skills (ignored)
research/intake/ original reference package and extracted PDF text (ignored)
```

I use the architecture handbook and physics atlas as research references. I implemented this code independently and did not execute or reuse the reference package's scripts or checkpoints. I retain the original 154-node concept graph and component mapping with provenance, and distinguish the 36,900 schematic registry entries from trained systems. Read [my research brief](research/brief.md), [architecture](research/architecture.md), [primary sources](research/sources.md), and [research roadmap](research/roadmap.md).

The evidence journal detects ordinary local tampering and rejects reuse of a recorded evaluation suite. It is not process isolation against malicious code. The candidate uses a validated finite program interpreter and does not rewrite the evaluator or execute arbitrary generated Python. The statistical gain bound assumes fresh independent bounded paired observations; retention gates are empirical and are not statistical certificates.

Install `.[reports]` to rebuild the measured report with `python scripts/build_report.py`. `requirements-lock.txt` records the complete verified Python 3.12 environment, including reporting tools; install its CPU PyTorch entry using `--extra-index-url https://download.pytorch.org/whl/cpu`. Selected evidence files use LF line endings so their hashes survive checkout on either platform. Run `python scripts/verify_release.py` to check them.

I have not assigned an open-source license to this research release. Original reference materials retain their source attribution.
