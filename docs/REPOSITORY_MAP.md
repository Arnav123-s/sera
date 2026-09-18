# Repository map

[Home](../README.md) · [Use SERA](START_HERE.md) · [Status](STATUS.md) · [Archive](RESEARCH_ARCHIVE.md)

| Location | Contents |
|---|---|
| [`src/sera/`](../src/sera/) | Core model, state, ownership and learning contracts |
| [`experiments/`](../experiments/) | Registered research implementations |
| [`experiments/self_chosen/`](../experiments/self_chosen/) | Elementary process teaching, learned question selection, conjectures and independent credit |
| [`experiments/discovery_frontier.py`](../experiments/discovery_frontier.py) | Broader missing-input investigations and composition of owned learned rules |
| [`experiments/discovery_observation.py`](../experiments/discovery_observation.py) | Source-attributed physical observations and separately verified empirical parameters |
| [`experiments/continuing_growth/`](../experiments/continuing_growth/) | Sustained twelve-strand practice through the persistent owner |
| [`experiments/operator_discovery/`](../experiments/operator_discovery/) | Internal proposal generation, rational certificates and executable discovered relations |
| [`experiments/autonomous_discovery/`](../experiments/autonomous_discovery/) | Self-selected investigations, prerequisite acquisition, exact credit and continuation |
| [`experiments/learning_progress/`](../experiments/learning_progress/) | Continuing practice, progress credit, procedure learning and independent replay |
| [`scripts/`](../scripts/) | Task entry points, resource supervisors and verification |
| [`tests/`](../tests/) | Behavioral, persistence, mathematical and integrity checks |
| [`workbench/`](../workbench/) | Local task tools and workbench interface |
| [`docs/`](./) | Current user and contributor navigation |
| [`research/`](../research/) | Architecture, references, direction and cross-study index |
| [`research-continuation/`](../research-continuation/) | Preserved protocols, results, audits and checkpoint archives |
| [`reports/`](../reports/) | Earlier consolidated reports and source comparisons |
| `runs/` (local, ignored) | Live revisions, downloaded data, logs and resumable checkpoints |

Historical sources and sealed records retain their paths because checkpoint contracts refer to their bytes. Current starting points are indexed above.

## Setup and verification

Create the environment with **Python 3.12.14**:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e ".[dev,reference]" numpy==2.5.3 scipy==1.18.1
.venv/Scripts/python.exe scripts/restore_test_artifacts.py
.venv/Scripts/python.exe scripts/reading_artifacts.py --restore
.venv/Scripts/python.exe scripts/books_artifacts.py --restore
.venv/Scripts/python.exe scripts/completion_artifacts.py --restore
.venv/Scripts/python.exe scripts/quest_artifacts.py --restore
.venv/Scripts/python.exe scripts/quest_retention_artifacts.py
.venv/Scripts/python.exe scripts/learning_artifacts.py
.venv/Scripts/python.exe -m scripts.discovery_artifacts
.venv/Scripts/python.exe -m scripts.autonomous_artifacts
.venv/Scripts/python.exe -m scripts.acquisition_artifacts
.venv/Scripts/python.exe -m scripts.growth_artifacts
.venv/Scripts/python.exe -m scripts.refinement_artifacts --stage 39
.venv/Scripts/python.exe -m scripts.refinement_artifacts --stage 40
.venv/Scripts/python.exe -m scripts.self_chosen_artifacts
.venv/Scripts/python.exe -m scripts.frontier_artifacts
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/verify_concept_release.py
```

On Linux, use `.venv/bin/python`. The [workflow](../.github/workflows/ci.yml) pins the qualified CPU profile. Fixture restoration checks hashes and refuses to overwrite divergent files; it restores test prerequisites without restarting training or replacing live stores.

In the managed research workspace, run numerical work through the existing bounded supervisors and resource ledger. The plain test command documents the clean-checkout entry point used in CI.

## Review a study

1. Read its report and frozen protocol.
2. Check sources, partitions and comparison models.
3. Inspect the independent audit and retained failures.
4. Verify archived bytes through the release manifest.
5. Read the latest continuation and live ledger before new work.

The current [self-chosen discovery release](../research-continuation/42_composed_discovery/README.md) contains elementary teaching, two sustained investigation studies, composed rules, independent reward audits and a measured-task return. Stages 38–42 store their full numerical archives as numbered [release assets](https://github.com/Arnav123-s/sera/releases/tag/research-2026-09-18-discovery). Each stage's manifest pins every part and archived member. The verifier downloads absent parts into `runs/artifact-cache`, checks their hashes and preserves existing local artifacts. Tensor deduplication retains exact semantic checkpoint recovery, including optimizer, RNG and event state; original local `.pt` files remain untouched.

The preserved [acquisition and discovery release](../research-continuation/37_constraint_acquisition/README.md) retains the identified gap, three acquired basis corrections, eight further relationships, exact continuation and frozen evaluation. Its compact archive reconstructs all four successor checkpoints; the [preceding autonomous experiment](../research-continuation/36_autonomous_discovery/README.md) retains its three checkpoints and unsuccessful discovery attempts.

The preserved [internal-discovery release](../research-continuation/35_operator_discovery/README.md) stores its inherited parent once and verifies exact reconstruction of all nine discovery checkpoints. The [learning-progress release](../research-continuation/34_learning_progress/README.md) publishes both successor transitions, both learned seeds, all controls and independent prediction witnesses in five compact archives. `scripts/learning_artifacts.py --restore` restores missing research endpoints without replacing divergent files.

The preserved [portfolio release](../research-continuation/33_capability_portfolio/README.md) separates the selected runtime checkpoint and parent from its full numerical research archive. `scripts/quest_artifacts.py --restore --research` restores both without replacing divergent files or restarting experiments. The [completion release](../research-continuation/32_verified_completion/README.md) remains preserved with its own restorer. Private assessor keys remain local; research replay and published certificates have separate entry points from live assessment authority.
