# Repository map

[Home](../README.md) · [Use SERA](START_HERE.md) · [Status](STATUS.md) · [Archive](RESEARCH_ARCHIVE.md)

| Location | Contents |
|---|---|
| [`src/sera/`](../src/sera/) | Core model, state, ownership and learning contracts |
| [`experiments/`](../experiments/) | Registered research implementations |
| [`experiments/intervention_model.py`](../experiments/intervention_model.py) · [`intervention_events.py`](../experiments/intervention_events.py) · [`intervention_assess.py`](../experiments/intervention_assess.py) | Owner-held temporal weights, explicit measurement meanings, monitored controls and independent qualification |
| [`experiments/structural_field.py`](../experiments/structural_field.py) · [`structural_inquiry.py`](../experiments/structural_inquiry.py) · [`structural_check.py`](../experiments/structural_check.py) | Owner-held physical weights, committed investigations, independent qualification and versioned conditional views |
| [`experiments/counterfactual_core.py`](../experiments/counterfactual_core.py) · [`counterfactual_loop.py`](../experiments/counterfactual_loop.py) · [`counterfactual_owner.py`](../experiments/counterfactual_owner.py) | Retained-weight imagination, evidence-selected follow-ups, procedure learning and compact guarded rules |
| [`experiments/solution_search.py`](../experiments/solution_search.py) · [`solution_owner.py`](../experiments/solution_owner.py) · [`solution_credit.py`](../experiments/solution_credit.py) | Imagined rational proposals, complete question lifecycles and corrected proof rewards on the actual owner |
| [`experiments/self_chosen/`](../experiments/self_chosen/) | Elementary process teaching, learned question selection, conjectures and independent credit |
| [`experiments/discovery_frontier.py`](../experiments/discovery_frontier.py) | Broader missing-input investigations and composition of owned learned rules |
| [`experiments/discovery_observation.py`](../experiments/discovery_observation.py) | Source-attributed physical observations and separately verified empirical parameters |
| [`experiments/continuing_growth/`](../experiments/continuing_growth/) | Sustained twelve-strand practice through the persistent owner |
| [`experiments/operator_discovery/`](../experiments/operator_discovery/) | Internal proposal generation, rational certificates and executable discovered relations |
| [`experiments/autonomous_discovery/`](../experiments/autonomous_discovery/) | Self-selected investigations, prerequisite acquisition, exact credit and continuation |
| [`experiments/learning_progress/`](../experiments/learning_progress/) | Continuing practice, progress credit, procedure learning and independent replay |
| [`scripts/`](../scripts/) | Task entry points, resource supervisors and verification |
| [`scripts/sera_current.py`](../scripts/sera_current.py) · [`restore_current.py`](../scripts/restore_current.py) | Current task interface and ordered restoration of its verified predecessor chain |
| [`tests/`](../tests/) | Behavioral, persistence, mathematical and integrity checks |
| [`workbench/`](../workbench/) | Local task tools and workbench interface |
| [`docs/`](./) | Current user and contributor navigation |
| [`research/`](../research/) | Architecture, references, direction and cross-study index |
| [`research-continuation/`](../research-continuation/) | Preserved protocols, results, audits and checkpoint archives |
| [`reports/`](../reports/) | Earlier consolidated reports and source comparisons |
| `runs/` (local, ignored) | Live revisions, downloaded data, logs and resumable checkpoints |

Historical sources and sealed records retain their paths because checkpoint contracts refer to their bytes. Current starting points are indexed above. The [single restoration command](../scripts/restore_current.py) retains the ordered historical restorers and their individual hash checks.

## Setup and verification

Create the environment with **Python 3.12.14**:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e ".[dev,reference]" numpy==2.5.3 scipy==1.18.1
.venv/Scripts/python.exe scripts/restore_current.py
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/verify_concept_release.py
```

On Linux, use `.venv/bin/python`. The [workflow](../.github/workflows/ci.yml) pins the qualified CPU profile. Fixture restoration checks hashes and refuses to overwrite divergent files; it restores test prerequisites without restarting training or replacing live stores.

In the managed research workspace, run current numerical work through `scripts/run_intervention_persistent.py` and the resource ledger. It applies the user's unrestricted local-time authorization, one CPU thread, a 2 GiB process-tree memory ceiling and a scoped Windows keep-awake request. Historical bounded supervisors remain preserved. The plain test command documents the clean-checkout entry point used in CI.

## Review a study

1. Read its report and frozen protocol.
2. Check sources, partitions and comparison models.
3. Inspect the independent audit and retained failures.
4. Verify archived bytes through the release manifest.
5. Read the latest continuation and live ledger before new work.

The current [intervention release](../research-continuation/47_intervention_understanding/README.md) contains the acquired models, control and outcome records, competing methods, complete owner, v17/v18 packets and independent replay. Its [release manifest](../research-continuation/47_intervention_understanding/release-manifest.json) pins the archive and scientific sources; the additive [portability contract](../research-continuation/47_intervention_understanding/publication/portability-contract.json) pins restoration across numerical platforms. [Follow one complete investigation](../research-continuation/47_intervention_understanding/publication/WORKED_EXAMPLE.md).

The preserved [self-chosen discovery release](../research-continuation/42_composed_discovery/README.md) contains elementary teaching, two sustained investigation studies, composed rules, independent reward audits and a measured-task return. Stages 38–42 store their full numerical archives as numbered [release assets](https://github.com/Arnav123-s/sera/releases/tag/research-2026-09-18-discovery). Each stage's manifest pins every part and archived member. The verifier downloads absent parts into `runs/artifact-cache`, checks their hashes and preserves existing local artifacts. Tensor deduplication retains exact semantic checkpoint recovery, including optimizer, RNG and event state; original local `.pt` files remain untouched.

The preserved [acquisition and discovery release](../research-continuation/37_constraint_acquisition/README.md) retains the identified gap, three acquired basis corrections, eight further relationships, exact continuation and frozen evaluation. Its compact archive reconstructs all four successor checkpoints; the [preceding autonomous experiment](../research-continuation/36_autonomous_discovery/README.md) retains its three checkpoints and unsuccessful discovery attempts.

The preserved [internal-discovery release](../research-continuation/35_operator_discovery/README.md) stores its inherited parent once and verifies exact reconstruction of all nine discovery checkpoints. The [learning-progress release](../research-continuation/34_learning_progress/README.md) publishes both successor transitions, both learned seeds, all controls and independent prediction witnesses in five compact archives. `scripts/learning_artifacts.py --restore` restores missing research endpoints without replacing divergent files.

The preserved [portfolio release](../research-continuation/33_capability_portfolio/README.md) separates the selected runtime checkpoint and parent from its full numerical research archive. `scripts/quest_artifacts.py --restore --research` restores both without replacing divergent files or restarting experiments. The [completion release](../research-continuation/32_verified_completion/README.md) remains preserved with its own restorer. Private assessor keys remain local; research replay and published certificates have separate entry points from live assessment authority.
