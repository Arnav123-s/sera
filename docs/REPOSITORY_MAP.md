# Repository map

[Home](../README.md) · [Use SERA](START_HERE.md) · [Status](STATUS.md) · [Archive](RESEARCH_ARCHIVE.md)

| Location | Contents |
|---|---|
| [`src/sera/`](../src/sera/) | Core model, state, ownership and learning contracts |
| [`experiments/`](../experiments/) | Registered research implementations |
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

The current [portfolio release](../research-continuation/33_capability_portfolio/README.md) separates the selected runtime checkpoint and parent from its full numerical research archive. `scripts/quest_artifacts.py --restore --research` restores both without replacing divergent files or restarting experiments. The [completion release](../research-continuation/32_verified_completion/README.md) remains preserved with its own restorer.
