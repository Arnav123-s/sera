# SERA

**State-Space Engine for Reasoning and Adaptation**

Research by [Arnav123-s](https://github.com/Arnav123-s).

[![Verification](https://github.com/Arnav123-s/sera/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Arnav123-s/sera/actions/workflows/ci.yml?query=branch%3Amain)

I am building one persistent learner that acquires executable knowledge, investigates missing information, imagines consequences and corrects its understanding with evidence. Learned weights, retained observations and independent verification have distinct roles.

## Start here

| I want to… | Open |
|---|---|
| Run a task with the saved learner | **[Use SERA](docs/START_HERE.md)** |
| See current results and verification | **[Current status](docs/STATUS.md)** |
| Understand the implementation | **[Repository map](docs/REPOSITORY_MAP.md)** |
| Find a past experiment, checkpoint or failure | **[Research archive](docs/RESEARCH_ARCHIVE.md)** |
| Understand the intended architecture | [Architecture](research/architecture.md) · [Project direction](research/project-context.md) |
| Check teaching-data quality and provenance | [Data guide](docs/DATA.md) |

## What the saved learner can do

| Capability | Use it | Measured evidence |
|---|---|---|
| Learn a body's motion from observations; refine and forecast | [Motion learning](research-continuation/28_concept_refinement/README.md) | [C01/C02 results](research-continuation/28_concept_refinement/report.md) |
| Learn verified operators and calculate polynomial motion | [Mathematical study](research-continuation/27_self_study/README.md) | [Acquisition and correction](research-continuation/27_self_study/report.md) |
| Interpret English, Spanish, French and German requests | [Language interfaces](docs/START_HERE.md#interpret-a-request) | [Language and retention](research-continuation/27_self_study/report.md) |
| Label English requests and extract entities into a file | [Request annotation](research-continuation/26_stream_curriculum/README.md) | [Held-out evaluation](research-continuation/26_stream_curriculum/report.md) |
| Resume conditional investigations and acquire observations | [Investigation](research-continuation/25_constraint_inquiry/README.md) | [Owner and evidence audit](research-continuation/25_constraint_inquiry/architecture-audit.md) |

Each result belongs to its stated teaching and evaluation conditions. I document supplied representations, data, algorithms and checkers separately from the mappings and coefficients the learner acquires.

## Current release

C01/C02 are integrated through the continuing StudyR1 owner. The release passed **436 local tests**, and verification passed on **Linux and Windows**. The [status page](docs/STATUS.md) links the exact code, results, costs and checks. Earlier failed workflow runs remain visible as historical records.

The next teaching cycle uses human-authored reading material with source attribution, deduplication and separate development/evaluation articles. Its progress and final decision are recorded on the status page.

## Development and preservation

Use Python **3.12.14**, CPU PyTorch **2.10.0**, NumPy **2.5.3** and SciPy **1.18.1** for the qualified persistence environment. [Setup and verification](docs/REPOSITORY_MAP.md#setup-and-verification) includes clean-checkout fixture restoration.

All prior research remains in place with an [indexed archive](docs/RESEARCH_ARCHIVE.md). Ignored `runs/` contains live local revisions. Sealed reports and checkpoints retain their original identities. The resource ledger governs local numerical work; printed balances are historical snapshots.

I have not assigned an open-source license to this research release. Original datasets and references retain their own licenses and attribution. See [data sources](docs/DATA.md) and [original references](research/references.md).
