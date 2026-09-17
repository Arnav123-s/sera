# SERA

**State-Space Engine for Reasoning and Adaptation**

Research by [Arnav123-s](https://github.com/Arnav123-s).

I am building one persistent learner that acquires executable knowledge, imagines consequences, investigates missing information, corrects its mistakes and retains useful abilities. Each release connects a measured capability to the existing parameter owner and preserves the evidence, checkpoints, unsuccessful candidates and research costs.

**Start here:** [research and evidence index](research/research-index.md) · [architecture](research/architecture.md) · [project direction](research/project-context.md) · [original sources](research/references.md).

## Current usable capabilities

| Task | Saved learner and instructions | Evidence |
|---|---|---|
| Learn checked equations from arXiv, retain corrected operators and calculate polynomial motion | [Continuing study learner](research-continuation/27_self_study/README.md) |256/256 polynomial and64/64 motion checks;128/128 fresh correction-transfer problems |
| Interpret Spanish, French and German requests with the continuing owner | [Language acquisition and retention](research-continuation/27_self_study/report.md) |63.13%,66.88%,61.88% final intent accuracy;78.75% English development retention after rehearsal |
| Label English requests and extract entities into a local file | [Request annotation](research-continuation/26_stream_curriculum/README.md) | 11,514 teaching requests; 74.54% intent accuracy, 50.20% entity-span F1 and 41.14% exact frames on 2,973 eligible held-out requests |
| Learn and apply a three-input numerical transformation from examples | [Persistent task learner](research-continuation/23_task_transfer/README.md) | Acquired structure admitted 24/24 related and 19/24 sparse-change tasks at the registered fitting budget |
| Interpret finite action requests, acquire missing simulated facts and resume investigation | [Constraint inquiry](research-continuation/25_constraint_inquiry/README.md) | Learned role binding, acquired dynamics and conditional settling through the continuing owner |
| Learn finite mathematical instruction forms and execute checked programs | [Grounded language](research-continuation/21_grounded_language/README.md) | 1,536/1,536 familiar-form held-out instructions per retained-core model; continuing task-time lessons |
| Process numerical data and continue adaptive simulated control | [Local workbench](research-continuation/20_live_workbench/README.md) | Held-out task checks, persistent revisions, observed corrections and guarded reuse |

The results apply to the taught tasks and stated evaluation protocols. Source labels, finite grammars, simulators, independent checkers and learning schedules are documented engineering inputs. Learned mappings and retained updates are reported separately. [The latest report](research-continuation/27_self_study/report.md) includes every comparison, failed candidate, exclusion and cost; completed final evaluation partitions remain closed to further tuning.

## Use the prepared local workspace

Run from the repository with its existing environment and trained checkpoints:

```powershell
.venv/Scripts/python.exe scripts/sera_study.py solve --domain integral --coefficients "1,2,3"
.venv/Scripts/python.exe scripts/sera_study.py motion --coefficients "2,3,1" --time 3 --position 5 --velocity=-1
.venv/Scripts/python.exe scripts/sera_study.py interpret --text "schalte das licht aus"
.venv/Scripts/python.exe scripts/sera_sources.py search 'ti:"Faulhaber" AND au:"Knuth"'
.venv/Scripts/python.exe scripts/sera_requests.py ask --id my-request --text "set an alarm for nine am"
.venv/Scripts/python.exe scripts/sera_requests.py result --id my-request --json
.venv/Scripts/python.exe scripts/sera_annotate.py --input requests.jsonl --output runs/my-inbox/annotations.jsonl
.venv/Scripts/python.exe scripts/sera_tasks.py status
.venv/Scripts/python.exe -X utf8 -m workbench
```

The workbench opens at `http://127.0.0.1:8765`. Request annotation writes intent and entity predictions for review. The [request guide](research-continuation/26_stream_curriculum/README.md) specifies input limits, model identities, exact continuation and checkpoint archives. Local commands honor the recorded numerical allowance: one CPU thread, a 2 GiB process-tree committed-memory limit and bounded invocation time.

## What has been evaluated

The continuing study release passed357 repository regression tests and28 targeted tests after its final metadata correction. Independent recounting verified2400 saved language predictions and1088 mathematical outcomes. Its source-learned numerical operators share the preserved owner with the request interface, while keeping their supplied representations and independent checkers explicit. See the [results](research-continuation/27_self_study/report.md), [architecture audit](research-continuation/27_self_study/architecture-audit.md) and [cost record](research-continuation/27_self_study/costs.json).

The preceding English interface added450219 trained parameters and assessed16 terminal checkpoints in one final cohort. Its successive-domain replay results and120756-record audit remain in [the preserved phase26 report](research-continuation/26_stream_curriculum/report.md).

Earlier studies cover shared recurrent ownership, guarded applicability, acquisition-dependent transfer, continuing control, sparse memory and weight correction. The [research index](research/research-index.md) gives their order and exact evidence locations. Quantum-inspired operators remain testable mechanisms; their mathematical validity and measured usefulness are evaluated separately.

## Development and preservation

The tested environment uses Python 3.12 and CPU PyTorch 2.10. On Linux/macOS use `.venv/bin/python` in place of `.venv/Scripts/python.exe`.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e ".[dev,reports,reference]"
```

`src/sera/` contains the original learning and ownership contracts; `experiments/` contains the registered continuations; `workbench/` contains local task tools; `tests/` contains regression checks. `research-continuation/` preserves protocols, result records, source identities, checkpoint archives and decisions. Ignored `runs/` stores live revisions and intermediate checkpoints. Completed experiments are resumed or inspected according to their own guides, with fresh outputs for new work. The [previous landing page](https://github.com/Arnav123-s/sera/blob/73dd66caecb7a0f3d3ab0bba21a725e2fc4d5678/README.md) retains the historical training and reproduction commands.

I have not assigned an open-source license to this research release. Original data and references retain their own licenses and attribution. MASSIVE teaching data is attributed to [FitzGerald et al.](https://github.com/alexa/massive), CC BY 4.0; detailed source identities accompany each study.
