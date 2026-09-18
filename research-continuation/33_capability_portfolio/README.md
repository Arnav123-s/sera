# Use verified alternatives

[Home](../../README.md) · [Results](report.md) · [Architecture audit](architecture-audit.md) · [Checklist](checklist.md)

The current live owner is saved in **runs/sera-quests-live**. It retains four independently checked conditional routes: time integration, impulse and positive/negative work–energy branches. The earlier completion store remains preserved.

## Solve with the saved owner

From D:/ai/projects/sera:

```powershell
.venv/Scripts/python.exe scripts/run_quests_bounded.py --seconds 45 --output runs/my-portfolio-answer-001 --module experiments.quest_portfolio.runtime -- solve --input research-continuation/33_capability_portfolio/example-impulse.json
```

The example supplies a 3 kg body, 12 N·s impulse and initial rest. SERA returns **4 m/s** through its qualified impulse route. Use **example-time.json** or **example-work.json** for consistent descriptions of the same 6 N, 3 kg, 2 s scenario. [All three checked outputs](alternative-example.json) preserve the method and assumptions.

Inputs declare SI units, a conditional mechanics origin and the relevant assumptions. Each returned result identifies its original goal and route. A changed owner requires fresh qualification before this interface deploys its portfolio.

## Inspect retained quests

```powershell
.venv/Scripts/python.exe scripts/run_quests_bounded.py --seconds 45 --output runs/my-portfolio-status-001 --module experiments.quest_portfolio.runtime -- status
```

The status includes the qualified original goal, the unfinished maintenance quest, preserved failures, assessments and checked policy updates. Status reads leave the store unchanged. Output directory names must be fresh; the existing allowance governs every local numerical invocation.

## Start a separate practice session

```powershell
.venv/Scripts/python.exe scripts/run_quests_bounded.py --seconds 120 --output runs/my-portfolio-learning-001 --module experiments.quest_portfolio.runtime -- learn --store runs/my-quest-session
```

This continues the saved CompletionR1 parent with the selected method weights. Five bounded attempts use balanced practice by default, retain independently checked alternatives and then assess the frozen owner on fresh cases. Add **--controller learned** to sample the learned selector and apply independently checked on-policy credit. The reserved fifth attempt explores a least-visited candidate. Renaming a quest preserves its semantic identity and spent attempts within the continuing session.

Assessment results qualify only their owner, scope and portfolio. Points remain cosmetic. Counterexamples are retained; an owner update makes earlier qualification stale. Conditional practice keeps factual observations and source admission separate.

## Restore a fresh clone

First follow [environment and predecessor restoration](../../docs/REPOSITORY_MAP.md#setup-and-verification), then run:

```powershell
.venv/Scripts/python.exe scripts/quest_artifacts.py --restore
```

The runtime archive restores the selected weights and exact saved parent without replacing divergent files. Starting a new practice session creates its own local assessor and new qualification. **--restore --research** also restores archived checkpoints, final records and audit evidence. Local assessor keys remain outside the public archive; published receipts preserve the historical audit, while fresh sessions obtain their own assessments.

## Inspect what learned

The trained head learned to retain all four valid routes in both evaluated cohorts: **512/512 correctly answered requests per seed**, compared with 232/512 for repeated-success selection. Balanced and greedy coverage also reached 512/512. The method programs, grammar, assumptions, teacher distribution and checkers were supplied; the scorer weights and recorded online investigation updates were learned. [Full teaching and comparison record](report.md).
