# Working on SERA

I welcome reproducible experiments, clear failure reports and focused implementation improvements. I use the following conventions to keep results comparable.

Install `.[dev]` in a local environment. Run `python -m ruff check src tests scripts` and `python -m pytest` before committing implementation changes. CI runs those checks on Linux and Windows with Python 3.12 and CPU PyTorch 2.10.

Experiments must record configuration, data-generator namespaces, source identity, dependencies, parameter count, working-state bytes and failures. Keep the original package and raw local runs separate from selected report artifacts. Store large model checkpoints outside Git history.

Choose mechanisms by task evidence. An architecture name, a valid density matrix or an appealing phase plot does not establish useful learning. Retain decisive classical controls and negative outcomes.

Final evaluation data belongs to the evaluator. Freeze a candidate before generating fresh promotion data. Do not use final test performance to select checkpoints or silently relabel an exposed test set as independent evidence. Claims about generality require new generators and task families, not only new random examples.

The finite interpreter is intentionally narrow. Extensions must define an input domain, execution budget, numerical/type contract and independent tests. Generated programs are data; do not add unrestricted `eval`, shell execution or automatic package installation to the learner.

Version 0.2 learns an intervention selector, while optimizers, task/domain routing, symbolic colors and program grammar remain supplied. Preserve those assumptions in summaries. A successful software lifecycle or a trained controller does not establish superiority over fixed methods; preserve the measured comparisons and failures.

The connected study freezes source and policy before final tests, retains support-record identifiers, and uses a separate namespace for each meta episode. Run `python scripts/audit_connected_run.py --input RUN_DIRECTORY` to reproduce exposed admission decisions; label that exercise as verification, not new generalization evidence. Report generation from committed evidence must work in a clean checkout.
