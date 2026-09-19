# Restoring the qualified owner across platforms

GitHub [run 53](https://github.com/Arnav123-s/sera/actions/runs/35413935752) passed on Windows (568 tests passed, 45 optional-fixture skips). Linux restored the archives and earlier interfaces, then stopped at the current owner's strict saved-qualification comparison. I retained the [complete run receipt](github-run-53.json), [original logs](github-run-53-logs.zip) and [Linux job log](github-job-105818748487.log).

I added a separately pinned restoration adapter. The [contract](portability-contract.json) pins its source and the unchanged scientific release manifest. It verifies both receipt hashes and retains the original comparison for all fields outside the independent numerical refit: the source, original goal, evidence, learned-weight identity, selection, adequacy and admission decisions. It then checks independent refit predictions within 0.000002 absolute probability and likelihood within 0.0000000001, with the original fit-qualification threshold and decision unchanged. Optimizer iteration counts and termination descriptions are recorded separately. No training, final-evaluation threshold or saved model is changed.

Twelve targeted checks passed locally, including exact owner restoration, rejection of changed source/goal/evidence/weights/adequacy/admission, and rejection of a materially changed independent fit. The adapter applies only while restoring through the current entry point. The original strict implementation and sealed archive remain available unchanged.

Native Linux replay identified one fallback in the current-interface run: `pulse_loss-47106-passive`. SciPy reported convergence by relative objective reduction instead of projected gradient. The independent refit's prediction discrepancy changed from `2.0801859124119915e-9` to `9.823485358495532e-10`; the maximum difference between the two independent fits was `1.0978373765624383e-9`. Source, goal, evidence, learned-weight identity, model adequacy and admission were unchanged. The recorded tolerances were fixed before this replay. [Structured diagnostics](native-portability-diagnostics.json) · [Original native log](github-job-105821417516.log).

Linux completed all 39 current example requests, passed 579 tests with 46 established optional-fixture skips, and passed every release check. The complete matrix result is recorded in [publication verification](VERIFICATION.md).

The new adapter is engineering for reproducible restoration. It adds no learned capability and changes no scientific conclusion.
