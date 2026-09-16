# The larger target: learn prerequisites in order to perform a task

The user's latest direction is a persistent task-driven researcher. Given a difficult objective, it should identify missing concepts, acquire the relevant language and domain skills, study current material, test itself, investigate and simulate, and attempt the original problem. The user now permits declining performance on older tasks when reacquisition is possible. Older states, sources and costs still remain preserved.

This is the project target, not a description of the present model. The current implementation cannot learn English from arbitrary raw documents, deeply interpret research papers, synthesize general proofs, or solve open mathematical problems. The numerical workbench is an executable early acquisition loop with limited task representations.

## Required loop

1. Represent the actual goal and how success could be checked. Detect ambiguity before optimizing a proxy.
2. Diagnose prerequisite gaps against demonstrated skills, with a dependency graph and explicit uncertainty. Generating the graph is itself a capability to evaluate.
3. Retrieve source material with provenance and timestamps. A document is input data, not authority to change the learner's objective or execute arbitrary code.
4. Learn usable representations and procedures from grounded examples. Distinguish retrieved context, stored facts K, adapted parameters, and independently improved investigator eta.
5. Generate exercises and candidate counterexamples, then check them using independent mechanisms. Self-generated model answers cannot certify themselves.
6. Attempt subgoals using acquired skills, counterfactual reasoning and permitted tools. Acquire new observations when the evidence is insufficient.
7. Validate the original objective and assumptions. Retain unsuccessful attempts, costs and predecessor states; consolidate only qualified procedures.
8. Revisit missing or degraded skills when a later task requires them. Compare warm reacquisition with scratch learning and account for the maintenance burden.

## The Riemann-hypothesis example

As checked for this continuation, the [Clay Mathematics Institute lists the Riemann hypothesis as unsolved](https://www.claymath.org/problem/unsolved/). There is no justified promise that SERA will solve it. The example specifies the desired acquisition process; it does not supply its missing capabilities.

A useful progression would require language and notation grounding, mathematical definitions, logic and proof construction, real and complex analysis, analytic number theory, and evidence-based understanding of the zeta function. These prerequisites are a research proposal rather than an automatically discovered, sufficient curriculum. Numerical investigations of zeros and correct answers to quizzes would not constitute a proof of the hypothesis.

For a formal route, the [Lean reference](https://lean-lang.org/doc/reference/latest/ValidatingProofs/) describes checking proof terms against the actual formal statement, definitions and axioms. A check must reject placeholders and audit assumptions; a formally checked statement still has to express the intended mathematical claim. Lean was not available on PATH in the current local environment, and no Lean proof or installation is claimed here. The current finite arithmetic checker is much narrower.

## What this release contributes

The workbench retrieves a matching local material record or accepts new paired examples, fits a compact numerical rule on request, selects it using a separate split, tests held-out examples, executes from registered R1 coefficients, and preserves acceptance or withholding. Live data tasks update as the inbox changes. The independent eta candidate is trained and tested across actual successor histories, but rejected against the strongest baseline.

The [test-time training paper](https://proceedings.mlr.press/v119/sun20b.html) studies adaptation using a specified self-supervised objective under distribution shift. It does not establish that an untrained language learner can understand arbitrary tutorials or become expert in an arbitrary domain. Our paired-example fitting is a narrower supplied-family acquisition mechanism, not a reproduction of that method.

The next foundational research question is a grounded task/language representation with independently checkable acquisition and transfer. Keep a small executable proof-and-language curriculum separate from the open-problem target. Increase scope only after demonstrations on unfamiliar tasks, adversarial misunderstanding, acquisition cost, and reacquisition after forgetting. An existing pretrained language model could be evaluated later as an explicitly labeled interface/control; it must not silently replace the from-scratch SERA research claim.
