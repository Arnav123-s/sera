# Fresh execution of pinned SERA

I checked out `a4047173fd33d27d33d8a151f32613eae343df56` in the isolated local source directory and executed its original suite using that checkout's `src` path. All **71 tests pass**. The supervised invocation took 29.377 seconds and reached 498,638,848 peak committed job bytes.

The reduced shared study also completed: one delta seed (19), two pretraining steps, two adaptation steps, support size eight, eight objective samples. Its retained-sample and typed-sample defaults remained 256 and 128. It wrote base, adapter, full, replay, scoped and scratch checkpoints, evaluation records and progress state. This is a plumbing check; the tiny optimization budget is not a model-quality benchmark or a fresh reproduction of the full 0.5 study.

The source identity is `a2e334bd8b13bf35754e51044da8d7e8988a91b46dcc6aa78e6310eb03e34853`. The study took 45.787 supervised wall seconds and 375,406,592 peak committed job bytes. Original full-training evidence remains preserved and separately verified.

[Test results](../14_release/pinned-SERA-tests.xml) · [Plumbing resource and command record](../14_release/pinned-SERA-plumbing-resources.json). Complete local source, logs and checkpoints remain in `00_sources/sera-pinned/` and `12_reproductions/sera-pinned/`.
