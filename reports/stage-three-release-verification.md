# SERA 0.3 release verification

I verified the final executable source `0d28049174092ae48ff91af9ff37c2478eedf51c159c41e3ccdcec6f39ce72bd` against the declared three-seed study.

- 55 local regression cases pass; Ruff and Git whitespace checks pass.
- All 45 original numerical checks and 36 original checkpoint checks pass; all 36 source model/seed combinations were trained from scratch.
- The final cohort records 75 episodes, 558 measured interventions, nine trained policy versions and fifteen persistent admission decisions.
- Independent audit reexecutes every saved action program, validates every version and replay, rechecks all fifteen decisions, and recomputes one full paired admission per seed.
- All fifteen predictor checkpoints reproduce ordinary, repeated and long-random scores: 45 comparisons with zero metric difference.
- Three full all-branch reference models are independently reconstructed; learning histories and evaluation agree, and trained per-event diagnostic sessions survive serialization.
- Ordinary typed inference and live-state continuation agree in fresh processes.
- The separately installed 0.3.0 wheel loads the new solver and the historical 0.2 solver. Its SHA-256 is `ded3019ee0bd6e4a5f1df2cc572500b978d5e59eb98cf26fc1496c1837b8ae35`.
- A working copy at `runs/sera-0.3-current` completes an additional CLI learning round; its result is separate from the frozen study.
- Twenty evidence artifacts and all 154 original source concepts pass release verification. The scientific figure was visually inspected.

Public Windows/Linux CI is the remaining publication check. Its outcome will be recorded after the push.

The measured results and limitations are in the [study](stage-three-study.md), [architecture audit](stage-three-architecture-audit.md), [artifact audit](stage-three-verification.json), [installation record](stage-three-installation.json) and [continued-learning record](stage-three-continuation.json). Some interrupted development cost records are unavailable and explicitly excluded from complete-cost claims.
