# Guarded trace consolidation and proof reuse

I implemented a bounded instance of the packet's executable-world direction. Successful, independently checked reasoning traces become parameterized programs with input/output types, necessary guards, semantic dependencies and evidence status. The existing trained R1 parent is loaded read-only and remains unchanged.

## What is acquired

The learner receives six successful traces for each relation, over the field of integers modulo 11:

- `a*x + b = c` becomes the reusable computation `(c-b) * inverse(a)` with `a != 0`.
- `x + b = c` becomes `c-b`, retained independently.
- After changing the first relation to `a*x - b = c`, six new traces produce `(c+b) * inverse(a)`.

The compiler generalizes varying values from their formal input provenance. It does not retain a table of final answers. All 1,331 finite input triples are checked for each proof; the affine guard admits 1,210 of them. Equation representation, variable-role mapping, field operations and local algebra rules are supplied. This is abstraction from reasoning, not unsupervised discovery of the algebra.

Each candidate is assessed on **1,024 renamed/new-value cases, 8,192 three-stage compositions, 512 violated-assumption cases and 1,024 post-correction cases**. Three paired library-growth conditions add 512 case occurrences each. These are exact finite computation tests; atomic proof already covers the entire finite domain, so I do not call them independent unseen physical worlds. Compositions and teaching bindings are separated, and the composition controller is supplied.

## First candidate and correction

GC-001 preserved conditions and returned no wrong guarded answers, but its full per-candidate count was **732,614**, worse than **621,064** for local reasoning. Its proof recomputed the same forward answers for every right-hand side. I retained that failed efficiency result.

GC-002 builds, for each fixed `a,b`, the exact map from all eleven possible `x` values to their observed right-hand sides once. It still checks every guarded input and all ambiguous preimages. This changes the verification procedure; it does not weaken proof or constitute learned eta.

| Method | Online work | Acquisition/proof/correction/restore | Total work proxy |
|---|---:|---:|---:|
| Repeated finite search | 976,904 | 0 | 976,904 |
| Local reasoning each time | 621,064 | 0 | 621,064 |
| GC-001 indexed dispatch | 403,976 | 328,638 | 732,614 |
| GC-002 indexed dispatch | 403,976 | 123,180 | 527,156 |

GC-002 reduces its declared per-candidate total by **15.1%** versus local reasoning and **46.0%** versus repeated search. These are heterogeneous counted operations, not FLOPs or a general hardware speedup. The full worker took 5.185 seconds including startup, evidence serialization and all controls. Failed GC-001 work and later repair/audit work are additional project costs, not silently amortized into a claimed net project-wide win.

All guarded methods answer every unique in-domain case correctly. All 512 zero-multiplier cases are unsupported; omitting the guard returns **512 invalid answers**. Changing the affine relation invalidates both its shortcut and its dependent composition definition, preserves the unrelated offset program, and allows correction and re-acquisition. The registered R1 owner retains its original identity.

At 128 irrelevant context entries, linear matching consumes 73,218 operations on the paired 512-case panel, versus 7,682 for indexed matching. Measured panel wall times are 0.0470 and 0.0207 seconds, respectively. This is a synthetic library-growth stress test, not evidence of 128 independently learned capabilities. Graph construction, index bytes, comparison bytes and stress costs are separately retained.

## Defects found after the cohort

An adversarial review found that indexing is unsound if the forward relation itself depends on the right-hand-side input. For example, a table built at `c=0` for `x+c=c` can incorrectly certify the program `x=c`. The [failed counterexample](../supervision/GC-002-proof-domain-counterexample/process.log) is preserved. The current checker rejects such relations explicitly. The cohort's affine and offset forward relations depend only on `a,b,x`, so this defect does not change their recorded answers.

I also found that the initial library trusted a caller-declared interpreter identifier. It now compares the actual compiler/proof/migration/shared-source and runtime fingerprint before execution. Old graph identifiers cannot silently authorize the repaired interpreter.

The [explicit migration](repair/result.json) preserves both source graphs, rebinds dependencies to current source, reproves every affected program and restores each successor. **1,024 fresh cases** pass against independent forward arithmetic, with the original trained parent unchanged. The additional repair work is **146,824 operations and 2.728 supervised seconds**. The usable checkpoint is [repair/corrected-library.json](repair/corrected-library.json); the original candidate graph remains [corrected-library.json](corrected-library.json). Their different identities are intentional. The 15.1% benchmark is the frozen candidate's result, not a remeasurement of the repaired lifecycle.

## Limits and integration decision

This implements finite guarded abstraction, persistent executable state, correction, transitive invalidation and explicit migration. The current repaired graph contains about 3.4 kB of serialized definitions, plus its derived index, teaching/proof records and inherited model checkpoint. Numeric R1 parameters added: **zero**. The experiments do not show that this knowledge improves a neural route, learns arbitrary semantic parsing, handles hidden-state physics, or discovers a better learning procedure. General frontier search and autonomous grammar expansion remain open.

I retain the repaired library as an explicit conditional component. Broader integration requires the packet's positive-transfer and retention comparison; shared ownership alone does not satisfy it.
