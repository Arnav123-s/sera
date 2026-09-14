# SERA workspace index

I preserve earlier work and give each location one explicit role. This index is the navigation layer; existing evidence paths stay stable so old scripts, journals and source references remain usable.

- **Current source:** [src/sera](../src/sera).
- **Active continuation solver:** [runs/sera-0.3-current](../runs/sera-0.3-current).
- **Frozen 0.3 study:** [runs/stage-three-complete](../runs/stage-three-complete), canonical seeds 0/1/2.
- **Current architecture judgment:** [source-packet comparison](../reports/source-packet-comparison.md).
- **Structured catalog:** [workspace-catalog.json](workspace-catalog.json).

Ignored local directories are available in this workspace; their links are not downloads from the public repository. Published measurements remain in reports. The catalog does not claim an off-device backup of local checkpoints.

## Preservation record

The review began with 1,972 work files totaling 192.13 MiB. A compressed per-file SHA-256 inventory is preserved at `runs/source-packet-review/preservation-baseline.json.gz`. Git history, the installed third-party environment and runtime caches are outside that work inventory.

Verification found 1,967 unchanged files, 5 corrected documents whose original bytes remain in commit `2f6207060473436891e31fcc15b799b13605a82a`, and 0 missing files. No existing work was moved, removed or deduplicated.

## Location and role

Counts below describe the starting inventory; the new audit has its own location. The worker subtree is counted separately from canonical study files.

| Location | Role | Files | MiB | Purpose |
|---|---|---:|---:|---|
| [.github](../.github) | active-verification | 1 | 0.00 | Public continuous integration configuration. |
| [build](../build) | build-snapshot | 33 | 0.26 | Existing generated build tree; never the source of truth and not updated by this review. |
| [dist](../dist) | release-artifacts | 11 | 4.79 | Versioned wheels/source archives and historical packaged solver. Existing filenames are never overwritten by this review. |
| [examples](../examples) | active-interface | 1 | 0.00 | Declared typed request examples. |
| [reports](../reports) | research-evidence | 38 | 11.43 | Versioned measurements and reviews. Current interpretation: source-packet-comparison.md; frozen 0.3 data: stage-three-data.json. |
| [repository-root](..) | project-entry | 6 | 0.01 | Project description, dependencies and contribution settings. |
| [research](../research) | research-record | 17 | 0.15 | Current design/provenance documents and explicitly dated historical protocols. The source-packet audit supersedes completion claims. |
| [research/intake](../research/intake) | historical-intake | 8 | 0.20 | Early source reading, naming and repository setup records; nested source and installation trees have separate roles. |
| [research/intake/Quantum_AGI_Research_Package](../research/intake/Quantum_AGI_Research_Package) | source-reference | 166 | 18.64 | Complete original packet extraction; keep embedded names and source bytes. |
| [research/intake/release-check](../research/intake/release-check) | historical-package-check | 45 | 0.47 | Early release verification files; not active code. |
| [research/intake/renders](../research/intake/renders) | source-visual-review | 2 | 0.34 | Original source page rendering artifacts. |
| [research/intake/sera-release-check](../research/intake/sera-release-check) | historical-package-check | 46 | 0.48 | Early SERA renamed-release checkout; preserved source snapshot, not a reference architecture. |
| [research/intake/visual-review](../research/intake/visual-review) | source-visual-review | 3 | 0.49 | Original visual review records. |
| [research/intake/wheel-check](../research/intake/wheel-check) | historical-package-check | 23 | 0.18 | Early installed-wheel snapshot; not active code. |
| [research/reference-materials](../research/reference-materials) | source-reference | 81 | 20.96 | Professionally named byte-identical references; original-package is a partial reading extraction. |
| [runs/baseline-v1](../runs/baseline-v1) | historical-evidence | 74 | 3.53 | First SERA mechanism cohort; changed task encoding and models relative to the original packet. |
| [runs/cli-learning-check](../runs/cli-learning-check) | historical-verification | 28 | 1.54 | 0.2 CLI continuation check; preserve its journal and preparation record. |
| [runs/connected-dev-r1](../runs/connected-dev-r1) | development | 2 | 0.07 | Early connected R1 model probe. |
| [runs/connected-smoke-v2](../runs/connected-smoke-v2) | smoke-check | 61 | 3.29 | Short 0.2 integration check; not release capability evidence. |
| [runs/connected-v2](../runs/connected-v2) | incomplete | 10 | 0.54 | Early 0.2 cohort with a manifest but no complete summary; exact termination is not reconstructed. |
| [runs/connected-v2-final](../runs/connected-v2-final) | historical-release-evidence | 234 | 11.53 | Completed 0.2 study, preserved with its own source and verification. |
| [runs/instrument-complex-0](../runs/instrument-complex-0) | development | 2 | 0.00 | Early single-instrument run; not an extra seed of the later suite. |
| [runs/instrument-suite-v1](../runs/instrument-suite-v1) | historical-evidence | 13 | 0.03 | Earlier separate real/complex EventInstrument suite. |
| [runs/integrated-v1](../runs/integrated-v1) | historical-evidence | 14 | 0.16 | First integration and adaptation evidence. |
| [runs/matched-delta-v1](../runs/matched-delta-v1) | historical-control | 11 | 0.63 | Early parameter-count control; use its own design and source identity. |
| [runs/original-reproduction-stage-three](../runs/original-reproduction-stage-three) | source-reproduction | 76 | 1.01 | Original 12 cores: numerical checks, 36 checkpoint evaluations and 36 fresh training/adaptation/scratch procedures. Separate from SERA models. |
| [runs/original-source-audit](../runs/original-source-audit) | historical-verification | 75 | 2.38 | Earlier source audit, rendered reference pages and baseline source archive. |
| [runs/package-check](../runs/package-check) | historical-package-check | 33 | 0.26 | Earlier package installation check; not source to edit. |
| [runs/release-check-v2](../runs/release-check-v2) | historical-package-check | 75 | 2.69 | Preserved 0.2 checkout/install verification tree. |
| [runs/sera-0.3-current](../runs/sera-0.3-current) | active-solver | 28 | 3.71 | Only designated continuation workspace; seed-zero 0.3 solver plus a separately recorded CLI learning attempt. Not a new study seed. |
| [runs/sera-current](../runs/sera-current) | historical-solver | 24 | 1.24 | Preserved 0.2 continuation workspace despite its old current name; not the active solver. |
| [runs/sera-v3-current](../runs/sera-v3-current) | superseded-solver | 27 | 3.65 | Working copy from the obsolete 0.3 cohort. Do not pool its results with the repaired release. |
| [runs/stage-three-audit](../runs/stage-three-audit) | verification | 6 | 10.95 | Independent reconstruction of trained reference memories and live sessions; not additional training-cohort seeds. |
| [runs/stage-three-complete](../runs/stage-three-complete) | frozen-release-evidence | 190 | 31.19 | Canonical repaired 0.3 study, seeds 0/1/2. Checkpoints, journals and measurements are immutable. |
| [runs/stage-three-complete/worker-runs](../runs/stage-three-complete/worker-runs) | preserved-worker-evidence | 195 | 31.04 | Original worker outputs. Canonical 0/1/2 copies belong to these same runs, not additional independent seeds. |
| [runs/stage-three-development](../runs/stage-three-development) | development | 10 | 1.27 | Tuning, pilots and probe results. Exposed development evidence; excluded from the final cohort. |
| [runs/stage-three-final](../runs/stage-three-final) | interrupted | 2 | 0.01 | Stopped after discovering the untrained legacy decoder in the study setup. Incomplete cost records. |
| [runs/stage-three-installed](../runs/stage-three-installed) | superseded-package-check | 42 | 0.37 | Installed pre-repair package labeled 0.3.0; source identity, not version string, distinguishes it. |
| [runs/stage-three-installed-final](../runs/stage-three-installed-final) | release-package-check | 42 | 0.37 | Installed repaired 0.3 package used for final installation checks. |
| [runs/stage-three-release](../runs/stage-three-release) | failed-cohort | 91 | 14.66 | Obsolete source ee62373c; seed one failed on all-missing likelihood targets. Preserve complete and partial outputs together. |
| [runs/stage-three-smoke-1](../runs/stage-three-smoke-1) | failed-smoke | 53 | 7.04 | Short integration run rejected by source-freeze checking; not capability evidence. |
| [runs/stage-three-verified](../runs/stage-three-verified) | interrupted | 6 | 0.12 | Repaired serial cohort stopped before independent-worker orchestration. Not the final cohort. |
| [scripts](../scripts) | active-verification | 14 | 0.10 | Reproduction, reporting and audit entry points. |
| [src/sable_learning.egg-info](../src/sable_learning.egg-info) | historical-package-metadata | 6 | 0.00 | Preserved early package-name metadata; not an active second project. |
| [src/sera](../src/sera) | active-code | 33 | 0.26 | SERA 0.3 executable source; edit here for subsequent versioned research. |
| [src/sera_learning.egg-info](../src/sera_learning.egg-info) | package-metadata | 6 | 0.00 | Existing editable-install metadata; generated, not model source. |
| [tests](../tests) | active-verification | 8 | 0.04 | Numerical and lifecycle regression suite. |
| [runs/source-packet-review](../runs/source-packet-review) | current-audit | new | new | Baseline, supplementary reproduction and integration probes from this review. |

## Avoiding duplicate work

The baseline contains 458 groups of byte-identical content, representing 72.92 MiB of additional logical file copies. This includes canonical/worker outputs, reference extractions, packaged copies and repeated checkpoint components. It is not a measurement of reclaimable disk space.

I retain these copies because they identify delivered sources, independent worker outputs, installations or historical workspaces. I do not count copies as independent evidence. New work uses one declared output directory; reports reference evidence instead of copying whole run trees. Installed packages and build snapshots are inspection artifacts, not parallel source branches.

## Revisiting earlier work

Read the directory role and its manifest first. Match the recorded source identity to the preserved source snapshots in the catalog. Several installed copies have the same package version and different source hashes; version 0.3.0 alone does not identify the repaired implementation. A run without recorded provenance remains labeled incomplete or development, rather than receiving an invented release identity.

To continue learning, start from the active continuation directory or use `scripts/prepare_solver.py` with an explicit completed seed and a fresh output. Do not train into canonical study directories or overwrite an earlier wheel/source archive. The historical `sable/data-v1` RNG namespace remains unchanged because changing it would change recorded datasets.

To verify preservation again:

```text
python scripts/catalog_workspace.py --baseline runs/source-packet-review/preservation-baseline.json.gz --check
```

The per-file inventory and hash checks establish preservation of the work present at this review's start. Earlier Git revisions preserve previous document claims. New audit outputs extend the record; they do not silently replace the 0.3 experiment.
