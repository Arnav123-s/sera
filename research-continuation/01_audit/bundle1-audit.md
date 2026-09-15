# First-bundle audit and reproduction

The first bundle's saved **accuracy and correctness outcomes reproduce completely**, but its original strict probability verifier does **not** pass on this Windows host. All 57 substantive checkpoints reproduce 603 condition records and 411,648 correctness elements exactly. Two NLL values exceed the unchanged `1e-6` tolerance. This is inference replay of old tests; no neural training run or upstream v0.5 study was rerun by this audit.

The machine-readable evidence, per-checkpoint hashes, commands, discrepancies, and labels are in [bundle1-evidence.json](D:/ai/projects/sera/research-continuation/01_audit/bundle1-evidence.json). The full raw-record/history audit is [bundle-data-audit.json](D:/ai/projects/sera/research-continuation/12_reproductions/bundle1/bundle-data-audit.json). Fresh replay scores and all comparison records are under [exhaustive-replay](D:/ai/projects/sera/research-continuation/12_reproductions/bundle1/exhaustive-replay).

## Identity and scope

| Object | Verified identity or scope |
|---|---|
| Original archive | `D:/SERA_Research_Bundle_2026-09-14.zip` |
| Archive SHA-256 | `e5dd7e67ae30cfe6cbb18f59cebdc253a6eedeefd7a2a15eba1c75c85ce55743` |
| Intake | `D:/ai/projects/sera/research/intake/probability-cloud/sera-bundle/SERA_Research` |
| Fresh working copy | `D:/ai/projects/sera/research-continuation/12_reproductions/bundle1/working` |
| Manifest | 176 of 176 entries match; 177 delivered files including the manifest itself |
| Pinned upstream referenced by bundle | `a4047173fd33d27d33d8a151f32613eae343df56` |
| Frozen `benchmark.py` | `65c83ffcc251255dfa2de2c616b6ed6e512d65442d030a35c6d15245ebe40e7e` |
| Frozen `followup.py` | `ce93af98d929a894851951b7786f45d2180ba392e3b82499fb579d99914ee475` |
| Original runtime | Python 3.13.5, Torch 2.10.0+cpu, NumPy 2.3.5, Linux |
| Continuation runtime | Python 3.12.14, Torch 2.10.0+cpu, NumPy 2.5.3, Windows 11; CPU only |

Both frozen code hashes match their protocol records. The original archive, intake, old results, and working-copy executable source bytes remain unchanged. Working-copy diagnostic outputs were deliberately regenerated; their archived counterparts remain in intake. The bundle contains 27 main and 30 follow-up substantive records, two excluded 100-update pilots, and five finite cyclic-program searches. Its 13 model-name strings represent 12 structural variants because `novelty_matched` repeats the soft novelty structure with a different initializer.

I read the complete README, handoff, report and cookbook Markdown, source/evidence ledgers, protocols, execution notes, every result log, and all model/evaluation/diagnostic source used below. Every substantive raw JSON was parsed and checked, including its complete saved score vectors, validation history, checkpoint identity, and metric bounds. PDFs were hashed as delivered renderings; visual layout QA was not repeated in this model-result audit.

## What was newly executed

| Check | Outcome on this host |
|---|---|
| Manifest and source identities | PASS: 176 entries, frozen protocol hashes, unchanged intake/code |
| Original verifier without adaptation | FAIL at import: Unix-only `resource` unavailable on Windows |
| Original verifier with import-only shim | FAIL after 49 completed checkpoints at its original NLL assertion |
| Exhaustive comparison using unchanged constructors/evaluator | 57/57 checkpoint states finite and reconstructable; 603/603 accuracy/vector records exact; 601/603 NLL checks below `1e-6` |
| All saved training-record consistency checks | PASS: 57 records, correct validation cadence, earliest best validation step, binary vector/accuracy consistency, finite bounded metrics |
| Recomputed aggregate tables | All capability means/SDs match; one old CPU mean differs by `1.78e-15` from summation order |
| Mathematical/numerical checks | PASS: all invariant assertions and expected counterexamples |
| Cyclic program search | PASS: 5 runs, identical non-timing outcomes, selected map `[0,1,2,3]`, all 25 test accuracies 100% |
| Split audit | Exact match across 8 training streams |
| Gate probes | 25 checkpoints; maximum scalar difference `1.7881393433e-7` |
| Density diagnostic | Numerical behavior reproduces; original speed ratio does not reproduce under this timer |
| Wolfram symbolic checks | Characteristic polynomial, reflection product, and bound/retention arithmetic reproduce |

The Windows shim only supplies an importable `resource` module; it raises if memory accounting is invoked. Model forward passes, data generators, weights, and the original verifier tolerance were not changed. The separate exhaustive audit avoids stopping at the first failed condition and also compares Brier scores. Its three disjoint checkpoint partitions are execution shards, not additional scientific seeds. The forensic replay used 196.40625 recorded CPU seconds, excluding the earlier attempted verifier and other audit work.

| Checkpoint | Condition | Replayed accuracy | Absolute NLL difference |
|---|---|---:|---:|
| `novelty_sharp_12` | `first_L256` | 74.609375% | `1.1424735931e-6` |
| `novelty_sharp_13` | `first_L256` | 59.1796875% | `1.1295969671e-6` |

The maximum Brier difference is `9.7706152857e-7`. The small numerical differences are consistent with different platform/library arithmetic, but that explanation was not isolated experimentally. The preserved original failure is [checkpoints-windows-replay.log](D:/ai/projects/sera/research-continuation/12_reproductions/bundle1/logs/checkpoints-windows-replay.log). **The working copy's inherited `results/checkpoint_verification.json` remains the bundle's previous record**, because the failing verifier never rewrote it; current status is [REPRODUCTION_STATUS.json](D:/ai/projects/sera/research-continuation/12_reproductions/bundle1/REPRODUCTION_STATUS.json).

## Positive findings and what they establish

**[C + D] History-aware writing is the strongest reproduced local mechanism result.** Across the original three main seeds, novelty delta has 100% first-value recall at L8, 63.9323% at L64, and 99.6745% latest-value recall at L64. The source-style delta has 57.4870%, 35.0911%, and 93.2292%, respectively. The local-delta control does not recover the novelty result merely by removing global decay. These are paired saved outcomes now replayed, not newly trained independent replications.

**[D + E] The mathematical boundary checks are useful and reproducible.** Complex-to-real multi-Kraus probabilities agree within `2.22e-16`; RLS agrees with batch ridge within `5.00e-16`; the rare-event example maps prior trace distance `1e-4` to posterior distance `1`; and the asymmetric erase transition has operator norm `1.14412` despite eigenvalues 0.5 and 1. The frozen-projector example again gives true derivative about `-0.26869247` versus surrogate `+0.04612236`. Finite checks complement the stated derivations; they are not proofs of stability under arbitrary feedback.

**[D] The finite-program success is reproducible within its supplied grammar.** All five searches identify the correct cyclic increment map from final labels and pass through length 1024. This supports retaining executable generators when the hypothesis class is appropriate. It establishes neither open-ended program discovery nor compactness after charging the full grammar/interpreter.

## Negative findings and limits to preserve

**[C + D] Initialization failures remain substantial.** Ordinary novelty delta's follow-up mean is about 99.7% at L8, whereas the exactly initializer-matched soft arm is about 73.2% with large variance. Sharp-gate seeds 10 and 14 reach only 35.35% and 35.94% first-value recall at L8; seed 11 reaches 100% through L256. QR alone does not remove these failures. The post-hoc matched control repairs an initializer confound, but cannot retroactively make the follow-up a sealed confirmation.

**[C + D] Long memory and arithmetic are unsolved.** Ordinary novelty delta falls to about 33.5% first-value recall at L256. Sharp mechanisms improve some long-sequence cases while retaining failed seeds. The follow-up modular results remain around 25% chance. Adding the tested rotor or density branch does not create a general algorithmic capability and can harm first-value behavior under this protocol.

**[D] Exact density's numerical reference value survives; its speed claim does not transfer directly.** The maximum final trace distance between truncated and full trajectories is `0.188952565`, matching the archived `0.188952506` closely. The +6,144-byte workspace change and approximately 17.1% total-core increase are arithmetic facts under the given layout. On this Windows host, both measured CPU medians are 31.25 ms, versus the archive's 16.84/14.35 ms and ratio 0.852. The observed process-time readings are quantized, so the tie is not evidence of equal underlying speeds. No task-accuracy advantage follows from this microbenchmark.

**[D] Finite-ID overlap is real.** There are no exact binding collisions at L8 in the eight audited streams, but 108–144 of 1,024 modular test strings already occurred during training. Length-24 and longer tests are disjoint by length. Paired first/latest queries and reused histories across architectures cannot be counted as independent discoveries.

**[A + C] Source evidence is strong for saved inference, narrower for acquisition.** Checkpoints contain inference state dictionaries without complete optimizer/RNG snapshots. Reported 4,377,600 training draws and 562.54347119 completed training-loop CPU seconds are internally consistent, but replay does not verify every historical optimizer step or the unaccounted interrupted work. The four key identities, explicit membership bits, both trained rules, finite cyclic grammar, and supplied task routing are privileged structure. The bundle does not implement learned entity discovery, learned applicability, persistent live-state migration, or improvement of the learning procedure across generations.

## Applicability to the current shared-owner SERA

**[G] Promote the history-aware write mechanism to an optional, initializer-matched integration experiment behind the existing shared owner.** Preserve the source-style and local-delta controls, early- and late-instruction tasks, query bypass, owned state, serialization, and variable-cardinality/renamed-key tests. Explicit membership should remain a disclosed control until learned uncertainty or novelty succeeds under comparable resource accounting.

**[G] Treat full density as a small-dimensional numerical reference and executable generators as a controlled acquisition path.** Each still needs actual shared-task training, budget controls, learned applicability or verified guards, and retained-capability testing. These findings do not establish that the current local R7 experiment or any later repository commit has passed those requirements.

**[B] Upstream v0.5 results remain separate.** This bundle audited selected pinned source and quoted committed reports; it never possessed or reran the full upstream shared study. Its 57 checkpoints are independent reduced models, not the upstream million-parameter shared owner. Current repository HEAD and pinned-repository test results belong to the main continuation's separate evidence chain.
