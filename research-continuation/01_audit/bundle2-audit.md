# Second-bundle audit and reproduction

The fresh analytic study completes and replays successfully, and its scientific conclusions agree with the archive. **Archived-versus-fresh strict numerical parity is incomplete:** two numeric leaves differ beyond the original tolerance in one main record, and 99 in one repair record. The archived repair verifier also fails an exact coefficient hash. These failures are preserved separately from successful fresh self-replay.

See [bundle2-evidence.json](D:/ai/projects/sera/research-continuation/01_audit/bundle2-evidence.json) for commands, code/protocol identities, all 360 record hashes, failures and evidence labels. [parity.json](D:/ai/projects/sera/research-continuation/12_reproductions/bundle2/independent-audit/parity.json) contains the exhaustive comparison of raw records; it does not rerun fitting or modify any source/result.

## Verified identities and completed work

The original archive is `C:/Users/admin/Downloads/SERA_Kavi_Astra_Research_Pack_v2b.zip`, 16,598,467 bytes, SHA-256 `4e9b829f67aca7bbb543b0c03eff022b019b67698a0f27f899fbd7a8186e814f`. All **726 manifest entries match**. Its embedded first ZIP has SHA-256 `e5dd7e67ae30cfe6cbb18f59cebdc253a6eedeefd7a2a15eba1c75c85ce55743`, exactly matching the separately delivered first archive. That nested material is duplicate provenance, not another independent experiment.

The package pins SERA at `a4047173fd33d27d33d8a151f32613eae343df56` and Kavi at `50f743cc44794b67bc1d30b927e25991197edce2`. Its own experiments are an independent NumPy analytic harness. Neither upstream test/training claim is established by this bundle's runs.

| Execution in this continuation | Result |
|---|---|
| Package regression tests | 55 passed |
| Original archive P1/meta replay | FAILED native source identity comparison |
| Portable source-map replay of archive | PASS: 1,050 evaluations, 240 teaching artifacts, 50 snapshots, 620,800 regenerated scoring labels; original `atol=rtol=1e-12` |
| Original archive P2 replay | FAILED: `AssertionError: Value root/artifact/parent_sha256` |
| Full new reproduction | COMPLETED: 7 jobs passed, 59.6842837 charged subprocess-wall seconds |
| Fresh P1/meta self-replay | PASS, same counts and tolerance as above |
| Fresh P2 self-replay | PASS: 100 paired trials, 102,400 sealed numeric predictions |
| Independent archive/fresh file coverage | All 250 main, 10 lifetime, and 100 repair records present; no extra/missing records |
| Independent record/protocol integrity | All 360 archived and 360 fresh outer record hashes and protocol identities valid |

I ran the seven-job reproduction under the 600-second cap with 180-second per-job limits. Its first invocation intentionally stopped after three jobs; the resumed invocation checked completed output/log hashes and continued. This is a controlled pause/resume, not an independent cohort or an unrecorded budget reset. The final command journal is [state.json](D:/ai/projects/sera/research-continuation/12_reproductions/bundle2/reproduction/state.json).

## Identity serialization and numerical failures

The archived experimental source digest is `ce205791aec0e94fbc6c9560375319b359701db3dd8bcb0d6d7cecd8fae62824`. The same bytes yield native Windows digest `4dea8dd2904332171f99fabfa3e6c6da19860fa4a58e6bd98dd6382de210b222`, because `run_study.source_identity()` uses native path separators. The documented [portable wrapper](D:/ai/projects/sera/research-continuation/01_audit/replay_bundle2_portable.py) computes canonical POSIX keys from unchanged file bytes; it does not copy an expected hash or alter inference/tolerance.

The broader source tree has a different digest by definition. Its root-relative POSIX code map produces `d86cacaeedd188acf6e1ee95fa62cd755b720e0b8b2cd206a451f9eb83c98baa`, exactly matching `RELEASE.json`. Its native runner map produces `c49a13f72951283160823385b9c3cf4634d9d48913011ce7d6081fd75ccc7da3`. These are different domains/serializations of source identity, not evidence of silent code substitution.

The exhaustive comparator retains metadata, hash and JSON-size differences separately, and applies the original `np.isclose(a,b,atol=1e-12,rtol=1e-12)` criterion to every floating leaf. It checks nonfloat decisions exactly. When floating program definitions change their content-addressed dictionary keys, it aligns only uniquely coefficient-equivalent definitions and records every changed key.

| Cohort | Float leaves compared | Records with numeric failures | Failed leaves |
|---|---:|---:|---:|
| Main | 198,000 | 1 / 250 | 2 |
| Meta | 473,480 | 0 / 10 | 0 |
| P2 repair | 127,380 | 1 / 100 | 99 |

`main_106_misspecified_frozen.json` has an already unstable rollout: archived mean MSE `2,518,425,956.382755` versus fresh `2,518,425,956.3861213`; one trajectory differs by `0.0538635254` near `4.02948e10`. The relative difference is about `1.34e-12`, slightly outside the declared tolerance. This does not create or remove the underlying catastrophic extrapolation failure.

All 99 P2 numeric failures are in `219_sine_4.json`: 10 coefficient leaves and 89 evaluation leaves. Both runs wrongly select frequency 2 and accept it. The largest error-vector discrepancy is `1.8785328848e-9`; repaired extrapolation MSE differs by `1.1360334895e-10` (about 2.16374 in both runs). The selected regularized normal matrix has condition number approximately 236,188, consistent with amplification of small arithmetic differences. This [conditioning diagnostic](D:/ai/projects/sera/research-continuation/12_reproductions/bundle2/independent-audit/conditioning.json) does not isolate which library/hardware change caused them.

Across all P2 records, 300 hash leaves and 77 deployed JSON byte counts differ. Last-bit float changes can alter exact coefficient hashes and decimal string lengths. Those are actual strict-verifier differences, and they are not erased to obtain a pass. The preserved archive failure is [repair_failure.json](D:/ai/projects/sera/research-continuation/12_reproductions/bundle2/portable-verification/repair_failure.json), explicitly labeled as a transcription of the execution exception rather than raw stderr.

## Reproduced scientific findings

**[C + D] The strongest fixed investigator remains useful.** In-basis interpolation MSE is `0.000450703` for random dense and `0.000229922` for active dense, about 49% lower. Active sparse retains 5.075 of six terms on average but has MSE `0.000240182`; its paired accuracy advantage over active dense is not established. These are small supplied-feature reset-design tasks, and the intervals are descriptive ten-seed cluster comparisons.

**[C + D] Misspecification overwhelms conditional confidence.** Active dense nominal 95% predictive coverage is **21.2890625%** on the omitted-sine family. Sparse active is 20.78125%, with two of ten models passing empirical validation. A posterior over coefficients inside a fixed basis does not represent uncertainty about missing mechanisms.

**[C + D] No reliable investigator improvement is established.** At generation four, interpolation MSE is `0.000420693` for initial knowledge/fixed policy, `0.000288952` for initial knowledge/current policy, `0.000276703` for current knowledge/fixed policy, and `0.000313640` for current knowledge/current policy. The policy-only paired interval is approximately `[-0.0004071, +0.0000375]`; with current knowledge it is `[-0.0000126, +0.0000952]`. Both include zero, and the latter point estimate is worse. Actual policy and knowledge succession occurs; sustained improvement of the learning procedure does not follow.

**[C + D] Repair improves prediction without reliably identifying mechanism.** The following decisions and frequency selections reproduce exactly:

| Family | Accepted / 20 | Correct frequency / 20 | Fixed extrapolation MSE | Repaired extrapolation MSE |
|---|---:|---:|---:|---:|
| Linear | 0 | N/A | 0.003836 | 0.003836 |
| Sine 2 | 0 | 0 | 0.028240 | 0.028240 |
| Sine 2.5, outside grammar | 0 | 0 | 0.232988 | 0.232988 |
| Sine 4 | 20 | 6 | 5.137595 | 0.515983 |
| Sine 6 | 20 | 20 | 6.824574 | 0.004668 |

For sine 4, accepted frequencies are 3 in eight runs, 4 in six, 5 in four, and 2 in two. The support/qualification regime is insufficient to identify the correct explanation in many runs. The next experiment should retain competing explanations and choose distinguishing permitted observations. No result supports calling the supplied sine grammar autonomous invention of trigonometry.

## Evaluator and source-contract limitations

The delivered checks are trusted-process reproducibility checks, not a protected independent evaluator. Several boundaries matter for subsequent work:

- Both replay scripts glob records without asserting the expected record set. Empty or incomplete directories can be reported as a smaller PASS. This audit independently checks complete expected sets from the protocols.
- `verify_results.py` checks the experimental source identity but does not explicitly check each record's protocol hash. This audit checks all protocol identities.
- P2's `runner_sha256` pins only `run_repair.py`, although it imports `features`, hashing, and seed-namespace code. The package manifest covers those bytes here; the P2 field alone does not.
- The P1 experimental identity excludes verifier, runner, analysis code, tests, and protocols. The runner's broader source identity still excludes tests/protocols; test/analysis jobs do not record output hashes. Future frozen evaluation requires explicit identity coverage of those objects.
- Posterior restoration checks mean/covariance against saved sufficient statistics. The archive verifier does not independently rebuild sufficient statistics from raw trace observations or repeat the entire acquisition policy. Fresh reproduction adds evidence but uses the same implementation.
- Exact definition preservation after the repaired snapshot copy is not learned applicability, shared neural behavioral retention, latent-state migration, or hostile-process isolation.

**[A] Preserved failures remain authoritative history.** The initial 260-record cohort had mutable program-library snapshots; the repaired canonical cohort corrected ownership without changing test banks or counting reruns as new seeds. The original hard-killed runner also undercounted completed work before durable reservation was introduced. Those failures and the old code are preserved in `evidence/` and `results/initial/`; they should not be removed from the evidence chain.

**[G] Integration remains proposed.** This bundle supports common provenance/ownership contracts, representation uncertainty, guarded generators, and a fixed information-gain baseline for further experiments. It does not implement a complete merged SERA/Kavi shared owner, learned guard discovery, unrestricted online causal learning, or recursive learning-algorithm improvement. Full repository baseline tests and later continuation experiments belong to separate evidence records.
