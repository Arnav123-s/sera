# QP-001: learning to retain verified alternatives

I extended the continuing CompletionR1 owner with a 337-parameter method selector, a persistent quest board and an independent portfolio assessor. SERA learns to select complementary checked methods. It retains alternative assumptions, unfinished questions, unsuccessful attempts and the evidence behind each reward. Its existing learned investigator, joint completion cloud, STOP action and analytic recommendation remain available on the same owner.

## What was taught

The supplied practice menu contains time integration, impulse balance and two signed work–energy branches, plus two defective candidates. Twenty-four exact aliases test identity handling. These are four distinct conditional routes in three mathematical families. The existing StudyR1 integration weights execute the time route; the impulse and energy programs, units and assumptions are supplied executable contracts.

The new selector receives twelve public features per candidate: unmet scope weights, claimed input coverage, cost, whether the candidate was visited, the attempt index and STOP. It learns its 12→24→1 scoring weights from checked outcomes. A fifth attempt is reserved for a least-visited alternative; its fixed choice earns no policy-gradient credit.

Both reward arms use two seeds, 300 Adam updates and 64 episodes per update. That is **19,200 repeated episode presentations per seed/arm**, 76,800 overall, and 384,000 attempt slots. Training draws from 2,048 supplied context vectors; development uses 256 separate contexts. These counts distinguish repetitions from independent examples.

The verified reward adds newly correct weighted coverage and a one-time 0.03 checked-method bonus, then subtracts 0.01 times method cost. The comparison pays for repeated success. Live credit maintains one cumulative high-water record across quests and aliases. Offline training episodes simulate separate acquisition lifetimes. Scientific evaluation always uses the common verified objective; the evaluation field named own_reward consequently refers to this common rollout objective, while training traces retain each arm's actual reward.

## Frozen results

Both verified-reward seeds selected all four valid routes on development data. Their utility tied; the frozen candidate order selected seed 3301. Selection was written before the final cohorts were opened.

Final evaluation uses 256 matched and 256 shifted scope contexts. A separately generated rational mechanics request is deployed for each context. These requests sample all four input scopes, while coverage weights describe the context's stated preferences. Returned-answer accuracy and preference-weighted coverage are therefore different measurements. Every portfolio is screened using practice evidence; the final answer never selects the route.

| Controller | Correct / 512 | Weighted coverage, matched | Weighted coverage, shifted | Cost, matched / shifted | Distinct valid routes |
|---|---:|---:|---:|---:|---:|
| Verified reward, seed 3301 | 512 | 100.00% | 100.00% | 8.50 / 8.50 | 4.00 |
| Verified reward, seed 3302 | 512 | 100.00% | 100.00% | 8.50 / 8.50 | 4.00 |
| Repeated success, either seed | 232 | 72.99% | 25.52% | 13.00 / 13.00 | 2.00 |
| Initial selector | 232 | 72.99% | 25.52% | 13.00 / 13.00 | 2.00 |
| Balanced practice | 512 | 100.00% | 100.00% | 8.50 / 8.50 | 4.00 |
| Useful-source random | 453 | 95.44% | 90.07% | 9.95 / 10.01 | See raw cohort records |
| Greedy complementary coverage | 512 | 100.00% | 100.00% | 8.50 / 8.50 | 4.00 |

The verified learner spends **34.6% less method cost than repeated-success selection** in this protocol. Its repeated-attempt fraction is zero; the repeated-success arm repeats 60% of its five slots. Every returned numerical answer passed; uncovered requests produced abstentions. All controllers preserved abstention on the separately labeled omitted-drag inputs. [Complete measurements](final.json) preserve both seeds and all controls.

Balanced and greedy coverage matched the learned policy. Balanced remains the default. The learned selector is retained for explicit checked on-policy practice, with its full comparison record. The evaluation supports learning a useful selection procedure over these supplied methods.

## Actual-owner acquisition, correction and return

The audited lifecycle began with an empty portfolio: 0/256 fresh checks and an open capability gap. Five balanced attempts retained four checked routes and a failed wrong-sign candidate. A separate process then assessed the frozen owner on 256 fresh cases: 256/256.

After an exact restart, a new on-policy practice choice updated the method weights. Its repeated known connection earned **−0.01**, its incurred cost, rather than another discovery bonus. The earlier qualification became stale. A further Stage 32 practice cycle also updated the retained investigator, increasing its checked update count from one to two. The owner then passed another 256 fresh checks. Attempt three's allocated alpha was 0.00208333 and its exact lower bound was **0.976172**.

The qualified live owner returned **4 m/s** for a 3 kg body receiving 12 N·s of impulse from rest. The worked examples also describe the same 6 N, 3 kg, 2 s scenario through time and work, preserving each route's assumptions. [Runnable guide](README.md) and [live audit](integration-audit.json) provide the inputs and retained state.

Qualification is tied to owner identity, semantic scope, the frozen portfolio and independent assessment. Every owner change invalidates eligibility until revalidation. Failed assessments spend their attempt and preserve counterexamples. Cosmetic points affect neither correctness, qualification nor resource permissions. Global allocation uses 0.05 / [j(j+1)t(t+1)] across canonical quests and attempts; renamed display IDs reuse the same quest.

## Independent verification and preservation

All four 150→300 training resumptions reproduced weights, optimizer, random generator, baseline and per-step traces exactly. An independent NumPy forward pass reproduced 512 actions with maximum logit difference 4.44e-16. A separately written latent-endpoint generator checked 128 fresh executions through the actual owner with zero numerical error. [Replay](replay.json).

Training retained all 194 predecessor tensors. The live audit intentionally updated the four Stage 32 investigator tensors; all other predecessor tensors remained equal. It preserved **128 four-language probes with zero logit difference, 26 physical meanings, exact mathematics, empirical records and the original force and momentum tasks**. Two exact restarts and 21 append-only audit revisions passed. The source and novel-definition gates remain attached to the inherited learner.

The v15 packet's manifest, 55 tests, 44 model replays and 360 portfolio selections passed. Strict diagnostic equality differed by at most 3.26e-16, with identical discrete outcomes. Its separate strict training-resume check reported a tensor-equality failure in this environment. That original failure is retained as a portability result; the four new local resumptions above passed exact comparison. [Packet qualification](packet-verification.json).

All ten targeted tests passed, covering alias handling, contradictory input, global credit, forged receipts, qualification scope, revocation, exploration budget, exact restart and interruption between decision and credit. Full local regression passed **485 tests with zero skips** in 421.33 seconds of test execution. Supervisor wall cost was 423.98 seconds. Full regression and resource logs are preserved under [checks](checks/); [costs](costs.json) includes failures and repeats under the existing one-thread, 2 GiB allowance. No final case trained either policy.

## Engineering and acquired behavior

I supplied the method programs, input features, conditional teachers, fixed exploration allocation, checker, qualification policy and independent assessor process. SERA learned the method-scoring weights and the subsequent checked changes to its investigation procedures. The current reward measures verified method and scope coverage. It does not pay for extra wording, aliases or unsupported branches.

The next integration should acquire a new human-sourced method through the existing reading and original-goal interface, verify its assumptions and program, and only then add its genuinely new coverage to this portfolio. That work uses new practice and evaluation cohorts; the final cohorts above remain sealed.
