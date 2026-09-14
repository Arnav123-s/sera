# Connected SERA study protocol

I set this protocol before running the final three-seed study. Development probes and the short seed-901 smoke run are excluded from the final tables.

## Engineering objective

I connect retained state, action-conditioned prediction, planning, admitted feedback, verified programs, a learned improvement policy and persistent version selection. The executable interface is `SolverStore.load()`: it restores all model components and accepted skill records. `sera evaluate`, `sera solve` and `sera learn` accept that persistent solver directory.

## Declared experiments

| Experiment | Teaching data and updates | Evaluation |
|---|---|---|
| Symbolic foundation | Four instructed tasks; 500 AdamW updates, 64 examples/update; fresh initialization | Five tasks after acquiring ordered-control and earliest-binding programs; ordinary inference after reload |
| R1 recurrent world learner | 512 independently collected eight-action trajectories; 30% of later sensor readings absent; 900 updates, 32 trajectories/update | 256 trajectories per condition: fully visible length 12, 40% missing length 12, 40% missing length 24; memory-reset ablation; goal execution with reactive/planned/partially observed controls |
| Learned improvement policy | Eight development worlds/episodes and four validation episodes per seed; each available intervention is actually constructed and evaluated; 500 policy fitting steps with validation selection | Four new reset-family worlds per seed after policy freezing; compare measured utility to each fixed method and the post-hoc best available method |
| Persistent improvement | Two symbolic skill acquisitions; one planned-feedback/replay proposal; two proposals selected by the trained policy | Fresh paired solver evaluations, previous-world retention, recorded rejection, cumulative round indices and rollback in a separate copy |
| R2 program learner | A separate permutation world; 256 eight-action traces; eight verified start/goal demonstrations; 200 likelihood/proposal updates | Four withheld start/goal combinations per seed; fixed search, untrained instrument and learned instrument, each limited to four executed candidates |

Final seeds are 0, 1 and 2. Inner policy-development interventions receive 16 update steps. The planned-feedback proposal receives 120 replay updates. A program intervention also receives a final update on its newly verified execution traces. Every count is recorded in raw artifacts.

## Sensor and feedback contract

Worlds have four hidden simulator states, four actions and four sensor colors. A sensor-color permutation separates the simulator's internal indexing from observations, but a visible color fully identifies the state within a world. This is a symbolic sensing model, not learned perception. Missing readings are represented by `-1`; their hidden colors do not enter admitted training trajectories. External rewards and the visible goal remain available. Evaluation retains truth separately to measure prediction during sensor gaps.

The neural encoder receives the visible/missing sensor, previous action, actual reward, goal and a deterministic encoding of the public world identifier. It never receives the transition table or hidden simulator state. Different world identities can therefore be learned without forcing contradictory rules into an indistinguishable input.

Program experiments explicitly permit resetting to a requested visible start color. This feedback is stronger than passive sequence supervision. Nonresettable policy episodes disable the program intervention. The fixed interpreter and its action/sequence/repeat/call grammar are supplied infrastructure. Programs are acquired from actual executions; this study does not establish discovery of an open-ended programming language.

## R1 and R2 mechanisms

R1 aggregates the current event embedding with retained associative state, then predicts the next observation and goal reward conditional on a proposed action. Its planner propagates the same model and executes its selected actions. A planned-evidence collection produces verified trajectories that enter replay and train a candidate. The default study uses a compact delta core. A separate executable preset implements the handbook's 256-wide embedding, 128-complex rotor, eight 32x32 memory heads and four 16x4 density factors. That preset's 35,840-byte core state and differentiation are tested; it is not substituted into the compact study or claimed trained here. Low-rank projection records discarded density mass.

R2 learns action-controlled CPTP maps and a shared projective event instrument with a learned basis. Conditioning, next-event probabilities and a program proposal head use the same predictive state. Unobserved events are marginalized through the nonselective instrument. A typed program is executed, independently re-executed for verification, stored with its domain and evidence, and used as training credit. The study reports execution and proposal-model costs separately.

## Policy objective and admission

The improvement policy predicts the measured utility of six supplied choices: no change, parameter update, replay update, additional evidence, increased planning, and program acquisition. Features come from observed diagnostic outcomes, confidence, support size and permitted reset access. Query outcomes are policy-training targets; they are not diagnostic input features. Final test episodes are produced after freezing the policy.

Utility is target-world score improvement minus twice the worst prior-world regression and `0.002 * log(1 + counted operations)`. Scores equally weight trajectory prediction accuracy and goal success. These weights are declared experimental choices. The policy learns which existing procedure to select; it does not invent an optimizer or architecture.

Promotion uses the same bounded paired-gain rule as the initial release, with fresh random seeds and a cumulative round index. Each world receives equal numbers of independently sampled evaluation episodes. Critical per-world retention gates are empirical; the gain lower bound has its stated conditional IID interpretation. Deterministic goal outcomes are enumerated and cached, then sampled through fresh independent start/goal draws from the declared finite distribution. This is not evidence of unlimited task diversity.

## Cost, provenance and interpretation

I retain candidate construction, replay draws, executed environment actions, verification executions, model proposal/planning operations, evaluation episodes and elapsed wall time. Their operation sum is a transparent admission/utility proxy; these unlike units are not FLOPs. Total study wall time includes counterfactual policy training and failed candidates. A deployment utility advantage would not by itself establish an advantage after paying all meta-training costs.

All models train locally from scratch. Numerical completeness, trace/rank behavior and program contracts are tested independently. The final report will include means and sample standard deviations where three independent seeds are available, individual generation decisions and negative results. I will not describe a passed software checklist as general intelligence, a quantum advantage, or unrestricted recursive self-improvement.
