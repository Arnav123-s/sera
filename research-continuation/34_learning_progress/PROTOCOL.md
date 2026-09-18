# LP-001: reward better acquisition procedures

I continue the actual Stage 33 owner, preserving its completed comparisons and applicability contracts. This finite W10 experiment also implements the source-matched practice portion of W03. It contains six curriculum strands and three existing computational readouts: dictionary, conversation, mathematics and science next-word prediction; source-sentence selection; and independently checked mechanics-method selection. The four prose strands share the same head. I retain all earlier language, algebra, memory, completion and portfolio records.

## Starting state and limits

SERA HEAD 5747d4937c9da1014450ce849d2d5c4948b0f8da; Kavi HEAD eb5139c2aa3bf6b4fcee39e585231e7b902e539e. The clean local repository, live quest revision 000002-b238cc73de13.json and relocated live assessor supersede the older packet and NEXT_ACTION path. The live numerical allowance is 1166.0691055996722 seconds, one worker/thread and 2 GiB committed process-tree memory. No completed final evaluation is reopened for optimization.

## Data and disclosure

Human prose and human SQuAD annotations come from the already attributed Stage 29/30 training/development sources. Deterministic whole-group partitions create two teaching transitions, two future acquisition cohorts and a retention cohort. These are disjoint within LP-001; the parent has historical exposure to these corpora. Consequently this measures continued acquisition and procedure transfer within retained task families, with exposure recorded, rather than claiming wholly new language or scientific understanding. Old sealed final datasets are not accessed. New mechanics examples use supplied typed methods, independently screened numerical validity and new seed-specific situations; they are conditional generated practice, not human facts.

Eight batches of eight examples per strand per transition, with deterministic wrapping if a small group requires it. Every arm sees the same ordered examples within each strand, the same 48 updates per transition, and the same diagnostic probes. Quotas match new-example exposure separately from updates. The only adaptive choices are lesson ordering and one of three supplied SGD rates (.0005, .002, .008). Fixed feature encoders are reused; old source gates stay closed. Four arms: balanced, random, measured progress and a learned reward predictor; seeds 3401 and 3402. Each continues through two successor transitions without resetting knowledge between them.

## Reward and procedure learning

Independent human labels / checked method labels determine the score 1/(1+cross-entropy) on a fixed, balanced practice probe. Credit is the signed mean gain, plus half the gain above the persistent per-strand high-water mark, plus one quarter of the change in mean accuracy. A forget/relearn cycle cannot create new high-water credit. All signed maintenance gains are retained. Points change neither qualification nor permissions. Evidence identities, original goal, knowledge and procedure identities bind every decision. Duplicate decision IDs are rejected. A 20% exploration component and equal per-strand quotas preserve breadth.

The actual owner receives a small shared candidate reward predictor, trained only from the realized independently scored update. Supplied features describe strand, current diagnostic loss/accuracy, recent measured progress, remaining quota and rate. Fitted reward prediction is the learned procedure; the menu, checker, reward and curriculum are engineering. Cost includes all probes, failed updates and comparisons. Confidence alone receives no credit; the proper loss is computed against external labels. Accuracy is reported separately.

## Separation and sealed evaluation

Save K0/eta0, K1/eta1 and K2/eta2 plus optimizer, RNG, cursors, high-water marks and full traces. After both transitions and development selection are frozen, assess (old K,old eta), (old K,new eta), (new K,old eta), (new K,new eta) at each successive transition. Starting K is identical within each procedure comparison. Future acquisition uses 24 updates (four per strand), matched examples and frozen eta; independent group-held probe labels never train eta. Compare balanced, random and measured-progress controls from the same K as well. The second transition uses K1, not a reset to K0. These diagnostic probe sets have historical parent exposure and are explicitly labeled accordingly.

Choose the development-best final checkpoint, tie favoring balanced. Integrate changed K only if aggregate future-probe score improves over its starting K and no strand loses more than .02 score; otherwise retain the predecessor and archive the challenger. Promote the learned scheduler only if it improves matched future acquisition over balanced in both successor comparisons; otherwise keep balanced as default. No final-driven hyperparameter revisions. Independent NumPy score/reward replay, exact checkpoint resume, protected tensor/language retention, forged/repeated reward rejection and prior portfolio preservation are required before release.

## Stop

Two transitions, one final opening, one independent replay. Save resumable state at a real limit. Additional method discovery retains the prior checked portfolio and its separate reward contract; this experiment does not count supplied procedures as self-invented methods.
