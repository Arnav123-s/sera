# Actual-owner development decision

I extended the synthetic weight result to the actual saved language owner `64b48d5f8192176ff6a4092b629822dfda6741d8f017b9cfbce295baf0eb7589`. This is a maintenance experiment: a copy of one readout neuron is damaged, then corrected using stored responses from before the damage. Those responses remain labeled model-generated checksums. They are not independent facts or teaching from new research material.

The one-fault pilot `CS-OWNER-PILOT-001` passed actual-forward, observation-access, partition, paired-baseline and LP checks. Sparse corrections restored withheld responses even though the measured feature columns were strongly correlated. Dense corrections were not recovered. The full owner and live pointer remained unchanged.

Review of the secondary coefficient score caught a reporting bug: applying the generic `relative` helper to `(updated - pristine, fault)` subtracted the fault again. A perfect repair therefore scored approximately one instead of zero. The pilot and its original audit remain preserved; `owner-pilot-metric-draft.zip` retains both affected reporters. The corrected definition is `norm(updated - pristine) / norm(fault)`, with a dedicated contract test at zero, half and complete residual fault. The independent final auditor computes it separately. This changes no trained weights, observations or primary prediction scores. A correction overlay will rescore the saved pilot coefficients without repeating the experiment.

After that repair, the final uses 12 fresh fault seeds beginning at 1310033 and fresh train/development/evaluation prompt samples. All 30 conditions per seed, five methods, sparse/dense faults and 32/64/96 anchors remain fixed. No source-only clone is claimed sufficient to rebuild the existing trained owner: the predecessor checkpoint chain is the one preserved by release 21.

The final will report correction accuracy, residual weight error, original-label agreement and semantic-slot accuracy separately. Matching an imperfect parent is not a new language capability. Passing this maintenance test does not authorize an automatic change to live SERA parameters.
