# F01: a mislabeled simple-system initial condition

Found during pre-final review after the first development comparison. No final data had been generated or evaluated. The simple family had c=0, but the generator still initialized F randomly in [-.3,.3]. Its force therefore decayed rather than staying zero. That is a hidden transient and does not implement the protocol's instantaneous control.

I set the simple family's initial hidden force to zero, keeping the same random draw consumption. Delayed and omitted families retain their original initialization. This is a teacher implementation repair, not a hyperparameter or admission-threshold change. Training, seeds, budgets, candidate mechanisms and final contract stay fixed. A regression test checks that every simple-family hidden-force entry is zero and that instantaneous identification recovers clean simple dynamics.

Preserved first attempt: `runs/CR-study`, including source-v1, all 10 fits, pilot, optimizer/RNG checkpoints, 14 development evaluations and their costs. Its freeze, metrics and selection remain at the top of this continuation directory. Those figures describe the mislabeled transient control and are not promotion evidence. The corrected run uses `runs/CR-study-r2` and the `r2/` report subdirectory. The original protocol remains unchanged; the repair identity joins the corrected frozen contracts.

Re-execute the predetermined training/development protocol on the corrected teacher, freeze selection, then open the previously unused final split once. Preserve both attempts in the final cost accounting.
