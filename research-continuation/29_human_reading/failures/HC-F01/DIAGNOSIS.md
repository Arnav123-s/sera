# HC-F01: keyword-only stable argsort

The first curriculum worker failed during the initial development measurement, before any teaching update or final evaluation. Torch 2.10 requires `dim` to be a keyword when the stable-sort overload is selected. The same unexecuted call in HR-001 was repaired prospectively. The failed source, log, cost and initial checkpoint remain preserved; the corrected curriculum uses `runs/HC-study-002`. No model mechanism, cohort, hyperparameter or final rule changed.
