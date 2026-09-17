# CI-002: fixed residual stopping, exploratory follow-up

Frozen after inspecting CI-001 and before generating this cohort. The primary
outcomes and source archive remain unchanged. No weight is retrained. CI-001
showed that 12 unconditional refinement steps waste work in ordinary cases and
that surface composition remains unreliable (50%); more training did not fix it.

Use seed 25400 and 128 new cases in each of the same four inspected families.
This is a fresh numeric bank, not a new protected family or confirmatory replication.
Compare uniform-adaptive, amortized-adaptive and the existing analytic reference.
Keep the already declared nominal distance target of 0.03m and 12-step maximum.
Stop a proposal cloud when a nominal witness meets that target. Report actual
iterations, all conditional evaluations including the gate, wall time and
assessment-only omitted-law outcomes. No threshold sweep, final retraining or
postselection. Analytic remains a legitimate stronger baseline.

The stopping policy is engineered. It establishes neither learned computation
allocation nor applicability. The runtime retains explicit manual step/resume
for investigation and adds solve for finite automatic stopping. All outputs
remain conditional and cannot answer historical occurrence. Language admission
is narrowed to the supplied active/passive productions; the entire failed held
surface family stays unresolved. This is a post-evaluation scope restriction,
not a repaired generalization score or a new calibration guarantee.
