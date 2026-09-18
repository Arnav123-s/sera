# HR-F03: distinguish forecast values from owner lineage

The second audit compared entire empirical prediction records. The `parent_owner` field correctly changed after registering the new reading weights, making complete-record equality inappropriate. The repaired audit independently checks both owner identities, then requires exact equality of every remaining field, including values, parameters, evidence, branch, assumptions and uncertainty. No model or evaluation result changes. The one-chapter textbook cohort is also explicitly reported without a between-chapter confidence interval.
