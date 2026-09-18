# GD-F01: audit encountered a markup-only glossary headword

The first integration audit stopped at m54593:fs-id1167063657626. Its original glossary term contains markup rather than direct text; the frozen intake retained a 51-character human meaning but an empty term, labeled other. Runtime validation correctly refused a nameless definition. The audit incorrectly assumed every teaching row was an executable interface input.

I preserve the original source, model, cohorts and final predictions. The audit now records this malformed name as rejected by the interface and continues auditing valid entries. It does not omit a physical-class error, add a usable concept, change a learned parameter or reopen a final evaluation. This repair concerns interface coverage accounting; future intake should collect nested term text before partitioning.
