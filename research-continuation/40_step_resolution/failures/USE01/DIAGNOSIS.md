# Preserved task-interface failure

The first batch adapter preferred `runs/sera-refinement-live`, an existing empirical store from an earlier stage, because the path existed. Its schema correctly failed validation before model loading or any mutation. The original adapter and failed job log are preserved here.

The repair gives potential new successors distinct store names (`sera-sustained-growth-live` and `sera-step-resolution-live`) and uses the portable lowercase `current.json` pointer name. The admitted Stage 38 store remains `sera-growth-live`. A regression test checks that the older empirical store is ignored. No scientific checkpoint, frozen evaluation, older store or learned weight changed for this interface repair.
