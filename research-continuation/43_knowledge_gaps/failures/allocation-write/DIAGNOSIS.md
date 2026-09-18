# Preserved allowance receipt-write failure

The first local allowance helper wrote its receipt and preserved the previous ledger, then raised `ValueError: Invalid suffix 'tmp'` at `path.with_suffix("tmp")`. It had not updated the ledger. A unit-test job subsequently used the existing allowance.

The correction verified that the grant ID was absent from the live ledger, preserved the first receipt as `resource-grant-prewrite-attempt.json`, preserved the then-current ledger as `budget-before-application.json`, and applied the single 3,600-second allocation with a valid `.tmp` path. `resource-grant.json` records that reconciliation. There was no duplicate grant and no research checkpoint was changed. Both original helper files remain under `runs/`.
