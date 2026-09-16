# Final maintenance compatibility check

The first release-verification receipt records a real failure: the historical applicability checker rejected `.github/workflows/ci.yml` after later releases added verification steps. The workflow is a maintenance file. Its original bytes are still present in the applicability release's recorded Git commit `9104839400d5053152ca423be6da1efe355aa71a`.

I added that single path to the checker's existing evolving-file list. The checker now verifies its exact historical SHA-256 from that fixed commit, just as it already does for research navigation and its own maintenance source. Current raw results, model records, protocols, frozen source archives and numerical-auditor revisions still have to match their preserved bytes. No scientific source or outcome changed.

This repair is outside the sealed numerical experiment and its release manifest. The failed receipt remains unchanged. `verification-receipt-v2.json` records the repaired check and the remaining release checks; the first two successful checks in the original receipt were not repeated. `maintenance-manifest.json` binds this supplement, the repaired checker, current workflow, receipts and original sealed manifest.
