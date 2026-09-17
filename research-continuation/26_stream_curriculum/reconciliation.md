# Local reconciliation and authority

This cycle starts from SERA commit
`8ae0b3ef55e2a4eda26d1dfff2c01fbcb505099f` on
`research/v3-calibration-consolidation`, with the previous release's work committed.
Kavi remains clean at `50f743cc44794b67bc1d30b927e25991197edce2` in
`research-continuation/00_sources/kavi-pinned`. The initial worktrees were clean;
the new files and later navigation edits belong to this cycle. No repository reset,
checkout replacement, old experiment restart or other-session job termination is
part of this work.

The current constraint store is revision `000014-dd392f7936e0.json`, whose SHA-256
is `dd392f7936e0e010984bbad334c1636aa88fd83cc50e8296ac55161b5fc2aaeb`.
The exact parent payload is pinned in `parent.json`. Its owner is
`5b1dc35e772f5c00ffe8fe0e43fbf1ea14f10378bbf990ebb50002ee7d728a7b`.
The workbench, task-transfer and inquiry stores remain predecessors, with their
unchanged pointers checked again by the preserving seal. Their original source
and artifact identities come from release 25, not stale packet TODOs.

The initial resource balance was 67.82661569984339 supervised seconds. The pilot,
preparation attempts, source audit and persistence checks were charged against it.
The user explicitly approved another 3,600 seconds for full-data training,
comparisons, retention and held-out evaluation. `resource-grant.json` and
`budget-before-extension.json` record the approval and exact ledger state. No
additional approval is inferred from a completed job or from elapsed time.

All numerical work uses the existing owned-job supervisor and shared resource
ledger, one thread and 2 GiB committed memory. A dispatch attempted while an
earlier worker still held the lock was refused before creating a second worker;
the prior job continued normally. The refused dispatch was not silently turned
into a parallel fit. Job logs and charged failures remain in `supervision.zip`.
Ordinary source retrieval, edits and static archival work are separately disclosed.

The v5–v10 source addenda, prior reproductions and GG-GUARD-002 applicability work
were already reconciled in release 25. I read the original handbook's learning
contracts and revisited v10's shared-owner/language cookbook while auditing this
extension. No standalone packet model replaced the actual owner. The user's newer
instruction to teach and measure broader tasks motivated the new real-request
curriculum; packet text and public dataset requests did not confer extra authority.

This note describes the reconciled anchors. `preservation.json` and the release
manifest contain the machine-checked end-of-cycle identities; they should not be
mistaken for a new snapshot retroactively collected before the pilot.
