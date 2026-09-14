# Resolution of the SERA 0.1 architecture audit

I retain the [original audit](architecture-alignment-audit.md) against the published 0.1 commit. This table records the connected 0.2 build and the requirements it leaves open. Handbook references identify the editable manuscript inside the ZIP recorded in the [source manifest](../research/source_manifest.json).

**Scope correction:** the subsequent [audit against the original ZIP and PDFs](original-source-alignment-audit.md) found remaining integration gaps and two implementation defects repaired in 0.2.1. This table resolves selected 0.1 findings; it is not a full-blueprint acceptance audit. In particular, live world-session persistence, automatic library composition, expressive history-preserving event branches and the complete failure/curriculum loop remain open.

| Audit finding | Implemented resolution | Verification and remaining scope |
|---|---|---|
| Separate observed-state MLP; lines 430–445 | `r1.py` fuses current events with retained state and conditions observation/reward heads on actions | Gradients reach encoder, memory and heads; visible/partial/long tests and memory-reset controls. Symbolic sensors remain supplied. |
| Planning feedback never trained recurrent memory; lines 443–459 | `connected.py` executes the same model's plans, admits actual trajectories and constructs a replay-trained candidate | A feedback proposal per final seed with fresh retention/admission. A connected loop does not guarantee useful learning. |
| Reference low-rank workspace absent; lines 420–426 | `lowrank.py` provides the reference dimensions, controlled factors, truncation and discarded-mass recording | 35,840-byte core and finite differentiation tested. Full preset remains untrained. |
| Instrument disconnected from programs; lines 485–523 | `r2.py` links controlled channels, shared event conditioning, proposals, execution and verified trace learning | Independent likelihood reference, missing-event marginalization and guided/fixed equal-execution trials. Finite supplied grammar. |
| Experience admission unused in world training; lines 1396–1400 | `EvidenceReplay` is the actual R1/instrument training input | Provenance, hidden-label and query-split exclusion plus serialization checks. The symbolic generator is a separate declared synthetic source. |
| Promoted skills absent from ordinary inference; lines 824–832 | `SolverStore.load()` restores components and executable skills; CLI resolves current solver directories | Fresh-process inference, later skill acquisition retaining earlier behavior and actual rollback. Explicit domain/task routing remains. |
| Each improvement restarted at v0 | Candidates load current state; immutable lineage and cumulative evaluation ledger | Successive acquisitions, learned choices, rejection and rollback; failed/invalid proposals preserve the incumbent. |
| Fixed diagnosis described as meta-learning; lines 844–871 | An intervention-value network learns from actual measured candidate outcomes | Frozen-policy reset-family evaluation and fixed-method controls. Methods are supplied; no optimizer invention. |
| Programs lacked composition syntax | Typed action/sequence/repeat/call interpreter with bounded fuel and recursion checks | Composition is executable. Useful learned abstraction transfer remains open; acquired programs mainly contain action sequences. |
| Verification/failed-candidate costs excluded | Acquisition, actions, verification, updates, planning/proposals and evaluation are counted; total study wall time retained | Operation proxy is not complete FLOP/energy accounting. Report aggregation avoids adding nested snapshots twice. |
| Narrow task distribution | Masked/longer trajectories, new permutation worlds, reset-family policy holdout and withheld goal combinations | Still finite four-state symbolic generators. |
| Multimodal decoding, R3–R8 and unrestricted self-improvement absent | Explicitly separate roadmap items | Not marked implemented or learned. |

The tests and connected study address the listed connections, with the qualifications in the later source audit. I assess stronger capability claims separately using [measured learning, retention and cost](connected-study.md). The source's recommended direction is preserved; its entire long-term architecture is not declared complete.
