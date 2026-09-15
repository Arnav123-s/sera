**Kavi pinned-source audit — 14 September 2026**

The pinned public Kavi checkout is available and its full upstream suite passes: **321 tests**, reported test time **10.480 s**, process wall time **13.844 s**. The best immediate SERA/Kavi seam is an explicit contract for executable identity, evidence and active-state ownership. The code supports that direction, but it does not implement a unified learned world, live migration between changed circuits, or a shared SERA/Kavi updater.

Evidence labels below distinguish **source-observed**, **executed here**, **historical report**, and **proposed**. Exact commands, raw test output, five reproducible contract probes, hashes for 51 inspected files and coverage limits are in [kavi-source-evidence.json](D:/ai/projects/sera/research-continuation/01_audit/kavi-source-evidence.json).

**Source identity and review boundary**

Kavi is detached at `50f743cc44794b67bc1d30b927e25991197edce2` in [the isolated checkout](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned). Read-only remote checks found both main and HEAD at that pin. The checkout remained clean after testing. The initial Windows schannel Git transport failed; a per-command OpenSSL backend succeeded. No global Git settings were changed.

The latest commit adds `phase/habitat.py`, its tests and the finite habitat course. Relevant prior changes are `88b8716` (independent phase continuations), `04eb8e3` (whole-circuit reconstruction), and `b2c6e2c` (hierarchical guarded configurations). This is a sequence of separate experimental implementations. The main CLI still dispatches the circuit/library commands or the older LiveRuntime; PCL is not the common application runtime. See [CLI](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/cli.py:110).

SERA's current local HEAD is `8dd87f9a02a3752c041f05a101fd250e73e82cd7`; the supplied packet cites `a4047173fd33d27d33d8a151f32613eae343df56`. The nine SERA core files inspected here have no committed difference between those revisions. Existing concurrent working-tree changes are recorded without modification. The suggested `session.py` does not exist: actual implementations are `models.py:Session`, `r1.py:WorldSession`, and `session_state.py`.

I reviewed Kavi's README/CONTRIBUTING, source design, experiment reports and the intake cookbook. The package explicitly calls its analytic code an independent research harness; it is not evidence that upstream architecture is merged.

**Actual retained structure and active computation — source-observed**

| Implementation | Retained executable structure | Active state and inference |
| --- | --- | --- |
| PCL | `PhaseConfiguration(moduli, impulses, couplings, outputs, ticks)`, all validated tuples; `CircuitGeneration` adds the fixed template and inner generation count | `PhaseActivity` stores only configuration, work, current phase tuple, closed and failed flags. Event impulses update phases modulo each modulus; each coupling tick reads one synchronous snapshot. Only `finish()` releases an integer port. Unknown input fails the invocation; unmatched or contradictory readouts return unresolved. |
| Learned habitat | `WorldGeneration(habitat, inner)`; habitat is disjoint event-role domains plus admitted role patterns | Complete event tuples must pass the habitat before inner execution. Habitat checks structural language membership, not real-world truth or semantic applicability. |
| Recurrent configuration | Moore-machine alphabet, transition table and state outputs | Prediction starts at state zero and keeps a single state index. Unknown/undefined transitions return unresolved. Its learner builds a prefix machine and performs consistency-constrained state merges. |
| Hierarchical configuration | Arity, named calls referencing earlier values, output reference or kernel name | Registry execution uses invocation-local value lists; expansion flattens a transitive call graph. Programs share intermediates through references. |
| Signal execution | A signal graph with seeds, triggers, dependencies and substrate | Latest values/revisions, signatures, pending activations, dependency fanout and completion/failure flags. Final output requires completed input, complete ports and a settled scheduler. This is separate from PhaseActivity. |

The PCL schema caps components at 64, each modulus at 256, and event ticks at 256. Its state is a finite modular transition system. The product of moduli bounds joint states; it does not establish useful knowledge in every state. Readouts can be exact phase patterns or interval conjunctions. The interval covering procedure is supplied and only constrains overlaps observed during fitting. [Model](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/phase/model.py:17), [runtime](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/phase/runtime.py:5), [regions](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/phase/regions.py:12).

`PhaseActivity.fork()` shares the immutable circuit and original Work budget, copies the current phase tuple reference safely, and gives the child its own status fields. Each later input replaces that child's tuple. It stores neither the parent nor an event history. This establishes independent continuations under unchanged dynamics. Appending hypothetical events is not automatically a causal intervention or a semantic counterfactual. The caller still owns branch lifetimes. [Fork](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/phase/runtime.py:13).

**What changes during learning — source-observed and exercised by upstream tests**

`reconstruct` materializes complete alternative impulse/coupling collections inside a supplied `CircuitTemplate`. It tries existing dynamics first, then supplied stochastic inheritance/redraw proposals. It reconstructs readouts from explicit protected and lesson examples. A compatible candidate must preserve every protected output; candidates rank by lesson errors then serialized size. A finite bank establishes finite-bank retention. Template moduli, tick semantics, coupling capacity, grammar, inheritance probabilities and search control are engineered. `teach` remains a separate local-edit comparison. [Reconstruction](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/phase/layers.py:55), [selection](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/phase/selection.py:5).

`infer_habitat` groups event symbols only when their complete observed one-hole context sets match. It exactly factors the finite confirmed language. It does not extrapolate to unseen arrangements. `teach_world` transiently expands the old habitat, obtains old targets from the old circuit, removes exact inputs explicitly contradicted by new lessons from protection, infers a successor habitat and reconstructs the circuit. It returns the pair only when all new/protected outputs pass and the old language remains covered. Interruption or rejection leaves the caller's old world available. [Habitat update](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/phase/habitat.py:122).

Two boundaries matter. First, most protection targets are old predictions, not independently verified truths. Correcting one target does not infer which other beliefs depended on it. Second, old boundary coverage is monotonic: negative structural evidence and retraction of an erroneously admitted arrangement are missing. Replay expansion is bounded at 4,096 inputs and words at 64 events. Replacement happens between invocations; there is no mapping from an existing PhaseActivity into a new circuit.

The arithmetic procedure library has supplied type and execution contracts, learned gate circuits/program bodies, bounded call/repeat/range/binary-fold semantics and earlier-procedure dependencies. The graph catalog pins the whole base-library digest and disallows replacing an earlier name. These are useful concrete contracts, but distinct from PCL's habitat or the proposed general relational object. [Procedure library](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/procedure_core.py:94), [catalog](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/configuration_composition.py:128).

**Guard and dependency ownership**

Hierarchical `Expansion` records hashes of every visited configuration and a substrate identity. Guarded execution rejects a missing/changed definition or changed substrate. Signal graphs perform corresponding checks during execution. The equation registry's substrate hashes its supplied module source and arithmetic library bytes at construction. Finite dependency invalidation is therefore implemented and tested. These are content guards, not a learned applicability predicate over physical or semantic conditions. [Expansion guard](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/composable_configurations.py:80), [substrate](D:/ai/projects/sera/research-continuation/00_sources/kavi-pinned/kavi/equation_kernels.py:47).

Five small **executed-here** probes expose the ownership boundary without changing repository source:

| Probe | Observed result | Integration consequence |
| --- | --- | --- |
| Habitat-only update | Accepted successor changes full world digest while `inner.generation` remains 0 and the inner object is unchanged | Identify the whole world record, schema and interpreter; never use the inner counter alone. |
| Mutable phase collection | PhaseConfiguration rejects a list of impulses | Preserve its strong tuple validation in any wrapper. |
| Mutable hierarchical calls | A frozen hierarchical Configuration accepts a caller-owned calls list; mutating it changes outputs through a copied registry from 5 to 6 | Frozen fields alone are insufficient. Canonicalize/deep-own all reachable containers. |
| Shared kernel mapping | Changing a copied registry's kernel changes a guarded expansion's output from 5 to 6 while its substrate string remains unchanged | Freeze executable handlers and their owner, or rebuild identities when handlers change. Registry.copy is not a full snapshot. |
| Negative work amount | Work(limit=1) accepts a -100 increment and then 50 operations without exhaustion | Validate nonnegative integer counts in the shared budget adapter. Do not compare heterogeneous sums as FLOPs. |

These are contract counterexamples, not evidence that published experiments silently mutated kernels or used negative accounting. Existing guarded tests correctly reject changed definition hashes; the weaknesses appear when callers provide mutable values or mutate shared execution infrastructure. The proposed adapter must close those cases before accepting arbitrary shared-library ownership.

**Comparison with actual SERA source**

`SharedR1` is the single trainable parameter owner. `SharedSequenceView` and `SharedTypedView` hold parameter-free references; `replace_shared_owner` reconnects both routes. `Solver.validate` rejects views detached from the registered owner. Shared typed/sequence batch routes nevertheless initialize separate activity on each forward call; common parameters alone do not provide one persistent cross-route situation. [Shared owner](D:/ai/projects/sera/src/sera/shared.py:217), [replacement](D:/ai/projects/sera/src/sera/shared.py:309).

`WorldSession` supplies a stronger model/encoder ownership boundary: model-content identity checks before observations, validated JSON save/load, admitted provenance, explicit current tensors and diagnostic history. It refuses a changed live model. `models.Session` is an older lighter contract and should not be confused with that implementation. `SolverStore.consider` freezes candidate identity before drawing fresh evaluation randomness, checks an expected predecessor, records failed/rejected attempts and changes the current pointer only after admission. [WorldSession](D:/ai/projects/sera/src/sera/r1.py:182), [admission](D:/ai/projects/sera/src/sera/solver.py:237).

SERA's EvidenceKind/Observation/Experience already distinguish origins, units, availability and verified/simulator targets. Kavi phase learners receive plain event/port pairs and leave provenance to the teacher. A bridge should preserve SERA's eligibility checks and label Kavi old-model obligations as behavioral consistency evidence. It must never promote model or imagined answers to factual labels.

The existing `SharedR1.programs` is restricted by `typed_programs.validate_program` to its supplied integer-rule grammar. Arbitrary Kavi records cannot simply be inserted there. SERA's R2 action library separately validates world identity, declared dependencies, optional dependency versions and expanded actions. A new executable format needs an explicit ownership/schema boundary and later solver serialization support. [Typed program validation](D:/ai/projects/sera/src/sera/typed_programs.py:44), [R2 library validation](D:/ai/projects/sera/src/sera/r2.py:257).

**Recommended concrete seam — proposed**

Implement a small common Event/MechanismRef boundary beside the existing source. Events carry immutable values, masks, units, entity references, sequence, role and origin. Mechanism references pin complete graph/parameter/schema/interpreter identities and dependency versions. Executable definitions live in a content-addressed immutable graph. A parameter-free reference checks the actual SharedR1 owner instead of copying it. A replacement is prepared against an exact parent identity, checks dependencies and invalidates the transitive dependent set.

Keep live factual situations and hypothetical branches separate. Branches may compute conditional results but cannot submit them as observed labels. A version change either explicitly replays admitted observations under an approved compatible schema or rejects the migration. Latent-vector migration remains a separate experiment. These are W01/W02 lifecycle contracts; adding them does not establish learned semantic grounding or positive neural/program transfer.

After that seam works, evaluate a small actual executable Kavi definition and the existing shared owner through the same event identities, then measure whether retained structure helps future prediction or acquisition. A parallel call router, independent analytic fit, or renamed six-feature pilot does not satisfy that capability claim.

**Validation and limits**

The full source suite ran under the existing Python 3.12.14 environment, PyTorch 2.10.0+cpu and Windows 11, with a 60-second external timeout. All 321 tests passed; a PyTorch scalar-conversion warning in a historical test was non-fatal. No packages were installed and no curriculum, background service, core edit or commit was made. Five ownership probes completed in 0.00136 seconds inside Python (0.286 seconds tool wall). Child CPU and peak RSS were not measured; CIM hardware queries were denied. The evidence file states those omissions.

All 12 source fingerprints in the published habitat report match LF-normalized UTF-8 bytes. Windows checkout CRLF explains differing raw hashes; both forms are recorded. The three published world checkpoint files are absent from the public clone. Their hashes are metadata, not available artifacts. Therefore the habitat source course, its three learned worlds and their historical reported scores were **not replayed here**. No broad language, world-model competence, learning-procedure improvement or merged-architecture claim follows from this audit.

