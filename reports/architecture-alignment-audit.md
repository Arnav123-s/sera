# SERA architecture alignment audit

I audited commit `c7cb959da73a94a11e9a2d470577ab81e899c127` against the architecture handbook in the original research package, its R1/R2 diagrams, and the relevant distinctions in the physics atlas and layered maps. I inspected the executable paths as well as the documentation and tests. I also re-evaluated the saved integrated-run checkpoint with and without its verified program.

**Verdict: SERA 0.1 is a working mechanism study and a partial first integration. It follows the recommended research direction, but it does not implement the complete proposed architecture. Several missing connections are architectural gaps, not merely smaller tensor dimensions.**

## What the reference actually prioritizes

The handbook's "Recommended architectures and research priorities" section specifies R1 with a separated program-learning pathway from R2 as the first build (editable manuscript lines 1256 and 1272). It recommends starting with associative memory and adding quantum-inspired branches when controlled experiments support them. Choosing delta memory as the default is therefore consistent with the source.

The source describes eight architectural families, not eight mandatory components of one model. SERA's eight `KINDS` are small cores and controls. They do not constitute implementations of R1 through R8. Keeping all 154 concept mappings is source coverage; it is not implementation coverage.

The atlas distinguishes representation, reduction, approximation and experimental evidence (Chapter 1, sections 1.1-1.3). Map 01 repeats that distinction. These materials constrain mathematical claims; they do not supply a completed intelligence objective. The handbook also explicitly distinguishes a mathematical operation from a learned cognitive capability.

## Requirement-to-code comparison

Manuscript line references below refer to `Quantum_AGI_Architecture_Handbook.md` inside the original ZIP. The original inputs are identified by SHA-256 in [the source manifest](../research/source_manifest.json).

| Reference requirement | What I found in SERA | Assessment |
|---|---|---|
| R1/B4: learned keys, queries, values, retention and corrective associative writes; lines 291-306, 430 | `DeltaMemory` implements the corrective write and learned addressing; tests check exact replacement for normalized keys. | Implemented primitive. |
| R1/B1: complex phase propagation with write/forgetting and a real control | `Rotor` implements these operations, with a no-phase control and measured comparisons. | Implemented primitive; no demonstrated general phase advantage. |
| R1 starting configuration: width 256, complex state 128, eight 32x32 heads, four 16x4 density factors; lines 420-426 | Defaults are width 32, two 8x8 memory heads; the optional density branch stores one full 4x4 matrix. The hybrid combines parallel branch readouts. | Substantially reduced configuration. The low-rank multiworkspace design is absent. |
| R1 transition prediction from retained hidden state and action, including observations and rewards; lines 430-445 | `ActionWorldModel` consumes one-hot observed state IDs and actions in a separate experiment. It does not consume `StatefulModel`'s retained state and has no reward head. | Major integration gap. |
| R1 planning, verified consequences and durable learning form a feedback loop; lines 443-459, 959-965 | The finite planner executes and checks plans. `integrated_run` runs neural training, world modeling, adaptation and skill discovery consecutively; their results are collected in one report. World-model feedback does not train the recurrent core. | Working component experiments, incomplete feedback loop. |
| R2: shared instrument for conditioning, probability and generation; lines 483-517 | `EventInstrument` has normalized Kraus operators, posterior updates, sequence likelihood and sampling. | Implemented small primitive. |
| R2: action control, program proposals from predictive state, execution events fed back to the instrument; lines 485-523 | The instrument experiment is separate. It has no action argument or program decoder. `discover` performs breadth-first transition identification without consulting an instrument. | Full R2 architecture is not implemented. |
| Explicit state ownership and serialization; lines 828-832, 1396 | `Session` saves fast tensors, owner, model/encoder version, position and configuration; restore checks ownership, shape and dtype. | Implemented fast-state contract; not a complete long-term experience system. |
| Learning consumes admitted, traceable experience; lines 1396-1400 | `Experience` rejects prediction-only targets when instantiated. The actual training path consumes `make_batch` directly and never accepts an `Experience` object. | Contract class exists, but a general admission-to-learning pipeline is missing. The current synthetic labels are generated from declared rules. |
| Verified executable skill acquisition; lines 967-973, 1402-1408 | Active queries infer a finite transition table; independent trajectories verify it; the library records domain, evidence, assumptions and a content hash. | Implemented for observable, deterministic, resettable finite worlds. |
| Programs compose and transfer to withheld structures; lines 519-523, 971 | Programs only iterate a transition table. There is no general DSL, library composition, learned applicability or program-structure transfer benchmark. | Missing capability. |
| Failure classification and curriculum choices; lines 844-869, 975-981 | The engine checks ordered-control accuracy against 0.95, then runs one predefined discovery method. | Fixed narrow policy; no learned diagnosis or curriculum. |
| Learning the update procedure; lines 97-105, 852-871 | There is no learned update-policy parameter set or meta-objective. Ordinary model training and a library update are implemented. | The third learning time scale is not implemented. |
| Candidate promotion persists into future behavior and subsequent versions; lines 824-832, 848-850 | Promotion writes a `current.json` pointer, but normal checkpoint evaluation does not load that pointer or its skill. Every `improve` invocation initializes `v0`; completed directories are rejected for another round. | Critical lifecycle gap. |
| Fresh paired admission, retention gates, archive and rollback; lines 897-911 | A fresh seed is drawn after freezing the candidate; paired gain and empirical retention gates are checked; the journal and version-pointer rollback work. | Implemented for one bounded intervention; not an independently isolated multigeneration evaluator. |
| Full cost of producing each improved version; lines 19-29 | The admission cost is discovery oracle calls plus discovery actions. Verification queries, updates and all failed-candidate/system costs are not combined in a full cost measure. | Partial accounting; recursive research-efficiency claims are unsupported. |
| Broad task mixtures and held-out rule families; lines 941-955 | Four synthetic tasks train jointly; a related fifth task probes adaptation. Tests mostly use fresh examples and longer sequences from the declared generators. | Limited evidence; not broad family-level transfer. |
| Multimodal inputs and typed outputs; lines 175-225, 445 | The executable encoder accepts 20 symbolic features and the decoder emits four classes. Other modality names are metadata declarations. | Text, image, audio and general typed decoding are not implemented. |
| R3-R8 architectural alternatives | No gauge-relational learner, normalized tensor-belief system, equilibrium-propagation learner, learned population search, parameter-cloud learner or joint-register architecture is provided. | Future research branches, as documented. |

## The most consequential missing connections

### The recurrent state does not drive world prediction

The R1 equations require prediction conditioned on `h_t` and `a_t`. In [world.py](../src/sera/world.py), `ActionWorldModel.forward` instead receives observed integer state IDs and actions. In [experiments.py](../src/sera/experiments.py), `integrated_run` invokes the world experiment separately and stores its report beside the neural and adaptation reports. A single runner is useful orchestration, but it does not establish the state-to-world-model-to-learning connection.

### The instrument does not guide acquired programs

[quantum.py](../src/sera/quantum.py) implements the instrument. [programs.py](../src/sera/programs.py) implements finite transition discovery. Neither the discovery procedure nor the improvement path calls the instrument. The R2 proposal-to-execution-to-observation learning cycle is consequently absent, despite both mechanisms existing independently.

### A promoted version is not the default executable solver

In [engine.py](../src/sera/engine.py), every improvement run starts with `version="v0"` and `skill_id=None`. The program is explicitly attached only while evaluating the candidate. In [cli.py](../src/sera/cli.py), `evaluate`, `adapt` and `improve` load a neural checkpoint through `load_model`; they do not resolve a promoted solver bundle from `current.json`.

I checked this with the existing `runs/integrated-v1` artifacts. Its current pointer names `v1` and a verified skill. On the recorded promotion dataset, checkpoint-only inference reproduces **66.2109375%**, while explicitly loading and attaching that skill reproduces **81.689453125%**. Both scores exactly match the saved report. This confirms that the measured intervention is real and that its benefit is not automatically available through ordinary checkpoint inference. Reusing this dataset checks software behavior; it is not fresh generalization evidence.

The current promotion/rollback test checks the pointer and candidate scores. It does not reload the current solver in another process, verify that normal inference uses the skill, then run a second improvement generation against that solver.

## Interpretation of the existing claims

I can support the claims that the release trains small models from scratch, implements several valid finite mathematical mechanisms, acquires a bounded executable procedure, and records a measured intervention with fresh admission evidence.

I cannot support a claim that the full R1/R2 systems, a unified learner/planner, general persistent skill use, a learned self-improver, or all eight architecture families have been built. Passing 26 tests validates their tested behaviors; it does not establish coverage of the source architecture.

The README already labels this a bounded release and lists several limitations, but its phrase "R1/R2 experimental path" needs to be read with the missing connections above. The brief's "three learning time scales" mapping overstates the current implementation: a fixed program-acquisition routine is not an implemented meta-update rule. Referring to R1 dimensions only as scaling proposals also obscures that the handbook provides them as a concrete starting configuration that this release deliberately reduced.

## Acceptance gates for the next implementation

1. **Persistent executable solver:** load neural state, verified skills, applicability rules and version lineage through one solver interface. In a fresh process, ordinary inference must reproduce the promoted behavior. A second improvement round must evaluate the current solver; rejection and rollback must restore its actual behavior and preserve cumulative evaluation accounting.
2. **Connected R1 loop:** condition world prediction on retained learned state and action; include observed consequences and appropriate reward targets. Use that model in planning, execute the plan, admit the resulting evidence and train a candidate with retention checks. Test withheld dynamics and partial observations, with observable-state access confined to explicitly labeled controls.
3. **Connected R2 loop:** add controlled actions and typed execution events to the instrument, let predictive state guide bounded program proposals, and train on verified execution traces. Compare instrument-guided search to the existing fixed discovery procedure at the same feedback budget.
4. **Learned improvement policy:** train choices among evidence acquisition, replay, updates, planning and program search using separate support/query tasks. Measure transfer, per-task retention and complete cost across several generations. Do not call a fixed router a learned controller.

These gates close the central architectural gaps before adding more optional physics branches or enlarging the model.
