**W01/W02 event and executable-ownership contracts**

Protocol frozen before implementation on 2026-09-14. Parent source: SERA 8dd87f9a02a3752c041f05a101fd250e73e82cd7; audited core files match packet pin a4047173fd33d27d33d8a151f32613eae343df56. Kavi reference: 50f743cc44794b67bc1d30b927e25991197edce2.

Initial protocol status: implementation pending. The executed outcome follows below. This tests lifecycle implementation and software contracts, with no curriculum, training or learned-semantic claim.

Hypothesis: an immutable common event schema and a content-addressed executable graph can preserve evidence origin and one SharedR1 owner across branch creation, serialization, correction and explicit same-schema migration. The counterexamples in kavi-source-audit.md define the hazards: shallow mutable configuration ownership, mutable shared handlers, inner-only generation identity and unvalidated work counts.

Scope: only new src/sera/event_ir.py, src/sera/world_graph.py and tests/test_world_graph.py, plus this report. Existing core and evaluator files are unchanged. Graph definitions contain data, never caller-supplied live kernel dictionaries. Interpreter manifests pin explicit source bytes. Parameter-free references check the existing SharedR1 object and its content identity. The subsequent trained-parent integration connects the format to solver serialization and verifies the existing neural routes.

Validation is deterministic software testing, not a final capability bank. No learning, search tuning or data split is involved. Tiny SharedR1 fixtures use seed 7 only for repeatability; these are not independent scientific trials. Acceptance requires correct owner sharing with no registered duplicate parameters, defensive immutable copies and content hashes, stale/transitive dependency rejection, transactional rejection/cancellation, masks and units, distinct factual versus hypothetical evidence, save/restore equality, model-version refusal and explicit same-schema replay preserving admitted observations. Incompatible schema changes must fail. Tests also cover invalid budget increments and shared branch budgets if a budget API is introduced.

Execution cap: each focused pytest invocation is externally limited to 60 seconds, lint to 20 seconds. Failures are recorded before repair; repetition is allowed only for a diagnosed failed check or changed source. Source files and final exact command outputs/hashes will be appended here. No full shared study, neural training, curriculum or paid compute is part of this subtask.

Boundary: event schemas, graph dependency semantics, evidence eligibility and the replay algorithm are supplied. Same-schema replay of observation records is not latent-state migration. A graph registry is not semantic grounding, causal understanding, a learned investigator or proof of a merged predictive architecture.


**Execution outcome — implemented and unit-tested**

The new files pass **19 focused cases** and Ruff. Final validation took **2.310 s wall**, **2.156 s CPU**, with **216,276,992 bytes peak RSS** in the actual Python process. All acceptance cases in this limited software protocol passed. W01/W02's broader requirements for an integrated neural/executable inference loop and actual latent-state migration remain open.

[event_ir.py](D:/ai/projects/sera/src/sera/event_ir.py) provides EventSchema, EventRole and Event. Values, masks and entity references are defensively owned immutable tuples. Unavailable values are zeroed before features or content identity can use them. Units/modality/scale changes require explicit conversion. Role and provenance remain distinct; a prediction carrying an observation-shaped payload is ineligible for factual admission.

[world_graph.py](D:/ai/projects/sera/src/sera/world_graph.py) provides SharedOwnerRef, content-addressed ExecutableDefinition/ExecutableGraph, MechanismRef, GraphStore, ExecutionBudget, LiveSituation and HypotheticalSituation. SharedOwnerRef registers no trainable parameters and can bind to an existing solver's r1 component. It refuses a replaced or different owner. Its fresh content hash uses SERA's identity format without the tensor-version cache, so raw tensor data mutations cannot evade identity checks.

Graph bodies are canonical immutable JSON data. No live kernel dictionary is copied from Kavi. Interpreter identities hash an explicit named source manifest. Graph construction validates schemas, interpreter identities and every declared dependency's content version. Replacement builds a candidate against an exact parent hash, removes transitively affected dependents unless explicitly rebuilt, calls an optional admission validator, and changes the pointer only after successful validation. Rejected and interrupted candidates leave the incumbent available. This validates declared dependency ownership; the eventual interpreter must separately validate the semantics of its program language.

LiveSituation retains a bounded admitted event record for replay, plus separately admitted verified/simulator labels. Factual observation and label acquisition counters count only new admissions. Restore and same-schema migration consume `replayed_observations` and `replayed_labels` in the preserved/shared execution budget. A budget failure can leave attempted work charged; it exposes no partially restored/migrated successor and leaves old factual state unchanged. Snapshot state identity and the resource-ledger hash are separate, since a branch legitimately consumes shared work while preserving facts.

HypotheticalSituation shares only immutable factual events, the frozen graph, the owner reference and the cumulative budget. Its assumptions require hypothetical role and prediction origin. It has no factual or label admission path. A hypothetical snapshot cannot restore as a factual situation. The mechanism guard rejects a parameter/schema/graph/interpreter change. Migration explicitly replays the same eligible observation and label records under exactly the same EventSchema; any schema change is refused. No latent tensors are migrated by this module.

**Meaningful verification and preserved failures**

[test_world_graph.py](D:/ai/projects/sera/tests/test_world_graph.py) covers nested list aliasing, hidden values, unit conversion refusal, source-manifest changes, dependency tampering, transitive invalidation, rebuilding dependent content references, rejection/cancellation, one-owner identity and registration, model mutation including raw tensor data writes, ordering and duplicate evidence, prediction-to-label rejection, imagined-branch isolation, immutable snapshots, charged replay, budget exhaustion, incompatible schemas and preserved observations after migration.

The first implementation run had seven passes and nine setup errors: the new fixture incorrectly requested one memory head although existing SERA requires at least two. Correcting the fixture resolved that setup error. A separate telemetry wrapper then failed after invoking pytest; its captured test result was lost and is not counted as success. The corrected wrapper produced a 16-test pass. Final review added the raw-tensor mutation check, producing a 17-test pass. My review then found that replay reused acquisition counters; the final revision separates replay work and adds the two exhaustion cases, producing the authoritative 19-case pass. Earlier passing results remain evidence about their earlier implementation, not extra independent trials.

Commands are equivalent to:

```powershell
& 'D:/ai/projects/sera/.venv/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_world_graph.py
& 'D:/ai/projects/sera/.venv/Scripts/python.exe' -B -m ruff check --no-cache src/sera/event_ir.py src/sera/world_graph.py tests/test_world_graph.py
```

The measured passes invoked pytest.main in the actual Python process under a 60-second exit watchdog. The final lint command exited 0. No existing SERA source, legacy verifier, evaluator artifact, baseline, commit or remote was changed by this subtask. The broader solver integration and full-suite validation are recorded separately.

**Final source identities**

| File | SHA-256 | Bytes |
| --- | --- | ---: |
| src/sera/event_ir.py | `0f063030fec0b52fd89964ec94a82342a0aa032814b7d79adda128f302615afa` | 6969 |
| src/sera/world_graph.py | `5bb5d9a0955f7b16cb60b2db5fc6494cb00a1e07438018225fed7931487e6a19` | 23640 |
| tests/test_world_graph.py | `a177ae4707632657e9ed930f002be8536c08a22a5e01313526e2ac4a9b5b84b3` | 18609 |

The first 2825 bytes of this report are the pre-implementation protocol; their SHA-256 is `20add732e9b5eed16677f3dbb401bacb1c743682cb47c3558ea4120f25d436d8`. The following machine-readable block retains the exact run outputs, failure interpretations, measured costs and source identities. Initial launcher-only telemetry is explicitly excluded from workload CPU/RSS claims.

```json
{
  "kind": "externally_authored_software_contract_validation",
  "status": "implemented_and_unit_tested",
  "capability_boundary": "No curriculum, learning, learned grounding, neural-latent migration or complete upstream predictive integration.",
  "protocol_sha256_before_implementation_results": "20add732e9b5eed16677f3dbb401bacb1c743682cb47c3558ea4120f25d436d8",
  "protocol_prefix_bytes": 2825,
  "source_files": [
    {
      "path": "src/sera/event_ir.py",
      "absolute_path": "D:\\ai\\projects\\sera\\src\\sera\\event_ir.py",
      "sha256": "0f063030fec0b52fd89964ec94a82342a0aa032814b7d79adda128f302615afa",
      "bytes": 6969,
      "lines": 164
    },
    {
      "path": "src/sera/world_graph.py",
      "absolute_path": "D:\\ai\\projects\\sera\\src\\sera\\world_graph.py",
      "sha256": "5bb5d9a0955f7b16cb60b2db5fc6494cb00a1e07438018225fed7931487e6a19",
      "bytes": 23640,
      "lines": 542
    },
    {
      "path": "tests/test_world_graph.py",
      "absolute_path": "D:\\ai\\projects\\sera\\tests\\test_world_graph.py",
      "sha256": "a177ae4707632657e9ed930f002be8536c08a22a5e01313526e2ac4a9b5b84b3",
      "bytes": 18609,
      "lines": 380
    }
  ],
  "initial_source_hashes": [
    {
      "Path": "D:\\ai\\projects\\sera\\src\\sera\\event_ir.py",
      "Hash": "0F063030FEC0B52FD89964EC94A82342A0AA032814B7D79ADDA128F302615AFA"
    },
    {
      "Path": "D:\\ai\\projects\\sera\\src\\sera\\world_graph.py",
      "Hash": "E48B7C6C288F1BA6A8A0B264E08D4D6439332039F62E55F827CA1F35EAB469DD"
    },
    {
      "Path": "D:\\ai\\projects\\sera\\tests\\test_world_graph.py",
      "Hash": "809F627A976C1A7DFECD43DCF714A74B9D9EBBCA95CCBDA65DB6D294EA53DA36"
    }
  ],
  "prior_17_test_source_hashes": [
    {
      "Path": "D:\\ai\\projects\\sera\\src\\sera\\event_ir.py",
      "Hash": "0F063030FEC0B52FD89964EC94A82342A0AA032814B7D79ADDA128F302615AFA"
    },
    {
      "Path": "D:\\ai\\projects\\sera\\src\\sera\\world_graph.py",
      "Hash": "14D702E154FCF07D8F8A26BCF95C531DD7FD02EE2E426AF6C767759FEC929114"
    },
    {
      "Path": "D:\\ai\\projects\\sera\\tests\\test_world_graph.py",
      "Hash": "507F1DF1793AB243E50934A70352F0FCE7C303720F5DC04D019603161CDA7162"
    }
  ],
  "attempts": [
    {
      "stage": "initial fixture",
      "tool_wall_seconds": 3.1469072,
      "command": [
        "D:\\ai\\projects\\sera\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_world_graph.py"
      ],
      "exit_code": 1,
      "timeout_seconds": 60,
      "timed_out": false,
      "wall_seconds": 2.9218009999999595,
      "stdout": ".......EEEEEEEEE                                                         [100%]\n=================================== ERRORS ====================================\n_ ERROR at setup of test_parameter_free_reference_checks_object_owner_and_registered_replacement _\n\n    @pytest.fixture\n    def owner():\n        torch.manual_seed(7)\n>       return SharedR1(width=8, heads=1, memory_dim=2)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntests\\test_world_graph.py:35: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\nsrc\\sera\\shared.py:47: in __init__\n    super().__init__(width, heads, memory_dim, kind, **options)\nsrc\\sera\\r1.py:44: in __init__\n    self.memory = StatefulModel(ModelConfig(**self.settings))\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n<string>:7: in __init__\n    ???\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\nself = ModelConfig(kind='delta', width=8, heads=1, memory_dim=2)\n\n    def __post_init__(self):\n        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:\n>           raise ValueError(\"Invalid model configuration\")\nE           ValueError: Invalid model configuration\n\nsrc\\sera\\models.py:39: ValueError\n_ ERROR at setup of test_mechanism_pins_complete_graph_owner_schema_and_interpreter _\n\n    @pytest.fixture\n    def owner():\n        torch.manual_seed(7)\n>       return SharedR1(width=8, heads=1, memory_dim=2)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntests\\test_world_graph.py:35: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\nsrc\\sera\\shared.py:47: in __init__\n    super().__init__(width, heads, memory_dim, kind, **options)\nsrc\\sera\\r1.py:44: in __init__\n    self.memory = StatefulModel(ModelConfig(**self.settings))\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n<string>:7: in __init__\n    ???\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\nself = ModelConfig(kind='delta', width=8, heads=1, memory_dim=2)\n\n    def __post_init__(self):\n        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:\n>           raise ValueError(\"Invalid model configuration\")\nE           ValueError: Invalid model configuration\n\nsrc\\sera\\models.py:39: ValueError\n_ ERROR at setup of test_live_situation_rejects_wrong_schema_order_duplicate_and_prediction _\n\n    @pytest.fixture\n    def owner():\n        torch.manual_seed(7)\n>       return SharedR1(width=8, heads=1, memory_dim=2)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntests\\test_world_graph.py:35: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\nsrc\\sera\\shared.py:47: in __init__\n    super().__init__(width, heads, memory_dim, kind, **options)\nsrc\\sera\\r1.py:44: in __init__\n    self.memory = StatefulModel(ModelConfig(**self.settings))\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n<string>:7: in __init__\n    ???\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\nself = ModelConfig(kind='delta', width=8, heads=1, memory_dim=2)\n\n    def __post_init__(self):\n        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:\n>           raise ValueError(\"Invalid model configuration\")\nE           ValueError: Invalid model configuration\n\nsrc\\sera\\models.py:39: ValueError\n_ ERROR at setup of test_imagination_cannot_mutate_live_state_or_admit_labels _\n\n    @pytest.fixture\n    def owner():\n        torch.manual_seed(7)\n>       return SharedR1(width=8, heads=1, memory_dim=2)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntests\\test_world_graph.py:35: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\nsrc\\sera\\shared.py:47: in __init__\n    super().__init__(width, heads, memory_dim, kind, **options)\nsrc\\sera\\r1.py:44: in __init__\n    self.memory = StatefulModel(ModelConfig(**self.settings))\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n<string>:7: in __init__\n    ???\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\nself = ModelConfig(kind='delta', width=8, heads=1, memory_dim=2)\n\n    def __post_init__(self):\n        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:\n>           raise ValueError(\"Invalid model configuration\")\nE           ValueError: Invalid model configuration\n\nsrc\\sera\\models.py:39: ValueError\n_ ERROR at setup of test_shared_budget_rejects_negative_counts_and_does_not_refresh_on_branch _\n\n    @pytest.fixture\n    def owner():\n        torch.manual_seed(7)\n>       return SharedR1(width=8, heads=1, memory_dim=2)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntests\\test_world_graph.py:35: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\nsrc\\sera\\shared.py:47: in __init__\n    super().__init__(width, heads, memory_dim, kind, **options)\nsrc\\sera\\r1.py:44: in __init__\n    self.memory = StatefulModel(ModelConfig(**self.settings))\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n<string>:7: in __init__\n    ???\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\nself = ModelConfig(kind='delta', width=8, heads=1, memory_dim=2)\n\n    def __post_init__(self):\n        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:\n>           raise ValueError(\"Invalid model configuration\")\nE           ValueError: Invalid model configuration\n\nsrc\\sera\\models.py:39: ValueError\n_ ERROR at setup of test_snapshot_restoration_preserves_evidence_and_charges_replay _\n\n    @pytest.fixture\n    def owner():\n        torch.manual_seed(7)\n>       return SharedR1(width=8, heads=1, memory_dim=2)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntests\\test_world_graph.py:35: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\nsrc\\sera\\shared.py:47: in __init__\n    super().__init__(width, heads, memory_dim, kind, **options)\nsrc\\sera\\r1.py:44: in __init__\n    self.memory = StatefulModel(ModelConfig(**self.settings))\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n<string>:7: in __init__\n    ???\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\nself = ModelConfig(kind='delta', width=8, heads=1, memory_dim=2)\n\n    def __post_init__(self):\n        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:\n>           raise ValueError(\"Invalid model configuration\")\nE           ValueError: Invalid model configuration\n\nsrc\\sera\\models.py:39: ValueError\n_ ERROR at setup of test_restoration_rejects_wrong_situation_model_and_budget_tampering _\n\n    @pytest.fixture\n    def owner():\n        torch.manual_seed(7)\n>       return SharedR1(width=8, heads=1, memory_dim=2)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntests\\test_world_graph.py:35: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\nsrc\\sera\\shared.py:47: in __init__\n    super().__init__(width, heads, memory_dim, kind, **options)\nsrc\\sera\\r1.py:44: in __init__\n    self.memory = StatefulModel(ModelConfig(**self.settings))\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n<string>:7: in __init__\n    ???\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\nself = ModelConfig(kind='delta', width=8, heads=1, memory_dim=2)\n\n    def __post_init__(self):\n        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:\n>           raise ValueError(\"Invalid model configuration\")\nE           ValueError: Invalid model configuration\n\nsrc\\sera\\models.py:39: ValueError\n_ ERROR at setup of test_identity_migration_replays_admitted_facts_without_reinterpreting_schema _\n\n    @pytest.fixture\n    def owner():\n        torch.manual_seed(7)\n>       return SharedR1(width=8, heads=1, memory_dim=2)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntests\\test_world_graph.py:35: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\nsrc\\sera\\shared.py:47: in __init__\n    super().__init__(width, heads, memory_dim, kind, **options)\nsrc\\sera\\r1.py:44: in __init__\n    self.memory = StatefulModel(ModelConfig(**self.settings))\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n<string>:7: in __init__\n    ???\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\nself = ModelConfig(kind='delta', width=8, heads=1, memory_dim=2)\n\n    def __post_init__(self):\n        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:\n>           raise ValueError(\"Invalid model configuration\")\nE           ValueError: Invalid model configuration\n\nsrc\\sera\\models.py:39: ValueError\n_ ERROR at setup of test_live_model_mutation_fails_before_new_evidence_is_admitted _\n\n    @pytest.fixture\n    def owner():\n        torch.manual_seed(7)\n>       return SharedR1(width=8, heads=1, memory_dim=2)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntests\\test_world_graph.py:35: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\nsrc\\sera\\shared.py:47: in __init__\n    super().__init__(width, heads, memory_dim, kind, **options)\nsrc\\sera\\r1.py:44: in __init__\n    self.memory = StatefulModel(ModelConfig(**self.settings))\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n<string>:7: in __init__\n    ???\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\nself = ModelConfig(kind='delta', width=8, heads=1, memory_dim=2)\n\n    def __post_init__(self):\n        if self.kind not in KINDS or min(self.width, self.heads, self.memory_dim) < 2:\n>           raise ValueError(\"Invalid model configuration\")\nE           ValueError: Invalid model configuration\n\nsrc\\sera\\models.py:39: ValueError\n=========================== short test summary info ===========================\nERROR tests/test_world_graph.py::test_parameter_free_reference_checks_object_owner_and_registered_replacement\nERROR tests/test_world_graph.py::test_mechanism_pins_complete_graph_owner_schema_and_interpreter\nERROR tests/test_world_graph.py::test_live_situation_rejects_wrong_schema_order_duplicate_and_prediction\nERROR tests/test_world_graph.py::test_imagination_cannot_mutate_live_state_or_admit_labels\nERROR tests/test_world_graph.py::test_shared_budget_rejects_negative_counts_and_does_not_refresh_on_branch\nERROR tests/test_world_graph.py::test_snapshot_restoration_preserves_evidence_and_charges_replay\nERROR tests/test_world_graph.py::test_restoration_rejects_wrong_situation_model_and_budget_tampering\nERROR tests/test_world_graph.py::test_identity_migration_replays_admitted_facts_without_reinterpreting_schema\nERROR tests/test_world_graph.py::test_live_model_mutation_fails_before_new_evidence_is_admitted\n",
      "stderr": "",
      "cpu_seconds": 0,
      "peak_rss_bytes": 4390912,
      "reported_result": "7 passed, 9 setup errors",
      "telemetry_scope": "Reported CPU=0 and RSS=4390912 apply to Windows venv launcher only, not actual workload; treat workload telemetry as unavailable.",
      "diagnosis": "heads=1 violates existing SERA ModelConfig minimum 2; new test fixture changed to heads=2."
    },
    {
      "stage": "post-fixture telemetry wrapper failure",
      "chunk_id": "a8e00d",
      "wall_time_seconds": 2.7902758,
      "exit_code": 1,
      "original_token_count": 38,
      "output": "Traceback (most recent call last):\r\n  File \"<string>\", line 14, in <module>\r\nctypes.ArgumentError: argument 1: OverflowError: int too long to convert\r\n",
      "reported_result": "pytest invoked, but captured result lost when telemetry failed; not counted as a passing run.",
      "diagnosis": "ctypes GetProcessMemoryInfo argument lacked pointer-width HANDLE argtypes; result logging now tolerates telemetry errors."
    },
    {
      "stage": "corrected fixture before raw-tensor check",
      "tool_wall_seconds": 2.767671,
      "command": [
        "D:\\ai\\projects\\sera\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_world_graph.py"
      ],
      "execution": "pytest.main in measured Python process with 60-second exit watchdog",
      "exit_code": 0,
      "wall_seconds": 2.295923799999855,
      "cpu_seconds": 2.265625,
      "peak_rss_bytes": 215171072,
      "timeout_seconds": 60,
      "stdout": "................                                                         [100%]\n",
      "stderr": "",
      "tests_passed": 16
    },
    {
      "stage": "fresh full owner fingerprint; before replay-accounting correction",
      "tool_wall_seconds": 2.6925584000000002,
      "command": [
        "D:\\ai\\projects\\sera\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_world_graph.py"
      ],
      "execution": "pytest.main in measured Python process with 60-second exit watchdog",
      "exit_code": 0,
      "wall_seconds": 2.2163376999997126,
      "cpu_seconds": 2.078125,
      "peak_rss_bytes": 215801856,
      "timeout_seconds": 60,
      "stdout": ".................                                                        [100%]\n",
      "stderr": "",
      "tests_passed": 17,
      "review_finding": "Restore/migration reprocessed old evidence through admission counters. Evidence payload stayed unique, but acquisition counters were incremented again."
    },
    {
      "stage": "final distinct replay accounting and transactional replay exhaustion",
      "tool_wall_seconds": 2.7850311,
      "command": [
        "D:\\ai\\projects\\sera\\.venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_world_graph.py"
      ],
      "execution": "pytest.main in measured Python process with 60-second exit watchdog",
      "exit_code": 0,
      "wall_seconds": 2.3097577000003184,
      "cpu_seconds": 2.15625,
      "peak_rss_bytes": 216276992,
      "timeout_seconds": 60,
      "stdout": "...................                                                      [100%]\n",
      "stderr": "",
      "tests_passed": 19
    }
  ],
  "lint_attempts": [
    {
      "stage": "initial",
      "chunk_id": "717da4",
      "wall_time_seconds": 0.2523592,
      "exit_code": 0,
      "original_token_count": 5,
      "output": "All checks passed!\n"
    },
    {
      "stage": "prior 17-test version",
      "chunk_id": "52d791",
      "wall_time_seconds": 0.2345882,
      "exit_code": 0,
      "original_token_count": 5,
      "output": "All checks passed!\n"
    },
    {
      "stage": "final",
      "chunk_id": "6296b5",
      "wall_time_seconds": 0.2278616,
      "exit_code": 0,
      "original_token_count": 5,
      "output": "All checks passed!\n"
    }
  ],
  "cost_scope": {
    "cpu_rss_scope": "For the final three passes, measurements come from the actual pytest Python process. Wall and CPU include imports and execution; peak RSS is lifetime PeakWorkingSetSize for that process. Wrapper startup and external research work are additional.",
    "negative_results_retained": true,
    "training_steps": 0,
    "curriculum_runs": 0,
    "installed_packages": 0,
    "paid_compute_calls": 0,
    "parameter_owner_count": 1,
    "added_trainable_parameters": 0
  }
}
```

