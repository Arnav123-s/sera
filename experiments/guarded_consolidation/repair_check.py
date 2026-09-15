"""Affected contract repair and fresh migration checks; not a rerun of GC-002."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from sera.storage import digest
from sera.world_graph import SharedOwnerRef

from .core import Library, Work, interpreter_identity
from .migration import migrate_verified_library
from .study import PARENT, ROOT, ReadOnlyStore, write


def test_explicit_repair_and_proof_replay():
    target = Path(os.environ["SERA_V3_REPAIR_OUTPUT"])
    target.mkdir(exist_ok=False)
    source = ROOT/"runs/v3-batch-001/GC-002-final/data"
    protocol = {"kind": "GC-002 affected-implementation repair", "source": source.relative_to(ROOT).as_posix(),
                "seed": 90339001, "new_cases_per_snapshot": 512,
                "current_interpreter": interpreter_identity(),
                "source_graph_files": {n: hashlib.sha256((source/f"{n}-library.json").read_bytes()).hexdigest()
                                       for n in ("initial", "corrected")},
                "changes": ["reject RHS-dependent forward relations before indexing", "enforce actual interpreter identity", "explicit finite-proof migration"],
                "scope": "No refitting, no old final tasks regenerated, no threshold changes; reprove saved programs and check fresh mathematical cases"}
    write(target/"protocol.json", {"payload": protocol, "sha256": digest(protocol)})
    started = time.perf_counter()
    solver = ReadOnlyStore(PARENT).load()
    parent_id = solver.identity()
    owner = SharedOwnerRef.from_solver(solver)
    work, rng, migrations = Work(), np.random.default_rng(protocol["seed"]), []
    checks = 0
    for name in ("initial", "corrected"):
        record = json.loads((source/f"{name}-library.json").read_text())
        with pytest.raises(ValueError, match="Stale executable interpreter"):
            Library.restore(record, owner, work, proof_backend="indexed")
        migrated, log = migrate_verified_library(record, owner, work, expected_parent=record["graph_sha256"])
        restored = Library.restore(migrated.record(), owner, work, proof_backend="indexed")
        assert restored.record() == migrated.record()
        relation = restored.store.current.definition("affine")
        with restored.batch(work):
            for _ in range(protocol["new_cases_per_snapshot"]):
                a, b, c = (int(x) for x in rng.integers(0, 11, 3))
                answer = restored.answer(relation.identity, dict(a=a, b=b, c=c), work)
                sign = 1 if name == "initial" else -1
                solutions = [x for x in range(11) if (a*x+sign*b) % 11 == c]
                assert answer == (solutions[0] if len(solutions) == 1 else None)
                checks += 1
        write(target/f"{name}-library.json", migrated.record())
        migrations.append(log)
    assert solver.identity() == parent_id
    result = {"status": "PASS", "migrations": migrations, "fresh_cases_checked": checks,
              "work": dict(work.counts), "operations": work.operations,
              "parent_solver_sha256": parent_id, "parent_unchanged": True,
              "wall_seconds": time.perf_counter()-started,
              "cost_scope": "Additional repair/migration validation cost, retained separately from both frozen candidate studies."}
    write(target/"result.json", result)
    print(json.dumps(result))
