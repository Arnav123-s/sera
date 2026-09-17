"""Trace actual artifact reads while replaying the affected integration tests."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

from .common import OUT, ROOT, RUNS, sha, write


def main():
    started = time.time()
    paths = set()

    def opened(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1]
        if isinstance(mode, str) and any(letter in mode for letter in "wax+"):
            return
        if not path.is_relative_to(ROOT):
            return
        name = path.relative_to(ROOT).as_posix()
        if (
            name.startswith(("runs/", "research/intake/"))
            and path.is_file()
            and path.stat().st_mtime < started
        ):
            paths.add(name)

    sys.addaudithook(opened)
    targets = [
        "tests/test_language_guard.py::test_selected_actual_owner_restores_without_changing_shared_aliases",
        "tests/test_math_runtime.py::test_language_parameters_persist_with_live_state_and_reproved_skills",
        "tests/test_self_study_lifecycle.py::test_resume_preserves_acquisition_order_and_owner",
        "tests/test_stream_interfaces.py::test_only_text_is_encoded_for_prediction",
        "tests/test_stream_interfaces.py::test_prepared_training_has_no_dev_or_final_ids",
        "tests/test_concept_refinement.py",
    ]
    code = pytest.main(["-q", *targets])
    # This completed-bank file is an existence gate in the new integration fixture;
    # tests use development observations, never score its final outcomes.
    paths.add((RUNS / "final.json").relative_to(ROOT).as_posix())
    write(
        OUT / "publication-hardening/fixture-reads.json",
        {
            "pytest_returncode": int(code),
            "targets": targets,
            "files": [
                {"path": name, "sha256": sha(ROOT / name), "bytes": (ROOT / name).stat().st_size}
                for name in sorted(paths)
            ],
        },
    )
    print(json.dumps({"returncode": int(code), "artifact_reads": len(paths)}))
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
