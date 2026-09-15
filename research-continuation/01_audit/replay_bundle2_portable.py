"""Replay unchanged packet source with explicit portable source-map serialization.

This repairs only the identity-map path separator at runtime. Numerical
tolerances, archived artifacts, and inference code are not changed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    code = args.source.resolve() / "code"
    sys.path.insert(0, str(code))
    import run_study
    from sklab.core import digest
    import verify_results

    files = sorted(list((code / "sklab").glob("*.py")) + [code / "run_study.py"])
    portable = {
        p.relative_to(code).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    expected = digest(portable)
    archived = {
        json.loads(p.read_text())["code_sha256"]
        for folder in ("main", "meta")
        for p in (args.source / "results" / folder).glob("*.json")
    }
    if archived != {expected}:
        raise ValueError("Portable map does not match every archived source digest")
    identity = {
        "native_source_identity": run_study.source_identity(),
        "portable_source_identity": expected,
        "file_hashes": portable,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Only relative-path serialization; unchanged source and numerical tolerance",
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    # Compute, rather than copy, the expected digest from the preserved file bytes.
    verify_results.source_identity = lambda: expected
    sys.argv = [str(code / "verify_results.py"), "--root", str(args.source),
                "--output", str(args.output / "replay.json")]
    started = time.perf_counter()
    try:
        verify_results.main()
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps({
            "error": repr(exc), "elapsed_seconds": time.perf_counter() - started,
            "numeric_atol": 1e-12, "numeric_rtol": 1e-12,
        }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
