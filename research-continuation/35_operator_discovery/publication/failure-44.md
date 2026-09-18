# Verification entry-point repair

GitHub run 44 preserved the learning archive checks, then both runners failed while importing the discovery verifier: direct script execution put `scripts/` on the import path, and `experiments` was unavailable. The corrected workflow invokes `python -m scripts.discovery_artifacts` from the repository root, matching the successful bounded local verification. Scientific sources, results, archives and seals are unchanged. The original [receipt](github-run-44.json) and [logs](github-run-44-logs.zip) remain preserved.
