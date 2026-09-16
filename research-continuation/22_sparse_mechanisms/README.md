# Sparse mechanisms and resilient task memory

I tested compressed sensing across mechanism discovery, memory, sparse weight adaptation, actual SERA readout repair and synthetic Fourier imaging. The [report](report.md) gives the results and failures; the [literature notes](literature.md) connect them to Tao's own work and the MRI papers. The [architecture audit](architecture-audit.md) identifies what this contributes to SERA and what remains unproved.

The release contains 23,096 fresh final records, independent saved-output audits, preserved development failures, exact source snapshots and a runnable small-artifact repair tool. The supported-answer challenger and the disagreement investigator both failed promotion. The live learner remains available at [the reasoning workspace](http://127.0.0.1:8765/?panel=reason).

## Use the tested repair example

From the repository root, this restores a copy of three retained SERA task contexts from an archive with four corrupted codeword cells per block:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/run_sparse_bounded.py --seconds 30 --output runs/my-memory-restore-worker --module experiments.sparse_mechanisms.repair -- restore --input research-continuation/22_sparse_mechanisms/CS-REAL-MEMORY-001/example-4-damaged-cells.zip --output runs/my-restored-contexts.json --expected-sha256 37d893785ed9c9daa7b723fc4ff5dbf1589246b7ecdc9089414f545b65e86380
```

Success reports `VERIFIED_EXACT_RESTORE`. Both the worker directory and output filename must be new. The restored JSON is an artifact copy; it is not automatically installed into the live model. The numerical allowance is the same cumulative allowance used by the workbench.

The utility also accepts your own file of 1–8,192 bytes:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/run_sparse_bounded.py --seconds 30 --output runs/my-memory-encode-worker --module experiments.sparse_mechanisms.repair -- protect --input path/to/small-artifact.json --output runs/my-protected-artifact.zip --seed 43
```

Retain the printed payload SHA-256 separately. Use it with `restore --expected-sha256` and a new output filename. A wrong digest, unsupported metadata, nonfinite values or failed payload reconstruction cause rejection before an output file is created. This protects against the declared finite-value corruption model with intact metadata; it is not a substitute for ordinary backups or a universally superior error-correcting code.

The numeric code takes about eight times the payload size before padding and metadata. The useful mechanism is sparse-error correction. The sparse-memory sketch is a separate experiment and was larger than direct index/value storage on known four-entry vectors.

## Inspect without repeating experiments

```powershell
.venv/Scripts/python.exe scripts/verify_sparse_release.py
```

This checks the frozen inventory, hashes, source archives, audit outcomes, targeted test reports and supervised costs. It performs no training or reconstruction. Original numerical records and arrays remain in the named study folders. Content-addressed array archives deduplicate common inputs losslessly.

For actual model restoration, retain the earlier local checkpoint chain documented in [release 21](../21_grounded_language/resume.md). This research release leaves those tensors and the current live pointer unchanged. [Resume details](resume.md) distinguish completed work from the next open research question.
