# Use and resume this release

The local service is available at [the reasoning workspace](http://127.0.0.1:8765/?panel=reason). Its other tabs perform numerical transformations from examples, process changing CSV observations and continue the saved simulated control task. The language route supports the displayed finite instruction families over residues modulo eleven. Unknown or weakened wording can trigger a supplied lesson; an unsupported interpretation is withheld.

The live workspace is `runs/sera-workbench`. Its `current.json` points to an immutable revision under `revisions/`. At this release it contains 15 transactions, 85 interaction observations, three previous numerical contexts and three language lessons using 432 examples. Revision 14 reuses the latest learned instruction and returns `x = 5` without another lesson. The complete owner identity is `64b48d5f8192176ff6a4092b629822dfda6741d8f017b9cfbce295baf0eb7589`.

If the service is already running, use it. After a deliberate host shutdown, start it from the repository with the existing environment:

```powershell
.venv/Scripts/python.exe -X utf8 -m workbench --port 8765
```

This loads existing state; it does not restart completed training. The owned host recorded for this release uses `runs/workbench-host-language-20260916-051800/pid.txt`. Recheck process ownership and start time before any future host replacement; a saved PID alone is insufficient.

Read-only evidence verification needs no training or model loading:

```powershell
.venv/Scripts/python.exe scripts/verify_grounded_release.py --local-checkpoints
```

`application-snapshot.json` is the frozen final session. `application-history.zip` preserves its earlier revisions, task requests/responses and the separate L11 continuing history. `checkpoint-manifest.json` identifies the preserved local tensors; `sources.zip` freezes the code closure. Tensor checkpoints remain in ignored `runs/`, so a source-only clone is insufficient to restore this trained owner. Keep the original local base solver and the checkpoint paths in the manifest together with the repository.

Interpreter checks depend on exact source bytes. For restoration in a fresh isolated checkout, use the frozen source archive; ordinary Git text normalization can change line endings and therefore strict interpreter identities. Do not extract an old source snapshot over uncommitted work or an active service.

The last explicit runtime migration changed execution plumbing while preserving the full owner hash and language checkpoint. Its receipt and old source archive are in this directory. Earlier L11 sessions refer to their original archived runtime; use that version in an isolated checkout for historical replay instead of relaxing integrity checks or replacing the live workspace. Do not rerun a completed cohort to recreate missing provenance.

Each task-time lesson saves its parent session, teaching/development examples, language tensor state, best selected state, optimizer, NumPy/Torch RNG state, parameter flags and step counter. Interrupted-lesson resumption was tested against uninterrupted learning. Completed jobs in this release have no unfinished optimizer work. Historical `pending.json` files are checkpoint journals; the corresponding completed lesson and transaction records take precedence.

All numerical work shares `runs/v3-batch-001/budget.json`, one worker thread and a 2 GiB committed process-tree cap. The frozen cost receipt is a release snapshot; the live balance changes with later tasks. The unchanged-file watcher performs no training until a CSV changes. The service retains outputs and reports a resource limit when it cannot reserve another task. No later scientific experiment is automatically scheduled by this release.
