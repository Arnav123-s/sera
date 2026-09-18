# Self-chosen investigations and verified discoveries

SERA learns the investigation process from three elementary examples, chooses questions from retained knowledge, proposes equations and text associations, saves its predictions before checking them, and retains scoped executable discoveries. [Results](report.md) · [All records](discoveries.json) · [Protocol](PROTOCOL.md) · [Audit](audit.json).

The independently selected SD owner is preserved at `runs/sera-self-discovery-live`. [Stage 42](../42_composed_discovery/README.md) continues from it by broadening missing-input combinations and composing previously learned rules. The ordinary batch interface selects the latest admitted owner automatically.

```powershell
.venv/Scripts/python.exe scripts/run_resolution_bounded.py --seconds 90 --output runs/my-discovery-batch-001 --module scripts.continuing_use -- --input research-continuation/42_composed_discovery/example-tasks.json --output runs/my-discovery-results-001.json
```

Use `discoveries` to inspect the actual retained records with their reserved evaluations. Use `solve_discovery` to supply a domain, target and available observations; SERA returns each applicable learned route, its required inputs and its conditional answer. The batch also retains the existing language, source reading, polynomial calculation and imagination commands.

All raw proposal commitments, intermediate states, final candidates and negative results are preserved in a content-hashed release archive. Large numerical artifacts are release assets; the repository holds reports, checksums, source, complete cost receipts and verification code. The verifier fetches a missing part only from this repository's fixed release URL, validates its size and SHA-256, and uses an owned local cache.
