# Discover connections from retained knowledge

[Home](../../README.md) · [Results and proofs](report.md) · [Exact records](results.json) · [Frozen protocol](PROTOCOL.md)

SERA constructed a new connection between its learned integration operator and time-weighting, then four verified extensions. The discovery run used **zero web queries and zero supplied completed equations**. It generated relationships by executing what it already knew, fitting candidate coefficients, and checking exact algebra. The operator grammar and checker are supplied engineering.

The newest continuing owner is `runs/sera-discovery-live`. This command uses a discovered alternative to calculate displacement from an acceleration polynomial:

```powershell
.venv/Scripts/python.exe scripts/run_discovery_bounded.py --seconds 45 --output runs/my-discovered-motion-001 --module experiments.operator_discovery.runtime -- solve --coefficients 2
```

For acceleration 2 and zero initial position/velocity, it returns the position polynomial **t²** through the newly retained relationship. Use comma-separated rational coefficients, constant term first, up to degree three. Existing primitive certificates check representability and exactness before returning a route.

```powershell
.venv/Scripts/python.exe scripts/run_discovery_bounded.py --seconds 45 --output runs/my-discovery-status-001 --module experiments.operator_discovery.runtime -- status
```

These commands use the prepared workspace, its existing allowance, one CPU thread and the 2 GiB process-tree cap. Use fresh output directories. The earlier learner and its qualified mechanics portfolio remain preserved at the [Stage 34 entry point](../34_learning_progress/README.md). The newest owner retains every inherited tensor and task record; its new algebraic routes carry their own exact certificates.

The public archive stores the inherited parent once and reconstructs all nine discovery checkpoints exactly. `scripts/discovery_artifacts.py` verifies the source hashes, checkpoint reconstruction and independent algebra certificates. All original checkpoint files remain locally preserved.
