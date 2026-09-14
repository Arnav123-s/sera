"""A declared parameter-count control, selected before observing its task scores."""

from dataclasses import asdict
from pathlib import Path

from sera.evaluation import evaluate
from sera.models import ModelConfig
from sera.storage import write_json
from sera.training import TrainConfig, train

configuration = ModelConfig(kind="delta", width=85, heads=2, memory_dim=12)
output = Path("runs/matched-delta-v1")
output.mkdir(parents=True, exist_ok=True)
write_json(
    output / "design.json",
    {
        "configuration": asdict(configuration),
        "selection": "Closest parameter count in widths 24..100, heads {2,4}, dimensions {8,12,16}; no task scores used",
        "hybrid_parameters": 11208,
        "delta_parameters": 11216,
        "limitations": "Parameter-count control, not matched state bytes or operator FLOPs. Architecture tuning budget remains minimal.",
    },
)
results = []
for seed in (0, 1, 2):
    run_dir = output / str(seed)
    model = train(configuration, TrainConfig(seed=seed), run_dir)
    row = {
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "core_state_bytes": model.state_bytes(),
    }
    for split, length in (("id", 12), ("ood", 24)):
        row[split], _ = evaluate(model, seed=seed, split=f"test-{split}", length=length)
    write_json(run_dir / "result.json", row)
    results.append(row)
    print(
        f"Matched delta seed {seed}: ID {row['id']['macro_accuracy']:.4f}, OOD {row['ood']['macro_accuracy']:.4f}",
        flush=True,
    )
write_json(output / "summary.json", {"config": asdict(configuration), "runs": results})
