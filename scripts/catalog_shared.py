"""Append the shared variants without removing historical models or their evidence."""

import hashlib
import json
from pathlib import Path

from sera.storage import write_json

ROOT = Path(__file__).resolve().parents[1]


def artifact(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    trials = [json.loads((ROOT / f"runs/shared-learner-repaired/{seed}/{kind}/trial.json").read_text(encoding="utf-8"))
              for seed in range(3) for kind in ("delta", "reference")]
    if any(r["status"] != "completed" for r in trials) or len({r["environment"]["source_sha256"] for r in trials}) != 1:
        raise ValueError("Catalog requires all six completed trials with one executable source")
    data = {"trials": trials, "source_sha256": trials[0]["environment"]["source_sha256"],
            "assembly": json.loads((ROOT / "runs/shared-learner-assembly-repaired/assembly.json").read_text(encoding="utf-8"))}
    path = ROOT / "research/variants.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    if any(v["id"].startswith("shared-") for v in registry["variants"]):
        raise ValueError("Shared variants already cataloged; review an explicit registry update")
    for kind, title in (("delta", "Associative"), ("reference", "Reference")):
        rows = [r for r in data["trials"] if r["kind"] == kind]
        registry["variants"].append({"id": f"shared-{kind}-v1", "name": f"SERA Shared {title}",
            "role": "jointly pretrained R1 owner for world, typed and sequence tasks", "status": "preserved-research-variant",
            "source_sha256": data["source_sha256"], "results": {"seeds": 3, "parameters": rows[0]["parameters"],
                "core_state_bytes": rows[0]["core_state_bytes"], "report": "reports/shared-learner-study.md"},
            "artifacts": [artifact(ROOT / f"runs/shared-learner-repaired/{seed}/{kind}/base.pt") for seed in range(3)],
            "scope": "Shared adapters and heads; same update/example budget across cores, unequal runtime and parameters. Earlier attempts and all adaptation controls remain in named study directories."})
    current = data["assembly"]["learning"]["current"]
    registry["variants"].append({"id": "shared-scoped-v1", "name": "SERA Shared Adaptation",
        "role": "persistent shared R1/R2 solver with scoped learned weight residuals", "status": "active-bounded-learner",
        "source_sha256": data["source_sha256"], "results": {"version": current["version"],
            "admission": data["assembly"]["learning"]["status"], "report": "reports/shared-learner-study.md"},
        "artifacts": [artifact(ROOT / f"runs/sera-0.5-current/versions/{current['version']}.pt")],
        "scope": "Seed 0 was preselected. First-binding applicability is supplied; the rank-limited weight residuals are learned. Base outputs outside that scope remain unchanged. This is not a separate complete typed model or a learned task router."})
    registry["current_shared_solver"] = {"path": "runs/sera-0.5-current", "version": current["version"],
                                         "solver_sha256": current["solver_sha256"]}
    write_json(path, registry)
    print("Preserved and cataloged", len(registry["variants"]), "variants")


if __name__ == "__main__":
    main()
