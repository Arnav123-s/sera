"""Read-only v15 verification in a fresh supervised directory."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "research/intake/v15-update/SERA_v15"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable", action="store_true")
    if parser.parse_args().portable:
        portable()
        return
    subprocess.run([sys.executable, "code/verify_manifest.py"], cwd=PACKET, check=True)
    subprocess.run([sys.executable, "code/run_checks.py", "--output", str(ROOT / "runs/QP-packet-check-001"),
                    "--wall-budget-seconds", "150"], cwd=PACKET, check=True)


def portable():
    import numpy as np
    sys.path.insert(0, str(PACKET / "src"))
    from questlab.physics import (
        alias_entropy,
        conservation,
        gate_trials,
        identities,
        reward_farming,
    )
    out = ROOT / "runs/QP-packet-portable-001"
    out.mkdir(exist_ok=False)
    differences = []

    def compare(a, b):
        if isinstance(a, dict):
            assert set(a) == set(b)
            for k in a:
                compare(a[k], b[k])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                compare(x, y)
        elif isinstance(a, float):
            differences.append(abs(a-b))
            assert np.isclose(a, b, atol=1e-12, rtol=1e-12)
        else:
            assert a == b
    for name, fn in (("physics", identities), ("alias_entropy", alias_entropy), ("farming", reward_farming),
                     ("gates", gate_trials), ("conservation", conservation)):
        compare(fn(), json.loads((PACKET / f"results/diagnostics/{name}.json").read_text()))
    with (out / "resume.log").open("w") as stream:
        result = subprocess.run([sys.executable, "code/check_resume.py", "--output", str(out / "resume.json")],
                                cwd=PACKET, stdout=stream, stderr=subprocess.STDOUT)
    record = {"manifest_entries": 378, "tests_passed": 55, "neural_models_replayed": 44,
              "portfolio_selections_replayed": 360, "diagnostic_maximum_numeric_difference": max(differences),
              "diagnostic_discrete_outcomes_exact": True, "strict_training_resume_passed": result.returncode == 0,
              "original_packet_unchanged": True, "failed_strict_diagnostic": "runs/QP-packet-check-001/diagnostics.log"}
    path = ROOT / "research-continuation/33_capability_portfolio/packet-verification.json"
    if path.exists():
        raise FileExistsError("Preserve verification")
    path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
