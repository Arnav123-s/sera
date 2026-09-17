"""Run the inspected v10 verifier within the project's resource supervisor."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    packet = ROOT / "research/intake/v6-v10-20260916/SERA_v10"
    output = ROOT / "research-continuation/25_constraint_inquiry/v10-replay"
    subprocess.run([sys.executable, "code/run.py", "--output", str(output),
                    "--max-jobs", "2"], cwd=packet, check=True)
    subprocess.run([sys.executable, "code/run.py", "--output", str(output),
                    "--max-jobs", "5"], cwd=packet, check=True)


if __name__ == "__main__":
    main()
