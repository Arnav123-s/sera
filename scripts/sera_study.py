"""Run a persistent, independently checked SERA study or calculation."""
import argparse
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("study", "solve", "motion", "status", "interpret"))
    p.add_argument("--store", default="runs/sera-study-live")
    p.add_argument("--id", default="new-goal")
    p.add_argument("--domain", choices=("sum", "integral"), default="sum")
    p.add_argument("--coefficients", default="0,2,0,3,0,1", help="constant-first rational polynomial coefficients")
    p.add_argument("--time", default="3")
    p.add_argument("--position", default="0")
    p.add_argument("--velocity", default="0")
    p.add_argument("--text")
    p.add_argument("--seconds", type=float, default=45)
    args = p.parse_args()
    extra = [args.action]
    for key in ("store", "id", "domain", "coefficients", "time", "position", "velocity", "text"):
        value = getattr(args, key)
        if value is not None:
            extra.append("--" + key + "=" + value)
    folder = ROOT / "runs" / ("study-command-" + str(uuid.uuid4()))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_self_study_bounded.py"),
                             "--seconds", str(args.seconds), "--output", str(folder),
                             "--module", "experiments.self_study.runtime", "--", *extra],
                            cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    log = folder / "process.log"
    print(log.read_text(encoding="utf-8") if log.exists() else result.stdout + result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
