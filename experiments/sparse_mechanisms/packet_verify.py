"""Run the reviewed packet verifier once against a disposable evidence copy."""

import json
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT/"research-continuation/00_sources/compressed-sensing-20260916/sera_cs_probe"
    target = ROOT/"runs/CS-packet-verification-001/evidence"
    shutil.copytree(source, target)
    sys.argv = [str(target/"verify.py"), "--results", str(target/"results")]
    try:
        runpy.run_path(str(target/"verify.py"), run_name="__main__")
    except BaseException as error:
        (target.parent/"failure.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
