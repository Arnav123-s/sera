"""Once-only replay of the inspected, unchanged v11 laboratory."""
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "research/intake/v11-20260917/SERA_v11"
OUT = ROOT / "research-continuation/27_self_study/v11-verification"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(PACKET / "src"))
    for script, args in (("verify_manifest", []),
                         ("verify_results", ["--out", str(OUT / "replay.json")]),
                         ("demo", ["--out", str(OUT / "demo.json")])):
        sys.argv = [script, *args]
        runpy.run_path(str(PACKET / "src" / f"{script}.py"), run_name="__main__")
    import pytest
    code = pytest.main(["-q", str(PACKET / "tests"), "--override-ini", "addopts="])
    (OUT / "tests.json").write_text(json.dumps({"exit_code": code}), encoding="utf-8")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
