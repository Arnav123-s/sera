"""The current qualified SERA task interface; historical interfaces stay available."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.intervention_use import main

if __name__ == "__main__":
    main()
