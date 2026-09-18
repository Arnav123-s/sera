"""Restore the published owner with read-only replay of historical assessments."""

from experiments.published_receipts import historical_replay
from scripts import gap_use

perform = gap_use.perform


def restore():
    with historical_replay():
        return gap_use.restore()


if __name__ == "__main__":
    with historical_replay():
        gap_use.main()
