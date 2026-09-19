"""Portable entry point for the unchanged, independently qualified IU-001 owner."""

from experiments.intervention_portability import installed
from scripts import intervention_use as original
from scripts.intervention_study import restore as strict_restore


def restore():
    with installed():
        return strict_restore()


def main():
    previous = original.restore
    original.restore = restore
    try:
        original.main()
    finally:
        original.restore = previous


if __name__ == "__main__":
    main()
