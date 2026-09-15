"""Extend the preserved v3 supervisor's allowlist; use its existing shared ledger."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"pytest", "experiments.cross_route_transfer.study",
                      "experiments.cross_route_transfer.audit"}

if __name__ == "__main__":
    supervisor.main()
