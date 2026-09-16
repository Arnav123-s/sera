"""Use the existing cumulative local allowance for the sparse challenger."""

import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.sparse_mechanisms.packet_verify", "experiments.sparse_mechanisms.study",
                      "experiments.sparse_mechanisms.audit", "experiments.sparse_mechanisms.investigate",
                      "experiments.sparse_mechanisms.variants",
                      "experiments.sparse_mechanisms.imaging", "experiments.sparse_mechanisms.audit_variants",
                      "experiments.sparse_mechanisms.audit_queries", "experiments.sparse_mechanisms.repair",
                      "experiments.sparse_mechanisms.report", "experiments.sparse_mechanisms.owner_weights",
                      "experiments.sparse_mechanisms.audit_owner", "pytest"}

if __name__ == "__main__":
    supervisor.main()
