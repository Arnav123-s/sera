"""Use the existing shared numerical allowance for the self-study continuation."""
import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.self_study.packet_verify", "experiments.self_study.study",
                      "experiments.self_study.runtime", "experiments.self_study.audit",
                      "experiments.self_study.multilingual", "experiments.self_study.correction",
                      "experiments.self_study.rehearsal", "pytest", "ruff"}
supervisor.ALLOWED.add("experiments.self_study.plot")

if __name__ == "__main__":
    supervisor.main()
