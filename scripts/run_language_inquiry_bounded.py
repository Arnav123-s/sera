"""Charge language/inquiry work to the existing one-thread, 2 GiB allowance."""
import run_v3_bounded as supervisor

supervisor.ALLOWED = {"experiments.language_inquiry.packet_verify",
                      "experiments.language_inquiry.study",
                      "experiments.language_inquiry.runtime",
                      "experiments.language_inquiry.evaluate",
                      "experiments.language_inquiry.audit", "pytest", "ruff", "sera"}

if __name__ == "__main__":
    supervisor.main()
