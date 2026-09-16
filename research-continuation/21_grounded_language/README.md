# Grounded language and task-time reacquisition

I added a qualified finite mathematical-language interface to the persistent SERA workbench. It can learn missing wording from supplied examples, check its interpretation, execute the task, withdraw weakened wording and relearn it on a later request. The existing data tasks, continuing control and verified finite programs remain available.

Start `scripts/start_workbench.ps1` and open [Verified reasoning](http://127.0.0.1:8765/?panel=reason). The existing [numerical task interface](http://127.0.0.1:8765/?panel=learn) accepts measured examples; [live data tasks](http://127.0.0.1:8765/?panel=data) update when their inbox CSV changes.

Try these mathematical requests in sequence:

1. `scale x by three then subtract two to get four modulo eleven`
2. `the product of x and three minus two equals four modulo eleven`
3. `subtract three from x then multiply by two to get four modulo eleven`

The first two should return `x = 2`; the third should return `x = 5`. A missing or weakened wording pattern triggers a local lesson. These are supplied finite sentence and arithmetic families, not unrestricted English. The original generic numerical task interface remains the practical route for new calibration and transformation tasks from your own examples.

Read the [complete results](report.md), [source-architecture audit](architecture-audit.md), [execution checklist](PLAN.md), [failure record](failures.md), [independent prediction audit](independent-audit.json), [qualification decision](qualification.json), and [corrective output-contract check](CONTRACT-001/result.json).

The [resumption record](resume.md) identifies the exact live owner, checkpoint dependencies and read-only verification command. This cycle used 36.78 minutes of supervised numerical work and leaves 24.58 minutes in the cumulative allowance at release. Later task requests use that live balance.

The nine prospective models and their eighteen targeted lessons are under `L10-FINAL-*`. The six-request continuing learner is under `L11-ONREQUEST-001`; it passed with five actual lessons and seven preserved revisions. Local neural/optimizer checkpoints remain under `runs/grounded-language/`, and application histories under `runs/sera-workbench/`. Checkpoint identities, sources, seeds and raw predictions are retained in the release records.

No pretrained language model, paid service, remote job or repository reset was used. General language acquisition, research-paper comprehension, proof invention and independently improved learning procedures remain open goals.
