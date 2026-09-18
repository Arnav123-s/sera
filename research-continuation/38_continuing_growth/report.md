# Sustained learning across twelve strands

I continued the actual Stage 37 owner through **6,144 decisions and 393,216 main practice presentations**. The selected successor is **self_directed-3802**, owner `11653a31b45b36be4d02a81c1a0d2aae5a30561ea9b4a305e0dc617a4751dd81`, saved in `runs/sera-growth-live`. It passed the frozen knowledge-admission gate.

The curriculum used retained human dictionary entries, conversations, grammar, philosophy, mathematics/science prose, source-reading annotations and four-language requests, plus independently checked conditional mechanics. No web answer retrieval or new task-specific hints entered the learner loop. Human labels, representations, action families and checkers are supplied resources. The learner updated its actual registered readout and procedure weights.

## Locked final results

| Strand | Parent correct | Successor correct | Cases | Parent loss | Successor loss |
|---|---:|---:|---:|---:|---:|
| dictionary | 45 | 58 | 192 | 4.27615 | 3.94748 |
| conversation | 26 | 26 | 192 | 4.52532 | 4.43817 |
| grammar | 34 | 40 | 192 | 4.21812 | 4.07272 |
| philosophy | 28 | 28 | 192 | 4.39931 | 4.26609 |
| mathematics | 49 | 50 | 166 | 3.13815 | 3.06415 |
| science | 64 | 60 | 192 | 2.80343 | 2.85078 |
| reading | 151 | 152 | 192 | 0.70293 | 0.70253 |
| en-US | 175 | 175 | 192 | 0.29272 | 0.32667 |
| es-ES | 181 | 178 | 192 | 0.27382 | 0.30710 |
| fr-FR | 180 | 179 | 192 | 0.27427 | 0.29492 |
| de-DE | 186 | 186 | 192 | 0.21467 | 0.21749 |
| methods | 36 | 45 | 192 | 2.48509 | 1.75326 |

Dictionary, conversation, grammar, philosophy, mathematics and science measure next-token prediction in retained human prose. Reading measures selection of a human-supported source sentence. Language rows measure request intent. Methods measure choice among supplied conditional mechanics procedures. These are distinct capabilities; the table is not a single measure of understanding. Prose accuracy includes the existing unknown-token class.

The final source groups were held back from this continuation, with translated requests kept in the same partition. The inherited parent had earlier exposure to the original training corpora. Existing completed final files remained closed. Source identities, exact groups and row IDs are in the archived data manifest.

## All matched main runs

| Controller / seed | Development before | Development after | Final macro | Accepted / attempted decisions |
|---|---:|---:|---:|---:|
| balanced-3801 | 0.449158 | 0.455741 | 0.446779 | 1524 / 1536 |
| self_directed-3801 | 0.449158 | 0.456820 | 0.444823 | 1528 / 1536 |
| balanced-3802 | 0.449158 | 0.456479 | 0.447218 | 1526 / 1536 |
| self_directed-3802 | 0.449158 | 0.457026 | 0.445148 | 1512 / 1536 |

Development alone selected the terminal checkpoint. No final result selected a different arm. Both procedures used four minibatches of sixteen examples per decision; balanced used the strongest supplied fixed rate. Rejected trials, weight restoration and their costs remain in every checkpoint.

## Learning procedure and independent checks

Twelve separate future-practice trials crossed old/new knowledge with old/new procedure and balanced scheduling, using 96 decisions and 6,144 presentations each. The learned policy did not satisfy the predeclared all-seed/all-knowledge superiority gate; **balanced remains the default**. Both learned policies and every trial are retained. Better continued knowledge is recorded separately from improvement in the procedure used to learn it.

Independent NumPy replay checked **24,722 prediction items**, with maximum loss difference 6.41e-07. Both 128-decision continuation segments reproduced exactly. **208 protected tensor records** and **64/64 fresh integral/summation cases** were retained. The novel-definition gate stayed closed.

The selected weights include useful improvements and small measured regressions; every strand is shown above. Development trajectories motivate a separately frozen successor that adds rehearsal and an original-accuracy floor. That successor must use its own held-back evidence.

## Preservation and use

[Use the saved owner](README.md) · [Finite protocol](PROTOCOL.md) · [Architecture audit](architecture-audit.md) · [Final measurements](final.json) · [Future procedure comparisons](procedure.json) · [Independent audit](audit.json) · [Costs](costs.json).

All predecessor stores, original source files, failed attempts, checkpoints and unfinished goals remain intact. Old signed portfolio qualifications remain associated with their original owner.
