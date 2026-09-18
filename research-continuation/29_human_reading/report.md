# Human reading and targeted research

I taught a new semantic embedding on the actual retained StudyR1 owner in the requested order: lexicographer definitions/examples, human-authored reading questions, manual meeting transcripts and publisher-authored physics prose. This preliminary lexical stage used WordNet. The subsequent user request for full ordinary dictionaries, everyday conversation, grammar, philosophy and mathematics is a new curriculum; those books are being acquired separately.

## Measured learning

Eight-candidate source-association accuracy. The unchanged initial embedding and a lexical BM25 control use the same final candidates. These are association tasks with supplied alternatives.

| Domain | Final pairs | Starting weights | Selected learned weights | BM25 |
|---|---:|---:|---:|---:|
| dictionary | 400 | 30.50% | 42.25% | 51.75% |
| sentences | 400 | 45.25% | 72.50% | 95.75% |
| conversation | 400 | 22.25% | 27.00% | 32.75% |
| textbook | 32 | 21.88% | 31.25% | 40.62% |

The selected embedding completed all four stages. The full development and final matrices preserve retention changes after every stage. Conversation accuracy after its own stage was 31.5%; later textbook teaching changed it to 27.0%. This motivates rehearsal in a fresh protocol. The textbook final has 32 pairs from one chapter, so it does not support a between-chapter confidence interval.

## Question-to-source reading

| Frozen condition | Final sentence accuracy |
|---|---:|
| bm25 | 83.10% |
| lexical-2901 | 82.60% |
| lexical-2902 | 82.60% |
| shared-2901 | 80.00% |
| shared-2902 | 79.90% |

Development selected `lexical-2902`. Its final accuracy was 82.6% across 1,000 human questions from disjoint articles, versus 83.1% for BM25. The selected-minus-BM25 article-bootstrap interval is -1.42% to 0.56%. The prespecified two-percentage-point noninferiority gate passed; I do not claim a reading advantage over BM25. Shared-memory feature candidates remain preserved; their additional features were not selected.

## What changed in the learner

SERA acquired 393,216 semantic embedding parameters and trained a small residual sentence scorer. I supplied tokenization, word hashing, source pairing, objectives, controls, selection rules and the source-verification boundary. The semantic embedding is registered on the same owner used by the existing mathematics, language and empirical-world interfaces. Common ownership alone is not measured positive transfer.

The integration audit verified 165 predecessor tensors unchanged, identical 128 four-language development probes, retained exact polynomial results and identical conditional physical predictions. Empirical output lineage correctly names the new owner. Checkpoint continuation from step 280 to 420 reproduced the final weights exactly; restore, tamper rejection, cohort separation and interrupted-task retention passed.

## arXiv connection

A live missing-source request about compressed sensing triggered current arXiv metadata search, retained two source identities and used the saved reader to rank source sentences. One paper supplied accessible full HTML, including an MRI/clinical-practice discussion; the other retained its abstract after a full-text access failure. The original question, gap, URLs, versions, timestamps and hashes are persisted. New or open investigations refresh metadata; completed investigations preserve their pinned evidence.

The result status is `REFERENCES_TO_VERIFY`. Finding or ranking a statement does not certify it. No arXiv prose entered this human-only foundation training and this search made zero weight updates. The existing independently checked algebra-acquisition route remains available. [Live source receipt](arxiv-live.json).

## Sources, costs and preservation

Teaching: 12,000 WordNet definition/example pairs, 12,000 SQuAD pairs, 8,000 adjacent eligible AMI utterance pairs and 478 OpenStax exposition pairs. AMI pairs use the next eligible utterance after length filtering; plausible alternatives are not labeled as impossible. Textbook parsing omits equation-bearing paragraphs to avoid flattening mathematical notation.

[Sources](curriculum-sources.json) · [Original reading protocol](PROTOCOL.md) · [Prospective curriculum amendment](CURRICULUM.md) · [Audit](integration-audit.json) · [Costs](costs.json). Full original data and all checkpoints remain locally indexed; the compact published archive restores the selected runtime without replacing existing files.

Three retained engineering failures: HC-F01 stable-sort API mismatch before teaching; HR-F02 guard capture order during owner extension; HR-F03 audit comparison mistakenly included the intentionally changed owner identity. Each has its source, log, diagnosis and cost under `failures/`. A full local regression exceeded its 120-second reservation; its partial log remains preserved. Targeted regression and remote full verification are recorded separately.

## Next experiment

The next integration uses complete dictionary entries and grounded interpretation: propose meaning, bind entities/quantities, imagine conditional consequences, check independent human examples, preserve unresolved senses, and retain only supported updates. Text association is reported as text association; the definition-to-situation connection requires its own held-out evidence.
