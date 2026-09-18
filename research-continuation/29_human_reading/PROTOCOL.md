# HR-001: human-authored source reading

Frozen before dataset preparation, training or evaluation. Parent release: `85a8bdde6f73cf82415e52c2afffae6eaa38f9ab`; current navigation-only successor preserves its models. Restore the actual Stage 28 empirical owner and register the reading weights on that object. All predecessor tensors stay fixed. No new recurrent component, external language model, arXiv teaching, generated prose or generated questions are used.

## Teaching and isolation

Use the original SQuAD 1.1 JSON files identified in sources.json. Questions and answer spans were written by human annotators over Wikipedia passages. Segment original passages without rewriting them. Require exact answer-text/offset alignment; 2–16 candidate sentences; context length at most 6,000 characters; and all accepted target spans wholly inside one candidate. Reject malformed records, blank questions, duplicate normalized question/context pairs and ambiguous cross-sentence target mappings, recording every count. The task is supporting-sentence selection, not exact-answer extraction.

Hash article titles with `HR-001:article:`. Residue zero modulo five from the official training articles is development; all other articles are teaching. Deterministically order questions by SHA-256 of their original ID. Use at most 12,000 teaching and 1,000 development questions, with at most 60 questions per article. Official development is the once-only final cohort, capped at 1,000 questions and 60 per article; exclude any article title or normalized passage occurring in teaching/development. Final labels stay closed until a selection record exists. No completed older final bank is reopened.

## Mechanisms and fixed training

Compare a supplied BM25 sentence scorer with two learned residual rankers, each with seeds 2901 and 2902. Both use identical lexical/position/question-form features and a 48-unit tanh head. The shared condition additionally uses 32 channels from the existing R1 after a pooled question and candidate sentence are passed through its actual memory. The lexical control zeros those channels. Existing text embeddings, projection, memory and all old weights stay frozen; common ownership alone is not a positive-transfer result.

Each fit receives 420 AdamW updates, batch size 32 questions, learning rate .002 and weight decay .0001. Human answer-containing sentence indices supply cross-entropy supervision. Baseline BM25 supplies initial logits, with a zero-initialized last residual layer. No hyperparameter search or final-data adaptation. Preserve RNG, optimizer state, source identities and the data cursor in exact checkpoints every 140 updates.

Select by development accuracy, preferring BM25 on ties. Report question-level accuracy, article-macro accuracy, reciprocal rank and article-cluster bootstrap intervals. Final admission of a selected trained ranker requires final accuracy at least baseline minus .02, exact old-tensor retention and persistence/evidence checks. Report a learning advantage only if measured; selection and the noninferiority gate do not establish broad reasoning or independent concept discovery.

## Integration and verification

Read an attributed local passage, rank original source sentences, and return source offsets/hash and an explicit candidate-support status. Ranking scores are not correctness probabilities. Changing a document changes its hash; prior predictions retain their old evidence identity. Imagined/retrieved sentences do not become observed events or algebraic certificates. Reuse the existing applicability boundary.

Verify the trained checkpoint, actual shared object identity, original tensor hashes, four-language development probes, retained exact polynomial calculations and the Stage 28 saved empirical forecast. Independently recount final predictions and bootstrap over articles. Test label isolation, input validation, tamper rejection and exact local restore. Keep all failed jobs and revisions.

Use only the existing one-thread, 2 GiB local allowance. The initial balance is 512.0372116000653 seconds, not a new grant. If a genuine limit interrupts training, retain the exact next update and optimizer/RNG state. Network acquisition time is recorded separately. CI runs are regression verification, not uncharged training.
