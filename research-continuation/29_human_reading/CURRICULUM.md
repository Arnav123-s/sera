# HC-001: ordered human-text acquisition

This prospective amendment implements the user's dictionary, sentences, recorded conversation, then textbook order. It preserves the already prepared HR-001 SQuAD partitions and the original protocol. No HR-001 model has been trained and no final labels have been opened when this amendment is written.

## Sources and tasks

1. Princeton WordNet 3.0: human lexicographer headwords, definitions and example sentences. Use a definition/example to identify a headword set, preserving synset identity. Hash whole synsets into 80/10/10 percent train/development/final partitions. At most 12,000/400/400 pairs. A supplied lexical database is not an independently discovered concept inventory.
2. Original SQuAD 1.1: use the existing training questions and their human-answer-containing sentences for teaching; development questions for development. The official final file remains unopened until selection. This measures source association, not free-form answering.
3. AMI manual annotation release 1.6.2: reconstruct human utterances from manual segment and word identifiers. Order utterances by start time and pair adjacent utterances, requiring 4–96 words in each. Split entire meeting series (including a/b/c/d sessions) by hash into 80/10/10. At most 8,000/400/400 pairs. The target is the recorded next utterance; alternatives may also be conversationally plausible. No acoustic model is trained.
4. OpenStax Physics: pin the publisher's XML and license. Pair adjacent authored exposition paragraphs within a section, excluding exercises, solutions, figures and teacher notes. Split whole textbook chapters 80/10/10 by hash. At most 4,000/400/400 pairs. This is a prose-continuation association task; it does not certify physical derivations.

All text remains from the cited human sources. Tokenization, grouping, pairing, negative sampling and supervision objectives are engineered. Exact duplicate normalized pairs are removed; development/final pairs sharing either text with an earlier partition are excluded. Raw final sources may be downloaded but final cohort preparation/evaluation occurs only after immutable selection. Record all counts and exclusions.

## Finite mechanism and controls

Register a 8,192 by 48 semantic embedding on the actual retained ConceptStudyR1 object, initialized from its existing multilingual embedding. Preserve every predecessor tensor. Mean token embeddings with unit normalization form the new source-association route. Train this one embedding sequentially for 400 dictionary, 300 sentence, 300 conversation and 300 textbook updates, batch 64, AdamW learning rate .003, weight decay .0001, in-batch contrastive cross entropy at temperature .15. Identical target strings are not negatives. Use seed 2911 and preserve optimizer, shuffled order, cursor and RNG every 100 updates; optimizer resets at each declared stage.

After every stage, evaluate only development records. Each question receives its original target and seven deterministic alternatives from the same split/domain. Rank using normalized embedding similarity. Compare the unchanged initial embedding and lexical BM25; chance with eight candidates is 12.5%. Report each domain and retention after later stages. Select one completed embedding by highest mean development accuracy across all four domains, preferring the earlier snapshot on ties. Keep every stage regardless of selection. Final records are evaluated once for all frozen snapshots; no final adaptation or selection.

The selected embedding also supplies the existing R1's question/candidate inputs in HR-001. Add its cosine similarity as one extra feature. The lexical control zeros all neural features; the shared condition uses the actual memory and this learned semantic feature. Keep HR-001's four seeds/conditions, 420 updates, development selection and final gate. This prospective change is the only amendment to its model comparison. Measure shared versus lexical results; common ownership alone is not transfer.

## Integration, references and limits

Expose attributed sentence ranking and missing-knowledge investigation through the same owner. arXiv searches are bounded, read-only and cached with URL/version, acquisition time and SHA-256. A search result is reference evidence, not an accepted fact, a training label, a solved goal or an algebraic certificate. Preserve the original question and gap. Missing network access or unsupported interpretation leaves a resumable open goal. The existing independently checked algebra acquisition route remains available.

Verify actual shared owner identity, exact predecessor tensors, four-language development probes, polynomial certificates, empirical forecasts, exact checkpoint restore and source tamper rejection. Preserve costs and unsuccessful stages. The current numerical balance is 506.128673800049 seconds, with one thread and a 2 GiB process-tree cap; this amendment grants no additional resources.
