# Teaching data

[Home](../README.md) · [Status](STATUS.md) · [Archive](RESEARCH_ARCHIVE.md)

I distinguish source prose, human annotations, engineered curricula and synthetic simulator evidence. A directory name is not a quality guarantee. New teaching material requires source identity, license, annotation checks and partitions that prevent evaluation leakage.

| Collection | Provenance and use |
|---|---|
| MASSIVE | Human-created and annotated multilingual requests; [publisher](https://github.com/alexa/massive), CC BY 4.0. Stages 26 and 27 document parallel semantic IDs and retention. |
| SQuAD 1.1 | Human-authored questions and answers over Wikipedia passages; [Stanford project](https://rajpurkar.github.io/SQuAD-explorer/) and [paper](https://aclanthology.org/D16-1264/), CC BY-SA 4.0. Selected for the new source-reading curriculum. |
| arXiv | Earlier source-acquisition evidence remains preserved. Targeted missing-knowledge searches are enabled with source hashes and dates. Retrieved papers remain reference evidence, separate from human-only foundation teaching. |
| Local D-drive collections | Preserved books, historical texts and research extracts. Titles, duplication, extraction quality and document metadata require checking before admission. |
| Simulated motion and generated algebra practice | Earlier research evidence, explicitly labeled in its original reports and separate from the new human-text curriculum. |

The renewed local check found Project Gutenberg's *The Strange Case of Dr. Jekyll and Mr. Hyde* inside `data_physics`. The text has a misleading folder label; bulk ingestion by directory name would not establish a physics curriculum. Quarantined collections remain excluded.

The new source-reading curriculum admits no language-model-generated questions, answers or paraphrases, and no new arXiv teaching material. Human answer spans supply supervision. Passage segmentation and candidate construction are engineered preprocessing. Evaluation distinguishes supporting-sentence selection from exact answer extraction. Human authorship is supported by the original dataset collection record; source quality is checked separately from that provenance claim.

The completed preliminary curriculum adds Princeton WordNet 3.0 lexicographer glosses, the AMI manual annotation release 1.6.2 (CC BY 4.0), and OpenStax *Physics* publisher XML (CC BY 4.0). WordNet is a lexical database; its association experiment is not a substitute for complete dictionary instruction. [Exact sources and selected counts](../research-continuation/29_human_reading/report.md).

The completed book intake contains *Webster's Unabridged Dictionary*, Baskervill and Sewell's *An English Grammar*, Plato's *Republic* (Jowett), Descartes' *Discourse on Method* (Veitch), Thompson's *Calculus Made Easy*, and original DailyDialog human-written everyday dialogues from the ACL supplementary archive. Raw books preserve pronunciation, grammatical labels, senses, usage and attribution. Automatically generated catalogue summaries are excluded. The bounded curriculum used 10,723 original training windows. [Counts and measured results](../research-continuation/30_grounded_books/report.md) · [Source identities](../research-continuation/30_grounded_books/sources.json).

The new grounded-definition bank uses original Webster sense spans and OpenStax *Physics* meanings with supplied type labels. A separately pinned *College Physics 2e* cohort supplies transfer evaluation. [Source-specific attribution and license notes](../research-continuation/30_grounded_books/ATTRIBUTION.md) distinguish the CC BY teaching content, CC BY-NC-SA evaluation source and unresolved DailyDialog archive license. Raw dialogue content remains local. Generated rational motion cases are diagnostic tests and do not count as human training data.
