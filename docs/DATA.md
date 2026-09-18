# Teaching data

[Home](../README.md) · [Status](STATUS.md) · [Archive](RESEARCH_ARCHIVE.md)

I distinguish source prose, human annotations, engineered curricula and synthetic simulator evidence. A directory name is not a quality guarantee. New teaching material requires source identity, license, annotation checks and partitions that prevent evaluation leakage.

| Collection | Provenance and use |
|---|---|
| MASSIVE | Human-created and annotated multilingual requests; [publisher](https://github.com/alexa/massive), CC BY 4.0. Stages 26 and 27 document parallel semantic IDs and retention. |
| SQuAD 1.1 | Human-authored questions and answers over Wikipedia passages; [Stanford project](https://rajpurkar.github.io/SQuAD-explorer/) and [paper](https://aclanthology.org/D16-1264/), CC BY-SA 4.0. Selected for the new source-reading curriculum. |
| arXiv | Versioned mathematics and science papers acquired through the existing read-only adapter. The new paper library retains metadata, extraction quality and source links; retrieved statements retain their status as source claims until checked. [API](https://info.arxiv.org/help/api/user-manual.html) · [Bulk access](https://info.arxiv.org/help/bulk_data.html). |
| Local D-drive collections | Preserved books, historical texts and research extracts. Titles, duplication, extraction quality and document metadata require checking before admission. |
| Simulated motion and generated algebra practice | Earlier research evidence, explicitly labeled in its original reports and separate from the new human-text curriculum. |

The renewed local check found Project Gutenberg's *The Strange Case of Dr. Jekyll and Mr. Hyde* inside `data_physics`. The text has a misleading folder label; bulk ingestion by directory name would not establish a physics curriculum. Quarantined collections remain excluded.

The new source-reading curriculum admits no language-model-generated questions, answers or paraphrases. Human answer spans supply supervision. Passage segmentation and candidate construction are engineered preprocessing. Evaluation distinguishes supporting-sentence selection from exact answer extraction.
