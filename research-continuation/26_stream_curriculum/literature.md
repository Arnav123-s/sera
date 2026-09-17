# Sources and mechanism choices

I used primary sources and the actual source bytes to select and delimit this cycle.

- [FitzGerald et al., MASSIVE](https://arxiv.org/html/2204.08582v2), introduction,
  collection/split descriptions and benchmark setup: real task-oriented utterances
  with intent and slot annotations. Intent accuracy, entity-span F1 and exact frames
  motivate the joint measurements here. The published pretrained-model scores use
  different training resources and are not comparisons with this SERA pilot.
- [Official MASSIVE repository](https://github.com/alexa/massive), dataset access,
  version note, record schema, licensing and attribution instructions: MASSIVE 1.1
  adds Catalan; English data are unchanged from 1.0. Both MASSIVE and its English
  seed source [SLURP, Bastianelli et al.](https://aclanthology.org/2020.emnlp-main.588/)
  receive attribution. The SLURP page is cited for provenance, not claimed as a
  separately reviewed method in this cycle.
- [Hugging Face dataset card](https://huggingface.co/datasets/AmazonScience/massive),
  schema/split/license information. The live Dataset Viewer reports unsupported
  dataset-script loading. I inspected the Hub metadata at commit
  `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740` and used original publisher bytes instead.
  I did not execute the remote dataset script.
- [Chaudhry et al., On Tiny Episodic Memories in Continual Learning](https://arxiv.org/html/1902.10486v4),
  abstract and sections 3–4: the current minibatch is combined with a sampled
  episodic memory, with reservoir sampling as one memory policy. I read the methods
  before SC-004 and used its task-matrix accuracy and forgetting definitions. This
  study uses repeated passes within each block and supplied NLU labels; the paper's
  single-pass experiments are not reproduced. SC-004 compares equal current-example
  access and equal total batch presentations, charging memory and bookkeeping.
- [Buzzega et al., Dark Experience for General Continual Learning](https://arxiv.org/abs/2004.07211),
  abstract: logit replay/distillation is a competing future mechanism. Its extra
  storage and teacher cost would need accounting separately from ordinary replay.

The full archive was 40,251,390 compressed bytes. Intake read it as a bounded
stream and retained only the English member and license, plus split files.
The license was encountered at the end, so this particular retrieval read the
whole compressed archive. The initial metadata incorrectly marked it partial;
the original metadata and an explicit correction are retained. Training itself
uses a bounded 128-record shuffle buffer with a saved byte cursor and RNG.

The public-source teaching labels supply semantic categories; SERA learns the
mapping from request text to those labels. The retrieval, schema, optimizer,
streaming machinery and checkpoint implementation are engineering contributions.
New interface weights and their measured request predictions are learned results.
