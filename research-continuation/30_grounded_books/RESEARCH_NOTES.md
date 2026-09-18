# Mechanism choice

I compared definition-trained representations, lexical classification, and existing recurrent state as competing mechanisms. [Hill et al. (2016)](https://aclanthology.org/Q16-1002/) train dictionary definitions toward word representations and compare recurrent and bag-of-words composition. That motivates a definition-to-type test; it does not establish executable physical meaning. [Tino (2020)](https://www.jmlr.org/beta/papers/v21/19-589.html) analyzes the temporal features induced by fixed recurrent dynamics. That supports measuring what the retained memory contributes rather than presuming that recurrence guarantees useful language features.

The new experiment adds independent conditional execution to the learned binding. The unit vocabulary, term labels, kinematic scope and verifier are supplied engineering. The learned quantities are readout weights fitted to original human text. Training a shared owner is distinct from demonstrating positive transfer, which is decided by the controls and retention tests.

The first chronological next-word run selected the ordered readout on development data. Its final loss remained above the unigram control, so I retain it as an experimental checkpoint rather than claiming a successful general language model. Its final cohort is closed. Grounding uses separately declared examples and endpoints.
