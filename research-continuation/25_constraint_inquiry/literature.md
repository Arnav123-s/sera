# Primary research and mechanism decisions

I reviewed primary abstracts and the relevant method sections below. This is a
targeted review, not a claim to have reproduced these papers' experiments.

- [Du, Li and Mordatch: compositional energy-based generation](https://arxiv.org/html/2004.06030v3).
  Learned energies can support compositional inference through numerical sampling.
  This motivates a language-selected compatibility objective; it does not provide
  SERA's semantics. My local objective is an engineered distance constraint around
  SERA's acquired dynamics, not this paper's visual generative training.
- [Scellier and Bengio: Equilibrium Propagation](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full).
  The method separates free and nudged equilibria and relates their difference to
  parameter gradients under regularity and stability conditions. That makes EqProp
  a candidate update mechanism. The current interfaces use ordinary Adam; numerical
  search over controls is not EqProp training. I read the introduction, fixed-point
  contract and theorem discussion; implementing C05 remains distinct work.
- [Han et al.: matrix-product-state generative modeling](https://arxiv.org/html/1709.01662v3).
  Tensor contraction supports tractable normalization and conditional sampling
  within the chosen representation. The learning and sampling sections motivate
  C06, but do not establish that a small bond retains every relevant alternative.
  The present route preserves seven explicit static parameter branches and makes
  no tensor compression claim.
- [Plenio and Huelga: dephasing-assisted transport](https://arxiv.org/abs/0807.4902).
  The primary abstract explains why local noise can assist transport in particular
  dissipative networks. This supports keeping mixed coherent/dissipative controllers
  as challengers rather than assuming maximal coherence is optimal. Here only the
  supplied v10 finite physics evidence was replayed; no new quantum controller was
  trained on SERA's task.
- [Deffner and Campbell: quantum speed limits](https://arxiv.org/abs/1705.08023).
  The abstract concerns limits on physical state evolution. It does not establish
  a computational advantage for simulating a cloud on this CPU.

A mistyped arXiv identifier during source lookup resolved to an unrelated ATLAS
paper and was discarded. No claim relies on that source.

My engineering inference: first make role selection, learned initialization,
actual acquired dynamics, branch persistence and evidence revision work together.
Then a recurrent energy learner, a learned correlated density or a mixed controller
has an actual downstream interface on which to compete. CI-001's grid and analytic
controls remain essential: fewer candidate points did not imply faster inference.
CI-002 subsequently tested a fixed early stopping rule on fresh numeric cases.
