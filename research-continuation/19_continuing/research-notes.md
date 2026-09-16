# Primary sources and implementation decisions

I reviewed 13 search results across three focused searches and fetched two primary pages. Duplicate versions of the same papers were not counted as independent scientific evidence.

- [Chua et al., PETS](https://arxiv.org/abs/1805.12114) motivates propagating dynamics uncertainty through hypothetical trajectories and using receding-horizon control. This implementation uses a tiny Bayesian linear action model and deterministic quadrature, not PETS' bootstrapped neural ensemble. Published benchmark gains are not evidence for this SERA component.
- [Klenske and Hennig, Dual Control for Approximate Bayesian Reinforcement Learning](https://www.jmlr.org/papers/v17/15-162.html) separates adaptive control from reasoning about the information value of future observations. The current controller updates from real measurements but does not explicitly optimize future belief changes. It must be described as adaptive model predictive control, not a learned or Bayes-optimal investigator.

The local causal question is whether conditional consequences improve action selection at the declared access/resource ceiling, beyond a strong reactive controller and a one-step predictor using the same learned dynamics procedure. An independently improved eta would require a separate old/new K by old/new eta intervention and future unfamiliar tasks. No such claim is made here.
