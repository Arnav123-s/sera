# Learning a procedure from actual outcomes

The current fixed numerical selector chooses a supplied update window using past measured prediction errors. Accumulating observations improves K; it does not independently establish learned eta.

[Learning Active Learning from Data](https://arxiv.org/abs/1703.03365) motivates fitting a decision rule from measured consequences of learning choices. Here I test a much narrower task: regress the subsequent observed loss of a numerical update method from information already available when choosing it. This implementation does not reproduce that paper or inherit its empirical results.

[Second-Order Non-Stationary Online Learning for Regression](https://www.jmlr.org/beta/papers/v16/moroshko15a.html) addresses changing predictors and adaptive forgetting. It supports using recency-sensitive controls and measuring performance under changes. No regret bound from that work applies automatically to this implementation.

The candidate is an eight-parameter ridge regressor over supplied features and update choices. Measured future prediction errors become training labels only after the observations arrive. An independently trained initial eta and later observation-trained successor etas remain separate from the retained numerical history K. Fresh final streams are never used for human-directed parameter selection. Confidence intervals cluster repeated forecasts by independent lifetime.

This remains a procedure-selection component, not unrestricted procedure synthesis, learning new physics, or general autonomous investigation. A failure against the fixed observed-error selector must leave that stronger control in the application.
